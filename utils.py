# utils.py
import time, os, math
import logging
import requests
from web3 import Web3
from decimal import Decimal
from config import w3, router, usdt, wmatic, OWNER, PRIVATE_KEY, ROUTER_ADDR, USDT_ADDR, WMATIC_ADDR

from dataclasses import dataclass, field
from typing import List, Dict, Optional

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from web3.middleware import geth_poa_middleware
from dotenv import load_dotenv

load_dotenv()

# ---------- Constants ----------
MAX_UINT = 2**256 - 1
GAS_LIMIT_APPROVE = 100_000
GAS_LIMIT_SWAP = 600_000
GAS_PRICE_LIMIT = 300 * (10**9)  # 300 gwei cap (skip tx if higher)

# ---------- Minimal ABIs ----------
ERC20_ABI = [
    {
        "constant": True,
        "inputs": [],
        "name": "name",
        "outputs": [{"name": "", "type": "string"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [],
        "name": "symbol",
        "outputs": [{"name": "", "type": "string"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [],
        "name": "totalSupply",
        "outputs": [{"name": "", "type": "uint256"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function",
    },
    {
        "constant": False,
        "inputs": [{"name": "_to", "type": "address"}, {"name": "_value", "type": "uint256"}],
        "name": "transfer",
        "outputs": [{"name": "success", "type": "bool"}],
        "type": "function",
    },
    {
        "constant": False,
        "inputs": [
            {"name": "_from", "type": "address"},
            {"name": "_to", "type": "address"},
            {"name": "_value", "type": "uint256"},
        ],
        "name": "transferFrom",
        "outputs": [{"name": "success", "type": "bool"}],
        "type": "function",
    },
    {
        "constant": False,
        "inputs": [
            {"name": "_spender", "type": "address"},
            {"name": "_value", "type": "uint256"},
        ],
        "name": "approve",
        "outputs": [{"name": "success", "type": "bool"}],
        "type": "function",
    },
    {
        "constant": True,
        "inputs": [
            {"name": "_owner", "type": "address"},
            {"name": "_spender", "type": "address"},
        ],
        "name": "allowance",
        "outputs": [{"name": "remaining", "type": "uint256"}],
        "type": "function",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "name": "owner", "type": "address"},
            {"indexed": True, "name": "spender", "type": "address"},
            {"indexed": False, "name": "value", "type": "uint256"},
        ],
        "name": "Approval",
        "type": "event",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "name": "from", "type": "address"},
            {"indexed": True, "name": "to", "type": "address"},
            {"indexed": False, "name": "value", "type": "uint256"},
        ],
        "name": "Transfer",
        "type": "event",
    },
]

# ---------- Utility helpers ----------
def to_decimals(amount: float, decimals: int) -> int:
    return int(Decimal(amount) * (10 ** decimals))

def from_decimals(amount_int: int, decimals: int) -> float:
    return float(Decimal(amount_int) / (10 ** decimals))

def get_token_decimals(token_contract):
    try:
        return token_contract.functions.decimals().call()
    except Exception:
        return 18

def get_nonce():
    """Track pending transactions to avoid nonce conflicts"""
    return w3.eth.get_transaction_count(OWNER, 'pending')

def get_safe_gas_price(multiplier=1.20, max_gwei=300, min_gwei=30):
    base = w3.eth.gas_price
    boosted = int(base * multiplier)
    boosted = max(boosted, Web3.to_wei(min_gwei, 'gwei'))  # avoid tx stuck at 1 gwei
    return min(boosted, Web3.to_wei(max_gwei, 'gwei'))

def gas_params():
    """Dynamic gas settings with ceiling control"""
    g = get_safe_gas_price()
    if g > GAS_PRICE_LIMIT:
        logging.warning(f"⚠️ Gas too high ({g/1e9:.1f} gwei), skipping transaction attempt.")
        return None
    return {"gasPrice": g, "chainId": w3.eth.chain_id}

def get_allowance(token_contract, owner, spender):
    """Check token allowance for router"""
    try:
        allowance = token_contract.functions.allowance(owner, spender).call()
        return allowance
    except Exception as e:
        logging.warning(f"Failed to check allowance: {e}")
        return 0

# ---------- Approve helper (expects amount in raw units, i.e. "wei/uint") ----------
def approve_if_needed(token_contract, spender, amount_wei):
    """
    Ensure router has allowance >= amount_wei for token_contract.
    - token_contract: web3 Contract (ERC20)
    - spender: router address
    - amount_wei: amount in token's smallest unit (int)
    Returns True on success, False on failure.
    """
    try:
        allowance = token_contract.functions.allowance(OWNER, spender).call()
        if allowance >= int(amount_wei):
            logging.info(f"✅ Sufficient allowance ({allowance}) — no approval needed.")
            return True

        logging.info(f"🔐 Approving router {spender} to spend token {token_contract.address} (MAX_UINT).")
        tx = token_contract.functions.approve(spender, MAX_UINT).build_transaction({
            "from": OWNER,
            "nonce": get_nonce(),
            "gas": GAS_LIMIT_APPROVE,
            **(gas_params() or {})
        })

        tx_hash = send_tx(tx)
        if not tx_hash:
            logging.warning("⚠️ Approval transaction failed or skipped.")
            return False

        # wait for receipt (best-effort)
        try:
            w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
        except Exception:
            logging.warning("⚠️ Approval tx did not confirm quickly; continuing and relying on on-chain state check later.")
        # verify allowance again
        new_allow = token_contract.functions.allowance(OWNER, spender).call()
        if new_allow >= int(amount_wei):
            logging.info("✅ Approval confirmed on-chain.")
            return True
        logging.error("❌ Approval not reflected on-chain after tx.")
        return False

    except Exception as exc:
        logging.exception("❌ approve_if_needed() error:")
        return False


# ---------- Safe swap with proper input-token approval & retry ----------
def safe_swap_exact_tokens_for_tokens(amount_in, amount_out_min, path, to, deadline):
    """
    Swap with:
     - dynamic approval of path[0] token,
     - gas ceiling guard via gas_params(),
     - up to 3 attempts with gas bump.
    Expects amount_in to be integer (raw units).
    """
    MAX_ATTEMPTS = 3
    GAS_BUMP = 1.20  # 20% per retry

    # quick gas guard
    if gas_params() is None:
        logging.warning("⏳ Gas too high — skipping swap.")
        return None

    # build contract object for input token (path[0])
    input_addr = path[0]
    input_token = w3.eth.contract(address=input_addr, abi=ERC20_ABI)

    # Ensure approval of input token for router
    ok = approve_if_needed(input_token, ROUTER_ADDR, int(amount_in))
    if not ok:
        logging.error("❌ Approval for input token failed, aborting swap.")
        return None

    # Attempt the swap, increasing gas each retry
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            params = gas_params()
            if params is None:
                logging.warning("⏳ Gas too high now — aborting swap attempt.")
                return None

            # apply gas bump for retry attempts
            if attempt > 1:
                # multiply gasPrice from fresh gas_params() snapshot to avoid stale values
                params["gasPrice"] = int(params["gasPrice"] * (GAS_BUMP ** (attempt - 1)))

            tx = router.functions.swapExactTokensForTokens(
                int(amount_in),
                int(amount_out_min),
                path,
                to,
                int(deadline)
            ).build_transaction({
                "from": OWNER,
                "nonce": get_nonce(),
                "gas": GAS_LIMIT_SWAP,
                **params
            })

            tx_hash = send_tx(tx)
            if tx_hash:
                logging.info(f"✅ Swap succeeded (attempt {attempt}): {tx_hash}")
                return tx_hash

            logging.warning(f"⚠️ swap call returned None on attempt {attempt}, retrying...")

        except Exception as e:
            # inspect error string to decide whether to retry
            em = str(e).lower()
            logging.warning(f"⚠️ Swap attempt {attempt} raised: {e}")
            retryable = False
            for token in ("underpriced", "replacement transaction", "nonce", "fee too low", "max fee per gas", "transfer_from_failed", "execution reverted"):
                if token in em:
                    retryable = True
                    break

            if not retryable:
                logging.error("❌ Non-retryable swap error, aborting.", exc_info=True)
                return None

            # retryable -> small sleep then continue to next attempt
            time.sleep(1 + attempt)
            continue

    logging.error("❌ Swap failed after max attempts.")
    return None


# ---------- send_tx with retry-on-underpriced / gas-bump ----------
def send_tx(tx, max_retries=2, gas_bump=1.4):
    """
    Sign & broadcast tx. If node rejects as underpriced, attempt a retry with bumped gasPrice.
    Returns tx_hash hex string or None.
    """
    # initial gas check
    params = gas_params()
    if params is None:
        logging.warning("⏳ Transaction skipped due to high gas.")
        return None

    # merge params into tx copy so we don't mutate caller's object unexpectedly
    tx_local = tx.copy()
    tx_local.update(params)

    # sign
    signed = w3.eth.account.sign_transaction(tx_local, private_key=PRIVATE_KEY)
    raw = getattr(signed, "raw_transaction", getattr(signed, "rawTransaction", None))
    if raw is None:
        raise AttributeError("Web3 signed transaction missing raw transaction field.")

    try:
        tx_hash = w3.eth.send_raw_transaction(raw)
        logging.info(f"✅ TX sent: {tx_hash.hex()}")
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        logging.info(f"🧾 TX confirmed in block {receipt.blockNumber}")
        return tx_hash.hex()

    except ValueError as e:
        err_s = str(e).lower()
        retryable = any(k in err_s for k in ["underpriced", "replacement transaction", "fee too low", "max fee per gas"])
        if not retryable:
            logging.error(f"❌ send_tx ValueError (not retryable): {e}")
            return None

        logging.warning("⚠️ send_tx detected underpriced/replacement error — attempting retries with bumped gasPrice.")
        for i in range(1, max_retries + 1):
            try:
                new_params = gas_params()
                if new_params is None:
                    logging.warning("⏳ Gas too high for retry; aborting.")
                    return None
                new_params["gasPrice"] = int(new_params["gasPrice"] * (gas_bump ** i))
                # update tx_local with bumped gas and fresh nonce
                tx_local.update({"nonce": get_nonce(), **new_params})
                signed_retry = w3.eth.account.sign_transaction(tx_local, private_key=PRIVATE_KEY)
                raw_retry = getattr(signed_retry, "raw_transaction", getattr(signed_retry, "rawTransaction", None))
                tx_hash_retry = w3.eth.send_raw_transaction(raw_retry)
                logging.info(f"🔄 Retry TX sent: {tx_hash_retry.hex()}")
                receipt = w3.eth.wait_for_transaction_receipt(tx_hash_retry)
                logging.info(f"🧾 Retry TX confirmed in block {receipt.blockNumber}")
                return tx_hash_retry.hex()

            except Exception as e2:
                logging.warning(f"⚠️ Retry {i} failed: {e2}")
                time.sleep(1 + i)
                continue

        logging.error("❌ All send_tx retries failed.")
        return None

    except Exception as e:
        logging.exception("❌ send_tx unexpected error:")
        return None



# ============================================================
# FETCH PRICE HISTORY (OKX public API)
# ============================================================
def fetch_price_history(days=3, interval="hourly"):
    """
    Fetch POL/USDT historical data from OKX public API.
    OKX granularity supports 1h, 4h, 1d, etc.
    """
    url = "https://www.okx.com/api/v5/market/candles"
    params = {"instId": "POL-USDT", "bar": "1H", "limit": str(days * 24)}

    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()

            # Log full response if API code != 0
            if data.get("code") != "0":
                logging.warning(f"⚠️ OKX API returned code={data.get('code')}, msg={data.get('msg')}")

            if "data" not in data or not data["data"]:
                logging.error("⚠️ No candle data returned from OKX.")
                return None

            df = pd.DataFrame(data["data"], columns=[
                "ts", "o", "h", "l", "c", "vol", "volCcy", "volCcyQuote", "confirm"
            ])
            df["timestamp"] = pd.to_datetime(df["ts"].astype(float), unit="ms")
            df["price"] = df["c"].astype(float)
            df = df.sort_values("timestamp").reset_index(drop=True)
            logging.info(f"✅ OKX price history fetched: {len(df)} records for POL/USDT.")
            return df[["timestamp", "price"]]

        except requests.exceptions.RequestException as e:
            logging.warning(f"⚠️ Attempt {attempt+1}/3 failed fetching OKX data: {e}")
            time.sleep(2)

    logging.error("❌ Failed to fetch price history after 3 attempts.")
    return None


def feature_engineer(df: pd.DataFrame, window=14):
    df = df.copy()
    df["ret1"] = df["price"].pct_change()
    df["sma"] = df["price"].rolling(window=window).mean()
    df["std"] = df["price"].rolling(window=window).std()
    df["momentum"] = df["price"] / df["price"].shift(window) - 1
    # RSI simple
    delta = df["price"].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(window=window).mean()
    roll_down = down.rolling(window=window).mean().replace(0, 1e-9)
    rs = roll_up / roll_down
    df["rsi"] = 100 - (100 / (1 + rs))
    df = df.dropna().reset_index(drop=True)
    return df

def train_small_model(df: pd.DataFrame):
    """
    Train a small RandomForestClassifier on historical data:
    Label = 1 if future 3-step return > threshold else 0
    This is a very lightweight 'AI' used to generate BUY/HOLD.
    """
    df = feature_engineer(df)
    # create label: next n periods return
    future_horizon = 3
    df["future_ret"] = df["price"].shift(-future_horizon) / df["price"] - 1
    threshold = 0.002  # require ~0.2% forward move to call it a positive sample
    df = df.dropna().reset_index(drop=True)
    df["label"] = (df["future_ret"] > threshold).astype(int)
    features = ["ret1", "sma", "std", "momentum", "rsi"]
    X = df[features]
    y = df["label"]
    if len(df) < 40:
        # not enough data -> fallback
        model = None
        return model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    return model


# ============================================================
# AI BUY SIGNAL (Simple SMA Crossover AI Logic)
# ============================================================
def ai_buy_signal():
    """
    Generate AI-based BUY signal using OKX price data.
    Returns: bool
    """
    try:
        df = fetch_price_history(days=3, interval="hourly")
        if df is None or df.empty:
            logging.error("❌ No price data available. Cannot generate AI signal.")
            return False

        # Simple AI logic: 5h SMA > 20h SMA triggers BUY
        df["sma_fast"] = df["price"].rolling(5).mean()
        df["sma_slow"] = df["price"].rolling(20).mean()

        latest_fast = df["sma_fast"].iloc[-1]
        latest_slow = df["sma_slow"].iloc[-1]
        signal = latest_fast > latest_slow

        logging.info(f"🤖 AI Signal computed | SMA_fast={latest_fast:.4f} | SMA_slow={latest_slow:.4f} | Signal={signal}")
        return signal

    except Exception as e:
        logging.error(f"AI signal generation failed, defaulting to False. Error: {e}", exc_info=True)
        return False

# ---------- Trading logic / position tracking ----------

@dataclass
class Position:
    lots_alloc: List[int] = field(default_factory=lambda: [1,1,2,3])  # sums to 7
    lot_size_usdt: float = 0.0
    buy_prices: List[float] = field(default_factory=list)  # price per lot executed
    amounts_wmatic: List[float] = field(default_factory=list)  # WMATIC amounts per buy
    total_usdt_spent: float = 0.0

    def total_wmatic(self):
        return sum(self.amounts_wmatic)

    def cost_basis(self):
        return self.total_usdt_spent  # USDT spent

    def current_value_usdt(self, price_wmatic_usdt: float):
        return self.total_wmatic() * price_wmatic_usdt

    def realized_profit_pct(self, price_wmatic_usdt: float):
        if self.cost_basis() == 0:
            return -999
        return (self.current_value_usdt(price_wmatic_usdt) / self.cost_basis() - 1) * 100.0

# helpers to read balances and price
def get_onchain_token_balance(token_contract, address):
    try:
        bal = token_contract.functions.balanceOf(address).call()
        dec = get_token_decimals(token_contract)
        return from_decimals(bal, dec)
    except Exception as e:
        logging.exception("Failed to read on-chain token balance.")
        return 0.0
		
def get_pol_price_from_okx():
    """
    Fetch latest POL/USDT price directly from OKX public API.
    Returns: float or None
    """
    try:
        url = "https://www.okx.com/api/v5/market/ticker"
        params = {"instId": "POL-USDT"}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()

        if data.get("code") != "0" or not data.get("data"):
            logging.warning(f"⚠️ Failed to fetch POL price. Code={data.get('code')}, Msg={data.get('msg')}")
            return None

        price = float(data["data"][0]["last"])
        logging.info(f"💰 Current POL price: {price:.6f} USDT")
        return price

    except Exception as e:
        logging.error(f"❌ Failed to fetch POL price from OKX: {e}")
        return None

def estimate_amounts_out(amount_in, path):
    try:
        amounts = router.functions.getAmountsOut(int(amount_in), path).call()
        return amounts
    except Exception:
        return None

def swap_usdt_to_wmatic(amount_usdt):
    """Swap USDT → WMATIC safely using allowance + gas guard"""
    try:
        if amount_usdt <= 0:
            logging.warning("⚠️ swap_usdt_to_wmatic called with zero or negative amount.")
            return None

        amount_in = int(Decimal(amount_usdt) * (10 ** 6))  # USDT decimals = 6
        path = [USDT_ADDR, WMATIC_ADDR]
        deadline = int(time.time()) + 600

        logging.info(f"🔁 Swapping {amount_usdt:.4f} USDT → WMATIC ...")

        return safe_swap_exact_tokens_for_tokens(
            amount_in=amount_in,
            amount_out_min=0,
            path=path,
            to=OWNER,
            deadline=deadline
        )

    except Exception as e:
        logging.error(f"❌ swap_usdt_to_wmatic failed: {e}", exc_info=True)
        return None


def swap_wmatic_to_usdt(amount_wmatic):
    """Swap WMATIC → USDT safely using allowance + gas guard"""
    try:
        if amount_wmatic <= 0:
            logging.warning("⚠️ swap_wmatic_to_usdt called with zero or negative amount.")
            return None

        amount_in = int(Decimal(amount_wmatic) * (10 ** 18))  # WMATIC decimals = 18
        path = [WMATIC_ADDR, USDT_ADDR]
        deadline = int(time.time()) + 600

        logging.info(f"🔁 Swapping {amount_wmatic:.4f} WMATIC → USDT ...")

        return safe_swap_exact_tokens_for_tokens(
            amount_in=amount_in,
            amount_out_min=0,
            path=path,
            to=OWNER,
            deadline=deadline
        )

    except Exception as e:
        logging.error(f"❌ swap_wmatic_to_usdt failed: {e}", exc_info=True)
        return None


