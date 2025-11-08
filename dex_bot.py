# dex_bot.py
"""
Polygon QuickSwap WMATIC/USDT Grid Trader (single-file)
- Splits USDT balance into 7 lots: 1,1,2,3
- Grid triggers: initial buy -> if price drops -3% -> buy 2nd (1 lot),
                  then -10% from previous -> buy 3rd (2 lots),
                  then -25% from previous -> buy 4th (3 lots)
- Close when profit >= +0.5% (portfolio value vs cost basis)
- AI BUY signal: tiny RandomForest trained on recent CoinGecko OHLC points (online each scan)
- Environment variables:
    RPC_URL      - Polygon RPC (e.g. https://polygon-rpc.com)
    PRIVATE_KEY  - deployer private key (keep secret)
    OWNER_ADDR   - your wallet address (checksum)
    USDT_ADDR    - optional override
    WMATIC_ADDR  - optional override
    ROUTER_ADDR  - optional override (QuickSwap router default)
- NOTE: This is a simple educational implementation. Test on small amounts / testnet first.
"""

import os
import time
import math
import logging
from decimal import Decimal
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import requests
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from web3 import Web3
from web3.middleware import geth_poa_middleware
from dotenv import load_dotenv

load_dotenv()

# ---------- Config / Addresses (defaults) ----------
RPC_URL = os.getenv("RPC_URL", "https://polygon-rpc.com")
PRIVATE_KEY = os.getenv("PRIVATE_KEY")
OWNER = Web3.to_checksum_address(os.getenv("OWNER_ADDR", ""))  # set in Railway env
USDT_ADDR = Web3.to_checksum_address(os.getenv("USDT_ADDR", "0xc2132D05D31c914a87C6611C10748AEb04B58e8F"))  # USDT on Polygon
WMATIC_ADDR = Web3.to_checksum_address(os.getenv("WMATIC_ADDR", "0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270"))
ROUTER_ADDR = Web3.to_checksum_address(os.getenv("ROUTER_ADDR", "0xa5E0829CaCEd8fFDD4De3c43696c57F7D7A678ff"))  # QuickSwap V2 Router

if not PRIVATE_KEY or not OWNER:
    raise RuntimeError("Set RPC_URL, PRIVATE_KEY, and OWNER_ADDR environment variables before running.")

# ---------- Web3 setup ----------
w3 = Web3(Web3.HTTPProvider(RPC_URL))
# polygon sometimes requires the POA middleware for some providers
w3.middleware_onion.inject(geth_poa_middleware, layer=0)

# ---------- Minimal ABIs ----------
ERC20_ABI = [
    {"constant": True, "inputs": [{"name": "_owner", "type": "address"}], "name": "balanceOf", "outputs": [{"name": "balance", "type": "uint256"}], "type": "function"},
    {"constant": True, "inputs": [], "name": "decimals", "outputs": [{"name": "", "type": "uint8"}], "type": "function"}
]

# QuickSwap/UniswapV2 Router minimal ABI
ROUTER_ABI = [
        {"inputs":[{"internalType":"address","name":"_factory","type":"address"},{"internalType":"address","name":"_WETH","type":"address"}],"stateMutability":"nonpayable","type":"constructor"},{"inputs":[],"name":"WETH","outputs":[{"internalType":"address","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"tokenA","type":"address"},{"internalType":"address","name":"tokenB","type":"address"},{"internalType":"uint256","name":"amountADesired","type":"uint256"},{"internalType":"uint256","name":"amountBDesired","type":"uint256"},{"internalType":"uint256","name":"amountAMin","type":"uint256"},{"internalType":"uint256","name":"amountBMin","type":"uint256"},{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"deadline","type":"uint256"}],"name":"addLiquidity","outputs":[{"internalType":"uint256","name":"amountA","type":"uint256"},{"internalType":"uint256","name":"amountB","type":"uint256"},{"internalType":"uint256","name":"liquidity","type":"uint256"}],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"token","type":"address"},{"internalType":"uint256","name":"amountTokenDesired","type":"uint256"},{"internalType":"uint256","name":"amountTokenMin","type":"uint256"},{"internalType":"uint256","name":"amountETHMin","type":"uint256"},{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"deadline","type":"uint256"}],"name":"addLiquidityETH","outputs":[{"internalType":"uint256","name":"amountToken","type":"uint256"},{"internalType":"uint256","name":"amountETH","type":"uint256"},{"internalType":"uint256","name":"liquidity","type":"uint256"}],"stateMutability":"payable","type":"function"},{"inputs":[],"name":"factory","outputs":[{"internalType":"address","name":"","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint256","name":"amountOut","type":"uint256"},{"internalType":"uint256","name":"reserveIn","type":"uint256"},{"internalType":"uint256","name":"reserveOut","type":"uint256"}],"name":"getAmountIn","outputs":[{"internalType":"uint256","name":"amountIn","type":"uint256"}],"stateMutability":"pure","type":"function"},{"inputs":[{"internalType":"uint256","name":"amountIn","type":"uint256"},{"internalType":"uint256","name":"reserveIn","type":"uint256"},{"internalType":"uint256","name":"reserveOut","type":"uint256"}],"name":"getAmountOut","outputs":[{"internalType":"uint256","name":"amountOut","type":"uint256"}],"stateMutability":"pure","type":"function"},{"inputs":[{"internalType":"uint256","name":"amountOut","type":"uint256"},{"internalType":"address[]","name":"path","type":"address[]"}],"name":"getAmountsIn","outputs":[{"internalType":"uint256[]","name":"amounts","type":"uint256[]"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint256","name":"amountIn","type":"uint256"},{"internalType":"address[]","name":"path","type":"address[]"}],"name":"getAmountsOut","outputs":[{"internalType":"uint256[]","name":"amounts","type":"uint256[]"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint256","name":"amountA","type":"uint256"},{"internalType":"uint256","name":"reserveA","type":"uint256"},{"internalType":"uint256","name":"reserveB","type":"uint256"}],"name":"quote","outputs":[{"internalType":"uint256","name":"amountB","type":"uint256"}],"stateMutability":"pure","type":"function"},{"inputs":[{"internalType":"address","name":"tokenA","type":"address"},{"internalType":"address","name":"tokenB","type":"address"},{"internalType":"uint256","name":"liquidity","type":"uint256"},{"internalType":"uint256","name":"amountAMin","type":"uint256"},{"internalType":"uint256","name":"amountBMin","type":"uint256"},{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"deadline","type":"uint256"}],"name":"removeLiquidity","outputs":[{"internalType":"uint256","name":"amountA","type":"uint256"},{"internalType":"uint256","name":"amountB","type":"uint256"}],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"token","type":"address"},{"internalType":"uint256","name":"liquidity","type":"uint256"},{"internalType":"uint256","name":"amountTokenMin","type":"uint256"},{"internalType":"uint256","name":"amountETHMin","type":"uint256"},{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"deadline","type":"uint256"}],"name":"removeLiquidityETH","outputs":[{"internalType":"uint256","name":"amountToken","type":"uint256"},{"internalType":"uint256","name":"amountETH","type":"uint256"}],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"token","type":"address"},{"internalType":"uint256","name":"liquidity","type":"uint256"},{"internalType":"uint256","name":"amountTokenMin","type":"uint256"},{"internalType":"uint256","name":"amountETHMin","type":"uint256"},{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"deadline","type":"uint256"}],"name":"removeLiquidityETHSupportingFeeOnTransferTokens","outputs":[{"internalType":"uint256","name":"amountETH","type":"uint256"}],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"token","type":"address"},{"internalType":"uint256","name":"liquidity","type":"uint256"},{"internalType":"uint256","name":"amountTokenMin","type":"uint256"},{"internalType":"uint256","name":"amountETHMin","type":"uint256"},{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"deadline","type":"uint256"},{"internalType":"bool","name":"approveMax","type":"bool"},{"internalType":"uint8","name":"v","type":"uint8"},{"internalType":"bytes32","name":"r","type":"bytes32"},{"internalType":"bytes32","name":"s","type":"bytes32"}],"name":"removeLiquidityETHWithPermit","outputs":[{"internalType":"uint256","name":"amountToken","type":"uint256"},{"internalType":"uint256","name":"amountETH","type":"uint256"}],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"token","type":"address"},{"internalType":"uint256","name":"liquidity","type":"uint256"},{"internalType":"uint256","name":"amountTokenMin","type":"uint256"},{"internalType":"uint256","name":"amountETHMin","type":"uint256"},{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"deadline","type":"uint256"},{"internalType":"bool","name":"approveMax","type":"bool"},{"internalType":"uint8","name":"v","type":"uint8"},{"internalType":"bytes32","name":"r","type":"bytes32"},{"internalType":"bytes32","name":"s","type":"bytes32"}],"name":"removeLiquidityETHWithPermitSupportingFeeOnTransferTokens","outputs":[{"internalType":"uint256","name":"amountETH","type":"uint256"}],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"tokenA","type":"address"},{"internalType":"address","name":"tokenB","type":"address"},{"internalType":"uint256","name":"liquidity","type":"uint256"},{"internalType":"uint256","name":"amountAMin","type":"uint256"},{"internalType":"uint256","name":"amountBMin","type":"uint256"},{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"deadline","type":"uint256"},{"internalType":"bool","name":"approveMax","type":"bool"},{"internalType":"uint8","name":"v","type":"uint8"},{"internalType":"bytes32","name":"r","type":"bytes32"},{"internalType":"bytes32","name":"s","type":"bytes32"}],"name":"removeLiquidityWithPermit","outputs":[{"internalType":"uint256","name":"amountA","type":"uint256"},{"internalType":"uint256","name":"amountB","type":"uint256"}],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"uint256","name":"amountOut","type":"uint256"},{"internalType":"address[]","name":"path","type":"address[]"},{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"deadline","type":"uint256"}],"name":"swapETHForExactTokens","outputs":[{"internalType":"uint256[]","name":"amounts","type":"uint256[]"}],"stateMutability":"payable","type":"function"},{"inputs":[{"internalType":"uint256","name":"amountOutMin","type":"uint256"},{"internalType":"address[]","name":"path","type":"address[]"},{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"deadline","type":"uint256"}],"name":"swapExactETHForTokens","outputs":[{"internalType":"uint256[]","name":"amounts","type":"uint256[]"}],"stateMutability":"payable","type":"function"},{"inputs":[{"internalType":"uint256","name":"amountOutMin","type":"uint256"},{"internalType":"address[]","name":"path","type":"address[]"},{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"deadline","type":"uint256"}],"name":"swapExactETHForTokensSupportingFeeOnTransferTokens","outputs":[],"stateMutability":"payable","type":"function"},{"inputs":[{"internalType":"uint256","name":"amountIn","type":"uint256"},{"internalType":"uint256","name":"amountOutMin","type":"uint256"},{"internalType":"address[]","name":"path","type":"address[]"},{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"deadline","type":"uint256"}],"name":"swapExactTokensForETH","outputs":[{"internalType":"uint256[]","name":"amounts","type":"uint256[]"}],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"uint256","name":"amountIn","type":"uint256"},{"internalType":"uint256","name":"amountOutMin","type":"uint256"},{"internalType":"address[]","name":"path","type":"address[]"},{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"deadline","type":"uint256"}],"name":"swapExactTokensForETHSupportingFeeOnTransferTokens","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"uint256","name":"amountIn","type":"uint256"},{"internalType":"uint256","name":"amountOutMin","type":"uint256"},{"internalType":"address[]","name":"path","type":"address[]"},{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"deadline","type":"uint256"}],"name":"swapExactTokensForTokens","outputs":[{"internalType":"uint256[]","name":"amounts","type":"uint256[]"}],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"uint256","name":"amountIn","type":"uint256"},{"internalType":"uint256","name":"amountOutMin","type":"uint256"},{"internalType":"address[]","name":"path","type":"address[]"},{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"deadline","type":"uint256"}],"name":"swapExactTokensForTokensSupportingFeeOnTransferTokens","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"uint256","name":"amountOut","type":"uint256"},{"internalType":"uint256","name":"amountInMax","type":"uint256"},{"internalType":"address[]","name":"path","type":"address[]"},{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"deadline","type":"uint256"}],"name":"swapTokensForExactETH","outputs":[{"internalType":"uint256[]","name":"amounts","type":"uint256[]"}],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"uint256","name":"amountOut","type":"uint256"},{"internalType":"uint256","name":"amountInMax","type":"uint256"},{"internalType":"address[]","name":"path","type":"address[]"},{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"deadline","type":"uint256"}],"name":"swapTokensForExactTokens","outputs":[{"internalType":"uint256[]","name":"amounts","type":"uint256[]"}],"stateMutability":"nonpayable","type":"function"},{"stateMutability":"payable","type":"receive"}
]

usdt = w3.eth.contract(address=USDT_ADDR, abi=ERC20_ABI)
wmatic = w3.eth.contract(address=WMATIC_ADDR, abi=ERC20_ABI)
router = w3.eth.contract(address=ROUTER_ADDR, abi=ROUTER_ABI)



# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

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
    return w3.eth.get_transaction_count(OWNER)

def send_tx(tx):
    signed = w3.eth.account.sign_transaction(tx, private_key=PRIVATE_KEY)
    raw = getattr(signed, "raw_transaction", getattr(signed, "rawTransaction", None))
    if raw is None:
        raise AttributeError("Web3 signed transaction missing raw transaction field.")
    tx_hash = w3.eth.send_raw_transaction(raw)
    logging.info(f"✅ Transaction sent: {tx_hash.hex()}")
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    logging.info(f"🧾 Tx confirmed in block {receipt.blockNumber}")
    return tx_hash.hex()


def gas_params():
    # Keep simple: use w3.eth.gas_price; you can swap to EIP-1559 style if you want.
    g = w3.eth.gas_price
    return {"gasPrice": g, "chainId": w3.eth.chain_id}

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

def get_matic_price_from_coingecko():
    # quick price retrieval for WMATIC vs USDT (using CoinGecko simple price)
    try:
        r = requests.get("https://api.coingecko.com/api/v3/simple/price", params={"ids":"matic-network","vs_currencies":"usd"}, timeout=10)
        r.raise_for_status()
        p = r.json().get("matic-network", {}).get("usd", None)
        return float(p) if p else None
    except Exception:
        return None

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

def approve_if_needed(token_contract, spender, owner_addr, amount):
    allowance = token_contract.functions.allowance(owner_addr, spender).call()
    dec = get_token_decimals(token_contract)
    logging.info(f"Current allowance: {allowance / (10 ** dec)} tokens")

    if allowance >= int(amount):
        logging.info("Sufficient allowance, no approval needed.")
        return True

    logging.info("Approving max allowance for spender...")
    max_approve = 2 ** 256 - 1
    tx = token_contract.functions.approve(spender, max_approve).build_transaction({
        "from": owner_addr,
        "nonce": get_nonce(),
        **gas_params()
    })
    tx_hash = send_tx(tx)
    logging.info(f"Approval tx sent: {tx_hash}")
    time.sleep(3)

    # Recheck after a delay
    new_allowance = token_contract.functions.allowance(owner_addr, spender).call()
    if new_allowance < int(amount):
        raise RuntimeError("Approval failed — allowance still insufficient!")
    return True


def swap_usdt_to_wmatic(amount_in_usdt, slippage=0.015):
    """
    Swap exact USDT -> WMATIC using router.swapExactTokensForTokensSupportingFeeOnTransferTokens
    amount_in_usdt is float (human)
    """
    dec_usdt = get_token_decimals(usdt)
    dec_wmatic = get_token_decimals(wmatic)
    amt_in = to_decimals(amount_in_usdt, dec_usdt)
    path = [USDT_ADDR, WMATIC_ADDR]
    # estimate out
    amounts = estimate_amounts_out(amt_in, path)
    if not amounts:
        raise RuntimeError("Failed to estimate amounts out.")
    out_est = amounts[-1]
    out_min = int(out_est * (1 - slippage))
    # ensure approval
    approve_token_direct(usdt, ROUTER_ADDR, amt_in)
    tx = router.functions.swapExactTokensForTokens(
        amt_in, out_min, path, OWNER, int(time.time()) + 60*10
    ).build_transaction({
        "from": OWNER,
        "nonce": get_nonce(),
        **gas_params()
    })
    tx_hash = send_tx(tx)
    logging.info(f"Swap USDT->WMATIC tx sent: {tx_hash}")
    time.sleep(2)
    return tx_hash


def approve_token_direct(token_contract, spender, amount):
    """
    Always send approval tx (without allowance check).
    """
    tx = token_contract.functions.approve(spender, amount).build_transaction({
        "from": OWNER,
        "nonce": get_nonce(),
        **gas_params()
    })
    tx_hash = send_tx(tx)
    logging.info(f"Approval tx sent: {tx_hash}")
    time.sleep(2)
    return tx_hash


def swap_wmatic_to_usdt(amount_in_wmatic, slippage=0.015):
    """
    Swap WMATIC -> USDT using QuickSwap (POL network).
    """
    try:
        dec_wmatic = get_token_decimals(wmatic)
        dec_usdt = get_token_decimals(usdt)

        # Convert to wei (use web3.to_wei to be consistent)
        amt_in = w3.to_wei(amount_in_wmatic, 'ether')  # WMATIC = 18 decimals

        path = [WMATIC_ADDR, USDT_ADDR]
        amounts = estimate_amounts_out(amt_in, path)
        if not amounts:
            raise RuntimeError("Failed to estimate amounts out.")
        
        out_est = amounts[-1]
        out_min = int(out_est * (1 - slippage))

        # ✅ Explicit approval (don’t rely on allowance logic)
        logging.info("Approving WMATIC for router (force refresh)...")
        approve_token_direct(wmatic, ROUTER_ADDR, amt_in)

        tx = router.functions.swapExactTokensForTokens(
            amt_in, out_min, path, OWNER, int(time.time()) + 60*10
        ).build_transaction({
            "from": OWNER,
            "nonce": get_nonce(),
            **gas_params()
        })

        tx_hash = send_tx(tx)
        logging.info(f"Swap WMATIC->USDT tx sent: {tx_hash}")
        time.sleep(2)
        return tx_hash

    except Exception as e:
        logging.error(f"swap_wmatic_to_usdt() failed: {e}")
        raise


# ---------- Main bot behavior ----------
def main_loop(poll_interval=60):
    logging.info("Starting DEX Grid Bot main loop...")
    in_position = False
    position: Optional[Position] = None
    while True:
        try:
            # 1) If not in position: check AI signal
            if not in_position:
                logging.info("Checking AI BUY signal...")
                buy_signal = ai_buy_signal()
                usdt_balance_onchain = get_onchain_token_balance(usdt, OWNER)
                logging.info(f"USDT balance: {usdt_balance_onchain:.6f}")
                if buy_signal and usdt_balance_onchain > 5:  # require min balance
                    # initialize position
                    lot_values = [1,1,2,3]
                    lot_total_units = sum(lot_values)  # 7
                    lot_size = usdt_balance_onchain / lot_total_units
                    position = Position(lots_alloc=lot_values, lot_size_usdt=lot_size)
                    logging.info(f"BUY signal accepted. Starting initial buy: lot_size={lot_size:.6f} USDT")
                    # execute initial buy (first lot)
                    logging.info("Executing initial buy (1 lot)...")
                    swap_usdt_to_wmatic(lot_size)
                    # read WMATIC balance diff to estimate amount bought
                    # For simplicity: read current WMATIC balance (assumes zero before)
                    wmatic_bal = get_onchain_token_balance(wmatic, OWNER)
                    position.buy_prices.append(get_matic_price_from_coingecko() or 0.0)
                    position.amounts_wmatic.append(wmatic_bal) # crude: in practice compute exact amount
                    position.total_usdt_spent += lot_size
                    in_position = True
                    logging.info(f"Position opened. WMATIC balance approx: {wmatic_bal:.6f}")
                else:
                    logging.info("No buy signal or insufficient USDT. Sleeping.")
            else:
                # in position: monitor grid and check for DCA triggers and TP
                price = get_pol_price_from_okx()
                if price is None:
                    logging.warning("Couldn't fetch price; skipping cycle.")
                    time.sleep(poll_interval)
                    continue
                logging.info(f"Current WMATIC price (USD): {price:.6f}")
                # compute profit percent
                profit_pct = position.realized_profit_pct(price)
                logging.info(f"Unrealized profit: {profit_pct:.4f}%")
                if profit_pct >= 0.5:  # close condition
                    logging.info("Profit threshold reached -> closing position (swap WMATIC -> USDT)")
                    total_wmatic = get_onchain_token_balance(wmatic, OWNER)
                    if total_wmatic > 0:
                        swap_wmatic_to_usdt(total_wmatic)
                    # reset
                    in_position = False
                    position = None
                    logging.info("Position closed. Will scan for next buy.")
                else:
                    # DCA triggers: check if price dropped relative to last executed buy price
                    last_price = position.buy_prices[-1] if position.buy_prices else position.buy_prices.append(price) or price
                    # grid triggers percentages
                    triggers = [-0.03, -0.10, -0.25]  # relative drops from previous order
                    for idx, trig in enumerate(triggers, start=1):
                        # only if we haven't executed that grid step yet (length control)
                        if len(position.amounts_wmatic) <= idx:
                            target_price = last_price * (1 + trig)
                            logging.info(f"Grid check idx={idx}: target_price={target_price:.6f} (current {price:.6f})")
                            if price <= target_price:
                                lot_to_buy = position.lots_alloc[idx]
                                amount_usdt = lot_to_buy * position.lot_size_usdt
                                logging.info(f"Trigger met for grid idx {idx}. Buying {lot_to_buy} lots => {amount_usdt:.6f} USDT")
                                swap_usdt_to_wmatic(amount_usdt)
                                # estimate new WMATIC delta by reading onchain balance and subtracting prior recorded
                                new_wmatic_bal = get_onchain_token_balance(wmatic, OWNER)
                                delta = new_wmatic_bal - sum(position.amounts_wmatic)
                                if delta <= 0:
                                    logging.warning("Unable to detect WMATIC delta after swap; recording approximate amount via estimate.")
                                    # estimate via getAmountsOut
                                    dec_usdt = get_token_decimals(usdt)
                                    amt_in = to_decimals(amount_usdt, dec_usdt)
                                    est = estimate_amounts_out(amt_in, [USDT_ADDR, WMATIC_ADDR])
                                    if est:
                                        delta = from_decimals(est[-1], get_token_decimals(wmatic))
                                position.amounts_wmatic.append(delta)
                                position.buy_prices.append(price)
                                position.total_usdt_spent += amount_usdt
                                last_price = price
                                break  # after a DCA, re-evaluate on next cycle
                    # else no DCA triggered this loop
            # sleep before next poll
            time.sleep(poll_interval)
        except Exception as exc:
            logging.exception("Main loop error, will continue after short sleep.")
            time.sleep(10)

if __name__ == "__main__":
    try:
        main_loop(poll_interval=60)
    except KeyboardInterrupt:
        logging.info("Exiting on user interrupt.")
