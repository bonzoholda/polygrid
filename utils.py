# utils.py
import os
import time
import logging
import requests
from decimal import Decimal
from typing import Optional, List, Dict, Any
from web3 import Web3
from web3.middleware import geth_poa_middleware

# import external objects from your config (must exist)
# expected in config.py: w3, router, usdt, wmatic, OWNER, PRIVATE_KEY, ROUTER_ADDR, USDT_ADDR, WMATIC_ADDR
from config import w3, router, usdt, wmatic, OWNER, PRIVATE_KEY, ROUTER_ADDR, USDT_ADDR, WMATIC_ADDR

# ---------- Configuration ----------
MAX_UINT = 2 ** 256 - 1
GAS_LIMIT_APPROVE = 120_000
GAS_LIMIT_SWAP = 700_000
# Gas price hard ceiling (wei). Keep high so aggressive mode can still operate.
GAS_PRICE_LIMIT = 1500 * (10 ** 9)  # 1500 gwei
# How long to wait for receipts before considering replacement attempts (seconds)
RECEIPT_TIMEOUT = 300  # 5 minutes
# send_tx retry policy (aggressive)
SENDTX_MAX_RETRIES = 3
SENDTX_GAS_BUMP = 1.5  # 50% bump per retry (aggressive mode)
# safe swap retry attempts
SWAP_MAX_ATTEMPTS = 3
SWAP_GAS_BUMP = 1.5  # bump per swap retry
# slippage multiplier steps when we progressively relax minOut (1.00 -> 0.98 -> 0.95)
SLIPPAGE_STEPS = [1.00, 0.98, 0.95]

# ---------- Caching configuration ----------
# Balance cache (reduce RPC calls)
BALANCE_CACHE = {
    "USDT": {"value": None, "timestamp": 0},
    "WMATIC": {"value": None, "timestamp": 0},
}
BALANCE_REFRESH_INTERVAL = 60  # refresh every 60 seconds

# Price cache for POL from OKX
PRICE_CACHE = {
    "POL": {"value": None, "timestamp": 0},
}
PRICE_REFRESH_INTERVAL = 60  # user selected option B -> 60 seconds

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ---------- Minimal ABI for ERC20 (we need decimals, balanceOf, approve, allowance) ----------
ERC20_ABI = [
    {"constant": True, "inputs": [], "name": "decimals", "outputs": [{"name": "", "type": "uint8"}], "type": "function"},
    {"constant": True, "inputs": [{"name": "_owner", "type": "address"}], "name": "balanceOf", "outputs": [{"name": "balance", "type": "uint256"}], "type": "function"},
    {"constant": False, "inputs": [{"name": "_spender", "type": "address"}, {"name": "_value", "type": "uint256"}], "name": "approve", "outputs": [{"name": "success", "type": "bool"}], "type": "function"},
    {"constant": True, "inputs": [{"name": "_owner", "type": "address"}, {"name": "_spender", "type": "address"}], "name": "allowance", "outputs": [{"name": "remaining", "type": "uint256"}], "type": "function"},
]

# ---------- Internal caches ----------
_DECIMALS_CACHE: Dict[str, int] = {}

# ---------- Helpers ----------
def _lower_addr(a: str) -> str:
    try:
        return a.lower()
    except Exception:
        return str(a)

def get_token_decimals(token_contract) -> int:
    """
    Safe ERC20 decimals() fetch with caching.
    token_contract: web3 contract instance
    """
    try:
        addr = _lower_addr(token_contract.address)
        if addr in _DECIMALS_CACHE:
            return _DECIMALS_CACHE[addr]

        decimals = token_contract.functions.decimals().call()
        _DECIMALS_CACHE[addr] = int(decimals)
        return int(decimals)

    except Exception as e:
        logging.error(f"❌ Failed to fetch decimals for {getattr(token_contract, 'address', '?')}: {e}")
        # fallback for known tokens
        try:
            if _lower_addr(token_contract.address) == _lower_addr(USDT_ADDR):
                return 6
            if _lower_addr(token_contract.address) == _lower_addr(WMATIC_ADDR):
                return 18
        except Exception:
            pass
        return 18


def safe_get_decimals(token_contract) -> int:
    """Simple guarded decimals fetch (no caching); kept for compatibility."""
    try:
        return token_contract.functions.decimals().call()
    except Exception:
        return 18


def to_raw(amount: Decimal, decimals: int) -> int:
    return int(Decimal(amount) * (10 ** decimals))


def from_raw(amount_int: int, decimals: int) -> float:
    return float(Decimal(amount_int) / (10 ** decimals))


def to_decimals(amount_float, decimals):
    """Convert float → raw integer token amount"""
    return int(Decimal(amount_float) * (10 ** decimals))


def from_decimals(amount_int, decimals):
    """Convert raw integer token amount → float"""
    return float(Decimal(amount_int) / (10 ** decimals))


# ---------- Router / token contract objects ----------
try:
    _usdt = usdt
except Exception:
    _usdt = w3.eth.contract(address=USDT_ADDR, abi=ERC20_ABI)

try:
    _wmatic = wmatic
except Exception:
    _wmatic = w3.eth.contract(address=WMATIC_ADDR, abi=ERC20_ABI)

_router = router  # should be contract in config

# ---------- Estimate amounts out (single canonical implementation) ----------
def estimate_amounts_out(amount_in_raw: int, path: List[str]) -> Optional[List[int]]:
    """
    Returns list of raw integer amounts per hop or None on failure.
    """
    try:
        amounts = _router.functions.getAmountsOut(int(amount_in_raw), path).call()
        return [int(a) for a in amounts]
    except Exception as e:
        logging.warning(f"estimate_amounts_out failed for path {path}: {e}")
        return None


# ---------- Cached token balance helper ----------
def get_cached_token_balance(token_name: str, token_contract, address: str) -> Optional[int]:
    """
    Returns raw integer token balance (not human decimals) from cache or on-chain.
    If RPC fails, returns the last cached raw integer or None if none exists.
    """
    now = time.time()
    token_name = str(token_name)

    # Prepare cache entry if missing
    if token_name not in BALANCE_CACHE:
        BALANCE_CACHE[token_name] = {"value": None, "timestamp": 0}

    cache_entry = BALANCE_CACHE[token_name]
    if cache_entry["value"] is not None:
        age = now - cache_entry["timestamp"]
        if age < BALANCE_REFRESH_INTERVAL:
            return cache_entry["value"]

    # otherwise fetch fresh
    try:
        fresh = token_contract.functions.balanceOf(address).call()
        BALANCE_CACHE[token_name] = {"value": int(fresh), "timestamp": now}
        return int(fresh)
    except Exception as e:
        logging.warning(f"[WARN] Failed to fetch {token_name} balance: {e}")
        # fallback to cached raw value (may be None)
        return cache_entry["value"]


# ---------- On-chain reads (human readable floats) ----------
def get_onchain_token_balance(token_contract, address: str) -> float:
    """
    Returns human float token balance using decimals conversion.
    Uses cached raw integer balances to reduce RPC calls.
    """
    try:
        # try to read raw cached value first
        # determine a friendly token_name for the cache (USDT / WMATIC) if possible
        token_addr = _lower_addr(getattr(token_contract, "address", ""))
        if token_addr == _lower_addr(USDT_ADDR):
            key = "USDT"
        elif token_addr == _lower_addr(WMATIC_ADDR):
            key = "WMATIC"
        else:
            key = token_addr  # fallback to address string

        raw = get_cached_token_balance(key, token_contract, address)
        if raw is None:
            # attempt one direct call if nothing cached
            raw = int(token_contract.functions.balanceOf(address).call())

        dec = safe_get_decimals(token_contract)
        return from_raw(raw, dec)
    except Exception as e:
        logging.exception("Failed to read on-chain token balance.")
        return 0.0


# ---------- Gas / nonce helpers ----------
def get_nonce() -> int:
    """Use 'pending' so replacement txs can be created safely."""
    try:
        return w3.eth.get_transaction_count(OWNER, "pending")
    except Exception as e:
        logging.warning(f"get_nonce RPC failed: {e} — attempting without 'pending'")
        try:
            return w3.eth.get_transaction_count(OWNER)
        except Exception as e2:
            logging.error(f"get_nonce fallback failed: {e2}")
            raise


def get_node_gas_price() -> int:
    """Return the provider's suggested gas price (wei)."""
    try:
        return w3.eth.gas_price
    except Exception as e:
        logging.warning(f"get_node_gas_price failed: {e}; using fallback 30 gwei")
        return Web3.to_wei(30, "gwei")


def gas_params(multiplier: float = 1.0, max_gwei: Optional[int] = None) -> Optional[dict]:
    """
    Returns gas params dict or None if gas price above ceiling.
    Multiplier multiplies provider gas_price to add safety buffer.
    """
    base = get_node_gas_price()
    price = int(base * multiplier)
    # enforce minimum reasonable gas (avoid 1 wei)
    min_gwei = Web3.to_wei(10, "gwei")
    price = max(price, min_gwei)
    # enforce global ceiling
    if price > GAS_PRICE_LIMIT:
        logging.warning(f"⚠️ Gas price {price/1e9:.1f} gwei > ceiling {GAS_PRICE_LIMIT/1e9:.1f} gwei → skipping tx.")
        return None
    # chainId may call RPC; try to get it but fallback to None if RPC fails
    try:
        chain_id = w3.eth.chain_id
    except Exception as e:
        logging.warning(f"Could not fetch chainId: {e}; leaving it out of gas params.")
        chain_id = None

    params = {"gasPrice": price}
    if chain_id is not None:
        params["chainId"] = chain_id
    return params


# ---------- Approvals ----------
def get_allowance(token_contract, owner_addr: str, spender_addr: str) -> int:
    try:
        return int(token_contract.functions.allowance(owner_addr, spender_addr).call())
    except Exception as e:
        logging.warning(f"Failed to read allowance: {e}")
        return 0


def approve_if_needed(token_contract, spender_addr: str, amount_required_raw: int) -> bool:
    """
    Ensure router has at least amount_required_raw allowance.
    Approve MAX_UINT once if insufficient.
    Returns True if allowance is sufficient (either already or after a confirmed approval tx).
    """
    try:
        allowance = get_allowance(token_contract, OWNER, spender_addr)
        if allowance >= int(amount_required_raw):
            logging.info(f"✅ Sufficient allowance ({allowance}) — no approval needed.")
            return True

        logging.info(f"🔐 Requesting approval (MAX_UINT) for {getattr(token_contract,'address', '?')} -> {spender_addr} ...")
        # Build approve tx
        tx = token_contract.functions.approve(spender_addr, MAX_UINT).build_transaction({
            "from": OWNER,
            "nonce": get_nonce(),
            "gas": GAS_LIMIT_APPROVE,
            **(gas_params(multiplier=1.0) or {}),
        })

        tx_hash = send_tx(tx)
        if not tx_hash:
            logging.warning("⚠️ Approval tx failed or skipped.")
            return False

        # best-effort wait and verify
        try:
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=RECEIPT_TIMEOUT)
            if receipt is None or getattr(receipt, "status", None) != 1:
                logging.warning("⚠️ Approval tx did not succeed (receipt.status != 1).")
            else:
                logging.info("✅ Approval tx confirmed successful.")
        except Exception:
            logging.warning("⚠️ Approval tx not confirmed quickly; continuing (will re-check allowance).")

        # confirm allowance
        new_allow = get_allowance(token_contract, OWNER, spender_addr)
        if new_allow >= int(amount_required_raw):
            logging.info("✅ Allowance sufficient after approval.")
            return True
        logging.error("❌ Allowance still insufficient after approval tx.")
        return False

    except Exception as e:
        logging.exception("approve_if_needed error:")
        return False


# ---------- send_tx (aggressive retry-on-underpriced / replacement strategy) ----------
def send_tx(tx: dict, max_retries: int = SENDTX_MAX_RETRIES, gas_bump: float = SENDTX_GAS_BUMP) -> Optional[str]:
    """
    Sign and broadcast tx. Waits for confirmation and returns tx_hash hex only if the tx confirmed successfully (status==1).
    On underpriced/replacement/node errors, tries replacement txs with bumped gas price (aggressive).
    Returns: tx_hash hex string on success, or None on failure/skip.
    """
    if gas_params() is None:
        logging.warning("⏳ Transaction skipped due to high gas.")
        return None

    tx_local: Dict[str, Any] = dict(tx)

    # apply initial gas params
    initial_params = gas_params(multiplier=1.0)
    if initial_params is None:
        logging.warning("⏳ Initial gas params blocked; skipping tx.")
        return None
    tx_local.update(initial_params)

    # sign initial tx
    try:
        signed = w3.eth.account.sign_transaction(tx_local, private_key=PRIVATE_KEY)
        raw = getattr(signed, "raw_transaction", None) or getattr(signed, "rawTransaction", None)
        if raw is None:
            raise AttributeError("Signed tx has no raw data.")
    except Exception as e:
        logging.exception("Failed to sign tx:")
        return None

    try:
        tx_hash = w3.eth.send_raw_transaction(raw)
        logging.info(f"✅ TX sent: {tx_hash.hex()} (waiting up to {RECEIPT_TIMEOUT}s for receipt)")
        # wait for receipt
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=RECEIPT_TIMEOUT)
        if receipt is None:
            logging.warning("⚠️ No receipt returned (None).")
            return None
        if getattr(receipt, "status", None) != 1:
            logging.error(f"❌ TX reverted or failed on-chain (status={getattr(receipt,'status',None)}). TxHash={tx_hash.hex()}")
            return None
        logging.info(f"🧾 TX confirmed in block {receipt.blockNumber}: {tx_hash.hex()}")
        return tx_hash.hex()

    except ValueError as e:
        # RPC returned an error immediately (e.g., underpriced, nonce)
        err_s = str(e).lower()
        logging.warning(f"⚠️ send_tx ValueError: {e}")
        # Check if retryable
        retryable_tokens = ("underpriced", "replacement transaction", "fee too low", "max fee per gas", "nonce", "insufficient funds")
        if not any(tok in err_s for tok in retryable_tokens):
            logging.error("❌ send_tx non-retryable ValueError.")
            return None

        # Retry loop with bumped gasPrice and fresh nonce
        for attempt in range(1, max_retries + 1):
            logging.info(f"🔁 send_tx replacement attempt {attempt}/{max_retries}")
            new_params = gas_params()
            if new_params is None:
                logging.warning("⏳ Gas too high for retry; aborting.")
                return None
            # bump gasPrice aggressively
            new_price = int(new_params["gasPrice"] * (gas_bump ** attempt))
            if new_price > GAS_PRICE_LIMIT:
                logging.warning("⚠️ Bumped gas exceeds ceiling; aborting retries.")
                return None
            new_params["gasPrice"] = new_price
            # refresh nonce (use pending to replace)
            try:
                new_nonce = get_nonce()
            except Exception as e_nonce:
                logging.warning(f"Could not refresh nonce for replacement attempt: {e_nonce}")
                return None
            tx_local.update({"nonce": new_nonce, **new_params})
            try:
                signed_retry = w3.eth.account.sign_transaction(tx_local, private_key=PRIVATE_KEY)
                raw_retry = getattr(signed_retry, "raw_transaction", None) or getattr(signed_retry, "rawTransaction", None)
                tx_hash_retry = w3.eth.send_raw_transaction(raw_retry)
                logging.info(f"🔄 Replacement TX sent: {tx_hash_retry.hex()} (waiting up to {RECEIPT_TIMEOUT}s)")
                receipt = w3.eth.wait_for_transaction_receipt(tx_hash_retry, timeout=RECEIPT_TIMEOUT)
                if receipt and getattr(receipt, "status", None) == 1:
                    logging.info(f"🧾 Replacement TX confirmed in block {receipt.blockNumber}")
                    return tx_hash_retry.hex()
                else:
                    logging.warning(f"⚠️ Replacement TX did not confirm successfully (status={getattr(receipt,'status',None)}).")
            except Exception as e2:
                logging.warning(f"⚠️ Replacement attempt {attempt} failed: {e2}")
            time.sleep(1 + attempt)

        logging.error("❌ All send_tx replacement retries failed.")
        return None

    except Exception as e:
        # This captures TimeExhausted or other unexpected issues. Attempt one replacement with bumped gas.
        logging.exception("❌ send_tx unexpected error — attempting a single replacement if possible:")
        try:
            new_params = gas_params()
            if new_params is None:
                logging.warning("⏳ Gas too high for replacement; aborting.")
                return None
            new_params["gasPrice"] = int(new_params["gasPrice"] * gas_bump)
            if new_params["gasPrice"] > GAS_PRICE_LIMIT:
                logging.warning("⚠️ Replacement gas would exceed ceiling; aborting.")
                return None
            tx_local.update({"nonce": get_nonce(), **new_params})
            signed_retry = w3.eth.account.sign_transaction(tx_local, private_key=PRIVATE_KEY)
            raw_retry = getattr(signed_retry, "raw_transaction", None) or getattr(signed_retry, "rawTransaction", None)
            tx_hash_retry = w3.eth.send_raw_transaction(raw_retry)
            logging.info(f"🔄 Replacement TX after unexpected error sent: {tx_hash_retry.hex()}")
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash_retry, timeout=RECEIPT_TIMEOUT)
            if receipt and getattr(receipt, "status", None) == 1:
                logging.info(f"🧾 Replacement TX confirmed in block {receipt.blockNumber}")
                return tx_hash_retry.hex()
            logging.warning("⚠️ Replacement after unexpected error did not confirm successfully.")
            return None
        except Exception as e3:
            logging.exception("⚠️ Replacement after unexpected error failed:")
            return None


# ---------- safe swap (uses send_tx + approve_if_needed) ----------
def safe_swap_exact_tokens_for_tokens(amount_in_raw: int, amount_out_min_raw: int, path: List[str], to: str, deadline: int) -> Optional[str]:
    """
    Robust swap wrapper:
      - ensures allowance (approve MAX_UINT if needed),
      - estimates expected out if caller passed amount_out_min_raw == 0,
      - tries up to SWAP_MAX_ATTEMPTS with progressive slippage (SLIPPAGE_STEPS)
      - bump gas aggressively between attempts.
    Returns confirmed tx_hash (hex) on success, otherwise None.
    """
    # quick gas check
    if gas_params() is None:
        logging.warning("⏳ Gas too high — skipping swap.")
        return None

    input_addr = path[0]
    input_token = w3.eth.contract(address=input_addr, abi=ERC20_ABI)

    # ensure approval (amount_in_raw is in token's raw units)
    if not approve_if_needed(input_token, ROUTER_ADDR, int(amount_in_raw)):
        logging.error("❌ Approval failed; aborting swap.")
        return None

    # estimate base_out if needed
    base_out = int(amount_out_min_raw) if int(amount_out_min_raw) != 0 else 0
    if base_out == 0:
        est = estimate_amounts_out(amount_in_raw, path)
        if est:
            base_out = int(est[-1])
            logging.info(f"🔍 Estimated output (raw): {base_out}")
        else:
            logging.warning("⚠️ Could not estimate amountsOut; will proceed with amountOutMin=0 (risk of revert).")
            base_out = 0

    for attempt in range(1, SWAP_MAX_ATTEMPTS + 1):
        multiplier = SLIPPAGE_STEPS[min(attempt - 1, len(SLIPPAGE_STEPS) - 1)]
        # when base_out==0 keep amount_out_min==0; else apply multiplier to relax minOut progressively
        adj_out_min = int(base_out * multiplier) if base_out else 0

        if attempt > 1:
            logging.warning(f"🔁 Swap retry attempt {attempt}: adj_out_min={adj_out_min} (multiplier={multiplier})")

        params = gas_params()
        if params is None:
            logging.warning("⏳ Gas too high now — abort swap attempt.")
            return None

        if attempt > 1:
            params["gasPrice"] = int(params["gasPrice"] * (SWAP_GAS_BUMP ** (attempt - 1)))
            if params["gasPrice"] > GAS_PRICE_LIMIT:
                logging.warning("⚠️ Bumped gas would exceed ceiling; aborting swap attempts.")
                return None
            logging.info(f"⬆️ Using gasPrice {params['gasPrice']/1e9:.2f} gwei for attempt {attempt}")

        # build tx
        try:
            tx = _router.functions.swapExactTokensForTokens(
                int(amount_in_raw),
                int(adj_out_min),
                path,
                to,
                int(deadline)
            ).build_transaction({
                "from": OWNER,
                "nonce": get_nonce(),
                "gas": GAS_LIMIT_SWAP,
                **params
            })
        except Exception as e:
            # if build_transaction fails (e.g., insufficient output amount for minimal out),
            # allow retry to change slippage; otherwise abort
            msg = str(e).lower()
            logging.warning(f"⚠️ Failed building swap tx attempt {attempt}: {e}")
            if "insufficient output amount" in msg or "execution reverted" in msg:
                # proceed to retry with more relaxed slippage
                time.sleep(1 + attempt)
                continue
            return None

        # send and wait for confirmed success via send_tx()
        tx_hash = send_tx(tx)
        if tx_hash:
            logging.info(f"✅ Swap successful (attempt {attempt}): {tx_hash}")
            return tx_hash
        else:
            logging.warning(f"⚠️ Swap attempt {attempt} did not confirm successfully (tx_hash None); will retry if attempts remain.")
            time.sleep(1 + attempt)

    logging.error("❌ Swap failed after all attempts.")
    return None


# ---------- convenience wrappers for USDT <-> WMATIC ----------
def swap_usdt_to_wmatic(amount_usdt: float, slippage: float = 0.02) -> Optional[str]:
    """
    Swap human amount_usdt (float) --> WMATIC. Returns confirmed tx_hash or None.
    Constructs a reasonable amountOutMin based on estimate and slippage.
    """
    try:
        if amount_usdt <= 0:
            logging.warning("⚠️ swap_usdt_to_wmatic called with non-positive amount.")
            return None

        dec_usdt = safe_get_decimals(_usdt)
        dec_wmatic = safe_get_decimals(_wmatic)

        amount_in_raw = int(Decimal(amount_usdt) * (10 ** dec_usdt))
        path = [USDT_ADDR, WMATIC_ADDR]
        deadline = int(time.time()) + 600

        # estimate outputs
        est = estimate_amounts_out(amount_in_raw, path)
        if est:
            expected_out = int(est[-1])
            amount_out_min = int(expected_out * (1 - slippage))
            logging.info(f"🔁 Swapping {amount_usdt:.6f} USDT -> expected_out {expected_out/(10**dec_wmatic):.6f} WMATIC (min {amount_out_min/(10**dec_wmatic):.6f})")
        else:
            amount_out_min = 0
            logging.warning("⚠️ Could not estimate expected output; using amountOutMin=0 (riskier).")

        return safe_swap_exact_tokens_for_tokens(amount_in_raw, amount_out_min, path, OWNER, deadline)

    except Exception as e:
        logging.exception("swap_usdt_to_wmatic failed:")
        return None


def swap_wmatic_to_usdt(amount_wmatic: float, slippage: float = 0.02) -> Optional[str]:
    """
    Swap human amount_wmatic (float) --> USDT. Returns confirmed tx_hash or None.
    """
    try:
        dec_wmatic = safe_get_decimals(_wmatic)
        dec_usdt = safe_get_decimals(_usdt)

        amount_in_raw = int(Decimal(amount_wmatic) * (10 ** dec_wmatic))
        path = [WMATIC_ADDR, USDT_ADDR]
        deadline = int(time.time()) + 600

        est = estimate_amounts_out(amount_in_raw, path)
        if est:
            expected_out = int(est[-1])
            amount_out_min = int(expected_out * (1 - slippage))
            logging.info(f"🔁 Swapping {amount_wmatic:.6f} WMATIC -> expected_out {expected_out/(10**dec_usdt):.6f} USDT (min {amount_out_min/(10**dec_usdt):.6f})")
        else:
            amount_out_min = 0
            logging.warning("⚠️ Could not estimate expected output; using amountOutMin=0 (riskier).")

        return safe_swap_exact_tokens_for_tokens(amount_in_raw, amount_out_min, path, OWNER, deadline)
    except Exception as e:
        logging.exception("swap_wmatic_to_usdt failed:")
        return None


# ---------- Price fetching with caching ----------
def get_pol_price_from_okx():
    """
    Fetch latest POL/USDT price from OKX public ticker with caching.
    Cached for PRICE_REFRESH_INTERVAL seconds (60s chosen).
    """
    now = time.time()
    try:
        # return cached if not expired
        if PRICE_CACHE["POL"]["value"] is not None:
            age = now - PRICE_CACHE["POL"]["timestamp"]
            if age < PRICE_REFRESH_INTERVAL:
                return PRICE_CACHE["POL"]["value"]

        # fetch fresh from OKX
        url = "https://www.okx.com/api/v5/market/ticker"
        params = {"instId": "POL-USDT"}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get("code") != "0" or not data.get("data"):
            logging.warning(f"⚠️ Failed to fetch POL price. Code={data.get('code')}, Msg={data.get('msg')}")
            return PRICE_CACHE["POL"]["value"]

        price = float(data["data"][0]["last"])
        PRICE_CACHE["POL"] = {"value": price, "timestamp": now}
        logging.info(f"💰 Current POL price: {price:.6f} USDT")
        return price

    except Exception as e:
        logging.error(f"❌ Failed to fetch POL price from OKX: {e}")
        return PRICE_CACHE["POL"]["value"]


# ---------- Compatibility alias ----------
def get_pol_price_from_okx_quiet():
    """Alias kept for compatibility if other modules call this name."""
    return get_pol_price_from_okx()


# ----------------------------
# End of utils.py
# ----------------------------
