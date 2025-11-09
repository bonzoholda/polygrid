# utils.py
import time
import logging
import requests
from web3 import Web3
from config import w3, router, usdt, wmatic, OWNER, PRIVATE_KEY, ROUTER_ADDR, USDT_ADDR, WMATIC_ADDR

# ---------- Helper Functions ----------

def get_nonce():
    return w3.eth.get_transaction_count(OWNER, 'pending')  # track pending txs

def gas_params():
    return {
        "gas": 250000,
        "gasPrice": w3.to_wei("40", "gwei"),
        "chainId": 137,
    }

def send_tx(tx, retries=3, gas_bump=1.2):
    for attempt in range(retries):
        try:
            signed = w3.eth.account.sign_transaction(tx, private_key=PRIVATE_KEY)
            tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
            return w3.to_hex(tx_hash)
        except ValueError as e:
            msg = str(e)
            if "replacement transaction underpriced" in msg or "already known" in msg:
                # bump gas price
                tx['gasPrice'] = int(tx['gasPrice'] * gas_bump)
                logging.warning(f"Tx underpriced, bumping gas and retrying... attempt {attempt+1}")
                time.sleep(2)
            else:
                raise
    raise RuntimeError("Failed to send tx after retries.")

def get_token_decimals(token_contract):
    return token_contract.functions.decimals().call()

def to_decimals(amount, decimals):
    return int(amount * (10 ** decimals))

def from_decimals(amount, decimals):
    return amount / (10 ** decimals)

def get_onchain_token_balance(token_contract, wallet):
    dec = get_token_decimals(token_contract)
    bal = token_contract.functions.balanceOf(wallet).call()
    return from_decimals(bal, dec)

# ---------- Price Fetching ----------

def get_pol_price_from_okx(symbol="POL-USDT", window="1h", limit=50):
    url = f"https://www.okx.com/api/v5/market/candles?instId={symbol}&bar={window}&limit={limit}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        candles = r.json().get("data", [])
        if not candles:
            logging.error("⚠️ No candle data returned from OKX.")
            return None
        close_prices = [float(c[4]) for c in candles]
        return sum(close_prices) / len(close_prices)
    except Exception as e:
        logging.error(f"Failed fetching OKX price: {e}")
        return None

# ---------- Swapping & Approvals ----------

def approve_if_needed(token_contract, spender, owner_addr, amount):
    allowance = token_contract.functions.allowance(owner_addr, spender).call()
    if allowance >= int(amount):
        return True
    dec = get_token_decimals(token_contract)
    max_approve = 2 ** 256 - 1
    tx = token_contract.functions.approve(spender, max_approve).build_transaction({
        "from": owner_addr,
        "nonce": get_nonce(),
        **gas_params()
    })
    tx_hash = send_tx(tx)
    logging.info(f"Approve tx sent: {tx_hash}")
    time.sleep(5)
    return True

def approve_token_direct(token_contract, spender, amount):
    """Force-approve a token without checking existing allowance."""
    tx = token_contract.functions.approve(
        spender, amount
    ).build_transaction({
        "from": OWNER,
        "nonce": get_nonce(),
        **gas_params()
    })
    tx_hash = send_tx(tx)
    logging.info(f"Direct approval tx sent: {tx_hash}")
    time.sleep(5)
    return tx_hash

def estimate_amounts_out(amount_in, path):
    try:
        return router.functions.getAmountsOut(amount_in, path).call()
    except Exception as e:
        logging.error(f"estimate_amounts_out() failed: {e}")
        return None

def swap_usdt_to_wmatic(amount_in_usdt, slippage=0.015):
    """
    Swap exact USDT -> WMATIC using QuickSwap (POL network).
    Uses explicit approval refresh to prevent allowance desync.
    amount_in_usdt is float (human units).
    """
    try:
        dec_usdt = get_token_decimals(usdt)
        dec_wmatic = get_token_decimals(wmatic)

        # Convert input amount to decimals
        amt_in = to_decimals(amount_in_usdt, dec_usdt)

        # Define swap path (USDT -> WMATIC)
        path = [USDT_ADDR, WMATIC_ADDR]

        # Estimate expected output
        amounts = estimate_amounts_out(amt_in, path)
        if not amounts:
            raise RuntimeError("Failed to estimate amounts out.")
        out_est = amounts[-1]
        out_min = int(out_est * (1 - slippage))

        # ✅ Try cached allowance logic first, then force approve if fails
        logging.info("Approving USDT for router (force refresh if needed)...")
        try:
            approve_if_needed(usdt, ROUTER_ADDR, OWNER, amt_in)
        except Exception as e:
            logging.warning(f"approve_if_needed() issue, retrying direct approve: {e}")
            approve_token_direct(usdt, ROUTER_ADDR, amt_in)

        # ✅ Execute swap
        tx = router.functions.swapExactTokensForTokens(
            amt_in,
            out_min,
            path,
            OWNER,
            int(time.time()) + 60 * 10
        ).build_transaction({
            "from": OWNER,
            "nonce": get_nonce(),
            **gas_params()
        })

        tx_hash = send_tx(tx)
        logging.info(f"✅ Swap USDT -> WMATIC sent successfully: {tx_hash}")
        time.sleep(5)
        return tx_hash

    except Exception as e:
        logging.error(f"swap_usdt_to_wmatic() failed: {e}")
        raise

def swap_wmatic_to_usdt(amount_in_wmatic, slippage=0.015):
    """
    Swap WMATIC -> USDT using QuickSwap (POL network).
    Uses explicit approval refresh to avoid allowance sync issues.
    """
    try:
        dec_wmatic = get_token_decimals(wmatic)
        dec_usdt = get_token_decimals(usdt)

        # Convert to wei (WMATIC = 18 decimals)
        amt_in = w3.to_wei(amount_in_wmatic, 'ether')

        # Define swap path
        path = [WMATIC_ADDR, USDT_ADDR]

        # Estimate expected output
        amounts = estimate_amounts_out(amt_in, path)
        if not amounts:
            raise RuntimeError("Failed to estimate amounts out.")
        out_est = amounts[-1]
        out_min = int(out_est * (1 - slippage))

        # ✅ Explicit approval (ignore allowance cache)
        logging.info("Approving WMATIC for router (force refresh)...")
        try:
            approve_if_needed(wmatic, ROUTER_ADDR, OWNER, amt_in)
        except Exception as e:
            logging.warning(f"approve_if_needed() issue, retrying direct approve: {e}")
            approve_token_direct(wmatic, ROUTER_ADDR, amt_in)

        # ✅ Execute swap using standard QuickSwap ABI
        tx = router.functions.swapExactTokensForTokens(
            amt_in,
            out_min,
            path,
            OWNER,
            int(time.time()) + 60 * 10
        ).build_transaction({
            "from": OWNER,
            "nonce": get_nonce(),
            **gas_params()
        })

        tx_hash = send_tx(tx)
        logging.info(f"✅ Swap WMATIC -> USDT sent successfully: {tx_hash}")
        time.sleep(5)
        return tx_hash

    except Exception as e:
        logging.error(f"swap_wmatic_to_usdt() failed: {e}")
        raise
