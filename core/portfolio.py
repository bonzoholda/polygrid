# core/portfolio.py
import logging
import sys
import os
import traceback

# Ensure root path included
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from config import usdt, wmatic
from uniswap_v3_manager import UniswapV3Manager


# ----------------------------------------------
# SLOT0 PRICE READER (ON-CHAIN WMATIC → USDT)
# ----------------------------------------------
def get_wmatic_price_slot0():
    """
    Fetch WMATIC price (USDT per WMATIC) using Uniswap V3 pool.slot0.
    We compute correctly as:
      sqrtPrice = sqrtPriceX96 / 2**96
      price_raw = sqrtPrice**2   # this is token1/token0 in raw units
      price_human = price_raw * 10**(dec1 - dec0)
    where dec0/dec1 are the decimals of token0 and token1 respectively.
    Returns float price (rounded to 6 decimals) or None on failure.
    """
    try:
        # Prefer explicit pool address in config if available (safe)
        try:
            from config import UNISWAP_POOL_ADDR
        except Exception:
            UNISWAP_POOL_ADDR = None

        # create manager with pool address when available
        if UNISWAP_POOL_ADDR:
            mgr = UniswapV3Manager(owner_address=None, pool_address=UNISWAP_POOL_ADDR)
        else:
            # If config has no pool addr, create manager without pool (will warn)
            mgr = UniswapV3Manager(owner_address=None)

        # ensure pool exists
        if not getattr(mgr, "pool", None):
            logging.error("❌ Failed to get slot0 price: UniswapV3Manager.pool is None — pool not loaded")
            return None

        # read slot0
        slot0 = mgr.pool.functions.slot0().call()
        sqrtPriceX96 = int(slot0[0])

        # decimals from manager (cached during init)
        dec0 = getattr(mgr, "dec0", None)
        dec1 = getattr(mgr, "dec1", None)
        # fallback safe defaults if missing
        if dec0 is None:
            dec0 = 18
        if dec1 is None:
            dec1 = 6

        # compute human price correctly
        sqrt_price = float(sqrtPriceX96) / (2 ** 96)          # sqrt(token1/token0)
        price_raw = sqrt_price * sqrt_price                   # token1/token0 (raw)
        dec_adj = dec1 - dec0                                 # IMPORTANT: dec1 - dec0
        price_human = float(price_raw) * (10 ** dec_adj)

        # numeric sanity: avoid absurd values, if insane, return None
        if not (0.0 < price_human < 1e9):
            # price_human can legitimately be >1e3 (if token is very expensive) but
            # astronomic values like 1e20 indicate scaling bug or wrong pool; log and return None
            logging.warning(f"⚠️ slot0 price out of expected range: {price_human}")
            # still return rounded value if it's reasonable; otherwise None
            # We'll allow up to 1e9 USDT/WMATIC (safety)
            if price_human <= 1e9:
                return round(price_human, 6)
            return None

        return round(price_human, 6)

    except Exception as e:
        logging.error(f"❌ Failed to get slot0 price via manager: {e}")
        return None


# ----------------------------------------------
# PORTFOLIO FETCHER
# ----------------------------------------------
def fetch_portfolio(uid: int):
    from dashboard.manager import get_user

    try:
        user = get_user(uid)
        if not user:
            return {"error": f"User {uid} not found"}

        owner_address = user["address"]

        # -------------------------------
        # Token balances (human units)
        # -------------------------------
        try:
            usdt_balance_raw = usdt.functions.balanceOf(owner_address).call()
            wmatic_balance_raw = wmatic.functions.balanceOf(owner_address).call()
        except Exception as e:
            logging.error(f"❌ Failed to read on-chain balances for {owner_address}: {e}")
            return {"error": "Failed to read on-chain balances"}

        usdt_balance = float(usdt_balance_raw) / 1e6
        wmatic_balance = float(wmatic_balance_raw) / 1e18

        # -------------------------------
        # Slot0 price
        # -------------------------------
        wmatic_price = get_wmatic_price_slot0()
        if wmatic_price is None:
            logging.error("❌ Cannot fetch WMATIC on-chain price (slot0). Falling back to OKX price as last resort.")
            # last-resort fallback: try OKX (keeps UI alive)
            try:
                from utils import get_pol_price_from_okx
                wmatic_price = get_pol_price_from_okx() or 0.0
            except Exception:
                wmatic_price = 0.0

        # -------------------------------
        # LP Position Valuation
        # -------------------------------
        lp_value_usdt = 0.0
        lp_assets_usdt = 0.0
        lp_assets_wmatic = 0.0
        has_lp = False

        try:
            # Initialize manager with user's address and pool (if configured)
            try:
                from config import UNISWAP_POOL_ADDR
                mgr = UniswapV3Manager(owner_address=owner_address, pool_address=UNISWAP_POOL_ADDR)
            except Exception:
                mgr = UniswapV3Manager(owner_address=owner_address)

            pos_id = mgr.get_active_position_id()

            if pos_id:
                u_val, m_val, total = mgr.get_position_asset_value(pos_id, wmatic_price)
                lp_value_usdt = float(total)
                lp_assets_usdt = float(u_val)
                lp_assets_wmatic = float(m_val)
                has_lp = True
                logging.info(f"🦄 LP found for user {uid}: {total:.2f} USDT")
            else:
                logging.info(f"User {uid} has no V3 LP positions.")

        except Exception as e:
            logging.warning(f"⚠️ LP fetch failed for user {uid}: {e}")

        # -------------------------------
        # Combined portfolio value
        # -------------------------------
        wallet_value = usdt_balance + (wmatic_balance * wmatic_price)
        total_value = wallet_value + lp_value_usdt

        return {
            "uid": uid,
            "owner": owner_address,
            "usdt_balance": usdt_balance,
            "wmatic_balance": wmatic_balance,
            "wmatic_price": wmatic_price,
            "wallet_value_usdt": wallet_value,
            "lp_value_usdt": lp_value_usdt,
            "lp_details": {
                "active": has_lp,
                "usdt": lp_assets_usdt,
                "wmatic": lp_assets_wmatic,
            },
            "total_value_usdt": total_value
        }

    except Exception as e:
        logging.error(f"❌ Portfolio error: {e}")
        return {"error": str(e)}
