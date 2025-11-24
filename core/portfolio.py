# core/portfolio.py

import logging
import sys
import os

# Ensure root path included
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from config import usdt, wmatic
from uniswap_v3_manager import UniswapV3Manager


# ============================================================
# SLOT0 PRICE FETCHER (USES INTERNAL MANAGER POOL)
# ============================================================

def get_wmatic_price_slot0():
    """
    Canonical WMATIC price fetched from Uniswap V3 pool.slot0().
    Uses the SAME pool address used inside UniswapV3Manager to
    avoid pricing mismatch.
    """
    try:
        # Initialize manager WITHOUT owner (just for pool access)
        mgr = UniswapV3Manager()

        # Manager already loads self.pool internally.
        pool = mgr.pool
        if pool is None:
            raise Exception("UniswapV3Manager.pool is None — pool not loaded")

        # slot0 returns:
        # sqrtPriceX96, tick, ..., etc.
        slot0 = pool.functions.slot0().call()
        sqrt_price_x96 = slot0[0]

        # Uniswap price formula:
        # price = (sqrtPriceX96² / 2^192)
        price_wmatic_usdt = (sqrt_price_x96 ** 2) / (2 ** 192)

        # Normalize WMATIC(18) → USDT(6)
        adjusted_price = price_wmatic_usdt * 10 ** (18 - 6)

        logging.info(f"📈 slot0 WMATIC price = {adjusted_price:.6f} USDT")

        return float(adjusted_price)

    except Exception as e:
        logging.error(f"❌ Failed to get slot0 price: {e}")
        return None



# ============================================================
# PORTFOLIO FETCHER (USER BALANCE + LP VALUE)
# ============================================================

def fetch_portfolio(uid: int):
    """
    Fetch wallet balances + LP value + total bot portfolio value.
    Based on WMATIC on-chain slot0 price.
    """
    from dashboard.manager import get_user

    try:
        # --------------------------
        # USER
        # --------------------------
        user = get_user(uid)
        if not user:
            return {"error": f"User {uid} not found"}

        owner_address = user["address"]

        # --------------------------
        # BALANCES
        # --------------------------
        usdt_balance = usdt.functions.balanceOf(owner_address).call() / 1e6
        wmatic_balance = wmatic.functions.balanceOf(owner_address).call() / 1e18

        # --------------------------
        # WMATIC PRICE (ON-CHAIN)
        # --------------------------
        wmatic_price = get_wmatic_price_slot0()
        if wmatic_price is None:
            return {"error": "Failed to get WMATIC price from Uniswap V3"}

        # --------------------------
        # LP VALUE
        # --------------------------
        lp_usdt_value = 0.0
        lp_assets_usdt = 0.0
        lp_assets_wmatic = 0.0
        has_lp = False

        try:
            # Manager initialized with owner address
            v3_mgr = UniswapV3Manager(owner_address=owner_address)

            pos_id = v3_mgr.get_active_position_id()

            if pos_id:
                # Uses SAME pricing as UniswapV3Manager → no mismatch
                usdt_val, wmatic_val, total_val = v3_mgr.get_position_asset_value(
                    pos_id,
                    wmatic_price
                )

                lp_usdt_value = total_val
                lp_assets_usdt = usdt_val
                lp_assets_wmatic = wmatic_val
                has_lp = True

                logging.info(f"🦄 LP found for user {uid}: {total_val:.2f} USDT")

            else:
                logging.info(f"🟡 User {uid} has no active LP position.")

        except Exception as e:
            logging.warning(f"⚠️ LP fetch failed for user {uid}: {e}")

        # --------------------------
        # FINAL TOTALS
        # --------------------------
        wallet_value = usdt_balance + (wmatic_balance * wmatic_price)
        total_value = wallet_value + lp_usdt_value

        return {
            "uid": uid,
            "owner": owner_address,

            "usdt_balance": usdt_balance,
            "wmatic_balance": wmatic_balance,
            "wmatic_price": wmatic_price,

            "wallet_value_usdt": wallet_value,

            "lp_value_usdt": lp_usdt_value,
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
