import logging
import sys
import os

# Ensure root path included
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from config import usdt, wmatic, UNISWAP_POOL_ADDR
from uniswap_v3_manager import UniswapV3Manager


# ----------------------------------------------
# SIMPLE WRAPPER TO READ EXACT MANAGER PRICE
# ----------------------------------------------

def get_wmatic_price_slot0():
    """
    Returns exact same price used inside UniswapV3Manager.
    No custom math, no duplication — prevent mismatches.
    """
    try:
        mgr = UniswapV3Manager(pool_address=UNISWAP_POOL_ADDR)
        price, _ = mgr.get_pool_price_and_tick()
        if price is None:
            raise RuntimeError("No price returned from manager")

        return float(price)

    except Exception as e:
        logging.error(f"❌ Slot0 fetch error: {e}")
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

        # ----------------------------------------------------
        # READ WALLET BALANCES (CORRECT DECIMALS)
        # ----------------------------------------------------
        usdt_balance = usdt.functions.balanceOf(owner_address).call() / 1e6
        wmatic_balance = wmatic.functions.balanceOf(owner_address).call() / 1e18

        # ----------------------------------------------------
        # ON-CHAIN PRICE (IDENTICAL TO BOT LOGIC)
        # ----------------------------------------------------
        wmatic_price = get_wmatic_price_slot0()
        if wmatic_price is None:
            return {"error": "Cannot fetch WMATIC price"}

        # ----------------------------------------------------
        # LP VALUE FROM EXACT UNISWAP MANAGER (NO RAW FORMULA)
        # ----------------------------------------------------
        lp_value_usdt = 0.0
        lp_assets_usdt = 0.0
        lp_assets_wmatic = 0.0
        has_lp = False

        try:
            v3_mgr = UniswapV3Manager(
                owner_address=owner_address,
                pool_address=UNISWAP_POOL_ADDR
            )

            pos_id = v3_mgr.get_active_position_id()

            if pos_id:
                # This already returns normalized floats in USDT and WMATIC
                usdt_part, wmatic_part, total = v3_mgr.get_position_asset_value(
                    pos_id,
                    wmatic_price,
                )

                lp_assets_usdt = float(usdt_part)
                lp_assets_wmatic = float(wmatic_part)
                lp_value_usdt = float(total)
                has_lp = True

                logging.info(
                    f"🦄 Portfolio: User {uid} LP total = {lp_value_usdt:.4f} USDT"
                )

        except Exception as e:
            logging.warning(f"⚠️ LP fetch failed for user {uid}: {e}")

        # ----------------------------------------------------
        # TOTAL PORTFOLIO VALUE
        # ----------------------------------------------------
        wallet_value = float(usdt_balance) + float(wmatic_balance * wmatic_price)
        total_value = wallet_value + lp_value_usdt

        # ----------------------------------------------------
        # FINAL CLEAN JSON
        # ----------------------------------------------------
        return {
            "uid": uid,
            "owner": owner_address,

            "usdt_balance": round(usdt_balance, 6),
            "wmatic_balance": round(wmatic_balance, 6),
            "wmatic_price": round(wmatic_price, 6),

            "wallet_value_usdt": round(wallet_value, 6),

            "lp_value_usdt": round(lp_value_usdt, 6),
            "lp_details": {
                "active": has_lp,
                "usdt": round(lp_assets_usdt, 6),
                "wmatic": round(lp_assets_wmatic, 6),
            },

            "total_value_usdt": round(total_value, 6),
        }

    except Exception as e:
        logging.error(f"❌ Portfolio error: {e}")
        return {"error": str(e)}
