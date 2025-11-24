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

def get_wmatic_price_slot0():
    try:
        try:
            from config import UNISWAP_POOL_ADDR
        except Exception:
            UNISWAP_POOL_ADDR = None

        if UNISWAP_POOL_ADDR:
            mgr = UniswapV3Manager(owner_address=None, pool_address=UNISWAP_POOL_ADDR)
        else:
            mgr = UniswapV3Manager(owner_address=None)

        pool_price, _ = mgr.get_pool_price_and_tick()
        if pool_price is None:
            logging.error("❌ Failed to get pool_price from UniswapV3Manager (slot0).")
            return None

        # round to 6 decimals for frontend friendliness
        return round(float(pool_price), 6)
    except Exception as e:
        logging.error(f"❌ Failed to get slot0 price via manager: {e}")
        return None

def fetch_portfolio(uid: int):
    from dashboard.manager import get_user

    try:
        user = get_user(uid)
        if not user:
            return {"error": f"User {uid} not found"}

        owner_address = user["address"]

        usdt_balance_raw = usdt.functions.balanceOf(owner_address).call()
        wmatic_balance_raw = wmatic.functions.balanceOf(owner_address).call()

        usdt_balance = float(usdt_balance_raw) / 1e6
        wmatic_balance = float(wmatic_balance_raw) / 1e18

        wmatic_price = get_wmatic_price_slot0()
        if wmatic_price is None:
            logging.error("❌ Cannot fetch WMATIC on-chain price (slot0). Falling back to 0.0")
            wmatic_price = 0.0

        lp_value_usdt = 0.0
        lp_assets_usdt = 0.0
        lp_assets_wmatic = 0.0
        has_lp = False

        try:
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
