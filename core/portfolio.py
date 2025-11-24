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


# ------------------------------
# SLOT0 ON-CHAIN PRICE
# ------------------------------

def get_wmatic_price_slot0():
    try:
        from config import UNISWAP_POOL_ADDR

        # Manager ONLY for price fetch
        mgr = UniswapV3Manager(
            owner_address=None,
            pool_address=UNISWAP_POOL_ADDR  # must exist in config
        )

        pool = mgr.pool
        if pool is None:
            raise Exception("Pool is None — invalid UNISWAP_POOL_ADDR")

        slot0 = pool.functions.slot0().call()
        sqrtPriceX96 = slot0[0]

        # price = (sqrt(x)/2^96)^2
        base_price = (sqrtPriceX96 ** 2) / (2 ** 192)

        # Adjust WMATIC (18) → USDT (6)
        adjusted = base_price * 1e12

        return float(adjusted)

    except Exception as e:
        logging.error(f"❌ Failed to get slot0 price: {e}")
        return None


# ------------------------------
# PORTFOLIO FETCHER
# ------------------------------

def fetch_portfolio(uid: int):
    from dashboard.manager import get_user

    try:
        user = get_user(uid)
        if not user:
            return {"error": f"User {uid} not found"}

        owner_address = user["address"]

        # ------------------------
        # WALLET BALANCES
        # ------------------------
        usdt_balance = usdt.functions.balanceOf(owner_address).call() / 1e6
        wmatic_balance = wmatic.functions.balanceOf(owner_address).call() / 1e18

        # ------------------------
        # PRICE (on-chain slot0)
        # ------------------------
        wmatic_price = get_wmatic_price_slot0()
        if wmatic_price is None:
            return {"error": "Failed to fetch WMATIC price from Uniswap"}

        # ------------------------
        # LP VALUE (via manager)
        # ------------------------
        lp_usdt_value = 0.0
        lp_assets_usdt = 0.0
        lp_assets_wmatic = 0.0
        has_lp = False

        try:
            # Manager ONLY for LP scanning
            v3_mgr = UniswapV3Manager(owner_address=owner_address)

            pos_id = v3_mgr.get_active_position_id()
            if pos_id:
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
                logging.info(f"User {uid} has no LP positions.")

        except Exception as e:
            logging.warning(f"⚠️ LP fetch failed for user {uid}: {e}")

        # ------------------------
        # TOTAL PORTFOLIO
        # ------------------------
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
