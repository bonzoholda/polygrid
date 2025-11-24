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
# SLOT0 PRICE READER (ON-CHAIN WMATIC → USDT)
# ----------------------------------------------

def get_wmatic_price_slot0():
    """
    Use UniswapV3Manager.get_pool_price_and_tick() so we reuse the exact same slot0 logic
    and avoid duplicate implementations / mismatches.
    Returns a float WMATIC price in USDT (rounded).
    """
    try:
        # instantiate manager with pool address from config (no owner needed)
        from config import UNISWAP_POOL_ADDR
        mgr = UniswapV3Manager(owner_address=None, pool_address=UNISWAP_POOL_ADDR)

        # call manager's helper which already returns (price_human, tick)
        pool_price, _ = mgr.get_pool_price_and_tick()

        if pool_price is None:
            raise RuntimeError("UniswapV3Manager returned no pool price")

        # Normalize: ensure it's a float and not some huge raw integer
        price_float = float(pool_price)

        # Round sensibly for frontend (6 decimals is good for USDT/WMATIC)
        return round(price_float, 6)

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
        # Token balances
        # -------------------------------
        usdt_balance = usdt.functions.balanceOf(owner_address).call() / 1e6
        wmatic_balance = wmatic.functions.balanceOf(owner_address).call() / 1e18

        # -------------------------------
        # Slot0 price
        # -------------------------------
        wmatic_price = get_wmatic_price_slot0()
        if wmatic_price is None:
            return {"error": "Cannot fetch WMATIC on-chain price"}

        # -------------------------------
        # LP Position Valuation
        # -------------------------------
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
                u_val, m_val, total = v3_mgr.get_position_asset_value(
                    pos_id,
                    wmatic_price
                )
                lp_value_usdt = total
                lp_assets_usdt = u_val
                lp_assets_wmatic = m_val
                has_lp = True

                logging.info(f"🦄 LP found for user {uid}: {total:.2f} USDT")

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
