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


def fetch_portfolio(uid: int):
    from dashboard.manager import get_user
    from core.state import get_lp_state

    try:
        user = get_user(uid)
        if not user:
            return {"error": f"User {uid} not found"}

        owner_address = user["address"]

        # -------- WALLET BALANCES --------
        usdt_balance_raw = usdt.functions.balanceOf(owner_address).call()
        wmatic_balance_raw = wmatic.functions.balanceOf(owner_address).call()

        usdt_balance = float(usdt_balance_raw) / 1e6
        wmatic_balance = float(wmatic_balance_raw) / 1e18

        # Defaults
        wmatic_price = 0.0
        lp_value_usdt = 0.0
        lp_details = {"active": False, "usdt": 0.0, "wmatic": 0.0}

        # -------- LP STATE (ALWAYS TRUST THIS) --------
        try:
            lp = get_lp_state()

            if lp and lp.get("active"):
                wmatic_price = lp["price"]
                lp_value_usdt = lp["lp_total_value"]

                lp_details = {
                    "active": True,
                    "usdt": lp["lp_usdt"],
                    "wmatic": lp["lp_wmatic"]
                }

                logging.info(f"🦄 Loaded LP state for uid={uid}: {lp_value_usdt:.2f} USDT")

            else:
                logging.info(f"No LP state for user {uid}")

        except Exception as e:
            logging.warning(f"⚠️ LP state read failed for user {uid}: {e}")

        # -------- PORTFOLIO CALCULATIONS --------
        wallet_value = usdt_balance + (wmatic_balance * wmatic_price)
        total_value = wallet_value + lp_value_usdt

        # -------- RETURN RESPONSE --------
        return {
            "uid": uid,
            "owner": owner_address,
            "usdt_balance": usdt_balance,
            "wmatic_balance": wmatic_balance,
            "wmatic_price": wmatic_price,
            "wallet_value_usdt": wallet_value,
            "lp_value_usdt": lp_value_usdt,
            "lp_details": lp_details,
            "total_value_usdt": total_value
        }

    except Exception as e:
        logging.error(f"❌ Portfolio error: {e}")
        return {"error": str(e)}
