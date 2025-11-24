# core/portfolio.py

import logging
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from config import usdt, wmatic
from core.state import get_lp_state


def fetch_portfolio(uid: int):
    from dashboard.manager import get_user

    try:
        user = get_user(uid)
        if not user:
            return {"error": f"User {uid} not found"}

        owner_address = user["address"]
        owner_id = user["id"]

        # Raw wallet balances
        usdt_bal = usdt.functions.balanceOf(owner_address).call() / 1e6
        wmatic_bal = wmatic.functions.balanceOf(owner_address).call() / 1e18

        # LP state updated by bot
        lp = get_lp_state(owner_id)

        if not lp:
            logging.info(f"No LP state for user {uid}")
            lp_value = 0
            lp_usdt = 0
            lp_wmatic = 0
            wmatic_price = 0
            lp_active = False
        else:
            wmatic_price = float(lp["price"])
            lp_value = float(lp["lp_total_value"])
            lp_usdt = float(lp["lp_usdt"])
            lp_wmatic = float(lp["lp_wmatic"])
            lp_active = lp["active"]

        # Wallet total (based on same price bot uses)
        wallet_value = usdt_bal + (wmatic_bal * wmatic_price)
        total_value = wallet_value + lp_value

        return {
            "uid": owner_id,
            "owner": owner_address,

            "usdt_balance": usdt_bal,
            "wmatic_balance": wmatic_bal,
            "wmatic_price": wmatic_price,

            "wallet_value_usdt": wallet_value,

            "lp_value_usdt": lp_value,
            "lp_details": {
                "active": lp_active,
                "usdt": lp_usdt,
                "wmatic": lp_wmatic,
            },

            "total_value_usdt": total_value
        }

    except Exception as e:
        logging.error(f"❌ Portfolio error: {e}")
        return {"error": str(e)}
