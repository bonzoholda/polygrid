import logging
import sys
import os

# Set up pathing to ensure imports work from the root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

# NOTE: The following imports are assumed to be available from your project structure
from config import usdt, wmatic, OWNER # Assuming OWNER is used in the mock
from core.state import get_lp_state
# Need a way to get the current price if the bot hasn't run yet
from utils import get_pol_price_from_okx

# --- Mock Dependencies (Replace with your actual implementation) ---
def get_user(user_id):
    """MOCK: Retrieves user data from the dashboard/manager."""
    # Assuming user ID 1 is the bot's configured user for testing
    if user_id == 1:
        # NOTE: Replace '0xOwnerAddressHere' with the actual address from config.
        return {"address": OWNER, "id": 1}
    return None
# --- End Mock Dependencies ---


def fetch_portfolio(uid: int):
    """
    Fetches the combined portfolio value (Wallet + LP Position) for a user.
    """
    try:
        user = get_user(uid)
        if not user:
            return {"error": f"User {uid} not found"}

        owner_address = user["address"]
        owner_id = user["id"]

        # Raw wallet balances
        # These contract calls are generally correct but require 'usdt' and 'wmatic' 
        # to be initialized contract objects.
        usdt_bal = usdt.functions.balanceOf(owner_address).call() / 1e6
        wmatic_bal = wmatic.functions.balanceOf(owner_address).call() / 1e18

        # LP state updated by bot (reads from core/state.py)
        lp = get_lp_state(owner_id)

        # --- 1. Determine WMATIC Price (Fallback Logic) ---
        wmatic_price = 0.0
        
        if lp:
            # Use the price saved by the bot, which is most current/relevant
            wmatic_price = float(lp["price"])
        else:
            # If no LP state or price is available, fetch the latest price as a fallback
            logging.warning(f"No bot-saved price for user {uid}. Fetching current CEX price.")
            wmatic_price = get_pol_price_from_okx()
            if not wmatic_price:
                logging.error("Failed to fetch WMATIC price, cannot calculate portfolio value.")
                return {"error": "Price lookup failed."}


        # --- 2. Process LP State ---
        if not lp:
            logging.info(f"No active LP state for user {uid}")
            lp_value = 0.0
            lp_usdt = 0.0
            lp_wmatic = 0.0
            lp_active = False
        else:
            # All fields are correctly extracted from the LP state
            lp_value = float(lp["lp_total_value"])
            lp_usdt = float(lp["lp_usdt"])
            lp_wmatic = float(lp["lp_wmatic"])
            lp_active = lp["active"]

        # --- 3. Calculate Totals ---
        # Wallet total (now uses the determined wmatic_price)
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
        # Log the full traceback for better debugging
        logging.exception(f"❌ Portfolio fetch error for user {uid}: {e}")
        return {"error": str(e)}
