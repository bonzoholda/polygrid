import logging
import sys
import os
import traceback

# 1. Fix Import Path (Ensure we can see the manager in root)
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from config import router, usdt, wmatic 
# Note: We don't even need to import OWNER anymore!

def fetch_portfolio(uid: int):
    from dashboard.manager import get_user
    
    try:
        # --- 1. Fetch User & Wallet Data ---
        user = get_user(uid)
        if not user:
            return {"error": f"User {uid} not found"}

        # This is the address we want to check (e.g. 0x5cd...)
        owner_address = user["address"]

        # Fetch Wallet Balances
        usdt_balance = usdt.functions.balanceOf(owner_address).call() / 1e6
        wmatic_balance = wmatic.functions.balanceOf(owner_address).call() / 1e18

        # Get Price
        price_path = [wmatic.address, usdt.address]
        amounts = router.functions.getAmountsOut(int(1e18), price_path).call()
        wmatic_price = amounts[-1] / 1e6

        # --- 2. LP Logic (Direct Check) ---
        lp_usdt_value = 0.0
        lp_assets_usdt = 0.0
        lp_assets_wmatic = 0.0
        has_lp = False
        
        try:
            # Import Manager
            from uniswap_v3_manager import UniswapV3Manager(owner_address = owner_address)
            
            # 🔥 DIRECT BYPASS: Initialize Manager with the specific user address
            # This forces the manager to look at THIS user's wallet, regardless of config.
            v3_mgr = UniswapV3Manager()
            
            # Scan for positions
            active_id = v3_mgr.get_active_position_id()
            
            if active_id:
                u_amt, m_amt, total_val = v3_mgr.get_position_asset_value(active_id, wmatic_price)
                lp_assets_usdt = u_amt
                lp_assets_wmatic = m_amt
                lp_usdt_value = total_val
                has_lp = True
                logging.info(f"🦄 User {uid} LP Found: ${total_val:.2f}")
            else:
                # It's normal for a user to have no LP, just log info
                logging.info(f"User {uid} has no active V3 positions.")

        except Exception as e:
            # Don't crash the whole portfolio if V3 check fails
            logging.warning(f"⚠️ Failed to fetch LP for user {uid}: {e}")

        # --- 3. Final Totals ---
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
                "wmatic": lp_assets_wmatic
            },
            "total_value_usdt": total_value
        }

    except Exception as e:
        logging.error(f"❌ Portfolio Error: {e}")
        return {"error": str(e)}
