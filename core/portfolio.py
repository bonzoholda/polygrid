# core/portfolio.py

import logging
import sys
import os

# Ensure root directory is in python path to find uniswap_v3_manager
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import w3, router, usdt, wmatic, OWNER

def fetch_portfolio(uid: int):
    from dashboard.manager import get_user
    """
    Fetch current portfolio state with VERBOSE DEBUGGING for LP calculation.
    """
    try:
        # --- 1. Basic User & Wallet Data ---
        user = get_user(uid)
        if not user:
            return {"error": f"User with id {uid} not found"}

        owner_address = user["address"]

        # Fetch balances
        usdt_balance = usdt.functions.balanceOf(owner_address).call() / 1e6
        wmatic_balance = wmatic.functions.balanceOf(owner_address).call() / 1e18

        # Get Price
        price_path = [wmatic.address, usdt.address]
        amounts = router.functions.getAmountsOut(int(1e18), price_path).call()
        wmatic_price = amounts[-1] / 1e6

        # --- 2. Uniswap V3 LP Calculation (Debug Section) ---
        lp_usdt_value = 0.0
        lp_assets_usdt = 0.0
        lp_assets_wmatic = 0.0
        has_lp = False

        logging.info(f"🔍 Checking LP for User: {owner_address}")
        logging.info(f"🔍 Config OWNER: {OWNER}")

        # Check Owner Match
        if owner_address.lower() == OWNER.lower():
            logging.info("✅ Owner Match confirmed. Attempting to load V3 Manager...")
            try:
                # debug: try import explicitly
                import uniswap_v3_manager
                logging.info(f"✅ Module imported: {uniswap_v3_manager}")
                
                from uniswap_v3_manager import UniswapV3Manager
                
                v3_mgr = UniswapV3Manager()
                logging.info("✅ V3 Manager Initialized in Portfolio.")

                active_id = v3_mgr.get_active_position_id()
                logging.info(f"🦄 Portfolio fetched Active ID: {active_id}")
                
                if active_id:
                    u_amt, m_amt, total_val = v3_mgr.get_position_asset_value(active_id, wmatic_price)
                    lp_assets_usdt = u_amt
                    lp_assets_wmatic = m_amt
                    lp_usdt_value = total_val
                    has_lp = True
                    logging.info(f"💰 LP Value Calculated: ${total_val}")
                else:
                    logging.warning("⚠️ Manager initialized but returned NO Active ID.")

            except ImportError as e:
                logging.error(f"❌ IMPORT ERROR: Could not import UniswapV3Manager. Path issue? {e}")
                logging.error(f"Current Sys Path: {sys.path}")
            except Exception as e:
                logging.exception(f"❌ CRITICAL ERROR inside LP block: {e}")
        else:
            logging.warning("⚠️ Owner Mismatch: Skipping LP check (User != Config Owner)")

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
        logging.error(f"❌ Failed to fetch portfolio for user {uid}: {e}")
        return {"error": str(e)}
