# core/portfolio.py
import logging
import sys
import os

# Ensure root path included
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from utils import w3
from config import usdt, wmatic
from uniswap_v3_manager import UniswapV3Manager
from core.state import get_lp_state

manager = UniswapV3Manager()
BOT_UID = int(os.getenv("BOT_UID", "0"))

def _normalize_price(price):
    try:
        if price is None:
            return 0.0
        return round(float(price), 6)
    except Exception:
        return 0.0

def fetch_portfolio(uid: int):
    from dashboard.manager import get_user

    # Initialize safe defaults
    lp_assets_usdt = 0.0
    lp_assets_wmatic = 0.0
    lp_value_usdt = 0.0
    wmatic_price = 0.0
    has_lp = False

    try:
        user = get_user(uid)
        if not user:
            return {"error": f"User {uid} not found"}

        owner_address = user["address"]
        owner_id = user["id"]
        # ensure checksum
        try:
            owner_address = w3.to_checksum_address(owner_address)
        except Exception:
            pass

        # -------------------------------
        # Token balances
        # -------------------------------
        try:
            usdt_balance_raw = usdt.functions.balanceOf(owner_address).call()
            wmatic_balance_raw = wmatic.functions.balanceOf(owner_address).call()
        except Exception as e:
            logging.error(f"❌ Failed to read on-chain balances for {owner_address}: {e}")
            return {"error": "Failed to read on-chain balances"}

        usdt_balance = float(usdt_balance_raw) / 1e6
        wmatic_balance = float(wmatic_balance_raw) / 1e18

        # -------------------------------
        # LP State
        # -------------------------------
        lp_state = get_lp_state(uid)
        logging.info(f"Extracting LP state for uid {uid}: {lp_state}")

        if lp_state:
            # Use stored state
            wmatic_price = _normalize_price(lp_state.get("wmatic_price", 0.0))
            lp_assets_usdt = float(lp_state.get("lp_usdt", 0.0))
            lp_assets_wmatic = float(lp_state.get("lp_wmatic", 0.0))
            lp_value_usdt = float(lp_state.get("lp_total_value", 0.0))
            has_lp = bool(lp_state.get("active", False))
            logging.info(f"Using stored LP state: price={wmatic_price}, total={lp_value_usdt}")
        else:
            # Fallback: fetch directly from UniswapV3Manager
            active_id = manager.get_active_position_id()
            if active_id:
                try:
                    # Get normalized pool price in USDT
                    wmatic_price = manager.get_pool_price_in_usdt()
            
                    # Get position liquidity in human-readable decimals
                    token0_raw, token1_raw = manager.get_position_liquidity(active_id)
                    usdt_amt = token0_raw / 1e6   # USDT has 6 decimals
                    wmatic_amt = token1_raw / 1e18  # WMATIC has 18 decimals
            
                    # Total LP value in USDT
                    total_value = usdt_amt + wmatic_amt * wmatic_price
            
                    # Assign LP details
                    lp_assets_usdt = usdt_amt
                    lp_assets_wmatic = wmatic_amt
                    lp_value_usdt = total_value
                    has_lp = True

                    # Optionally update core state for future reads
                    state_data = {
                        "wmatic_price": wmatic_price,
                        "lp_usdt": lp_assets_usdt,
                        "lp_wmatic": lp_assets_wmatic,
                        "lp_total_value": lp_value_usdt,
                        "active": True
                    }
                    logging.info(f"Updating core state: {state_data}")
                    # update_lp_state(uid, state_data)  # uncomment if you want to persist
                except Exception as e:
                    logging.error(f"❌ Failed to fetch on-chain LP: {e}")
                    wmatic_price = 0.0
                    lp_assets_usdt = 0.0
                    lp_assets_wmatic = 0.0
                    lp_value_usdt = 0.0
                    has_lp = False

        # -------------------------------
        # Total portfolio
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
                "wmatic": lp_assets_wmatic
            },
            "total_value_usdt": total_value
        }

    except Exception as e:
        logging.error(f"❌ Portfolio error: {e}")
        return {"error": str(e)}
