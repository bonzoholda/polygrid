# core/portfolio.py
import logging
import sys
import os

# Ensure root path included
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

# reuse w3 from utils for checksumming addresses
from utils import w3
from config import usdt, wmatic
from uniswap_v3_manager import UniswapV3Manager
from core.state import get_lp_state


lp_value_usdt = 0.0
lp_assets_usdt = 0.0
lp_assets_wmatic = 0.0
has_lp = False
wmatic_price = None

def _normalize_price(price):
    try:
        if price is None:
            return None
        price_f = float(price)
        # small safety: if price looks like raw X96 squared integer, it's likely wrong; but manager returns normalized
        # We still round to 6 decimals for human readability
        return round(price_f, 6)
    except Exception:
        return None

def fetch_portfolio(uid: int):
    from dashboard.manager import get_user

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
            # fallback: use as-is
            pass

        # -------------------------------
        # Token balances (human units)
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
        # Prefer reading LP state (set by the running bot)
        # -------------------------------
        lp_state = get_lp_state(owner_id)

        wmatic_price = lp_state.get("price")
        lp_assets_usdt = float(lp_state.get("lp_usdt", 0.0))
        lp_assets_wmatic = float(lp_state.get("lp_wmatic", 0.0))
        lp_value_usdt = float(lp_state.get("lp_total_value", 0.0))
        has_lp = bool(lp_state.get("active", False))
        logging.info(f"Using stored LP state for uid {uid}: price={wmatic_price}, lp_total={lp_value_usdt}")

        
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
                "wmatic": lp_assets_wmatic
            },
            "total_value_usdt": total_value
        }

    except Exception as e:
        logging.error(f"❌ Portfolio error: {e}")
        return {"error": str(e)}
