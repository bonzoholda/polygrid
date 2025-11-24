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
        lp_state = get_lp_state(uid)
        lp_value_usdt = 0.0
        lp_assets_usdt = 0.0
        lp_assets_wmatic = 0.0
        has_lp = False
        wmatic_price = None

        if lp_state:
            # Use stored state (this was updated by uniswap_v3_manager for that uid)
            try:
                wmatic_price = lp_state.get("price") or lp_state.get("wmatic_price")
                lp_assets_usdt = float(lp_state.get("lp_usdt", 0.0))
                lp_assets_wmatic = float(lp_state.get("lp_wmatic", 0.0))
                lp_value_usdt = float(lp_state.get("lp_total_value", lp_state.get("lp_value_usdt", 0.0)))
                has_lp = bool(lp_state.get("active", False))
                logging.info(f"Using stored LP state for uid {uid}: price={wmatic_price}, lp_total={lp_value_usdt}")
            except Exception as e:
                logging.warning(f"Failed to parse lp_state for uid {uid}: {e}")
                lp_state = None

        if lp_state is None:
            # No live state — attempt to read directly on-chain using UniswapV3Manager
            try:
                # try to use pool address from config if available; manager will warn/fallback itself
                try:
                    from config import UNISWAP_POOL_ADDR
                    mgr = UniswapV3Manager(owner_address=owner_address, pool_address=UNISWAP_POOL_ADDR)
                except Exception:
                    mgr = UniswapV3Manager(owner_address=owner_address)

                # get price (on-chain pool preferred)
                pool_price, _ = mgr.get_pool_price_and_tick()
                if pool_price is not None:
                    wmatic_price = pool_price
                else:
                    # fallback: try using current wallet logic (router) — but we keep consistent with manager's fallback
                    # ask manager for OKX fallback price if needed
                    try:
                        # manager.get_position_asset_value will itself call OKX fallback if needed
                        pass
                    except Exception:
                        pass

                pos_id = mgr.get_active_position_id()
                if pos_id:
                    u_amt, m_amt, total_val = mgr.get_position_asset_value(pos_id, wmatic_price)
                    lp_assets_usdt = float(u_amt)
                    lp_assets_wmatic = float(m_amt)
                    lp_value_usdt = float(total_val)
                    has_lp = True
                    logging.info(f"🦄 Direct on-chain LP calc for user {uid}: ${lp_value_usdt:.2f}")
                else:
                    logging.info(f"User {uid} has no V3 LP positions (direct on-chain).")
            except Exception as e:
                logging.warning(f"⚠️ LP direct fetch failed for user {uid}: {e}")

        # -------------------------------
        # Final price normalization
        # -------------------------------
        wmatic_price = _normalize_price(wmatic_price) if wmatic_price is not None else None
        # If still None, set to 0 to avoid crazy arithmetic in frontend
        if wmatic_price is None:
            wmatic_price = 0.0

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
