# core/portfolio.py

import logging
from config import w3, router, usdt, wmatic, OWNER  # Added OWNER for verification

def fetch_portfolio(uid: int):
    from dashboard.manager import get_user  # 👈 to access wallet address and keys per user
    """
    Fetch current portfolio state for a given user (by uid):
    - USDT + WMATIC balances
    - WMATIC/USDT price
    - Uniswap V3 Active LP Value (if applicable)
    - Total portfolio value in USDT
    """
    try:
        # --- Get user info from DB
        user = get_user(uid)
        if not user:
            return {"error": f"User with id {uid} not found"}

        owner_address = user["address"]

        # --- Fetch balances (Wallet)
        usdt_balance = usdt.functions.balanceOf(owner_address).call() / 1e6  # USDT = 6 decimals
        wmatic_balance = wmatic.functions.balanceOf(owner_address).call() / 1e18  # WMATIC = 18 decimals

        # --- Get WMATIC→USDT price from router
        # We need this for both wallet calculation AND V3 LP calculation
        price_path = [wmatic.address, usdt.address]
        amounts = router.functions.getAmountsOut(int(1e18), price_path).call()
        wmatic_price = amounts[-1] / 1e6  # 1 WMATIC in USDT

        # --- NEW: Fetch Uniswap V3 LP Value ---
        lp_usdt_value = 0.0
        lp_assets_usdt = 0.0
        lp_assets_wmatic = 0.0
        has_lp = False

        # Only check LP if the portfolio user matches the Bot Config Owner
        # (Because UniswapV3Manager is currently hardcoded to config.OWNER)
        if owner_address.lower() == OWNER.lower():
            try:
                # Local import to prevent circular dependency
                from uniswap_v3_manager import UniswapV3Manager
                
                # Initialize Manager
                v3_mgr = UniswapV3Manager()
                
                # Get Active ID
                active_id = v3_mgr.get_active_position_id()
                
                if active_id:
                    # Calculate Asset Value using the current price we just fetched
                    u_amt, m_amt, total_val = v3_mgr.get_position_asset_value(active_id, wmatic_price)
                    
                    lp_assets_usdt = u_amt
                    lp_assets_wmatic = m_amt
                    lp_usdt_value = total_val
                    has_lp = True
                    logging.info(f"🦄 V3 LP Found: ${lp_usdt_value:.2f} ({u_amt:.2f} USDT, {m_amt:.4f} WMATIC)")
            except Exception as e:
                logging.warning(f"⚠️ Failed to fetch V3 LP data (skipping): {e}")

        # --- Calculate total value (Wallet + LP)
        wallet_value = usdt_balance + (wmatic_balance * wmatic_price)
        total_value = wallet_value + lp_usdt_value

        logging.info(
            f"💰 [User {uid}] Portfolio — "
            f"Wallet: ${wallet_value:.2f} | LP: ${lp_usdt_value:.2f} | "
            f"Total: ${total_value:.2f}"
        )

        return {
            "uid": uid,
            "owner": owner_address,
            "usdt_balance": usdt_balance,
            "wmatic_balance": wmatic_balance,
            "wmatic_price": wmatic_price,
            "wallet_value_usdt": wallet_value, # Explicit wallet only value
            "lp_value_usdt": lp_usdt_value,    # Explicit LP value
            "lp_details": {                    # Details for UI if needed
                "active": has_lp,
                "usdt": lp_assets_usdt,
                "wmatic": lp_assets_wmatic
            },
            "total_value_usdt": total_value    # Grand Total
        }

    except Exception as e:
        logging.error(f"❌ Failed to fetch portfolio for user {uid}: {e}")
        return {"error": str(e)}
