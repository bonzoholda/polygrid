# core/portfolio.py

import logging
from config import w3, router, usdt, wmatic



def fetch_portfolio(uid: int):
    from dashboard.manager import get_user  # 👈 to access wallet address and keys per user
    """
    Fetch current portfolio state for a given user (by uid):
    - USDT + WMATIC balances
    - WMATIC/USDT price
    - Total portfolio value in USDT
    """
    try:
        # --- Get user info from DB
        user = get_user(uid)
        if not user:
            return {"error": f"User with id {uid} not found"}

        owner_address = user["address"]

        # --- Fetch balances
        usdt_balance = usdt.functions.balanceOf(owner_address).call() / 1e6  # USDT = 6 decimals
        wmatic_balance = wmatic.functions.balanceOf(owner_address).call() / 1e18  # WMATIC = 18 decimals

        # --- Get WMATIC→USDT price from router
        price_path = [wmatic.address, usdt.address]
        amounts = router.functions.getAmountsOut(int(1e18), price_path).call()
        wmatic_price = amounts[-1] / 1e6  # 1 WMATIC in USDT

        # --- Calculate total value
        total_value = usdt_balance + (wmatic_balance * wmatic_price)

        logging.info(
            f"💰 [User {uid}] Portfolio — USDT: {usdt_balance:.4f}, "
            f"WMATIC: {wmatic_balance:.4f}, Total: {total_value:.4f} USDT"
        )

        return {
            "uid": uid,
            "owner": owner_address,
            "usdt_balance": usdt_balance,
            "wmatic_balance": wmatic_balance,
            "wmatic_price": wmatic_price,
            "total_value_usdt": total_value
        }

    except Exception as e:
        logging.error(f"❌ Failed to fetch portfolio for user {uid}: {e}")
        return {"error": str(e)}
