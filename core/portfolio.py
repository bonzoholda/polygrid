# main.py (or core/portfolio.py)

import logging
from config import w3, OWNER, usdt, wmatic

def fetch_portfolio():
    """
    Fetch current portfolio state: USDT + WMATIC balance + total value.
    Returns dict with both balances and total value.
    """
    try:
        usdt_balance = usdt.functions.balanceOf(OWNER).call() / 1e6  # assuming USDT has 6 decimals
        wmatic_balance = wmatic.functions.balanceOf(OWNER).call() / 1e18

        # Get current WMATIC/USDT price from router
        price_path = [wmatic.address, usdt.address]
        amounts = router.functions.getAmountsOut(int(1e18), price_path).call()
        wmatic_price = amounts[-1] / 1e6  # WMATIC in USDT

        total_value = usdt_balance + (wmatic_balance * wmatic_price)

        logging.info(f"💰 Portfolio — USDT: {usdt_balance:.4f}, WMATIC: {wmatic_balance:.4f}, Total: {total_value:.4f} USDT")
        return {
            "usdt_balance": usdt_balance,
            "wmatic_balance": wmatic_balance,
            "wmatic_price": wmatic_price,
            "total_value_usdt": total_value
        }

    except Exception as e:
        logging.error(f"❌ Failed to fetch portfolio: {e}")
        return {"error": str(e)}
