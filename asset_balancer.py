import time
import logging
from config import usdt, wmatic, OWNER, USDT_ADDR, WMATIC_ADDR
from utils import get_onchain_token_balance, get_pol_price_from_okx, swap_usdt_to_wmatic, swap_wmatic_to_usdt

# Ensure logging works even when launched by FastAPI
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

TARGET_RATIO = 0.5      # 50% USDT / 50% WMATIC
THRESHOLD = 0.10        # Trigger rebalance if one side deviates >10%
TRADE_PORTION = 0.5     # Trade only 50% of the deviation
COOLDOWN = 60 * 30      # 30-minute minimum between rebalances


def get_portfolio_value():
    """Return USDT balance, WMATIC balance, their USDT values, and total."""
    price = get_pol_price_from_okx()
    if not price or price <= 0:
        logging.warning("⚠️ Invalid price data received, skipping rebalance.")
        return 0, 0, 0, 0, 0

    # Adjust this depending on utils.py signature
    usdt_bal = get_onchain_token_balance(usdt, OWNER)
    wmatic_bal = get_onchain_token_balance(wmatic, OWNER)

    wmatic_val = wmatic_bal * price
    total = usdt_bal + wmatic_val
    return usdt_bal, wmatic_bal, wmatic_val, total, price


def rebalance_once():
    usdt_bal, wmatic_bal, wmatic_val, total, price = get_portfolio_value()
    if total == 0:
        logging.warning("⚠️ Portfolio total is zero or invalid, skipping rebalance.")
        return False

    target_val = total * TARGET_RATIO
    delta = wmatic_val - target_val

    logging.info(f"📊 Portfolio: USDT=${usdt_bal:.3f} | WMATIC=${wmatic_val:.3f} | Total=${total:.3f}")

    # WMATIC grew → sell some to USDT
    if delta > total * THRESHOLD:
        trade_val = delta * TRADE_PORTION
        trade_amount = trade_val / price
        logging.info(f"🔼 WMATIC grew +{delta/total*100:.2f}%, selling {trade_amount:.4f} WMATIC → USDT")
        swap_wmatic_to_usdt(trade_amount)
        return True

    # WMATIC dropped → buy some with USDT
    elif delta < -total * THRESHOLD:
        trade_val = abs(delta) * TRADE_PORTION
        logging.info(f"🔻 WMATIC dropped {abs(delta)/total*100:.2f}%, buying ${trade_val:.4f} USDT worth of WMATIC")
        swap_usdt_to_wmatic(trade_val)
        return True

    else:
        logging.info("🟢 Portfolio within balance range. No action.")
        return False


def run_asset_balancer():
    """Main loop for Asset Balancer strategy."""
    logging.info("🚀 Starting Asset Balancer Strategy...")
    last_rebalance = 0

    while True:
        now = time.time()
        if now - last_rebalance >= COOLDOWN:
            if rebalance_once():
                last_rebalance = now
        else:
            remaining = COOLDOWN - (now - last_rebalance)
            logging.info(f"⏳ Cooldown active ({remaining:.0f}s left)")
        time.sleep(60)

# ---------- Entry Point ----------
if __name__ == "__main__":
    import logging
    import time

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    logging.info("🚀 Asset Balancer bot initialized.")
    try:
        # Import your main entry function here if it exists (e.g., start_asset_balancer)
        # Otherwise, simulate the continuous balancing loop:
        logging.info("⚙️ Starting Asset Balancer main loop...")

        # Example: placeholder loop until real implementation
        while True:
            # Call your periodic rebalance function here
            time.sleep(10)
            logging.info("🔄 Asset Balancer heartbeat tick...")

    except KeyboardInterrupt:
        logging.info("🛑 Manual stop received. Exiting Asset Balancer gracefully...")
    except Exception as e:
        logging.exception(f"❌ Unexpected error in Asset Balancer: {e}")
