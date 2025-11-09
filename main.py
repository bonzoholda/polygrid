# --- Prevent DeprecationWarnings for pkg_resources (Python 3.12+)
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")

# --- Core system imports ---
import time
import logging
from typing import Optional
from utils import (
    get_pol_price_from_okx,
    get_onchain_token_balance,
    swap_usdt_to_wmatic,
    swap_wmatic_to_usdt,
    get_token_decimals,
    estimate_amounts_out,
    to_decimals,
    from_decimals,
)
from ai_module import MLSignalGeneratorOKX
from config import usdt, wmatic, OWNER, USDT_ADDR, WMATIC_ADDR


# ---------- Position class ----------
class Position:
    def __init__(self, lots_alloc=None, lot_size_usdt=0.0):
        self.lots_alloc = lots_alloc or [1, 1, 2, 3]
        self.lot_size_usdt = lot_size_usdt
        self.buy_prices = []
        self.amounts_wmatic = []
        self.total_usdt_spent = 0.0

    def realized_profit_pct(self, current_price):
        if not self.buy_prices or not self.amounts_wmatic:
            return 0.0
        avg_cost = sum(self.buy_prices) / len(self.buy_prices)
        return ((current_price - avg_cost) / avg_cost) * 100


# ---------- Initialize AI module ----------
ml_signal = MLSignalGeneratorOKX()


# ---------- Main bot behavior ----------
def main_loop(poll_interval=60):
    logging.info("🚀 Starting DEX Grid Bot main loop (with enhanced AI + trailing) ...")
    in_position = False
    position: Optional[Position] = None
    trail_active = False
    trail_peak = 0.0
    TRAIL_LOCK_STEP = 0.002  # 0.2% trailing gap
    MIN_PROFIT_LOCK = 0.009  # trigger trailing after +0.9%

    while True:
        try:
            # --- 1. Entry section ---
            if not in_position:
                logging.info("Checking AI BUY signal ...")
                ai_signal = ml_signal.generate_signal()

                usdt_balance_onchain = get_onchain_token_balance(usdt, OWNER)
                logging.info(f"USDT balance: {usdt_balance_onchain:.6f}")
                logging.info(f"🤖 AI module signal: {ai_signal}")

                if ai_signal == "BUY" and usdt_balance_onchain > 5:
                    lot_values = [1, 1, 2, 3]
                    lot_total_units = sum(lot_values)
                    lot_size = usdt_balance_onchain / lot_total_units
                    position = Position(lots_alloc=lot_values, lot_size_usdt=lot_size)

                    logging.info(f"🟢 BUY signal accepted. Executing initial buy: {lot_size:.6f} USDT")
                    swap_usdt_to_wmatic(lot_size)

                    wmatic_bal = get_onchain_token_balance(wmatic, OWNER)
                    price_now = get_pol_price_from_okx() or 0.0
                    position.buy_prices.append(price_now)
                    position.amounts_wmatic.append(wmatic_bal)
                    position.total_usdt_spent += lot_size
                    in_position = True
                    logging.info(f"Position opened. WMATIC balance: {wmatic_bal:.6f}")
                else:
                    logging.info("No buy signal or insufficient USDT. Sleeping.")
                    time.sleep(poll_interval)
                    continue

            # --- 2. Managing open position ---
            price = get_pol_price_from_okx()
            if price is None:
                logging.warning("Couldn't fetch price; skipping cycle.")
                time.sleep(poll_interval)
                continue

            profit_pct = position.realized_profit_pct(price)
            avg_cost = sum(position.buy_prices) / len(position.buy_prices)
            drawdown_pct = ((price / avg_cost) - 1) * 100
            logging.info(f"📈 Price={price:.6f} | Profit={profit_pct:.2f}% | Drawdown={drawdown_pct:.2f}%")

            # --- 3. Profit & Trailing Logic ---
            if not trail_active and profit_pct >= MIN_PROFIT_LOCK * 100:
                trail_active = True
                trail_peak = profit_pct
                logging.info(f"🔒 Trailing profit lock activated at +{trail_peak:.2f}%")

            if trail_active:
                trail_peak = max(trail_peak, profit_pct)
                if profit_pct <= trail_peak - TRAIL_LOCK_STEP * 100:
                    logging.info(f"💰 Trailing stop triggered: locked {trail_peak:.2f}%, now {profit_pct:.2f}%")
                    total_wmatic = get_onchain_token_balance(wmatic, OWNER)
                    if total_wmatic > 0:
                        swap_wmatic_to_usdt(total_wmatic)
                    in_position = False
                    position = None
                    trail_active = False
                    continue

            # --- 4. DCA Logic (AI + Grid Combo) ---
            ai_signal = ml_signal.generate_signal()
            last_price = position.buy_prices[-1]
            triggers = [-0.03, -0.07, -0.15]

            for idx, trig in enumerate(triggers, start=1):
                if len(position.amounts_wmatic) <= idx:
                    target_price = last_price * (1 + trig)
                    if price <= target_price or (ai_signal == "BUY" and drawdown_pct <= -3):
                        lot_to_buy = position.lots_alloc[idx]
                        amount_usdt = lot_to_buy * position.lot_size_usdt
                        logging.info(f"🟡 DCA Trigger idx={idx}: buying {lot_to_buy} lots ({amount_usdt:.2f} USDT)")
                        swap_usdt_to_wmatic(amount_usdt)

                        new_wmatic_bal = get_onchain_token_balance(wmatic, OWNER)
                        delta = new_wmatic_bal - sum(position.amounts_wmatic)
                        if delta <= 0:
                            dec_usdt = get_token_decimals(usdt)
                            amt_in = to_decimals(amount_usdt, dec_usdt)
                            est = estimate_amounts_out(amt_in, [USDT_ADDR, WMATIC_ADDR])
                            if est:
                                delta = from_decimals(est[-1], get_token_decimals(wmatic))

                        position.amounts_wmatic.append(delta)
                        position.buy_prices.append(price)
                        position.total_usdt_spent += amount_usdt
                        logging.info(f"✅ DCA executed at {price:.6f}")
                        break

            # --- 5. Stop-loss fallback ---
            if price < avg_cost * 0.88:
                logging.warning("🔻 Stop-loss hit → selling to preserve capital.")
                total_wmatic = get_onchain_token_balance(wmatic, OWNER)
                if total_wmatic > 0:
                    swap_wmatic_to_usdt(total_wmatic)
                in_position = False
                position = None
                trail_active = False
                continue

            time.sleep(poll_interval)

        except Exception as exc:
            logging.exception("Main loop error, retrying after short sleep.")
            time.sleep(10)


# ---------- Entry Point ----------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logging.info("🚀 Bot container initialized. Starting trading loop...")
    try:
        main_loop(poll_interval=60)  # adjust polling interval (seconds)
    except KeyboardInterrupt:
        logging.info("🛑 Manual stop received. Exiting gracefully...")
