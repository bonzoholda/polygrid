# --- Prevent DeprecationWarnings for pkg_resources (Python 3.12+)
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="pkg_resources is deprecated")

import time, logging
from typing import Optional
from collections import deque
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

# ---------- Real-time bot state for dashboard ----------
MAX_LOG_LINES = 50
bot_state = {
    "usdt_balance": 0.0,
    "ai_signal": "--",
    "confidence": 0.0,
    "rsi": 0.0,
    "momentum": 0.0,
    "logs": deque(maxlen=MAX_LOG_LINES)
}

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
    logging.info("🚀 Starting DEX Grid Bot main loop ...")
    in_position = False
    position: Optional[Position] = None
    trail_active = False
    trail_peak = 0.0
    TRAIL_LOCK_STEP = 0.002  # 0.2% trailing gap
    MIN_PROFIT_LOCK = 0.009  # 0.9%

    while True:
        try:
            # --- 1. Entry ---
            if not in_position:
                ai_signal = ml_signal.generate_signal()
                usdt_balance_onchain = get_onchain_token_balance(usdt, OWNER)

                # Update bot_state
                bot_state["usdt_balance"] = usdt_balance_onchain
                bot_state["ai_signal"] = ai_signal
                bot_state["confidence"] = getattr(ml_signal, "last_confidence", 0.0)
                bot_state["rsi"] = getattr(ml_signal, "last_rsi", 0.0)
                bot_state["momentum"] = getattr(ml_signal, "last_momentum", 0.0)

                log_line = f"{time.strftime('%H:%M:%S')} | USDT: {usdt_balance_onchain:.4f} | AI: {ai_signal}"
                bot_state["logs"].append(log_line)
                logging.info(log_line)

                if ai_signal == "BUY" and usdt_balance_onchain > 5:
                    lot_values = [1, 1, 2, 3]
                    lot_total_units = sum(lot_values)
                    lot_size = usdt_balance_onchain / lot_total_units
                    position = Position(lots_alloc=lot_values, lot_size_usdt=lot_size)
                    swap_usdt_to_wmatic(lot_size)
                    wmatic_bal = get_onchain_token_balance(wmatic, OWNER)
                    price_now = get_pol_price_from_okx() or 0.0
                    position.buy_prices.append(price_now)
                    position.amounts_wmatic.append(wmatic_bal)
                    position.total_usdt_spent += lot_size
                    in_position = True
                    bot_state["logs"].append(f"{time.strftime('%H:%M:%S')} | BUY executed: {lot_size:.4f} USDT")
                else:
                    time.sleep(poll_interval)
                    continue

            # --- 2. Manage position ---
            price = get_pol_price_from_okx()
            if not price:
                time.sleep(poll_interval)
                continue

            profit_pct = position.realized_profit_pct(price)
            avg_cost = sum(position.buy_prices)/len(position.buy_prices)
            drawdown_pct = ((price/avg_cost)-1)*100

            bot_state["logs"].append(f"{time.strftime('%H:%M:%S')} | Price={price:.4f} | Profit={profit_pct:.2f}%")

            # Trailing
            if not trail_active and profit_pct >= MIN_PROFIT_LOCK*100:
                trail_active = True
                trail_peak = profit_pct
                bot_state["logs"].append(f"🔒 Trailing started at +{trail_peak:.2f}%")

            if trail_active:
                trail_peak = max(trail_peak, profit_pct)
                if profit_pct <= trail_peak - TRAIL_LOCK_STEP*100:
                    total_wmatic = get_onchain_token_balance(wmatic, OWNER)
                    if total_wmatic > 0:
                        swap_wmatic_to_usdt(total_wmatic)
                    in_position = False
                    position = None
                    trail_active = False
                    bot_state["logs"].append(f"💰 Trailing stop triggered. Locked profit: {trail_peak:.2f}%")
                    continue

            time.sleep(poll_interval)

        except Exception as exc:
            logging.exception("Bot loop error")
            bot_state["logs"].append(f"{time.strftime('%H:%M:%S')} | ERROR: {str(exc)}")
            time.sleep(5)

# ---------- Entry point ----------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    from threading import Thread
    Thread(target=main_loop, daemon=True).start()
    logging.info("🚀 Bot container initialized.")
