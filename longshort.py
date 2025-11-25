# --- Prevent DeprecationWarnings for pkg_resources (Python 3.12+)
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")

# --- Core system imports ---
import time
import logging
import threading
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
from uniswap_v3_manager import run_uniswap_v3_loop

# Optional import of asset_balancer
try:
    from asset_balancer import run_asset_balancer
except ImportError:
    run_asset_balancer = None


# ---------- Position class ----------
class Position:
    def __init__(self, lots_alloc=None, lot_size_usdt=0.0):
        # keep default allocation for backward compatibility; new loop uses 5 lots logic separately
        self.lots_alloc = lots_alloc or [1, 1, 2, 3]
        self.lot_size_usdt = lot_size_usdt
        self.buy_prices = []
        self.amounts_wmatic = []
        self.total_usdt_spent = 0.0
        # new fields for simple long/short position tracking
        self.position_type = None  # "long" or "short"
        self.wmatic_amount_open = 0.0
        self.usdt_amount_open = 0.0

    def realized_profit_pct(self, current_price):
        if self.position_type is None or self.position_type == "short" and self.wmatic_amount_open == 0:
            return 0.0
        if self.position_type == "long":
            if not self.buy_prices or not self.amounts_wmatic:
                return 0.0
            avg_cost = sum(self.buy_prices) / len(self.buy_prices)
            return ((current_price - avg_cost) / avg_cost) * 100
        # For short we can compute approx P/L using the open amounts (not used currently)
        return 0.0


# ---------- Initialize AI module ----------
ml_signal = MLSignalGeneratorOKX()


# ---------- Signal-guided Long/Short Loop ----------
def longshort_loop(poll_interval=60):
    """
    Modified main loop:
    - Uses ai_module to get BUY/SELL/HOLD.
    - Splits USDT into 5 lots; each trade costs 1 lot.
    - If USDT_value(WMATIC) < USDT_balance -> wait BUY to open LONG (buy WMATIC), close on SELL.
    - If USDT_value(WMATIC) > USDT_balance -> wait SELL to open SHORT (sell WMATIC), close on BUY.
    - While Long open: ignore BUY signals. While Short open: ignore SELL signals.
    - Repeat after each close.
    """
    logging.info("🚀 Starting Signal-Guided Long/Short loop ...")
    in_position = False
    position: Optional[Position] = None

    while True:
        try:
            # fetch balances & price
            usdt_balance_onchain = get_onchain_token_balance(usdt, OWNER)
            wmatic_balance = get_onchain_token_balance(wmatic, OWNER)
            price = get_pol_price_from_okx()
            if price is None:
                logging.warning("Couldn't fetch price; retrying after sleep.")
                time.sleep(poll_interval)
                continue

            # estimate WMATIC value in USDT
            wmatic_value_usdt = wmatic_balance * price
            logging.info(f"USDT balance: {usdt_balance_onchain:.6f} | WMATIC balance: {wmatic_balance:.6f} (~{wmatic_value_usdt:.6f} USDT)")
            
            # compute lot size (split USDT balance into 5 lots)
            # Use the current USDT balance as base for lot calculation (if base is zero, use estimated from WMATIC)
            lot_base_usdt = max(usdt_balance_onchain, 0.0)
            lots_count = 5
            # avoid division by zero
            if lot_base_usdt <= 0:
                logging.info("USDT balance is zero or negative — cannot compute lots. Sleeping.")
                time.sleep(poll_interval)
                continue
            lot_size = lot_base_usdt / lots_count
            logging.debug(f"Lot size (USDT) = {lot_size:.6f}")

            # get AI signal
            ai_signal = ml_signal.generate_signal()
            logging.info(f"🤖 AI signal: {ai_signal}")

            # ---------- If no position -> decide which side to wait for ----------
            if not in_position:
                # Decide market side by comparing balances
                if wmatic_value_usdt < usdt_balance_onchain:
                    # More USDT than WMATIC value -> prefer LONG on BUY signal
                    logging.debug("Preference: LONG (USDT > WMATIC_value). Waiting BUY to open LONG.")
                    if ai_signal == "BUY":
                        # attempt to open long using 1 lot
                        amount_usdt_to_spend = lot_size
                        if amount_usdt_to_spend < 1:  # minimal sanity check
                            logging.warning("Lot size too small to open position. Sleeping.")
                            time.sleep(poll_interval)
                            continue

                        logging.info(f"🟢 Opening LONG: swapping {amount_usdt_to_spend:.6f} USDT -> WMATIC")
                        pre_wmatic = get_onchain_token_balance(wmatic, OWNER)
                        success = swap_usdt_to_wmatic(amount_usdt_to_spend)
                        if not success:
                            logging.warning("⚠️ swap_usdt_to_wmatic failed. Will retry later.")
                            time.sleep(poll_interval)
                            continue
                        time.sleep(2)  # small wait to let onchain reflect (optional)
                        post_wmatic = get_onchain_token_balance(wmatic, OWNER)
                        delta_wmatic = max(post_wmatic - pre_wmatic, 0.0)

                        if delta_wmatic <= 0:
                            # fallback: estimate via price if swap didn't update quickly
                            delta_wmatic = amount_usdt_to_spend / price
                            logging.warning("Swap result not immediately visible; using estimate for WMATIC amount.")

                        # record position
                        position = Position(lot_size_usdt=lot_size)
                        position.position_type = "long"
                        position.wmatic_amount_open = delta_wmatic
                        position.usdt_amount_open = amount_usdt_to_spend
                        position.buy_prices.append(price)
                        position.amounts_wmatic.append(delta_wmatic)
                        position.total_usdt_spent += amount_usdt_to_spend
                        in_position = True
                        logging.info(f"🔓 LONG opened: WMATIC +{delta_wmatic:.6f} @ price {price:.6f}")
                    else:
                        logging.debug("No BUY signal yet. Sleeping.")
                        time.sleep(poll_interval)
                        continue

                else:
                    # WMATIC value > USDT -> prefer SHORT on SELL signal
                    logging.debug("Preference: SHORT (WMATIC_value > USDT). Waiting SELL to open SHORT.")
                    if ai_signal == "SELL":
                        # open short by selling WMATIC amount equivalent to 1 lot USDT
                        # compute wmatic amount to sell
                        wmatic_to_sell = lot_size / price
                        # ensure we have enough WMATIC
                        if wmatic_to_sell > wmatic_balance:
                            logging.warning(f"Not enough WMATIC to open short. Required {wmatic_to_sell:.6f}, available {wmatic_balance:.6f}. Sleeping.")
                            time.sleep(poll_interval)
                            continue

                        logging.info(f"🔴 Opening SHORT: swapping {wmatic_to_sell:.6f} WMATIC -> USDT (approx {lot_size:.6f} USDT)")
                        pre_usdt = get_onchain_token_balance(usdt, OWNER)
                        success = swap_wmatic_to_usdt(wmatic_to_sell)
                        if not success:
                            logging.warning("⚠️ swap_wmatic_to_usdt failed. Will retry later.")
                            time.sleep(poll_interval)
                            continue
                        time.sleep(2)
                        post_usdt = get_onchain_token_balance(usdt, OWNER)
                        delta_usdt = max(post_usdt - pre_usdt, 0.0)

                        if delta_usdt <= 0:
                            # fallback: estimate
                            delta_usdt = wmatic_to_sell * price
                            logging.warning("Swap result not immediately visible; using estimate for USDT received.")

                        # record position
                        position = Position(lot_size_usdt=lot_size)
                        position.position_type = "short"
                        position.wmatic_amount_open = wmatic_to_sell
                        position.usdt_amount_open = delta_usdt
                        position.buy_prices.append(price)
                        position.amounts_wmatic.append(-wmatic_to_sell)  # negative to denote sold
                        position.total_usdt_spent += 0.0  # for shorts, total_usdt_spent not used similarly
                        in_position = True
                        logging.info(f"🔓 SHORT opened: WMATIC -{wmatic_to_sell:.6f} @ price {price:.6f}")
                    else:
                        logging.debug("No SELL signal yet. Sleeping.")
                        time.sleep(poll_interval)
                        continue

            # ---------- If in position -> wait for opposite signal to close ----------
            else:
                # refresh latest info
                usdt_balance_onchain = get_onchain_token_balance(usdt, OWNER)
                wmatic_balance = get_onchain_token_balance(wmatic, OWNER)
                price = get_pol_price_from_okx()
                if price is None:
                    logging.warning("Couldn't fetch price; skipping cycle.")
                    time.sleep(poll_interval)
                    continue

                logging.info(f"In position: {position.position_type} | Open WMATIC={position.wmatic_amount_open:.6f} | Open USDT={position.usdt_amount_open:.6f} | Current price={price:.6f}")
                # re-fetch ai signal
                ai_signal = ml_signal.generate_signal()
                logging.info(f"🤖 AI signal while in-position: {ai_signal}")

                if position.position_type == "long":
                    # while long, ignore BUY signals, close only on SELL
                    if ai_signal == "SELL":
                        logging.info("🟠 SELL signal detected -> closing LONG (sell opened WMATIC).")
                        # attempt to sell the exact wmatic amount opened (or as much as possible)
                        # fetch current wmatic balance to ensure we don't oversell other holdings
                        current_wmatic = get_onchain_token_balance(wmatic, OWNER)
                        amount_to_sell = min(position.wmatic_amount_open, current_wmatic)
                        if amount_to_sell <= 0:
                            logging.warning("No WMATIC available to sell. Marking position closed.")
                            in_position = False
                            position = None
                            time.sleep(poll_interval)
                            continue

                        success = swap_wmatic_to_usdt(amount_to_sell)
                        if not success:
                            logging.warning("⚠️ swap_wmatic_to_usdt failed when closing LONG. Retrying later.")
                            time.sleep(poll_interval)
                            continue

                        logging.info(f"✅ LONG closed: sold {amount_to_sell:.6f} WMATIC.")
                        in_position = False
                        position = None
                        # continue loop for next cycle
                        time.sleep(poll_interval)
                        continue
                    else:
                        logging.debug("LONG open -> ignoring BUY/HOLD. Waiting for SELL.")
                        time.sleep(poll_interval)
                        continue

                elif position.position_type == "short":
                    # while short, ignore SELL signals, close only on BUY
                    if ai_signal == "BUY":
                        logging.info("🔵 BUY signal detected -> closing SHORT (buy WMATIC back using USDT).")
                        # attempt to buy WMATIC using the USDT we got when opening the short (or available USDT)
                        # determine amount_usdt_to_use (use min of opened USDT and current USDT balance)
                        current_usdt = get_onchain_token_balance(usdt, OWNER)
                        amount_usdt_available = min(position.usdt_amount_open, current_usdt)
                        if amount_usdt_available <= 0:
                            logging.warning("No USDT available to buy WMATIC to close short. Marking position closed.")
                            in_position = False
                            position = None
                            time.sleep(poll_interval)
                            continue

                        success = swap_usdt_to_wmatic(amount_usdt_available)
                        if not success:
                            logging.warning("⚠️ swap_usdt_to_wmatic failed when closing SHORT. Retrying later.")
                            time.sleep(poll_interval)
                            continue

                        logging.info(f"✅ SHORT closed: bought WMATIC using {amount_usdt_available:.6f} USDT.")
                        in_position = False
                        position = None
                        time.sleep(poll_interval)
                        continue
                    else:
                        logging.debug("SHORT open -> ignoring SELL/HOLD. Waiting for BUY.")
                        time.sleep(poll_interval)
                        continue

            # small sleep for safety (shouldn't usually reach here)
            time.sleep(poll_interval)

        except Exception as exc:
            logging.exception("Main loop error, retrying after short sleep.")
            time.sleep(10)


# ---------- Strategy Switcher ----------
active_strategy = None
bot_thread = None


def start_bot(strategy: str = "grid_dca"):
    global bot_thread, active_strategy
    if active_strategy:
        logging.warning(f"⚠️ {active_strategy} is already running. Stop it first.")
        return

    if strategy not in ["grid_dca", "asset_balancer", "uniswap_v3", "long_short"]:
        logging.error(f"❌ Invalid strategy: {strategy}")
        return

    def runner():
        if strategy == "grid_dca":
            grid_dca_loop(poll_interval=60)
        elif strategy == "asset_balancer":
            run_asset_balancer()
        elif strategy == "uniswap_v3":  # <--- NEW
            run_uniswap_v3_loop(poll_interval=60)
        elif strategy == "long_short":  # <--- NEW
            longshort_loop(poll_interval=60)    
        else:
            logging.error("Strategy module not found.")
        logging.info(f"🛑 {strategy} loop exited.")

    active_strategy = strategy
    bot_thread = threading.Thread(target=runner, daemon=True)
    bot_thread.start()
    logging.info(f"🎯 Strategy started: {strategy}")


def stop_bot():
    global active_strategy
    if not active_strategy:
        logging.warning("⚠️ No active strategy running.")
        return
    logging.info(f"🛑 Stopping {active_strategy} gracefully...")
    active_strategy = None


# ---------- Entry Point ----------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logging.info("🚀 Bot container initialized.")

    # Choose strategy manually here (you can replace with UI control later)
    # Available: "grid_dca" or "asset_balancer"
    selected_strategy = "long_short"

    logging.info(f"🎯 Selected strategy: {selected_strategy}")
    try:
        start_bot(selected_strategy)
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        stop_bot()
        logging.info("🛑 Manual stop received. Exiting gracefully...")
