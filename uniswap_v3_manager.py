# uniswap_v3_manager.py
import math
import time
import logging
import os
from web3 import Web3

# --- Import from utils ---
from utils import (
    w3,
    send_tx,
    approve_if_needed,
    get_pol_price_from_okx,
    get_onchain_token_balance,
    swap_usdt_to_wmatic,
    swap_wmatic_to_usdt,
    ERC20_ABI
)

import config
from config import usdt, wmatic, USDT_ADDR, WMATIC_ADDR, UNISWAP_POOL_ADDR

from core.state import update_lp_state, get_lp_state
from core.lp_helper import push_lp_stat

# --- Constants ---
NFT_MANAGER_ADDR = "0xC36442b4a4522E871399CD717aBDD847Ab11FE88"
pool_address = UNISWAP_POOL_ADDR
POOL_FEE = 3000
TICK_SPACING = 60

# Minimal pool ABI
POOL_ABI = [
    {
        "inputs": [], "name": "slot0",
        "outputs": [
            {"internalType": "uint160", "name": "sqrtPriceX96", "type": "uint160"},
            {"internalType": "int24", "name": "tick", "type": "int24"},
            {"internalType": "uint16", "name": "observationIndex", "type": "uint16"},
            {"internalType": "uint16", "name": "observationCardinality", "type": "uint16"},
            {"internalType": "uint16", "name": "observationCardinalityNext", "type": "uint16"},
            {"internalType": "uint8", "name": "feeProtocol", "type": "uint8"},
            {"internalType": "bool", "name": "unlocked", "type": "bool"}
        ],
        "stateMutability": "view", "type": "function"
    }
]

# NFT_MANAGER_ABI (same as before)
NFT_MANAGER_ABI = [
    {"inputs":[{"internalType":"struct MintParams","name":"params","type":"tuple","components":[{"internalType":"address","name":"token0","type":"address"},{"internalType":"address","name":"token1","type":"address"},{"internalType":"uint24","name":"fee","type":"uint24"},{"internalType":"int24","name":"tickLower","type":"int24"},{"internalType":"int24","name":"tickUpper","type":"int24"},{"internalType":"uint256","name":"amount0Desired","type":"uint256"},{"internalType":"uint256","name":"amount1Desired","type":"uint256"},{"internalType":"uint256","name":"amount0Min","type":"uint256"},{"internalType":"uint256","name":"amount1Min","type":"uint256"},{"internalType":"address","name":"recipient","type":"address"},{"internalType":"uint256","name":"deadline","type":"uint256"}]}],"name":"mint","outputs":[{"internalType":"uint256","name":"tokenId","type":"uint256"},{"internalType":"uint128","name":"liquidity","type":"uint128"},{"internalType":"uint256","name":"amount0","type":"uint256"},{"internalType":"uint256","name":"amount1","type":"uint256"}],"stateMutability":"payable","type":"function"},
    {"inputs":[{"internalType":"struct DecreaseLiquidityParams","name":"params","type":"tuple","components":[{"internalType":"uint256","name":"tokenId","type":"uint256"},{"internalType":"uint128","name":"liquidity","type":"uint128"},{"internalType":"uint256","name":"amount0Min","type":"uint256"},{"internalType":"uint256","name":"amount1Min","type":"uint256"},{"internalType":"uint256","name":"deadline","type":"uint256"}]}],"name":"decreaseLiquidity","outputs":[{"internalType":"uint256","name":"amount0","type":"uint256"},{"internalType":"uint256","name":"amount1","type":"uint256"}],"stateMutability":"payable","type":"function"},
    {"inputs":[{"internalType":"struct CollectParams","name":"params","type":"tuple","components":[{"internalType":"uint256","name":"tokenId","type":"uint256"},{"internalType":"address","name":"recipient","type":"address"},{"internalType":"uint128","name":"amount0Max","type":"uint128"},{"internalType":"uint128","name":"amount1Max","type":"uint128"}]}],"name":"collect","outputs":[{"internalType":"uint256","name":"amount0","type":"uint256"},{"internalType":"uint256","name":"amount1","type":"uint256"}],"stateMutability":"payable","type":"function"},
    {"inputs":[{"internalType":"address","name":"owner","type":"address"}],"name":"balanceOf","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"internalType":"address","name":"owner","type":"address"},{"internalType":"uint256","name":"index","type":"uint256"}],"name":"tokenOfOwnerByIndex","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"internalType":"uint256","name":"tokenId","type":"uint256"}],"name":"positions","outputs":[{"internalType":"uint96","name":"nonce","type":"uint96"},{"internalType":"address","name":"operator","type":"address"},{"internalType":"address","name":"token0","type":"address"},{"internalType":"address","name":"token1","type":"address"},{"internalType":"uint24","name":"fee","type":"uint24"},{"internalType":"int24","name":"tickLower","type":"int24"},{"internalType":"int24","name":"tickUpper","type":"int24"},{"internalType":"uint128","name":"liquidity","type":"uint128"},{"internalType":"uint256","name":"feeGrowthInside0LastX128","type":"uint256"},{"internalType":"uint256","name":"feeGrowthInside1LastX128","type":"uint256"},{"internalType":"uint128","name":"tokensOwed0","type":"uint128"},{"internalType":"uint128","name":"tokensOwed1","type":"uint128"}],"stateMutability":"view","type":"function"}
]

class UniswapV3Manager:
    def __init__(self, owner_address=None, owner_private_key=None):
        print("DEBUG: Initializing UniswapV3Manager class...")

        if owner_address:
            self.owner = w3.to_checksum_address(owner_address)
        else:
            self.owner = w3.to_checksum_address(config.OWNER)

        self.owner_private_key = owner_private_key if owner_private_key else config.PRIVATE_KEY

        try:
            self.nft_manager = w3.eth.contract(address=NFT_MANAGER_ADDR, abi=NFT_MANAGER_ABI)
            self.token0_obj = w3.eth.contract(address=WMATIC_ADDR, abi=ERC20_ABI)
            self.token1_obj = w3.eth.contract(address=USDT_ADDR, abi=ERC20_ABI)

            if int(WMATIC_ADDR, 16) < int(USDT_ADDR, 16):
                self.token0 = WMATIC_ADDR
                self.token1 = USDT_ADDR
                self.is_wmatic_zero = True
            else:
                self.token0 = USDT_ADDR
                self.token1 = WMATIC_ADDR
                self.is_wmatic_zero = False

            try:
                self.dec0 = int(self.token0_obj.functions.decimals().call())
            except Exception:
                self.dec0 = 18 if self.is_wmatic_zero else 6

            try:
                self.dec1 = int(self.token1_obj.functions.decimals().call())
            except Exception:
                self.dec1 = 6 if self.is_wmatic_zero else 18

        except Exception as e:
            print(f"CRITICAL ERROR in V3 __init__: {e}")
            raise e

    def _with_user_creds(self, func, *args, **kwargs):
        orig_owner = getattr(config, "OWNER", None)
        orig_priv = getattr(config, "PRIVATE_KEY", None)
        try:
            config.OWNER = self.owner
            config.PRIVATE_KEY = self.owner_private_key
            return func(*args, **kwargs)
        finally:
            if orig_owner is not None:
                config.OWNER = orig_owner
            else:
                try: delattr(config, "OWNER")
                except Exception: pass
            if orig_priv is not None:
                config.PRIVATE_KEY = orig_priv
            else:
                try: delattr(config, "PRIVATE_KEY")
                except Exception: pass

    def _send_tx_local(self, tx_dict):
        try:
            if "nonce" not in tx_dict:
                tx_dict["nonce"] = w3.eth.get_transaction_count(self.owner, "pending")
            if "gasPrice" not in tx_dict:
                tx_dict["gasPrice"] = w3.eth.gas_price
            signed = w3.eth.account.sign_transaction(tx_dict, private_key=self.owner_private_key)
            raw = getattr(signed, "raw_transaction", None) or getattr(signed, "rawTransaction", None)
            if raw is None:
                logging.error("❌ Local sign failed: missing raw tx")
                return None
            tx_hash = w3.eth.send_raw_transaction(raw)
            logging.info(f"✅ Local TX sent: {tx_hash.hex()} (waiting for receipt)")
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=300)
            if receipt and getattr(receipt, "status", None) == 1:
                logging.info(f"🧾 Local TX confirmed in block {receipt.blockNumber}")
                return tx_hash.hex()
            logging.error(f"❌ Local TX reverted or failed (status={getattr(receipt,'status',None)})")
            return None
        except Exception as e:
            logging.exception(f"❌ _send_tx_local error: {e}")
            return None

    def get_tick_from_price(self, price_float):
        try:
            if price_float is None or price_float == 0:
                return 0
            exp = (self.dec1 - self.dec0)
            raw_price = float(price_float) * (10 ** exp)
            if raw_price <= 0:
                return 0
            tick = math.log(raw_price) / math.log(1.0001)
            return int(round(tick))
        except Exception as e:
            logging.error(f"Error in get_tick_from_price: {e}")
            return 0

    def align_tick(self, tick):
        try:
            return int(math.floor(tick / TICK_SPACING) * TICK_SPACING)
        except Exception:
            return (tick // TICK_SPACING) * TICK_SPACING

    def balance_wallet_50_50(self, current_price_usdt):
        logging.info("⚖️ Checking wallet balance for 50:50 split...")
        try:
            bal_usdt = get_onchain_token_balance(usdt, self.owner)
            bal_wmatic = get_onchain_token_balance(wmatic, self.owner)
        except Exception as e:
            logging.error(f"Error reading balances for 50:50: {e}")
            return

        val_usdt = float(bal_usdt)
        val_wmatic = float(bal_wmatic) * float(current_price_usdt)
        total_val = val_usdt + val_wmatic
        if total_val < 5.0:
            logging.warning("⚠️ Wallet balance too low (< $5) to balance.")
            return

        try:
            usdt_ratio = val_usdt / total_val
        except Exception:
            usdt_ratio = 1.0
        logging.info(f"⚖️ Ratio: USDT {usdt_ratio*100:.1f}% | WMATIC {(1-usdt_ratio)*100:.1f}%")

        if usdt_ratio > 0.60:
            surplus_usdt = val_usdt - (total_val * 0.5)
            logging.info(f"⚖️ Rebalancing: Swapping {surplus_usdt:.2f} USDT -> WMATIC")
            try: self._with_user_creds(swap_usdt_to_wmatic, surplus_usdt)
            except Exception as e: logging.error(f"swap_usdt_to_wmatic failed: {e}")
            time.sleep(5)
        elif usdt_ratio < 0.40:
            surplus_wmatic_val = val_wmatic - (total_val * 0.5)
            swap_amt = surplus_wmatic_val / current_price_usdt
            logging.info(f"⚖️ Rebalancing: Swapping {swap_amt:.2f} WMATIC -> USDT")
            try: self._with_user_creds(swap_wmatic_to_usdt, swap_amt)
            except Exception as e: logging.error(f"swap_wmatic_to_usdt failed: {e}")
            time.sleep(5)
        else:
            logging.info("✅ Balance is healthy (near 50:50). No swap needed.")

    def mint_position(self, center_price, range_pct=0.05, usdt_alloc=5.0):
        logging.info(f"🦄 Calculating V3 Mint params. Center: {center_price}, Range: {range_pct*100}%")
        lower_price = center_price * (1 - range_pct)
        upper_price = center_price * (1 + range_pct)
        tick_lower = self.align_tick(self.get_tick_from_price(lower_price))
        tick_upper = self.align_tick(self.get_tick_from_price(upper_price))
        if tick_lower > tick_upper:
            tick_lower, tick_upper = tick_upper, tick_lower

        if self.is_wmatic_zero:
            dec_wmatic, dec_usdt = self.dec0, self.dec1
        else:
            dec_wmatic, dec_usdt = self.dec1, self.dec0

        usdt_wei_target = int(usdt_alloc * (10 ** dec_usdt))
        try:
            wmatic_wei_target = int((usdt_alloc / max(center_price, 1e-12)) * (10 ** dec_wmatic))
        except Exception:
            wmatic_wei_target = 0

        try:
            bal_t0 = int(self.token0_obj.functions.balanceOf(self.owner).call())
            bal_t1 = int(self.token1_obj.functions.balanceOf(self.owner).call())
        except Exception as e:
            logging.error(f"❌ Failed to read on-chain balances: {e}")
            return None

        target0, target1 = (wmatic_wei_target, usdt_wei_target) if self.is_wmatic_zero else (usdt_wei_target, wmatic_wei_target)
        amount0_final = min(target0, int(bal_t0 * 0.999))
        amount1_final = min(target1, int(bal_t1 * 0.999))

        if amount0_final == 0 and amount1_final == 0:
            logging.error("❌ MINT FAILED: Cannot execute mint due to zero balance in both tokens.")
            return None

        logging.info(f"DEBUG: Approving {amount0_final} (T0) and {amount1_final} (T1) for mint...")
        try:
            if amount0_final > 0 and not self._with_user_creds(approve_if_needed, self.token0_obj, NFT_MANAGER_ADDR, amount0_final):
                logging.error("Approval failed for Token0"); return None
            if amount1_final > 0 and not self._with_user_creds(approve_if_needed, self.token1_obj, NFT_MANAGER_ADDR, amount1_final):
                logging.error("Approval failed for Token1"); return None
        except Exception as e:
            logging.error(f"Approval exception: {e}"); return None

        params = {
            'token0': self.token0, 'token1': self.token1, 'fee': POOL_FEE,
            'tickLower': tick_lower, 'tickUpper': tick_upper,
            'amount0Desired': amount0_final, 'amount1Desired': amount1_final,
            'amount0Min': 0, 'amount1Min': 0,
            'recipient': self.owner, 'deadline': int(time.time()) + 300
        }

        try:
            tx_build = self.nft_manager.functions.mint(params).build_transaction({
                'from': self.owner, 'nonce': w3.eth.get_transaction_count(self.owner, 'pending'),
                'gas': 900000, 'gasPrice': w3.eth.gas_price
            })
            return self._with_user_creds(send_tx, tx_build)
        except Exception as e:
            logging.error(f"Mint build failed: {e}"); return None

    # --- Active LP & Close ---
    def check_position_status(self, token_id, current_price):
        try:
            pos = self.nft_manager.functions.positions(token_id).call()
            tick_lower, tick_upper, liquidity = pos[5], pos[6], pos[7]
            current_tick = self.get_tick_from_price(current_price)
            logging.info(f"🔍 Position Status: Tick {current_tick} vs Range [{tick_lower} - {tick_upper}]")
            if current_tick < tick_lower or current_tick > tick_upper:
                logging.warning(f"🚨 Price Out of Range! Closing Position {token_id}...")
                self.close_position(token_id)
                return False
            if liquidity == 0:
                logging.warning("⚠️ Position has 0 liquidity (already closed?).")
                return False
            return True
        except Exception as e:
            logging.error(f"Error checking position status: {e}")
            return True

    def close_position(self, token_id):
        logging.info(f"🔥 Closing Position NFT ID: {token_id}")
        try:
            pos = self.nft_manager.functions.positions(token_id).call()
            liquidity = pos[7]
        except Exception as e:
            logging.error(f"Failed reading position for close: {e}"); return

        if liquidity > 0:
            try:
                params_dec = {'tokenId': token_id, 'liquidity': liquidity, 'amount0Min': 0, 'amount1Min': 0, 'deadline': int(time.time())+300}
                tx = self.nft_manager.functions.decreaseLiquidity(params_dec).build_transaction({
                    'from': self.owner, 'nonce': w3.eth.get_transaction_count(self.owner, 'pending'),
                    'gas': 700000, 'gasPrice': w3.eth.gas_price
                })
                self._with_user_creds(send_tx, tx)
                time.sleep(2)
            except Exception as e:
                logging.error(f"Error decreasing liquidity: {e}")

        try:
            params_col = {'tokenId': token_id, 'recipient': self.owner, 'amount0Max': 2**128-1, 'amount1Max': 2**128-1}
            tx2 = self.nft_manager.functions.collect(params_col).build_transaction({
                'from': self.owner, 'nonce': w3.eth.get_transaction_count(self.owner, 'pending'),
                'gas': 200000, 'gasPrice': w3.eth.gas_price
            })
            self._with_user_creds(send_tx, tx2)
            logging.info("✅ Position Closed and Funds Collected.")
        except Exception as e:
            logging.error(f"Error collecting funds from position {token_id}: {e}")

    def get_active_position_id(self):
        try:
            balance = self.nft_manager.functions.balanceOf(self.owner).call()
            if balance == 0:
                return None
            logging.info(f"NFT Manager reports {balance} active positions.")
            return self.nft_manager.functions.tokenOfOwnerByIndex(self.owner, 0).call()
        except Exception as e:
            logging.warning(f"Failed to retrieve active position ID: {e}")
            return None

    def get_position_asset_value(self, token_id, current_price):
        try:
            pos = self.nft_manager.functions.positions(token_id).call()
            tick_lower, tick_upper, liquidity = pos[5], pos[6], pos[7]
            if liquidity == 0: return 0, 0, 0

            # simplified: assume 50:50
            bal_usdt = get_onchain_token_balance(usdt, self.owner)
            bal_wmatic = get_onchain_token_balance(wmatic, self.owner)
            total_val = bal_usdt + bal_wmatic * current_price
            return bal_usdt, bal_wmatic, total_val
        except Exception as e:
            logging.error(f"Failed computing asset value: {e}")
            return 0, 0, 0

# -----------------------------
# Runner Loop (thread-safe, updates core.state)
# -----------------------------
import threading

def run_uniswap_v3_loop(uid: int, poll_interval=60, pool_address: str = None):
    logging.info(f"🦄 Starting UniswapV3Manager loop for UID {uid}")
    manager = UniswapV3Manager(pool_address=pool_address)

    while True:
        try:
            # Get price
            pool_price, _ = manager.get_pool_price_and_tick()
            price = pool_price if pool_price else get_pol_price_from_okx()
            if not price:
                time.sleep(10)
                continue

            # Active position
            active_id = manager.get_active_position_id()
            if active_id:
                usdt_amt, wmatic_amt, total_val = manager.get_position_asset_value(active_id, price)
                state_data = {
                    "wmatic_price": float(price),
                    "lp_usdt": float(usdt_amt),
                    "lp_wmatic": float(wmatic_amt),
                    "lp_total_value": float(total_val),
                    "active": True
                }
                update_lp_state(uid, state_data)
                push_lp_stat(uid, state_data)  # optional helper call
            else:
                state_data = {
                    "wmatic_price": float(price),
                    "lp_usdt": 0.0,
                    "lp_wmatic": 0.0,
                    "lp_total_value": 0.0,
                    "active": False
                }
                update_lp_state(uid, state_data)

                # Rebalancing + mint if enough balance
                manager.balance_wallet_50_50(price)
                usdt_balance = manager._with_user_creds(get_onchain_token_balance, usdt, manager.owner)
                alloc_size = min(usdt_balance * 0.9, 50.0)
                if alloc_size >= 5.0:
                    manager.mint_position(center_price=price, range_pct=0.10, usdt_alloc=alloc_size)

            time.sleep(poll_interval)
        except Exception as e:
            logging.exception(f"CRITICAL ERROR in UniswapV3 loop: {e}")
            time.sleep(10)

# Helper to start in same process as FastAPI
def start_uniswap_v3_runner(uid: int, poll_interval=60, pool_address: str = None):
    thread = threading.Thread(target=run_uniswap_v3_loop, args=(uid, poll_interval, pool_address), daemon=True)
    thread.start()
    return thread


# ---------- Entry Point ----------
if __name__ == "__main__":
    import logging
    import time
    import os

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    BOT_UID = int(os.getenv("BOT_UID", "0"))

    logging.info("🚀 UniswapV3 bot initialized.")
    try:
        logging.info("⚙️ Starting UniswapV3 Strategy Loop...")
        run_uniswap_v3_loop(uid=BOT_UID)  # 👈 pass the UID here
    except KeyboardInterrupt:
        logging.info("🛑 Manual stop received. Exiting Asset Balancer gracefully...")
    except Exception as e:
        logging.exception(f"❌ Unexpected error in Asset Balancer: {e}")
