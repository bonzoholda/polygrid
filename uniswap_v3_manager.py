import math
import time
import logging
import traceback
import importlib
from web3 import Web3

# --- Import from your robust utils ---
from utils import (
    w3,
    send_tx, approve_if_needed,
    get_pol_price_from_okx,
    get_onchain_token_balance, # Used for auto-split
    swap_usdt_to_wmatic,       # Used for auto-split
    swap_wmatic_to_usdt,       # Used for auto-split
    ERC20_ABI
)

# import config so we can temporarily override OWNER/PRIVATE_KEY when calling utils helpers
import config

from config import usdt, wmatic, USDT_ADDR, WMATIC_ADDR

# --- V3 Constants ---
NFT_MANAGER_ADDR = "0xC36442b4a4522E871399CD717aBDD847Ab11FE88"
POOL_FEE = 3000
TICK_SPACING = 60

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
            # fallback to global config owner if not provided
            self.owner = w3.to_checksum_address(config.OWNER)

        # store per-user private key (plaintext expected)
        self.owner_private_key = owner_private_key if owner_private_key else config.PRIVATE_KEY

        try:
            self.nft_manager = w3.eth.contract(address=NFT_MANAGER_ADDR, abi=NFT_MANAGER_ABI)
            # instantiate token contract objects using ERC20_ABI you import from utils
            self.token0_obj = w3.eth.contract(address=WMATIC_ADDR, abi=ERC20_ABI)
            self.token1_obj = w3.eth.contract(address=USDT_ADDR, abi=ERC20_ABI)

            # Determine Ordering (keep original check)
            if int(WMATIC_ADDR, 16) < int(USDT_ADDR, 16):
                self.token0 = WMATIC_ADDR
                self.token1 = USDT_ADDR
                self.is_wmatic_zero = True
            else:
                self.token0 = USDT_ADDR
                self.token1 = WMATIC_ADDR
                self.is_wmatic_zero = False

            # Cache decimals (safe calls)
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

    # ---------------------------
    # Helper: temporary patch config.OWNER / config.PRIVATE_KEY
    # so utils functions that rely on them operate with the user's creds.
    # ---------------------------
    def _with_user_creds(self, func, *args, **kwargs):
        """
        Temporarily set config.OWNER and config.PRIVATE_KEY to this manager's values,
        call func(*args, **kwargs), then restore originals.
        """
        orig_owner = getattr(config, "OWNER", None)
        orig_priv = getattr(config, "PRIVATE_KEY", None)
        try:
            config.OWNER = self.owner
            config.PRIVATE_KEY = self.owner_private_key
            # reload utils module attributes that may have cached OWNER/PRIVATE_KEY if needed
            # (not strictly necessary unless utils cached them at import-time)
            result = func(*args, **kwargs)
            return result
        finally:
            # restore
            if orig_owner is not None:
                config.OWNER = orig_owner
            else:
                try:
                    delattr(config, "OWNER")
                except Exception:
                    pass
            if orig_priv is not None:
                config.PRIVATE_KEY = orig_priv
            else:
                try:
                    delattr(config, "PRIVATE_KEY")
                except Exception:
                    pass

    # ---------------------------
    # Small helper for building and sending tx signed by user private key
    # (used for txs we build locally rather than via utils.send_tx)
    # ---------------------------
    def _send_tx_local(self, tx_dict):
        """
        Sign with self.owner_private_key and send raw tx.
        Uses pending nonce for safety.
        Returns tx_hash hex or None.
        """
        try:
            # ensure nonce and gas/gasPrice present
            if "nonce" not in tx_dict:
                tx_dict["nonce"] = w3.eth.get_transaction_count(self.owner, "pending")
            # Fill defaults for gasPrice if not provided
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

    # ---------------------------
    # get_tick_from_price & align_tick (improved & kept names)
    # ---------------------------
    def get_tick_from_price(self, price_float):
        try:
            if price_float is None:
                return 0
            if price_float == 0:
                return 0

            exp = (self.dec1 - self.dec0)
            raw_price = float(price_float) * (10 ** exp)

            if raw_price <= 0:
                return 0

            tick = math.log(raw_price) / math.log(1.0001)
            tick_int = int(round(tick))
            return tick_int
        except Exception as e:
            logging.error(f"Error in get_tick_from_price: {e}")
            return 0

    def align_tick(self, tick):
        try:
            aligned = int(math.floor(tick / TICK_SPACING) * TICK_SPACING)
            return aligned
        except Exception:
            return (tick // TICK_SPACING) * TICK_SPACING

    # --- Auto-Balancer (uses user's balances) ---
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
            swap_amt = surplus_usdt
            logging.info(f"⚖️ Rebalancing: Swapping {swap_amt:.2f} USDT -> WMATIC")
            try:
                # use utils.swap_usdt_to_wmatic but with user creds patched
                self._with_user_creds(swap_usdt_to_wmatic, swap_amt)
            except Exception as e:
                logging.error(f"swap_usdt_to_wmatic failed: {e}")
            time.sleep(5)

        elif usdt_ratio < 0.40:
            surplus_wmatic_val = val_wmatic - (total_val * 0.5)
            surplus_wmatic_amt = surplus_wmatic_val / current_price_usdt
            swap_amt = surplus_wmatic_amt
            logging.info(f"⚖️ Rebalancing: Swapping {swap_amt:.2f} WMATIC -> USDT")
            try:
                self._with_user_creds(swap_wmatic_to_usdt, swap_amt)
            except Exception as e:
                logging.error(f"swap_wmatic_to_usdt failed: {e}")
            time.sleep(5)
        else:
            logging.info("✅ Balance is healthy (near 50:50). No swap needed.")

    # --- Mint (keeps names and logs) ---
    def mint_position(self, center_price, range_pct=0.05, usdt_alloc=5.0):
        logging.info(f"🦄 Calculating V3 Mint params. Center: {center_price}, Range: {range_pct*100}%")

        lower_price = center_price * (1 - range_pct)
        upper_price = center_price * (1 + range_pct)

        tick_lower = self.align_tick(self.get_tick_from_price(lower_price))
        tick_upper = self.align_tick(self.get_tick_from_price(upper_price))

        if tick_lower > tick_upper:
            tick_lower, tick_upper = tick_upper, tick_lower

        # decimals
        if self.is_wmatic_zero:
            dec_wmatic = self.dec0
            dec_usdt = self.dec1
        else:
            dec_wmatic = self.dec1
            dec_usdt = self.dec0

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

        if self.is_wmatic_zero:
            target0 = wmatic_wei_target
            target1 = usdt_wei_target
        else:
            target0 = usdt_wei_target
            target1 = wmatic_wei_target

        try:
            amount0_final = min(target0, int(bal_t0 * 0.999))
            amount1_final = min(target1, int(bal_t1 * 0.999))
        except Exception as e:
            logging.error(f"❌ Error clamping token amounts: {e}")
            return None

        if amount0_final == 0 and amount1_final == 0:
            logging.error("❌ MINT FAILED: Cannot execute mint due to zero balance in both tokens.")
            return None

        logging.info(f"DEBUG: Approving {amount0_final} (T0) and {amount1_final} (T1) for mint...")

        # Use approve_if_needed from utils but patched to use this user's creds
        try:
            if amount0_final > 0:
                ok = self._with_user_creds(approve_if_needed, self.token0_obj, NFT_MANAGER_ADDR, amount0_final)
                if not ok:
                    logging.error("Approval failed for Token0")
                    return None
            if amount1_final > 0:
                ok = self._with_user_creds(approve_if_needed, self.token1_obj, NFT_MANAGER_ADDR, amount1_final)
                if not ok:
                    logging.error("Approval failed for Token1")
                    return None
        except Exception as e:
            logging.error(f"Approval exception: {e}")
            return None

        deadline = int(time.time()) + 300

        params = {
            'token0': self.token0,
            'token1': self.token1,
            'fee': POOL_FEE,
            'tickLower': tick_lower,
            'tickUpper': tick_upper,
            'amount0Desired': amount0_final,
            'amount1Desired': amount1_final,
            'amount0Min': 0,
            'amount1Min': 0,
            'recipient': self.owner,
            'deadline': deadline
        }

        logging.info("🦄 Sending Mint Transaction...")
        try:
            tx_build = self.nft_manager.functions.mint(params).build_transaction({
                'from': self.owner,
                'nonce': w3.eth.get_transaction_count(self.owner, 'pending'),
                'gas': 900000,
                'gasPrice': w3.eth.gas_price
            })
            # send via utils.send_tx but patched to use user's creds
            return self._with_user_creds(send_tx, tx_build)
        except Exception as e:
            logging.error(f"Mint build failed: {e}")
            return None

    # --- Check Active Position ---
    def check_position_status(self, token_id, current_price):
        try:
            pos = self.nft_manager.functions.positions(token_id).call()
            tick_lower = pos[5]
            tick_upper = pos[6]
            liquidity = pos[7]

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
            logging.error(f"Failed reading position for close: {e}")
            return

        if liquidity > 0:
            try:
                params_dec = {
                    'tokenId': token_id,
                    'liquidity': liquidity,
                    'amount0Min': 0,
                    'amount1Min': 0,
                    'deadline': int(time.time())+300
                }
                tx = self.nft_manager.functions.decreaseLiquidity(params_dec).build_transaction({
                    'from': self.owner, 'nonce': w3.eth.get_transaction_count(self.owner, 'pending'),
                    'gas': 700000, 'gasPrice': w3.eth.gas_price
                })
                # use utils.send_tx patched to user's creds
                self._with_user_creds(send_tx, tx)
                time.sleep(2)
            except Exception as e:
                logging.error(f"Error decreasing liquidity: {e}")

        try:
            params_col = {
                'tokenId': token_id, 'recipient': self.owner,
                'amount0Max': 2**128-1, 'amount1Max': 2**128-1
            }
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

            token_id = self.nft_manager.functions.tokenOfOwnerByIndex(self.owner, 0).call()
            return token_id

        except Exception as e:
            logging.warning(f"Failed to retrieve active position ID: {e}")
            return None

    def get_position_asset_value(self, token_id, current_price):
        try:
            pos = self.nft_manager.functions.positions(token_id).call()
            tick_lower = pos[5]
            tick_upper = pos[6]
            liquidity = pos[7]

            if liquidity == 0:
                logging.info(f"Position ID {token_id} has zero liquidity.")
                return 0.0, 0.0, 0.0

            current_tick = self.get_tick_from_price(current_price)

            L = float(liquidity)

            sqrt_price = 1.0001 ** (current_tick / 2.0)
            sqrt_price_lower = 1.0001 ** (tick_lower / 2.0)
            sqrt_price_upper = 1.0001 ** (tick_upper / 2.0)

            amount0_wei = 0.0
            amount1_wei = 0.0

            if current_tick <= tick_lower:
                amount0_wei = L * ((sqrt_price_upper - sqrt_price_lower) / (sqrt_price_lower * sqrt_price_upper))
                amount1_wei = 0.0

            elif current_tick >= tick_upper:
                amount0_wei = 0.0
                amount1_wei = L * (sqrt_price_upper - sqrt_price_lower)

            else:
                amount0_wei = L * ((sqrt_price_upper - sqrt_price) / (sqrt_price * sqrt_price_upper))
                amount1_wei = L * (sqrt_price - sqrt_price_lower)

            if self.is_wmatic_zero:
                try:
                    amount_wmatic = float(amount0_wei) / (10 ** self.dec0)
                    amount_usdt = float(amount1_wei) / (10 ** self.dec1)
                except Exception:
                    amount_wmatic = float(amount0_wei) / 1e18
                    amount_usdt = float(amount1_wei) / 1e6
            else:
                try:
                    amount_usdt = float(amount0_wei) / (10 ** self.dec0)
                    amount_wmatic = float(amount1_wei) / (10 ** self.dec1)
                except Exception:
                    amount_usdt = float(amount0_wei) / 1e6
                    amount_wmatic = float(amount1_wei) / 1e18

            total_usdt_value = amount_usdt + (amount_wmatic * current_price)

            return amount_usdt, amount_wmatic, total_usdt_value

        except Exception as e:
            logging.error(f"Error calculating position value for ID {token_id}: {e}")
            return 0.0, 0.0, 0.0


# --- Runner (keeps names & logs) ---
def run_uniswap_v3_loop(uid=None, poll_interval=60):
    print("DEBUG: Thread started for Uniswap V3 Strategy...")
    logging.info("🦄 Uniswap V3 Strategy Started.")

    manager = None

    while True:
        try:
            if manager is None:
                # create manager using global creds by default; your start_bot should create per-user manager
                manager = UniswapV3Manager()

            price = get_pol_price_from_okx()
            if not price:
                time.sleep(10)
                continue
            logging.info(f"💰 Current WMATIC price: {price:.4f} USDT")

            active_id = manager.get_active_position_id()

            if active_id:

                usdt_amt, wmatic_amt, total_value = manager.get_position_asset_value(active_id, price)

                logging.info("----------------------------------------------------------------")
                logging.info(f"🦄 Active Position ID: {active_id}")
                logging.info(f"💰 Position Assets: {usdt_amt:.2f} USDT | {wmatic_amt:.4f} WMATIC")
                logging.info(f"💵 **TOTAL LP VALUE (USD): ${total_value:.2f}**")
                logging.info("----------------------------------------------------------------")

                is_active = manager.check_position_status(active_id, price)
                if not is_active:
                    logging.info("♻️ Position closed. Preparing to re-enter...")
                    time.sleep(5)
                    continue
                else:
                    logging.info(f"🦄 Holding active position ID {active_id}. Price {price}")

            else:
                logging.info(f"🦄 No active position. Preparing entry around {price}...")
                manager.balance_wallet_50_50(price)

                usdt_balance = get_onchain_token_balance(usdt, manager.owner)
                alloc_size = min(usdt_balance * 0.9, 50.0)

                manager.mint_position(center_price=price, range_pct=0.10, usdt_alloc=alloc_size)

            time.sleep(poll_interval)

        except Exception as e:
            print(f"CRITICAL THREAD ERROR: {e}")
            traceback.print_exc()
            time.sleep(10)


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

        # ✅ Create the manager instance first
        manager = UniswapV3Manager()  # optionally pass owner_address/private_key/pool_address

        # Start the loop
        manager.run_uniswap_v3_loop(uid=BOT_UID)

    except KeyboardInterrupt:
        logging.info("🛑 Manual stop received. Exiting Asset Balancer gracefully...")
    except Exception as e:
        logging.exception(f"❌ Unexpected error in Asset Balancer: {e}")
