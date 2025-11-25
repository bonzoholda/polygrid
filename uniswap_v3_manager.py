# uniswap_v3_manager.py
import math
import time
import logging
import threading
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

# NFT_MANAGER_ABI (unchanged)
NFT_MANAGER_ABI = [
    {"inputs":[{"internalType":"struct MintParams","name":"params","type":"tuple","components":[{"internalType":"address","name":"token0","type":"address"},{"internalType":"address","name":"token1","type":"address"},{"internalType":"uint24","name":"fee","type":"uint24"},{"internalType":"int24","name":"tickLower","type":"int24"},{"internalType":"int24","name":"tickUpper","type":"int24"},{"internalType":"uint256","name":"amount0Desired","type":"uint256"},{"internalType":"uint256","name":"amount1Desired","type":"uint256"},{"internalType":"uint256","name":"amount0Min","type":"uint256"},{"internalType":"uint256","name":"amount1Min","type":"uint256"},{"internalType":"address","name":"recipient","type":"address"},{"internalType":"uint256","name":"deadline","type":"uint256"}]}],"name":"mint","outputs":[{"internalType":"uint256","name":"tokenId","type":"uint256"},{"internalType":"uint128","name":"liquidity","type":"uint128"},{"internalType":"uint256","name":"amount0","type":"uint256"},{"internalType":"uint256","name":"amount1","type":"uint256"}],"stateMutability":"payable","type":"function"},
    {"inputs":[{"internalType":"struct DecreaseLiquidityParams","name":"params","type":"tuple","components":[{"internalType":"uint256","name":"tokenId","type":"uint256"},{"internalType":"uint128","name":"liquidity","type":"uint128"},{"internalType":"uint256","name":"amount0Min","type":"uint256"},{"internalType":"uint256","name":"amount1Min","type":"uint256"},{"internalType":"uint256","name":"deadline","type":"uint256"}]}],"name":"decreaseLiquidity","outputs":[{"internalType":"uint256","name":"amount0","type":"uint256"},{"internalType":"uint256","name":"amount1","type":"uint256"}],"stateMutability":"payable","type":"function"},
    {"inputs":[{"internalType":"struct CollectParams","name":"params","type":"tuple","components":[{"internalType":"uint256","name":"tokenId","type":"uint256"},{"internalType":"address","name":"recipient","type":"address"},{"internalType":"uint128","name":"amount0Max","type":"uint128"},{"internalType":"uint128","name":"amount1Max","type":"uint128"}]}],"name":"collect","outputs":[{"internalType":"uint256","name":"amount0","type":"uint256"},{"internalType":"uint256","name":"amount1","type":"uint256"}],"stateMutability":"payable","type":"function"},
    {"inputs":[{"internalType":"address","name":"owner","type":"address"}],"name":"balanceOf","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"internalType":"address","name":"owner","type":"address"},{"internalType":"uint256","name":"index","type":"uint256"}],"name":"tokenOfOwnerByIndex","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"internalType":"uint256","name":"tokenId","type":"uint256"}],"name":"positions","outputs":[{"internalType":"uint96","name":"nonce","type":"uint96"},{"internalType":"address","name":"operator","type":"address"},{"internalType":"address","name":"token0","type":"address"},{"internalType":"address","name":"token1","type":"address"},{"internalType":"uint24","name":"fee","type":"uint24"},{"internalType":"int24","name":"tickLower","type":"int24"},{"internalType":"int24","name":"tickUpper","type":"int24"},{"internalType":"uint128","name":"liquidity","type":"uint128"},{"internalType":"uint256","name":"feeGrowthInside0LastX128","type":"uint256"},{"internalType":"uint256","name":"feeGrowthInside1LastX128","type":"uint256"},{"internalType":"uint128","name":"tokensOwed0","type":"uint128"},{"internalType":"uint128","name":"tokensOwed1","type":"uint128"}],"stateMutability":"view","type":"function"}
]

class UniswapV3Manager:
    def __init__(self, owner_address=None, owner_private_key=None, pool_address=None):
        logging.info("Initializing UniswapV3Manager...")
        self.pool_address = pool_address or UNISWAP_POOL_ADDR

        self.owner = w3.to_checksum_address(owner_address) if owner_address else w3.to_checksum_address(config.OWNER)
        self.owner_private_key = owner_private_key if owner_private_key else config.PRIVATE_KEY

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

    # -----------------------
    # User credentials wrapper
    # -----------------------
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

    # -----------------------
    # Local TX helper
    # -----------------------
    def _send_tx_local(self, tx_dict):
        try:
            if "nonce" not in tx_dict:
                tx_dict["nonce"] = w3.eth.get_transaction_count(self.owner, "pending")
            if "gasPrice" not in tx_dict:
                tx_dict["gasPrice"] = w3.eth.gas_price
            signed = w3.eth.account.sign_transaction(tx_dict, private_key=self.owner_private_key)
            raw = getattr(signed, "raw_transaction", None) or getattr(signed, "rawTransaction", None)
            if raw is None:
                logging.error("Local sign failed: missing raw tx")
                return None
            tx_hash = w3.eth.send_raw_transaction(raw)
            logging.info(f"TX sent: {tx_hash.hex()}")
            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=300)
            if receipt and getattr(receipt, "status", None) == 1:
                logging.info(f"TX confirmed in block {receipt.blockNumber}")
                return tx_hash.hex()
            logging.error(f"TX failed (status={getattr(receipt,'status',None)})")
            return None
        except Exception as e:
            logging.exception(f"_send_tx_local error: {e}")
            return None

    # -----------------------
    # Pool price & tick
    # -----------------------
    def get_pool_price_and_tick(self):
        try:
            pool = w3.eth.contract(address=self.pool_address, abi=POOL_ABI)
            slot0 = pool.functions.slot0().call()
            sqrtPriceX96, tick = slot0[0], slot0[1]
            price = (sqrtPriceX96 ** 2 / 2 ** 192) * (10 ** (self.dec1 - self.dec0))
            return float(price), tick
        except Exception as e:
            logging.error(f"Failed to fetch pool price: {e}")
            return None, None

    # -----------------------
    # Tick helpers
    # -----------------------
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

    # -----------------------
    # Wallet balancing
    # -----------------------
    def balance_wallet_50_50(self, current_price_usdt):
        logging.info("Checking wallet balance for 50:50 split...")
        try:
            bal_usdt = get_onchain_token_balance(usdt, self.owner)
            bal_wmatic = get_onchain_token_balance(wmatic, self.owner)
        except Exception as e:
            logging.error(f"Error reading balances: {e}")
            return

        val_usdt = float(bal_usdt)
        val_wmatic = float(bal_wmatic) * float(current_price_usdt)
        total_val = val_usdt + val_wmatic
        if total_val < 5.0:
            logging.warning("Wallet balance too low (<$5) to balance.")
            return

        usdt_ratio = val_usdt / total_val if total_val > 0 else 1.0
        logging.info(f"Ratio: USDT {usdt_ratio*100:.1f}% | WMATIC {(1-usdt_ratio)*100:.1f}%")

        if usdt_ratio > 0.60:
            surplus_usdt = val_usdt - (total_val * 0.5)
            logging.info(f"Swapping {surplus_usdt:.2f} USDT -> WMATIC")
            try: self._with_user_creds(swap_usdt_to_wmatic, surplus_usdt)
            except Exception as e: logging.error(f"swap_usdt_to_wmatic failed: {e}")
            time.sleep(5)
        elif usdt_ratio < 0.40:
            surplus_wmatic_val = val_wmatic - (total_val * 0.5)
            swap_amt = surplus_wmatic_val / current_price_usdt
            logging.info(f"Swapping {swap_amt:.2f} WMATIC -> USDT")
            try: self._with_user_creds(swap_wmatic_to_usdt, swap_amt)
            except Exception as e: logging.error(f"swap_wmatic_to_usdt failed: {e}")
            time.sleep(5)
        else:
            logging.info("Balance healthy (near 50:50). No swap needed.")

    # -----------------------
    # Mint position
    # -----------------------
    def mint_position(self, center_price, range_pct=0.05, usdt_alloc=5.0):
        logging.info(f"Calculating V3 Mint params. Center: {center_price}, Range: {range_pct*100}%")
        lower_price = center_price * (1 - range_pct)
        upper_price = center_price * (1 + range_pct)
        tick_lower = self.align_tick(self.get_tick_from_price(lower_price))
        tick_upper = self.align_tick(self.get_tick_from_price(upper_price))
        if tick_lower > tick_upper:
            tick_lower, tick_upper = tick_upper, tick_lower

        dec_wmatic, dec_usdt = (self.dec0, self.dec1) if self.is_wmatic_zero else (self.dec1, self.dec0)
        usdt_wei_target = int(usdt_alloc * (10 ** dec_usdt))
        wmatic_wei_target = int((usdt_alloc / max(center_price, 1e-12)) * (10 ** dec_wmatic))

        try:
            bal_t0 = int(self.token0_obj.functions.balanceOf(self.owner).call())
            bal_t1 = int(self.token1_obj.functions.balanceOf(self.owner).call())
        except Exception as e:
            logging.error(f"Failed to read on-chain balances: {e}")
            return None

        target0, target1 = (wmatic_wei_target, usdt_wei_target) if self.is_wmatic_zero else (usdt_wei_target, wmatic_wei_target)
        amount0_final = min(target0, int(bal_t0 * 0.999))
        amount1_final = min(target1, int(bal_t1 * 0.999))

        if amount0_final == 0 and amount1_final == 0:
            logging.error("MINT FAILED: zero balance in both tokens.")
            return None

        logging.info(f"Approving {amount0_final} (T0) and {amount1_final} (T1) for mint...")
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

    # -----------------------
    # Active LP & Close
    # -----------------------
    def check_position_status(self, token_id, current_price):
        try:
            pos = self.nft_manager.functions.positions(token_id).call()
            tick_lower, tick_upper, liquidity = pos[5], pos[6], pos[7]
            current_tick = self.get_tick_from_price(current_price)
            logging.info(f"Position Status: Tick {current_tick} vs Range [{tick_lower}-{tick_upper}]")
            if current_tick < tick_lower or current_tick > tick_upper:
                logging.warning(f"Price out of range! Closing position {token_id}")
                self.close_position(token_id)
                return False
            if liquidity == 0:
                logging.warning("Position has 0 liquidity")
                return False
            return True
        except Exception as e:
            logging.error(f"Error checking position status: {e}")
            return True

    def close_position(self, token_id):
        logging.info(f"Closing Position NFT ID: {token_id}")
        try:
            pos = self.nft_manager.functions.positions(token_id).call()
            liquidity = pos[7]
        except Exception as e:
            logging.error(f"Failed reading position: {e}"); return

        if liquidity > 0:
            try:
                params_dec = {'tokenId': token_id, 'liquidity': liquidity, 'amount0Min': 0, 'amount1Min': 0, 'deadline': int(time.time())+300}
                tx = self.nft_manager.functions.decreaseLiquidity(params_dec).build_transaction({
                    'from': self.owner, 'nonce': w3.eth.get_transaction_count(self.owner, 'pending'),
                    'gas': 700000, 'gasPrice': w3.eth.gas_price
                })
                self._with_user_creds(send_tx, tx)
            except Exception as e:
                logging.error(f"DecreaseLiquidity failed: {e}")

        try:
            params_collect = {'tokenId': token_id, 'recipient': self.owner, 'amount0Max': 2**128-1, 'amount1Max': 2**128-1}
            txc = self.nft_manager.functions.collect(params_collect).build_transaction({
                'from': self.owner, 'nonce': w3.eth.get_transaction_count(self.owner, 'pending'),
                'gas': 400000, 'gasPrice': w3.eth.gas_price
            })
            self._with_user_creds(send_tx, txc)
        except Exception as e:
            logging.error(f"Collect failed: {e}")

    # -----------------------
    # Active position helper
    # -----------------------
    def get_active_position_id(self):
        """
        Returns the NFT ID of the active LP position.
        Checks all NFTs owned by the bot and selects one with non-zero liquidity.
        """
        try:
            balance = self.nft_manager.functions.balanceOf(self.owner).call()
            if balance == 0:
                return None
    
            for i in range(balance):
                token_id = self.nft_manager.functions.tokenOfOwnerByIndex(self.owner, i).call()
                pos = self.nft_manager.functions.positions(token_id).call()
                liquidity = pos[7]  # liquidity
                token0_pos, token1_pos = pos[2], pos[3]
    
                # Only consider the LP matching our WMATIC-USDT pool
                tokens_match = (
                    (token0_pos.lower() == self.token0.lower() and token1_pos.lower() == self.token1.lower()) or
                    (token0_pos.lower() == self.token1.lower() and token1_pos.lower() == self.token0.lower())
                )
    
                if liquidity > 0 and tokens_match:
                    return token_id
            return None
        except Exception as e:
            logging.error(f"get_active_position_id error: {e}")
            return None


    
    # -----------------------
    # Portfolio value (wallet + LP)
    # -----------------------
    def get_position_asset_value(self):
        try:
            bal_usdt = get_onchain_token_balance(usdt, self.owner)
            bal_wmatic = get_onchain_token_balance(wmatic, self.owner)
            pool_price, _ = self.get_pool_price_and_tick()
            if pool_price is None:
                pool_price = get_pol_price_from_okx("WMATICUSDT")
            total_val = float(bal_usdt) + float(bal_wmatic) * float(pool_price)
            return float(bal_usdt), float(bal_wmatic), total_val
        except Exception as e:
            logging.error(f"Error in get_position_asset_value: {e}")
            return 0.0, 0.0, 0.0

    # -----------------------
    # Runner loop
    # -----------------------
    def run_uniswap_v3_loop(self, uid, loop_interval=30):
        logging.info(f"Starting Uniswap V3 manager loop for UID {uid}")
        while True:
            try:
                price, _ = self.get_pool_price_and_tick()
                if price is None:
                    price = get_pol_price_from_okx("WMATICUSDT")
                self.balance_wallet_50_50(price)
                bal_usdt, bal_wmatic, total_val = self.get_position_asset_value()
                state_data = {
                    "price": price,
                    "bal_usdt": bal_usdt,
                    "bal_wmatic": bal_wmatic,
                    "total_val": total_val,
                    "timestamp": int(time.time())
                }
                update_lp_state(uid, state_data)
                push_lp_stat(uid, state_data)
            except Exception as e:
                logging.error(f"Loop error: {e}")
            time.sleep(loop_interval)


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
