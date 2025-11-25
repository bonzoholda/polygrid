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
POOL_ADDRESS = UNISWAP_POOL_ADDR
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
    def __init__(self, owner_address=None, owner_private_key=None, pool_address: str = None):
        logging.info("DEBUG: Initializing UniswapV3Manager class...")

        self.owner = w3.to_checksum_address(owner_address) if owner_address else w3.to_checksum_address(config.OWNER)
        self.owner_private_key = owner_private_key if owner_private_key else config.PRIVATE_KEY

        # Pool
        self.pool_address = w3.to_checksum_address(pool_address) if pool_address else None
        self.pool = w3.eth.contract(address=self.pool_address, abi=POOL_ABI) if self.pool_address else None
        if not self.pool:
            logging.warning("⚠️ No pool_address provided — LP valuation will fallback to OKX price.")

        # NFT Manager
        self.nft_manager = w3.eth.contract(address=NFT_MANAGER_ADDR, abi=NFT_MANAGER_ABI)

        # Token objects
        self.token0_obj = w3.eth.contract(address=WMATIC_ADDR, abi=ERC20_ABI)
        self.token1_obj = w3.eth.contract(address=USDT_ADDR, abi=ERC20_ABI)

        # Determine token order
        if int(WMATIC_ADDR, 16) < int(USDT_ADDR, 16):
            self.token0, self.token1 = WMATIC_ADDR, USDT_ADDR
            self.is_wmatic_zero = True
        else:
            self.token0, self.token1 = USDT_ADDR, WMATIC_ADDR
            self.is_wmatic_zero = False

        # Decimals
        self.dec0 = int(self.token0_obj.functions.decimals().call()) if self.token0_obj else 18
        self.dec1 = int(self.token1_obj.functions.decimals().call()) if self.token1_obj else 6

    # --- Patch utils helpers with this user's creds ---
    def _with_user_creds(self, func, *args, **kwargs):
        orig_owner, orig_priv = getattr(config, "OWNER", None), getattr(config, "PRIVATE_KEY", None)
        try:
            config.OWNER = self.owner
            config.PRIVATE_KEY = self.owner_private_key
            return func(*args, **kwargs)
        finally:
            if orig_owner is not None:
                config.OWNER = orig_owner
            if orig_priv is not None:
                config.PRIVATE_KEY = orig_priv

    # --- Pool helpers ---
    def get_pool_slot0(self):
        if not self.pool:
            return None, None
        try:
            slot0 = self.pool.functions.slot0().call()
            return int(slot0[0]), int(slot0[1])
        except Exception as e:
            logging.warning(f"⚠️ Failed to read slot0: {e}")
            return None, None

    def get_pool_price_and_tick(self):
        sqrtPriceX96, tick = self.get_pool_slot0()
        if sqrtPriceX96 is None:
            return None, None
        try:
            sqrtPrice = float(sqrtPriceX96) / (2 ** 96)
            price_raw = sqrtPrice * sqrtPrice
            price_human = price_raw * (10 ** (self.dec0 - self.dec1))
            return price_human, tick
        except Exception as e:
            logging.error(f"Error converting sqrtPriceX96: {e}")
            return None, tick

    def get_tick_from_price(self, price_float):
        if price_float is None or price_float <= 0:
            return 0
        exp = (self.dec1 - self.dec0)
        raw_price = price_float * (10 ** exp)
        tick = math.log(raw_price) / math.log(1.0001)
        return int(round(tick))

    def align_tick(self, tick):
        return int(math.floor(tick / TICK_SPACING) * TICK_SPACING)

    # --- Auto-Balancer ---
    def balance_wallet_50_50(self, current_price_usdt):
        try:
            bal_usdt = get_onchain_token_balance(usdt, self.owner)
            bal_wmatic = get_onchain_token_balance(wmatic, self.owner)
            val_usdt = float(bal_usdt)
            val_wmatic = float(bal_wmatic) * float(current_price_usdt)
            total_val = val_usdt + val_wmatic
            if total_val < 5.0:
                return

            usdt_ratio = val_usdt / total_val
            if usdt_ratio > 0.60:
                surplus_usdt = val_usdt - (total_val * 0.5)
                self._with_user_creds(swap_usdt_to_wmatic, surplus_usdt)
            elif usdt_ratio < 0.40:
                surplus_wmatic_val = val_wmatic - (total_val * 0.5)
                self._with_user_creds(swap_wmatic_to_usdt, surplus_wmatic_val / current_price_usdt)
        except Exception as e:
            logging.error(f"Balance 50:50 error: {e}")

    # --- Mint position ---
    def mint_position(self, center_price, range_pct=0.05, usdt_alloc=5.0):
        tick_lower = self.align_tick(self.get_tick_from_price(center_price * (1 - range_pct)))
        tick_upper = self.align_tick(self.get_tick_from_price(center_price * (1 + range_pct)))
        if tick_lower > tick_upper:
            tick_lower, tick_upper = tick_upper, tick_lower

        dec_wmatic, dec_usdt = (self.dec0, self.dec1) if self.is_wmatic_zero else (self.dec1, self.dec0)
        usdt_wei_target = int(usdt_alloc * (10 ** dec_usdt))
        wmatic_wei_target = int((usdt_alloc / max(center_price, 1e-12)) * (10 ** dec_wmatic))

        bal_t0 = int(self.token0_obj.functions.balanceOf(self.owner).call())
        bal_t1 = int(self.token1_obj.functions.balanceOf(self.owner).call())

        target0, target1 = (wmatic_wei_target, usdt_wei_target) if self.is_wmatic_zero else (usdt_wei_target, wmatic_wei_target)
        amount0_final = min(target0, int(bal_t0 * 0.999))
        amount1_final = min(target1, int(bal_t1 * 0.999))
        if amount0_final == 0 and amount1_final == 0:
            return None

        if amount0_final > 0:
            if not self._with_user_creds(approve_if_needed, self.token0_obj, NFT_MANAGER_ADDR, amount0_final):
                return None
        if amount1_final > 0:
            if not self._with_user_creds(approve_if_needed, self.token1_obj, NFT_MANAGER_ADDR, amount1_final):
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

        tx_build = self.nft_manager.functions.mint(params).build_transaction({
            'from': self.owner,
            'nonce': w3.eth.get_transaction_count(self.owner, 'pending'),
            'gas': 900000,
            'gasPrice': w3.eth.gas_price
        })
        return self._with_user_creds(send_tx, tx_build)

    # --- Get active position ID ---
    def get_active_position_id(self):
        balance = self.nft_manager.functions.balanceOf(self.owner).call()
        if balance == 0:
            return None
        return self.nft_manager.functions.tokenOfOwnerByIndex(self.owner, 0).call()

    # --- Position value ---
    def get_position_asset_value(self, token_id, current_price):
        pos = self.nft_manager.functions.positions(token_id).call()
        tick_lower, tick_upper, liquidity = pos[5], pos[6], pos[7]
        if liquidity == 0:
            return 0.0, 0.0, 0.0

        current_tick = self.get_tick_from_price(current_price)
        L = float(liquidity)
        sqrt_price = 1.0001 ** (current_tick / 2.0)
        sqrt_price_lower = 1.0001 ** (tick_lower / 2.0)
        sqrt_price_upper = 1.0001 ** (tick_upper / 2.0)

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
            amount_wmatic = amount0_wei / (10 ** self.dec0)
            amount_usdt = amount1_wei / (10 ** self.dec1)
        else:
            amount_usdt = amount0_wei / (10 ** self.dec0)
            amount_wmatic = amount1_wei / (10 ** self.dec1)

        total_usdt_value = amount_usdt + (amount_wmatic * current_price)
        return amount_usdt, amount_wmatic, total_usdt_value

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
