import math
import time
import logging
from web3 import Web3

# --- Import from your robust utils ---
from utils import (
    w3, OWNER, PRIVATE_KEY, 
    WMATIC_ADDR, USDT_ADDR, 
    send_tx, approve_if_needed, 
    get_pol_price_from_okx, 
    get_token_decimals, 
    get_onchain_token_balance
)
from config import wmatic, usdt  # Assuming these are the contract objects

# --- V3 Constants ---
# Polygon NonfungiblePositionManager
NFT_MANAGER_ADDR = "0xC36442b4a4522E871399CD717aBDD847Ab11FE88"
POOL_FEE = 3000  # 0.3% fee tier (Standard for volatile pairs like MATIC/USDT)
TICK_SPACING = 60 # Associated with 3000 fee tier

# Minimal ABI for NFT Manager (Mint, Burn, Collect)
NFT_MANAGER_ABI = [
    {"inputs":[{"internalType":"struct MintParams","name":"params","type":"tuple","components":[{"internalType":"address","name":"token0","type":"address"},{"internalType":"address","name":"token1","type":"address"},{"internalType":"uint24","name":"fee","type":"uint24"},{"internalType":"int24","name":"tickLower","type":"int24"},{"internalType":"int24","name":"tickUpper","type":"int24"},{"internalType":"uint256","name":"amount0Desired","type":"uint256"},{"internalType":"uint256","name":"amount1Desired","type":"uint256"},{"internalType":"uint256","name":"amount0Min","type":"uint256"},{"internalType":"uint256","name":"amount1Min","type":"uint256"},{"internalType":"address","name":"recipient","type":"address"},{"internalType":"uint256","name":"deadline","type":"uint256"}]}],"name":"mint","outputs":[{"internalType":"uint256","name":"tokenId","type":"uint256"},{"internalType":"uint128","name":"liquidity","type":"uint128"},{"internalType":"uint256","name":"amount0","type":"uint256"},{"internalType":"uint256","name":"amount1","type":"uint256"}],"stateMutability":"payable","type":"function"},
    {"inputs":[{"internalType":"struct DecreaseLiquidityParams","name":"params","type":"tuple","components":[{"internalType":"uint256","name":"tokenId","type":"uint256"},{"internalType":"uint128","name":"liquidity","type":"uint128"},{"internalType":"uint256","name":"amount0Min","type":"uint256"},{"internalType":"uint256","name":"amount1Min","type":"uint256"},{"internalType":"uint256","name":"deadline","type":"uint256"}]}],"name":"decreaseLiquidity","outputs":[{"internalType":"uint256","name":"amount0","type":"uint256"},{"internalType":"uint256","name":"amount1","type":"uint256"}],"stateMutability":"payable","type":"function"},
    {"inputs":[{"internalType":"struct CollectParams","name":"params","type":"tuple","components":[{"internalType":"uint256","name":"tokenId","type":"uint256"},{"internalType":"address","name":"recipient","type":"address"},{"internalType":"uint128","name":"amount0Max","type":"uint128"},{"internalType":"uint128","name":"amount1Max","type":"uint128"}]}],"name":"collect","outputs":[{"internalType":"uint256","name":"amount0","type":"uint256"},{"internalType":"uint256","name":"amount1","type":"uint256"}],"stateMutability":"payable","type":"function"},
    {"inputs":[{"internalType":"address","name":"owner","type":"address"}],"name":"balanceOf","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"internalType":"address","name":"owner","type":"address"},{"internalType":"uint256","name":"index","type":"uint256"}],"name":"tokenOfOwnerByIndex","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"internalType":"uint256","name":"tokenId","type":"uint256"}],"name":"positions","outputs":[{"internalType":"uint96","name":"nonce","type":"uint96"},{"internalType":"address","name":"operator","type":"address"},{"internalType":"address","name":"token0","type":"address"},{"internalType":"address","name":"token1","type":"address"},{"internalType":"uint24","name":"fee","type":"uint24"},{"internalType":"int24","name":"tickLower","type":"int24"},{"internalType":"int24","name":"tickUpper","type":"int24"},{"internalType":"uint128","name":"liquidity","type":"uint128"},{"internalType":"uint256","name":"feeGrowthInside0LastX128","type":"uint256"},{"internalType":"uint256","name":"feeGrowthInside1LastX128","type":"uint256"},{"internalType":"uint128","name":"tokensOwed0","type":"uint128"},{"internalType":"uint128","name":"tokensOwed1","type":"uint128"}],"stateMutability":"view","type":"function"}
]

class UniswapV3Manager:
    def __init__(self):
        self.nft_manager = w3.eth.contract(address=NFT_MANAGER_ADDR, abi=NFT_MANAGER_ABI)
        
        # Determine Token0/Token1 ordering (Uniswap requirement)
        if int(WMATIC_ADDR, 16) < int(USDT_ADDR, 16):
            self.token0 = WMATIC_ADDR
            self.token1 = USDT_ADDR
            self.token0_obj = wmatic
            self.token1_obj = usdt
            self.is_wmatic_zero = True # Price is USDT/WMATIC
        else:
            self.token0 = USDT_ADDR
            self.token1 = WMATIC_ADDR
            self.token0_obj = usdt
            self.token1_obj = wmatic
            self.is_wmatic_zero = False # Price is WMATIC/USDT (inverted)

    # --- Math Helpers ---
    def get_tick_from_price(self, price_float):
        """
        Converts human price (USDT per WMATIC) to V3 Tick.
        """
        # 1. Adjust price to Raw (taking decimals into account)
        # Price_Raw = Price_Human * 10^(Dec1 - Dec0)
        # Usually Dec1(USDT)=6, Dec0(WMATIC)=18 => 10^-12
        
        if self.is_wmatic_zero:
            # Token0=WMATIC, Token1=USDT. Price = Token1/Token0
            raw_price = price_float * (10 ** (6 - 18))
        else:
            # Token0=USDT, Token1=WMATIC. Price = Token1/Token0
            # If input is USDT/WMATIC, we must invert for math
            if price_float == 0: return 0
            raw_price = (1 / price_float) * (10 ** (18 - 6))

        # 2. Log base 1.0001
        tick = math.log(raw_price) / math.log(1.0001)
        return int(tick)

    def align_tick(self, tick):
        """Aligns tick to the pool spacing"""
        return (tick // TICK_SPACING) * TICK_SPACING

    # --- Core Actions ---
    def mint_position(self, center_price, range_pct=0.15, usdt_alloc=10.0):
        """
        Mints a V3 position centered on price +/- range_pct.
        Uses utils.approve_if_needed and utils.send_tx
        """
        logging.info(f"🦄 Calculating V3 Mint params. Center: {center_price}, Range: {range_pct*100}%")
        
        # 1. Calculate Range
        lower_price = center_price * (1 - range_pct)
        upper_price = center_price * (1 + range_pct)
        
        tick_lower = self.align_tick(self.get_tick_from_price(lower_price))
        tick_upper = self.align_tick(self.get_tick_from_price(upper_price))
        
        # Ensure Order
        if tick_lower > tick_upper:
            tick_lower, tick_upper = tick_upper, tick_lower

        # 2. Calculate Amounts (Rough estimation for single-sided or balanced)
        # We try to supply provided USDT and equivalent WMATIC
        usdt_wei = int(usdt_alloc * 1e6)
        wmatic_wei = int((usdt_alloc / center_price) * 1e18) 

        # 3. Approvals (Using your Utils!)
        if not approve_if_needed(self.token0_obj, NFT_MANAGER_ADDR, wmatic_wei if self.is_wmatic_zero else usdt_wei):
            return None
        if not approve_if_needed(self.token1_obj, NFT_MANAGER_ADDR, usdt_wei if self.is_wmatic_zero else wmatic_wei):
            return None

        # 4. Build Tx
        deadline = int(time.time()) + 300
        
        params = {
            'token0': self.token0,
            'token1': self.token1,
            'fee': POOL_FEE,
            'tickLower': tick_lower,
            'tickUpper': tick_upper,
            'amount0Desired': wmatic_wei if self.is_wmatic_zero else usdt_wei,
            'amount1Desired': usdt_wei if self.is_wmatic_zero else wmatic_wei,
            'amount0Min': 0, # Slippage set to 0 for simplicity (use with care or add buffer)
            'amount1Min': 0,
            'recipient': OWNER,
            'deadline': deadline
        }

        logging.info("🦄 Sending Mint Transaction...")
        # Note: We build the tx from the contract function
        tx_build = self.nft_manager.functions.mint(params).build_transaction({
            'from': OWNER,
            'nonce': w3.eth.get_transaction_count(OWNER, 'pending'),
            'gasPrice': w3.eth.gas_price
        })
        
        # 5. Execute via robust send_tx
        tx_hash = send_tx(tx_build)
        return tx_hash

    def close_position(self, token_id):
        """
        1. Decrease Liquidity to 0 (Burn)
        2. Collect Fees + Principal
        """
        logging.info(f"🔥 Closing Position NFT ID: {token_id}")
        
        # A. Get Liquidity Amount
        pos = self.nft_manager.functions.positions(token_id).call()
        liquidity = pos[7]
        
        if liquidity > 0:
            # Decrease Liquidity
            params_decrease = {
                'tokenId': token_id,
                'liquidity': liquidity,
                'amount0Min': 0,
                'amount1Min': 0,
                'deadline': int(time.time()) + 300
            }
            tx_dec = self.nft_manager.functions.decreaseLiquidity(params_decrease).build_transaction({
                'from': OWNER,
                'nonce': w3.eth.get_transaction_count(OWNER, 'pending'),
                'gasPrice': w3.eth.gas_price
            })
            if not send_tx(tx_dec):
                logging.error("❌ Failed to decrease liquidity")
                return False

        # B. Collect Tokens (Fees + Principal)
        MAX_UINT128 = 2**128 - 1
        params_collect = {
            'tokenId': token_id,
            'recipient': OWNER,
            'amount0Max': MAX_UINT128,
            'amount1Max': MAX_UINT128
        }
        tx_col = self.nft_manager.functions.collect(params_collect).build_transaction({
             'from': OWNER,
             'nonce': w3.eth.get_transaction_count(OWNER, 'pending'),
             'gasPrice': w3.eth.gas_price
        })
        
        res = send_tx(tx_col)
        if res:
            logging.info(f"💰 Position closed & funds collected. TX: {res}")
            return True
        return False

    def get_active_position_id(self):
        """Check if wallet holds a V3 NFT for this pool"""
        bal = self.nft_manager.functions.balanceOf(OWNER).call()
        if bal == 0:
            return None
        
        # Iterate to find one relevant to our pool (simplified: just grab last)
        # In prod, check token0/token1 match our pool
        token_id = self.nft_manager.functions.tokenOfOwnerByIndex(OWNER, bal - 1).call()
        
        # Optional: Check if liquidity > 0
        pos = self.nft_manager.functions.positions(token_id).call()
        liq = pos[7]
        if liq == 0:
            return None # We have an empty NFT
            
        return token_id

# --- Main Runner for Bot ---
def run_uniswap_v3_loop(poll_interval=60):
    logging.info("🦄 Uniswap V3 Strategy Started.")
    manager = UniswapV3Manager()
    
    while True:
        try:
            # 1. Get Data
            price = get_pol_price_from_okx()
            if not price:
                time.sleep(10)
                continue
                
            active_id = manager.get_active_position_id()
            
            # 2. Logic
            if not active_id:
                logging.info(f"🦄 No active position. Price: {price}. Minting new range...")
                # Create a range: e.g., Current Price to -27% (mimicking your grid)
                # Or balanced: +/- 10%
                manager.mint_position(center_price=price, range_pct=0.10, usdt_alloc=20.0)
            else:
                logging.info(f"🦄 Active Position Found (ID: {active_id}). Monitoring...")
                # [Optimization] Here you can add logic:
                # If price moves > 10% away from entry, call manager.close_position(active_id)
                
            time.sleep(poll_interval)
            
        except Exception as e:
            logging.exception("V3 Loop Error")
            time.sleep(10)
