import math
import time
import logging
import traceback  # checking stack trace
from web3 import Web3

# --- Import from your robust utils ---
# We use addresses from utils to ensure we are using the same config
from utils import (
    w3, OWNER, PRIVATE_KEY, 
    WMATIC_ADDR, USDT_ADDR, 
    send_tx, approve_if_needed, 
    get_pol_price_from_okx, 
    ERC20_ABI # Importing ABI from utils to ensure we can create contracts
)

# --- V3 Constants ---
NFT_MANAGER_ADDR = "0xC36442b4a4522E871399CD717aBDD847Ab11FE88"
POOL_FEE = 3000  
TICK_SPACING = 60 

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
        print("DEBUG: Initializing UniswapV3Manager class...")
        try:
            self.nft_manager = w3.eth.contract(address=NFT_MANAGER_ADDR, abi=NFT_MANAGER_ABI)
            
            # Explicitly create contract objects to avoid import errors from config
            self.token0_obj = w3.eth.contract(address=WMATIC_ADDR, abi=ERC20_ABI)
            self.token1_obj = w3.eth.contract(address=USDT_ADDR, abi=ERC20_ABI)
            
            # Determine Ordering
            if int(WMATIC_ADDR, 16) < int(USDT_ADDR, 16):
                self.token0 = WMATIC_ADDR
                self.token1 = USDT_ADDR
                self.is_wmatic_zero = True 
            else:
                self.token0 = USDT_ADDR
                self.token1 = WMATIC_ADDR
                self.is_wmatic_zero = False
                
            print(f"DEBUG: V3 Manager initialized. WMATIC is Token{'0' if self.is_wmatic_zero else '1'}")
            
        except Exception as e:
            print(f"CRITICAL ERROR in V3 __init__: {e}")
            logging.exception("V3 Init Failed")
            raise e

    def get_tick_from_price(self, price_float):
        if self.is_wmatic_zero:
            raw_price = price_float * (10 ** (6 - 18))
        else:
            if price_float == 0: return 0
            raw_price = (1 / price_float) * (10 ** (18 - 6))

        tick = math.log(raw_price) / math.log(1.0001)
        return int(tick)

    def align_tick(self, tick):
        return (tick // TICK_SPACING) * TICK_SPACING

    def mint_position(self, center_price, range_pct=0.15, usdt_alloc=5.0):
        logging.info(f"🦄 Calculating V3 Mint params. Center: {center_price}, Range: {range_pct*100}%")
        
        lower_price = center_price * (1 - range_pct)
        upper_price = center_price * (1 + range_pct)
        
        tick_lower = self.align_tick(self.get_tick_from_price(lower_price))
        tick_upper = self.align_tick(self.get_tick_from_price(upper_price))
        
        if tick_lower > tick_upper:
            tick_lower, tick_upper = tick_upper, tick_lower

        # Rough amount calculation
        usdt_wei = int(usdt_alloc * 1e6)
        wmatic_wei = int((usdt_alloc / center_price) * 1e18) 

        print(f"DEBUG: Approving {usdt_alloc} USDT and ~{usdt_alloc/center_price:.2f} WMATIC...")

        if not approve_if_needed(self.token0_obj if self.is_wmatic_zero else self.token1_obj, NFT_MANAGER_ADDR, wmatic_wei if self.is_wmatic_zero else usdt_wei):
            logging.error("Approval failed for Token0")
            return None
        if not approve_if_needed(self.token1_obj if self.is_wmatic_zero else self.token0_obj, NFT_MANAGER_ADDR, usdt_wei if self.is_wmatic_zero else wmatic_wei):
            logging.error("Approval failed for Token1")
            return None

        deadline = int(time.time()) + 300
        
        params = {
            'token0': self.token0,
            'token1': self.token1,
            'fee': POOL_FEE,
            'tickLower': tick_lower,
            'tickUpper': tick_upper,
            'amount0Desired': wmatic_wei if self.is_wmatic_zero else usdt_wei,
            'amount1Desired': usdt_wei if self.is_wmatic_zero else wmatic_wei,
            'amount0Min': 0, 
            'amount1Min': 0,
            'recipient': OWNER,
            'deadline': deadline
        }

        logging.info("🦄 Sending Mint Transaction...")
        try:
            tx_build = self.nft_manager.functions.mint(params).build_transaction({
                'from': OWNER,
                'nonce': w3.eth.get_transaction_count(OWNER, 'pending'),
                'gasPrice': w3.eth.gas_price
            })
            return send_tx(tx_build)
        except Exception as e:
            logging.error(f"Mint build failed: {e}")
            return None

    def get_active_position_id(self):
        try:
            bal = self.nft_manager.functions.balanceOf(OWNER).call()
            if bal == 0: return None
            # Just grabbing the last one for now
            token_id = self.nft_manager.functions.tokenOfOwnerByIndex(OWNER, bal - 1).call()
            
            # Verify it has liquidity
            pos = self.nft_manager.functions.positions(token_id).call()
            liq = pos[7]
            if liq == 0: return None
            return token_id
        except Exception as e:
            logging.error(f"Error checking position: {e}")
            return None

# --- Main Runner ---
def run_uniswap_v3_loop(poll_interval=60):
    print("DEBUG: Thread started for Uniswap V3 Strategy...") # Direct console output
    logging.info("🦄 Uniswap V3 Strategy Started.")
    
    manager = None
    
    while True:
        try:
            # Lazy initialization inside the loop to catch startup errors
            if manager is None:
                manager = UniswapV3Manager()
                print("DEBUG: Manager successfully instantiated.")

            # 1. Get Data
            price = get_pol_price_from_okx()
            if not price:
                print("DEBUG: Failed to fetch price, retrying...")
                time.sleep(10)
                continue
                
            logging.info(f"🦄 V3 Cycle | Price: {price} USDT")
            
            # 2. Check Positions
            active_id = manager.get_active_position_id()
            
            if not active_id:
                logging.info(f"🦄 No active position. Minting range around {price}...")
                # Try to mint with 10 USDT worth (adjust 'usdt_alloc' as needed)
                tx = manager.mint_position(center_price=price, range_pct=0.10, usdt_alloc=5.0)
                if tx:
                    logging.info(f"✅ Minted! TX: {tx}")
            else:
                logging.info(f"🦄 Active Position Found (ID: {active_id}). Holding.")
                
            time.sleep(poll_interval)
            
        except Exception as e:
            print(f"CRITICAL THREAD ERROR: {e}")
            traceback.print_exc() # Prints full stack trace to console
            logging.exception("V3 Loop Error")
            time.sleep(10)


# ---------- Entry Point ----------
if __name__ == "__main__":
    import logging
    import time

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    logging.info("🚀 UniswapV3 bot initialized.")
    try:
        logging.info("⚙️ Starting UniswapV3 Strategy Loop...")
        run_uniswap_v3_loop()  # 👈 this calls your actual rebalancing logic
    except KeyboardInterrupt:
        logging.info("🛑 Manual stop received. Exiting Asset Balancer gracefully...")
    except Exception as e:
        logging.exception(f"❌ Unexpected error in Asset Balancer: {e}")
