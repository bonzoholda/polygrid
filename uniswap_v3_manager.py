import math
import time
import logging
import traceback
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

from config import usdt, wmatic, OWNER, USDT_ADDR, WMATIC_ADDR


# --- V3 Constants ---
NFT_MANAGER_ADDR = "0xC36442b4a4522E871399CD717aBDD847Ab11FE88"
POOL_FEE = 3000  
TICK_SPACING = 60 

# (Keep NFT_MANAGER_ABI the same as before - omitted here for brevity)
NFT_MANAGER_ABI = [
    {"inputs":[{"internalType":"struct MintParams","name":"params","type":"tuple","components":[{"internalType":"address","name":"token0","type":"address"},{"internalType":"address","name":"token1","type":"address"},{"internalType":"uint24","name":"fee","type":"uint24"},{"internalType":"int24","name":"tickLower","type":"int24"},{"internalType":"int24","name":"tickUpper","type":"int24"},{"internalType":"uint256","name":"amount0Desired","type":"uint256"},{"internalType":"uint256","name":"amount1Desired","type":"uint256"},{"internalType":"uint256","name":"amount0Min","type":"uint256"},{"internalType":"uint256","name":"amount1Min","type":"uint256"},{"internalType":"address","name":"recipient","type":"address"},{"internalType":"uint256","name":"deadline","type":"uint256"}]}],"name":"mint","outputs":[{"internalType":"uint256","name":"tokenId","type":"uint256"},{"internalType":"uint128","name":"liquidity","type":"uint128"},{"internalType":"uint256","name":"amount0","type":"uint256"},{"internalType":"uint256","name":"amount1","type":"uint256"}],"stateMutability":"payable","type":"function"},
    {"inputs":[{"internalType":"struct DecreaseLiquidityParams","name":"params","type":"tuple","components":[{"internalType":"uint256","name":"tokenId","type":"uint256"},{"internalType":"uint128","name":"liquidity","type":"uint128"},{"internalType":"uint256","name":"amount0Min","type":"uint256"},{"internalType":"uint256","name":"amount1Min","type":"uint256"},{"internalType":"uint256","name":"deadline","type":"uint256"}]}],"name":"decreaseLiquidity","outputs":[{"internalType":"uint256","name":"amount0","type":"uint256"},{"internalType":"uint256","name":"amount1","type":"uint256"}],"stateMutability":"payable","type":"function"},
    {"inputs":[{"internalType":"struct CollectParams","name":"params","type":"tuple","components":[{"internalType":"uint256","name":"tokenId","type":"uint256"},{"internalType":"address","name":"recipient","type":"address"},{"internalType":"uint128","name":"amount0Max","type":"uint128"},{"internalType":"uint128","name":"amount1Max","type":"uint128"}]}],"name":"collect","outputs":[{"internalType":"uint256","name":"amount0","type":"uint256"},{"internalType":"uint256","name":"amount1","type":"uint256"}],"stateMutability":"payable","type":"function"},
    {"inputs":[{"internalType":"address","name":"owner","type":"address"}],"name":"balanceOf","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"internalType":"address","name":"owner","type":"address"},{"internalType":"uint256","name":"index","type":"uint256"}],"name":"tokenOfOwnerByIndex","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"internalType":"uint256","name":"tokenId","type":"uint256"}],"name":"positions","outputs":[{"internalType":"uint96","name":"nonce","type":"uint96"},{"internalType":"address","name":"operator","type":"address"},{"internalType":"address","name":"token0","type":"address"},{"internalType":"address","name":"token1","type":"address"},{"internalType":"uint24","name":"fee","type":"uint24"},{"internalType":"int24","name":"tickLower","type":"int24"},{"internalType":"int24","name":"tickUpper","type":"int24"},{"internalType":"uint128","name":"liquidity","type":"uint128"},{"internalType":"uint256","name":"feeGrowthInside0LastX128","type":"uint256"},{"internalType":"uint256","name":"feeGrowthInside1LastX128","type":"uint256"},{"internalType":"uint128","name":"tokensOwed0","type":"uint128"},{"internalType":"uint128","name":"tokensOwed1","type":"uint128"}],"stateMutability":"view","type":"function"}
]

class UniswapV3Manager:
    def __init__(self, owner_address=None):
        print("DEBUG: Initializing UniswapV3Manager class...")

        # 💡 Ensure your internal logic uses the passed address if available
        if owner_address:
            self.owner = w3.to_checksum_address(owner_address)
        else:
            self.owner = w3.to_checksum_address(OWNER)        
        
        try:
            self.nft_manager = w3.eth.contract(address=NFT_MANAGER_ADDR, abi=NFT_MANAGER_ABI)
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
        except Exception as e:
            print(f"CRITICAL ERROR in V3 __init__: {e}")
            raise e

    # --- Math Helpers (Keep same) ---
    def get_tick_from_price(self, price_float):
        # ... (Same as previous code) ...
        if self.is_wmatic_zero:
            raw_price = price_float * (10 ** (6 - 18))
        else:
            if price_float == 0: return 0
            raw_price = (1 / price_float) * (10 ** (18 - 6))
        return int(math.log(raw_price) / math.log(1.0001))

    def align_tick(self, tick):
        return (tick // TICK_SPACING) * TICK_SPACING

    # --- NEW: Auto-Balancer ---
    def balance_wallet_50_50(self, current_price_usdt):
        """
        Check USDT and WMATIC balance. 
        If imbalance > 20%, swap to restore ~50:50 ratio.
        """
        logging.info("⚖️ Checking wallet balance for 50:50 split...")
        
        # 1. Get Balances (Float)
        bal_usdt = get_onchain_token_balance(usdt, OWNER)
        bal_wmatic = get_onchain_token_balance(wmatic, OWNER)
        
        # 2. Calculate Total Value in USDT
        val_usdt = bal_usdt
        val_wmatic = bal_wmatic * current_price_usdt
        total_val = val_usdt + val_wmatic
        
        if total_val < 5.0:
            logging.warning("⚠️ Wallet balance too low (< $5) to balance.")
            return

        # 3. Check Ratio
        usdt_ratio = val_usdt / total_val
        logging.info(f"⚖️ Ratio: USDT {usdt_ratio*100:.1f}% | WMATIC {(1-usdt_ratio)*100:.1f}%")

        # 4. Swap Logic (Buffer 0.4 - 0.6 is acceptable)
        if usdt_ratio > 0.60:
            # Too much USDT, Swap half of excess to WMATIC
            # Target USDT = 50% of total
            surplus_usdt = val_usdt - (total_val * 0.5)
            logging.info(f"⚖️ Rebalancing: Swapping {surplus_usdt:.2f} USDT -> WMATIC")
            swap_usdt_to_wmatic(surplus_usdt)
            time.sleep(5) # Wait for blockchain update

        elif usdt_ratio < 0.40:
            # Too much WMATIC, Swap half of excess to USDT
            surplus_wmatic_val = val_wmatic - (total_val * 0.5)
            surplus_wmatic_amt = surplus_wmatic_val / current_price_usdt
            logging.info(f"⚖️ Rebalancing: Swapping {surplus_wmatic_amt:.2f} WMATIC -> USDT")
            swap_wmatic_to_usdt(surplus_wmatic_amt)
            time.sleep(5)
        else:
            logging.info("✅ Balance is healthy (near 50:50). No swap needed.")

# --- Action: Mint (With Clamping Logic) ---
    def mint_position(self, center_price, range_pct=0.05, usdt_alloc=5.0):
        logging.info(f"🦄 Calculating V3 Mint params. Center: {center_price}, Range: {range_pct*100}%")
        
        lower_price = center_price * (1 - range_pct)
        upper_price = center_price * (1 + range_pct)
        
        tick_lower = self.align_tick(self.get_tick_from_price(lower_price))
        tick_upper = self.align_tick(self.get_tick_from_price(upper_price))
        
        if tick_lower > tick_upper:
            tick_lower, tick_upper = tick_upper, tick_lower

        # 1. Calculate Theoretical Desired Amounts (Wei)
        # Use a distinct name for targets
        usdt_wei_target = int(usdt_alloc * 1e6)
        wmatic_wei_target = int((usdt_alloc / center_price) * 1e18) 

        # 2. Get Current WEI Balances (HIGH PRECISION CHECK)
        bal_t0 = self.token0_obj.functions.balanceOf(OWNER).call()
        bal_t1 = self.token1_obj.functions.balanceOf(OWNER).call()
        
        # 3. Determine the desired amounts for Token0 and Token1 based on price
        if self.is_wmatic_zero: # Token0=WMATIC, Token1=USDT
            target0 = wmatic_wei_target
            target1 = usdt_wei_target
        else:                   # Token0=USDT, Token1=WMATIC
            target0 = usdt_wei_target
            target1 = wmatic_wei_target

        # 4. CLAMP AND ASSIGN FINAL DESIRED AMOUNTS (THE CRUCIAL STEP)
        # We clamp the target to the actual balance
        amount0_final = min(target0, int(bal_t0 * 0.999))
        amount1_final = min(target1, int(bal_t1 * 0.999))
        
        if amount0_final == 0 or amount1_final == 0:
             logging.error("❌ MINT FAILED: Cannot execute mint due to zero balance in one token.")
             return None
        
        logging.info(f"DEBUG: Approving {amount0_final} (T0) and {amount1_final} (T1) for mint...")

        # 5. Approvals (Using final amounts for clarity)
        if not approve_if_needed(self.token0_obj, NFT_MANAGER_ADDR, amount0_final):
            logging.error("Approval failed for Token0")
            return None
        if not approve_if_needed(self.token1_obj, NFT_MANAGER_ADDR, amount1_final):
            logging.error("Approval failed for Token1")
            return None

        # 6. Build Params (USING CLAMPED FINAL AMOUNTS)
        deadline = int(time.time()) + 300
        
        params = {
            'token0': self.token0,
            'token1': self.token1,
            'fee': POOL_FEE,
            'tickLower': tick_lower,
            'tickUpper': tick_upper,
            'amount0Desired': amount0_final, # <-- CORRECTED
            'amount1Desired': amount1_final, # <-- CORRECTED
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

    # --- NEW: Check Active Position ---
    def check_position_status(self, token_id, current_price):
        """
        Checks if the current price is outside the position's range.
        If yes, closes the position.
        """
        try:
            # Fetch position details from contract
            # Returns: (nonce, operator, token0, token1, fee, tickLower, tickUpper, liquidity, ...)
            pos = self.nft_manager.functions.positions(token_id).call()
            tick_lower = pos[5]
            tick_upper = pos[6]
            liquidity = pos[7]

            current_tick = self.get_tick_from_price(current_price)

            logging.info(f"🔍 Position Status: Tick {current_tick} vs Range [{tick_lower} - {tick_upper}]")

            # EXIT CONDITION: Out of Range
            if current_tick < tick_lower or current_tick > tick_upper:
                logging.warning(f"🚨 Price Out of Range! Closing Position {token_id}...")
                self.close_position(token_id)
                return False # Position no longer active
            
            if liquidity == 0:
                logging.warning("⚠️ Position has 0 liquidity (already closed?).")
                return False

            return True # Position still active and healthy
        except Exception as e:
            logging.error(f"Error checking position status: {e}")
            return True # Assume active to prevent panic loop

    def close_position(self, token_id):
        # ... (Use the close_position logic provided in step 1) ...
        logging.info(f"🔥 Closing Position NFT ID: {token_id}")
        
        # 1. Get Liquidity
        pos = self.nft_manager.functions.positions(token_id).call()
        liquidity = pos[7]
        
        if liquidity > 0:
            # Decrease Liquidity to 0
            params_dec = {
                'tokenId': token_id, 
                'liquidity': liquidity, 
                'amount0Min': 0, 
                'amount1Min': 0, 
                'deadline': int(time.time())+300
            }
            tx = self.nft_manager.functions.decreaseLiquidity(params_dec).build_transaction({
                'from': OWNER, 'nonce': w3.eth.get_transaction_count(OWNER, 'pending'), 'gasPrice': w3.eth.gas_price
            })
            send_tx(tx)
            time.sleep(2)

        # 2. Collect Fees/Tokens
        params_col = {
            'tokenId': token_id, 'recipient': OWNER, 
            'amount0Max': 2**128-1, 'amount1Max': 2**128-1
        }
        tx2 = self.nft_manager.functions.collect(params_col).build_transaction({
            'from': OWNER, 'nonce': w3.eth.get_transaction_count(OWNER, 'pending'), 'gasPrice': w3.eth.gas_price
        })
        send_tx(tx2)
        logging.info("✅ Position Closed and Funds Collected.")

    def get_active_position_id(self):
        """
        Retrieves the first NFT ID owned by the bot (OWNER) 
        that represents an active position.
        """
        try:
            # Check how many NFTs (positions) the owner holds
            balance = self.nft_manager.functions.balanceOf(self.owner).call()
            
            if balance == 0:
                return None
            
            logging.info(f"NFT Manager reports {balance} active positions.")
            
            # For a simple bot, we usually just manage one position. 
            # We fetch the first NFT ID (index 0) held by the owner.
            # Uniswap V3 returns token ID, not index.
            token_id = self.nft_manager.functions.tokenOfOwnerByIndex(self.owner, 0).call()
            return token_id
            
        except Exception as e:
            # This can happen if the NFT balance is 0 or if there's a RPC issue
            logging.warning(f"Failed to retrieve active position ID: {e}")
            return None

    def get_position_asset_value(self, token_id, current_price):
        """
        Calculates the amount of Token0 and Token1 held by the active position 
        based on the current price. Returns the total value in USDT.
        """
        try:
            # 1. Fetch Position Data
            pos = self.nft_manager.functions.positions(token_id).call()
            tick_lower = pos[5]
            tick_upper = pos[6]
            liquidity = pos[7]

            if liquidity == 0:
                logging.info(f"Position ID {token_id} has zero liquidity.")
                return 0.0, 0.0, 0.0 # T0, T1, Total USD Value

            # 2. Get Current Tick
            current_tick = self.get_tick_from_price(current_price)

            # 3. Uniswap V3 Math: Calculate Token Amounts
            # This requires complex math based on Liquidity (L) and the current/boundary ticks.
            # For simplicity and robust calculation, we will use a library approximation 
            # or pre-calculated formulas. We'll stick to the core Web3 logic here:
            
            # Convert ticks to sqrtPrice (P = 1.0001^tick)
            sqrt_price = math.sqrt(1.0001 ** current_tick)
            sqrt_price_lower = math.sqrt(1.0001 ** tick_lower)
            sqrt_price_upper = math.sqrt(1.0001 ** tick_upper)

            amount0_wei = 0
            amount1_wei = 0
            
            # Case A: Price is below the range (100% Token0)
            if current_tick <= tick_lower:
                amount0_wei = liquidity * ((sqrt_price_upper - sqrt_price_lower) / (sqrt_price_lower * sqrt_price_upper))
                amount1_wei = 0
            
            # Case B: Price is above the range (100% Token1)
            elif current_tick >= tick_upper:
                amount0_wei = 0
                amount1_wei = liquidity * (sqrt_price_upper - sqrt_price_lower)
            
            # Case C: Price is inside the range (Mixed T0 and T1)
            else:
                amount0_wei = liquidity * ((sqrt_price_upper - sqrt_price) / (sqrt_price * sqrt_price_upper))
                amount1_wei = liquidity * (sqrt_price - sqrt_price_lower)

            # 4. Convert to Human-Readable Floats (Handling Token Decimals)
            # Assuming WMATIC is 18 decimals and USDT is 6 decimals.
            if self.is_wmatic_zero:
                # Token0 = WMATIC (18), Token1 = USDT (6)
                amount_wmatic = amount0_wei / 1e18
                amount_usdt = amount1_wei / 1e6
            else:
                # Token0 = USDT (6), Token1 = WMATIC (18)
                amount_usdt = amount0_wei / 1e6
                amount_wmatic = amount1_wei / 1e18
            
            # 5. Calculate Total Value in USDT
            total_usdt_value = amount_usdt + (amount_wmatic * current_price)
            
            return amount_usdt, amount_wmatic, total_usdt_value

        except Exception as e:
            logging.error(f"Error calculating position value for ID {token_id}: {e}")
            return 0.0, 0.0, 0.0



# --- Updated Runner ---
def run_uniswap_v3_loop(poll_interval=60):
    print("DEBUG: Thread started for Uniswap V3 Strategy...")
    logging.info("🦄 Uniswap V3 Strategy Started.")
    
    manager = None
    
    while True:
        try:
            if manager is None: manager = UniswapV3Manager()

            # 1. Get Data
            price = get_pol_price_from_okx()
            if not price:
                time.sleep(10)
                continue
            logging.info(f"💰 Current WMATIC price: {price:.4f} USDT") # Log price here
            
            # 2. Check Active ID
            active_id = manager.get_active_position_id()
            
            if active_id:
                
                # --- NEW: Get and Log Position Value ---
                usdt_amt, wmatic_amt, total_value = manager.get_position_asset_value(active_id, price)
                
                logging.info("----------------------------------------------------------------")
                logging.info(f"🦄 Active Position ID: {active_id}")
                logging.info(f"💰 Position Assets: {usdt_amt:.2f} USDT | {wmatic_amt:.4f} WMATIC")
                logging.info(f"💵 **TOTAL LP VALUE (USD): ${total_value:.2f}**")
                logging.info("----------------------------------------------------------------")    
                
                # --- NEW: Check if we need to Close ---
                is_active = manager.check_position_status(active_id, price)
                if not is_active:
                    # If it returned False, it means it just closed it.
                    # Loop will restart, find no active ID, and enter Mint logic below.
                    logging.info("♻️ Position closed. Preparing to re-enter...")
                    time.sleep(5)
                    continue 
                else:
                    logging.info(f"🦄 Holding active position ID {active_id}. Price {price}")
            
            else:
                # --- NEW: Auto-Balance before Minting ---
                logging.info(f"🦄 No active position. Preparing entry around {price}...")
                
                # Dynamic Split: Checks USDT/WMATIC and swaps if needed
                manager.balance_wallet_50_50(price)
                
                # Mint Logic (Dynamic Allocation based on balance)
                usdt_balance = get_onchain_token_balance(usdt, OWNER)
                alloc_size = min(usdt_balance * 0.9, 50.0) # Use 90% of USDT or max 50 USDT
                
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
