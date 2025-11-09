# config.py
from web3 import Web3
import os
from dotenv import load_dotenv

load_dotenv()

# ---------- Blockchain Config ----------
RPC_URL = os.getenv("RPC_URL", "https://polygon-rpc.com")
PRIVATE_KEY = os.getenv("PRIVATE_KEY")
OWNER = os.getenv("OWNER_ADDR")

w3 = Web3(Web3.HTTPProvider(RPC_URL))
CHAIN_ID = 137  # Polygon Mainnet

# ---------- Contract Addresses ----------
ROUTER_ADDR = Web3.to_checksum_address("0xa5E0829CaCED8fFDD4De3c43696c57F7D7A678ff")  # QuickSwap
USDT_ADDR = Web3.to_checksum_address("0xC2132D05D31c914a87C6611C10748AEb04B58e8F")
WMATIC_ADDR = Web3.to_checksum_address("0x0d500B1d8E8eF31E21C99d1Db9A6444d3ADf1270")

# ---------- ABIs ----------
# You can put minimal ERC20 + Router ABIs here
ERC20_ABI = [
    {"constant": True, "inputs": [], "name": "decimals", "outputs": [{"name": "", "type": "uint8"}], "type": "function"},
    {"constant": True, "inputs": [{"name": "_owner", "type": "address"}, {"name": "_spender", "type": "address"}],
     "name": "allowance", "outputs": [{"name": "remaining", "type": "uint256"}], "type": "function"},
    {"constant": False, "inputs": [{"name": "_spender", "type": "address"}, {"name": "_value", "type": "uint256"}],
     "name": "approve", "outputs": [{"name": "success", "type": "bool"}], "type": "function"},
    {"constant": True, "inputs": [{"name": "_owner", "type": "address"}],
     "name": "balanceOf", "outputs": [{"name": "balance", "type": "uint256"}], "type": "function"},
]

ROUTER_ABI = [
    {
        "name": "swapExactTokensForTokensSupportingFeeOnTransferTokens",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "amountIn", "type": "uint256"},
            {"name": "amountOutMin", "type": "uint256"},
            {"name": "path", "type": "address[]"},
            {"name": "to", "type": "address"},
            {"name": "deadline", "type": "uint256"},
        ],
        "outputs": [],
    },
    {
        "name": "getAmountsOut",
        "type": "function",
        "stateMutability": "view",
        "inputs": [{"name": "amountIn", "type": "uint256"}, {"name": "path", "type": "address[]"}],
        "outputs": [{"name": "amounts", "type": "uint256[]"}],
    },
]

# ---------- Instantiate Contracts ----------
usdt = w3.eth.contract(address=USDT_ADDR, abi=ERC20_ABI)
wmatic = w3.eth.contract(address=WMATIC_ADDR, abi=ERC20_ABI)
router = w3.eth.contract(address=ROUTER_ADDR, abi=ROUTER_ABI)
