from threading import Lock
import logging

# Set up a lock to safely access the shared dictionary across threads
_state_lock = Lock()

# per-user LP states stored as dict of dicts
# { uid: { "price", "lp_usdt", "lp_wmatic", "lp_total_value", "active" } }
_user_lp_states = {}     


def save_lp_state(uid: int, state: dict):
    """
    Saves the complete LP position state dictionary for a specific user ID (uid).
    This function is called by the Uniswap V3 bot thread.
    
    Args:
        uid: The unique identifier for the user.
        state: A dictionary containing the LP data calculated by the bot.
    """
    logging.debug(f"Attempting to save LP state for user {uid}")
    
    # Safely update the state under the lock
    with _state_lock:
        _user_lp_states[uid] = {
            # Use .get() for safety and ensure type casting
            "price": float(state.get("price", 0.0)),
            "lp_usdt": float(state.get("lp_usdt", 0.0)),
            "lp_wmatic": float(state.get("lp_wmatic", 0.0)),
            "lp_total_value": float(state.get("lp_total_value", 0.0)),
            "active": bool(state.get("active", False)),
            # "token_id": state.get("token_id") # Optional: include token ID if present
        }
    logging.debug(f"Successfully saved LP state for user {uid}")


def get_lp_state(uid: int):
    """
    Retrieve LP state for a specific user.
    This function is called by the portfolio fetching process (API/Dashboard).
    
    Args:
        uid: The unique identifier for the user.
        
    Returns:
        dict: The LP state dictionary, or None if the state has never been saved.
    """
    # Safely retrieve the state under the lock
    with _state_lock:
        return _user_lp_states.get(uid)
