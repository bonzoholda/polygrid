# core/state.py
from threading import Lock

# Per-user LP state store (keyed by uid)
_state_lock = Lock()
_user_lp_state = {}  # uid -> { price, lp_usdt, lp_wmatic, lp_total_value, active }

def update_lp_state(uid: int, state_dict: dict):
    """
    Update the LP state for a given uid.
    state_dict expected keys: wmatic_price (or price), lp_usdt, lp_wmatic, lp_value_usdt (or lp_total_value), active
    """
    if uid is None:
        return
    with _state_lock:
        # Normalize keys for compatibility
        normalized = {
            "price": None,
            "lp_usdt": 0.0,
            "lp_wmatic": 0.0,
            "lp_total_value": 0.0,
            "active": False
        }
        # Accept multiple key names used across code
        if state_dict is None:
            state_dict = {}

        # mapping helpers
        if "wmatic_price" in state_dict:
            normalized["price"] = state_dict.get("wmatic_price")
        elif "price" in state_dict:
            normalized["price"] = state_dict.get("price")

        normalized["lp_usdt"] = float(state_dict.get("lp_usdt", state_dict.get("usdt", normalized["lp_usdt"])))
        normalized["lp_wmatic"] = float(state_dict.get("lp_wmatic", state_dict.get("wmatic", normalized["lp_wmatic"])))
        normalized["lp_total_value"] = float(state_dict.get("lp_value_usdt", state_dict.get("lp_total_value", normalized["lp_total_value"])))
        normalized["active"] = bool(state_dict.get("active", normalized["active"]))

        _user_lp_state[int(uid)] = normalized

def get_lp_state(uid: int):
    """
    Returns a copy of the stored LP state for uid, or None if not present.
    """
    with _state_lock:
        s = _user_lp_state.get(int(uid))
        if s is None:
            return None
        return s.copy()

def clear_lp_state(uid: int):
    with _state_lock:
        _user_lp_state.pop(int(uid), None)
