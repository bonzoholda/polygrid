# core/state.py
from threading import Lock

_state_lock = Lock()

# map uid -> state dict
# each state: {"price": float, "lp_usdt": float, "lp_wmatic": float, "lp_total_value": float, "active": bool}
_lp_state_map = {}

def update_lp_state(uid: int):
    """
    Update LP state for a specific user (uid).
    Keeps simple structure and is thread-safe.
    """
    with _state_lock:
        _lp_state_map[int(uid)] = {
            "price": price,
            "lp_usdt": usdt,
            "lp_wmatic": wmatic,
            "lp_total_value": total,
            "active": bool(active)
        }

def get_lp_state(uid: int):
    """
    Return a *copy* of the stored state for uid, or None if not present.
    """
    with _state_lock:
        s = _lp_state_map.get(int(uid))
        if s is None:
            return None
        return s.copy()

def clear_lp_state(uid: int):
    """
    Optional helper to remove a user's LP state (useful on bot stop).
    """
    with _state_lock:
        _lp_state_map.pop(int(uid), None)
        
