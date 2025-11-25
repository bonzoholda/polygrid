from core.state import update_lp_state
from threading import Lock

# Per-user LP state store (keyed by uid)
_state_lock = Lock()
_user_lp_state = {}  # uid -> { price, lp_usdt, lp_wmatic, lp_total_value, active }

def push_lp_stat(uid: int, stat: dict):
    """
    Update LP state in core/state.py
    Can be called from both runner and FastAPI route
    """
    update_lp_state(uid, stat)
    return stat


def get_lp_stat(uid: int):
    """
    Returns a copy of the stored LP state for uid, or None if not present.
    """
    with _state_lock:
        s = _user_lp_state.get(int(uid))
        if s is None:
            return None
        return s.copy()
