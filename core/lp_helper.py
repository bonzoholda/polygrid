from core.state import update_lp_state
from threading import Lock

# Per-user LP state store (keyed by uid)
_state_lock = Lock()
_user_lp_state = {}  # uid -> { price, lp_usdt, lp_wmatic, lp_total_value, active }

def push_lp_stat(uid: int, stat: dict):
    """
    Update LP state in core/state.py and also in memory.
    Can be called from both runner and FastAPI route.
    """
    # Update DB / persistent state
    update_lp_state(uid, stat)

    # Update in-memory store
    with _state_lock:
        _user_lp_state[int(uid)] = stat.copy()

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
