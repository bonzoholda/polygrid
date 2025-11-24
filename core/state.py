# core/state.py
from threading import Lock

_state_lock = Lock()

# Store LP state per user
# lp_states[uid] = { price, lp_usdt, lp_wmatic, lp_total_value, active }
lp_states = {}


def update_lp_state(uid, price, usdt, wmatic, total, active=True):
    """
    Update LP state for a specific user.
    """
    with _state_lock:
        lp_states[uid] = {
            "price": price,
            "lp_usdt": usdt,
            "lp_wmatic": wmatic,
            "lp_total_value": total,
            "active": active,
        }


def get_lp_state(uid):
    """
    Fetch LP state for a user. Returns None if no state exists.
    """
    with _state_lock:
        return lp_states.get(uid, None)
