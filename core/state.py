# core/state.py

from threading import Lock

_state_lock = Lock()

# Store LP state PER USER
# Example:
# {
#     1: {"price":..., "lp_usdt":..., ...},
#     2: {"price":..., "lp_usdt":..., ...},
# }
lp_state = {}


def update_lp_state(uid, price, usdt, wmatic, total, active=True):
    """
    Store LP state per user (thread-safe)
    """
    with _state_lock:
        lp_state[uid] = {
            "price": price,
            "lp_usdt": usdt,
            "lp_wmatic": wmatic,
            "lp_total_value": total,
            "active": active
        }


def get_lp_state(uid):
    """
    Get LP state for a specific user (thread-safe)
    """
    with _state_lock:
        return lp_state.get(uid, {
            "price": None,
            "lp_usdt": None,
            "lp_wmatic": None,
            "lp_total_value": None,
            "active": False
        }).copy()
