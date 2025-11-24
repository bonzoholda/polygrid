# core/state.py

from threading import Lock

_state_lock = Lock()

lp_state = {
    "price": None,
    "lp_usdt": None,
    "lp_wmatic": None,
    "lp_total_value": None,
    "active": False
}

def update_lp_state(price, usdt, wmatic, total, active=True):
    with _state_lock:
        lp_state["price"] = price
        lp_state["lp_usdt"] = usdt
        lp_state["lp_wmatic"] = wmatic
        lp_state["lp_total_value"] = total
        lp_state["active"] = active

def get_lp_state():
    with _state_lock:
        return lp_state.copy()
