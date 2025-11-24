# core/state.py

from threading import Lock

_state_lock = Lock()

# per-user LP states stored as dict of dicts
_user_lp_states = {}     # { uid: { price, lp_usdt, lp_wmatic, lp_total_value, active } }


def update_lp_state(uid: int, price, usdt, wmatic, total, active=True):
    """
    Store LP state for a specific UID.
    This is called from uniswap_v3_manager.py (inside the bot process).
    """
    with _state_lock:
        _user_lp_states[uid] = {
            "price": float(price) if price is not None else None,
            "lp_usdt": float(usdt) if usdt is not None else None,
            "lp_wmatic": float(wmatic) if wmatic is not None else None,
            "lp_total_value": float(total) if total is not None else None,
            "active": bool(active)
        }


def get_lp_state(uid: int):
    """
    Retrieve LP state for a specific user.
    """
    with _state_lock:
        return _user_lp_states.get(uid)
