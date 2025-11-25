from core.state import update_lp_state

def push_lp_stat(uid: int, stat: dict):
    """
    Update LP state in core/state.py
    Can be called from both runner and FastAPI route
    """
    update_lp_state(uid, stat)
    return stat
