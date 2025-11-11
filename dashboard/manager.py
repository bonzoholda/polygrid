import subprocess, signal, sqlite3, time, logging, os, sys
from cryptography.fernet import Fernet
import bcrypt


# Allow imports from root project folder
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.portfolio import fetch_portfolio

DB_FILE = os.path.join(os.path.dirname(__file__), "data", "bots.db")
LOG_DIR = os.path.join(os.path.dirname(__file__), "data", "logs")
KEY_FILE = os.path.join(os.path.dirname(__file__), "data", "fernet.key")

# Ensure directories exist
os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Generate encryption key if not exists
if not os.path.exists(KEY_FILE):
    with open(KEY_FILE, "wb") as f:
        f.write(Fernet.generate_key())
fernet = Fernet(open(KEY_FILE, "rb").read())

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
processes = {}  # user_id -> subprocess.Popen
bot_state = {}


# ------------------------------
# Database utilities
# ------------------------------
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS bots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password_hash TEXT,
            name TEXT,
            address TEXT,
            encrypted_key TEXT,
            is_active INTEGER DEFAULT 0,
            log_file TEXT,
            strategy TEXT DEFAULT 'grid_dca',
            init_portfolio_value REAL,
            start_timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()


def add_user(username, password, name, address, private_key):
    password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    encrypted_key = fernet.encrypt(private_key.encode()).decode()
    conn = sqlite3.connect(DB_FILE)
    conn.execute(
        "INSERT INTO bots (username, password_hash, name, address, encrypted_key) VALUES (?, ?, ?, ?, ?)",
        (username, password_hash, name, address, encrypted_key)
    )
    conn.commit()
    conn.close()


def get_user_by_username(username):
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM bots WHERE username=?", (username,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_user(uid):
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM bots WHERE id=?", (uid,)).fetchone()
    conn.close()
    return dict(row) if row else None


def verify_user(username, password):
    user = get_user_by_username(username)
    if not user:
        return False
    return bcrypt.checkpw(password.encode(), user["password_hash"].encode())


def decrypt_key(enc_key):
    return fernet.decrypt(enc_key.encode()).decode()


def set_active(uid, active):
    conn = sqlite3.connect(DB_FILE)
    conn.execute("UPDATE bots SET is_active=? WHERE id=?", (1 if active else 0, uid))
    conn.commit()
    conn.close()


def update_strategy(uid, strategy):
    conn = sqlite3.connect(DB_FILE)
    conn.execute("UPDATE bots SET strategy=? WHERE id=?", (strategy, uid))
    conn.commit()
    conn.close()


def tail_log(uid, n=50):
    path = os.path.join(LOG_DIR, f"user_{uid}.log")
    if not os.path.exists(path):
        return ["(no log yet)"]
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    return lines[-n:]


def get_users():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM bots").fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ------------------------------
# Bot process management
# ------------------------------
def start_bot(uid, strategy="grid_dca"):
    """
    Start a bot for a user with the chosen strategy.
    """
    user = get_user(uid)
    if not user or processes.get(uid):
        return False

    log_file = os.path.join(LOG_DIR, f"user_{uid}.log")
    os.makedirs(LOG_DIR, exist_ok=True)

    # Save strategy choice to DB
    update_strategy(uid, strategy)

    project_root = os.path.dirname(os.path.dirname(__file__))
    if strategy == "asset_balancer":
        bot_path = os.path.join(project_root, "asset_balancer.py")
    else:
        bot_path = os.path.join(project_root, "main.py")

    if not os.path.exists(bot_path):
        logging.error(f"❌ Bot file not found at {bot_path}")
        return False

    env = os.environ.copy()
    env["OWNER_ADDR"] = user["address"]
    env["PRIVATE_KEY"] = decrypt_key(user["encrypted_key"])
    env["BOT_STRATEGY"] = strategy

    with open(log_file, "a", buffering=1, encoding="utf-8") as log_handle:
        log_handle.write(f"\n=== Starting bot for {user['name']} (strategy: {strategy}) ===\n")
        log_handle.flush()

    # --- init stat recording
    portfolio = fetch_portfolio(uid)
    init_value = portfolio.get("total_value_usdt", 0)
    now = datetime.utcnow().isoformat()

    with sqlite3.connect(DB_FILE) as conn:
        c = conn.cursor()
        c.execute(
            "UPDATE bots SET init_portfolio_value=?, start_timestamp=? WHERE id=?",
            (init_value, now, uid)
        )
        conn.commit()
    # ------ 
    
    # ⚡ Launch Python in unbuffered mode (-u)
    proc = subprocess.Popen(
        [sys.executable, "-u", bot_path],
        cwd=project_root,
        env=env,
        stdout=open(log_file, "a", buffering=1, encoding="utf-8"),
        stderr=subprocess.STDOUT,
    )

    processes[uid] = proc
    set_active(uid, True)
    logging.info(f"🚀 Started {strategy} bot {uid} ({user['name']}) [PID {proc.pid}]")
    return True


def stop_bot(uid):
    proc = processes.get(uid)
    if proc:
        try:
            os.kill(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        processes.pop(uid, None)
        set_active(uid, False)
        logging.info(f"🛑 Stopped bot {uid}")
        return True
    set_active(uid, False)
    return False


def auto_resume():
    users = get_users()
    for u in users:
        if u["is_active"]:
            strategy = u.get("strategy", "grid_dca")
            logging.info(f"🔁 Resuming bot {u['id']} ({u['name']}) with strategy {strategy}")
            start_bot(u["id"], strategy)


def record_bot_stat(uid):
    """Record initial portfolio value and start time."""
    try:
        data = fetch_portfolio(uid)
        if "total_value_usdt" not in data:
            return {"error": "Portfolio fetch failed"}
        bot_state[uid] = {
            "start_time": time.time(),
            "initial_value": data["total_value_usdt"],
        }
        return {"message": "Bot stat recorded", **bot_state[uid]}
    except Exception as e:
        return {"error": str(e)}

def get_bot_stat(uid):
    """Return current runtime duration and growth stats."""
    if uid not in bot_state:
        return {"error": "Bot not started or no stats recorded yet"}

    try:
        current = fetch_portfolio(uid)
        start_time = bot_state[uid]["start_time"]
        initial_value = bot_state[uid]["initial_value"]
        duration_sec = time.time() - start_time

        # Format duration as dd:hh:mm
        days = int(duration_sec // 86400)
        hours = int((duration_sec % 86400) // 3600)
        minutes = int((duration_sec % 3600) // 60)
        duration_str = f"{days:02d}:{hours:02d}:{minutes:02d}"

        current_value = current["total_value_usdt"]
        growth_pct = ((current_value - initial_value) / initial_value) * 100

        return {
            "uid": uid,
            "initial_value": initial_value,
            "current_value": current_value,
            "growth_pct": round(growth_pct, 3),
            "duration": duration_str,
        }
    except Exception as e:
        return {"error": str(e)}
