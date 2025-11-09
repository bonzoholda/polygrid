import subprocess, signal, sqlite3, time, logging, os, sys

# Allow imports from root project folder
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

DB_FILE = os.path.join(os.path.dirname(__file__), "bots.db")
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

processes = {}  # user_id -> subprocess.Popen


# ------------------------------
# Database utilities
# ------------------------------
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS bots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            address TEXT,
            private_key TEXT,
            is_active INTEGER DEFAULT 0,
            log_file TEXT
        )
    """)
    conn.commit()
    conn.close()


def get_users():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    users = conn.execute("SELECT * FROM bots").fetchall()
    conn.close()
    return [dict(u) for u in users]


def get_user(uid):
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    user = conn.execute("SELECT * FROM bots WHERE id=?", (uid,)).fetchone()
    conn.close()
    return dict(user) if user else None


def add_user(name, address, private_key):
    conn = sqlite3.connect(DB_FILE)
    conn.execute("INSERT INTO bots (name, address, private_key) VALUES (?, ?, ?)", (name, address, private_key))
    conn.commit()
    conn.close()


def set_active(uid, active):
    conn = sqlite3.connect(DB_FILE)
    conn.execute("UPDATE bots SET is_active=? WHERE id=?", (1 if active else 0, uid))
    conn.commit()
    conn.close()


# ------------------------------
# Bot process management
# ------------------------------
def start_bot(uid):
    """Start the trading bot subprocess for the given user."""
    user = get_user(uid)
    if not user or processes.get(uid):
        return False

    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, f"user_{uid}.log")

    # Path to main bot script in the project root
    project_root = os.path.dirname(os.path.dirname(__file__))
    bot_path = os.path.join(project_root, "main.py")

    if not os.path.exists(bot_path):
        logging.error(f"❌ Bot file not found at {bot_path}")
        return False

    # Environment variables for bot process
    env = os.environ.copy()
    env["OWNER_ADDR"] = user["address"]
    env["PRIVATE_KEY"] = user["private_key"]

    with open(log_file, "a") as f:
        f.write(f"\n=== Starting bot for {user['name']} ===\n")

    # Start bot process in root folder
    proc = subprocess.Popen(
        ["python", bot_path],
        cwd=project_root,
        env=env,
        stdout=open(log_file, "a"),
        stderr=subprocess.STDOUT,
    )

    processes[uid] = proc
    set_active(uid, True)
    logging.info(f"🚀 Started bot {uid} ({user['name']}) [PID {proc.pid}]")
    return True


def stop_bot(uid):
    """Stop the running bot subprocess for the given user."""
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
    """Restart bots that were active before shutdown."""
    users = get_users()
    for u in users:
        if u["is_active"]:
            logging.info(f"🔁 Resuming bot {u['id']} ({u['name']})")
            start_bot(u["id"])


def tail_log(uid, n=50):
    """Return the last N lines of a user's bot log."""
    path = os.path.join(LOG_DIR, f"user_{uid}.log")
    if not os.path.exists(path):
        return ["(no log yet)"]
    with open(path, "r") as f:
        lines = f.readlines()
    return lines[-n:]
