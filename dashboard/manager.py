import subprocess, signal, sqlite3, time, logging, os, sys
from cryptography.fernet import Fernet
import bcrypt

# Allow imports from root project folder
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

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
            log_file TEXT
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

def tail_log(uid, n=50):
    path = os.path.join(LOG_DIR, f"user_{uid}.log")
    if not os.path.exists(path):
        return ["(no log yet)"]
    with open(path, "r") as f:
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
def start_bot(uid):
    user = get_user(uid)
    if not user or processes.get(uid):
        return False

    log_file = os.path.join(LOG_DIR, f"user_{uid}.log")
    os.makedirs(LOG_DIR, exist_ok=True)

    project_root = os.path.dirname(os.path.dirname(__file__))
    bot_path = os.path.join(project_root, "main.py")
    if not os.path.exists(bot_path):
        logging.error(f"❌ Bot file not found at {bot_path}")
        return False

    env = os.environ.copy()
    env["OWNER_ADDR"] = user["address"]
    env["PRIVATE_KEY"] = decrypt_key(user["encrypted_key"])

    log_handle = open(log_file, "a")
    log_handle.write(f"\n=== Starting bot for {user['name']} ===\n")

    proc = subprocess.Popen(
        ["python", bot_path],
        cwd=project_root,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )

    processes[uid] = proc
    set_active(uid, True)
    logging.info(f"🚀 Started bot {uid} ({user['name']}) [PID {proc.pid}]")
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
            logging.info(f"🔁 Resuming bot {u['id']} ({u['name']})")
            start_bot(u["id"])
