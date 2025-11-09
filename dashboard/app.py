import os, sys, hashlib, secrets, threading
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

# Allow imports from root
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from manager import init_db, add_user, get_users, get_user, decrypt_key
from manager import start_bot, stop_bot, auto_resume, tail_log

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=os.environ.get("SESSION_SECRET", "supersecret"))

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# initialize database
init_db()

# resume active bots on startup
@app.on_event("startup")
def startup_event():
    threading.Thread(target=auto_resume, daemon=True).start()


# -----------------
# Auth utilities
# -----------------
def hash_password(password: str):
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(password: str, hashed: str):
    return hash_password(password) == hashed


def get_current_user(request: Request):
    username = request.session.get("username")
    if not username:
        return None
    for u in get_users():
        if u.get("username") == username:
            return u
    return None


# -----------------
# Routes
# -----------------
@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    return templates.TemplateResponse("index.html", {"request": request, "user": user})


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/login")
def login(request: Request, username: str = Form(...), password: str = Form(...)):
    user = None
    for u in get_users():
        if u.get("username") == username:
            user = u
            break
    if not user or not verify_password(password, user.get("password_hash", "")):
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid credentials"})
    request.session["username"] = username
    return RedirectResponse("/", status_code=303)


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


@app.get("/register", response_class=HTMLResponse)
def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})


@app.post("/register")
def register(
    request: Request,
    username: str = Form(...),
    name: str = Form(...),
    address: str = Form(...),
    private_key: str = Form(...),
    password: str = Form(...),
):
    # store user with hashed password & encrypted key
    password_hash = hash_password(password)
    add_user(name=name, address=address, private_key=private_key)  # encrypted in db.py
    # manually add username & password_hash in db (for auth)
    conn = sqlite3.connect(os.path.join(os.path.dirname(__file__), "data", "bots.db"))
    conn.execute("ALTER TABLE IF NOT EXISTS bots ADD COLUMN username TEXT")
    conn.execute("ALTER TABLE IF NOT EXISTS bots ADD COLUMN password_hash TEXT")
    conn.execute(
        "UPDATE bots SET username=?, password_hash=? WHERE name=? AND address=?",
        (username, password_hash, name, address),
    )
    conn.commit()
    conn.close()
    request.session["username"] = username
    return RedirectResponse("/", status_code=303)


# -----------------
# Bot actions
# -----------------
@app.get("/start")
def start(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    start_bot(user["id"])
    return RedirectResponse("/", status_code=303)


@app.get("/stop")
def stop(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    stop_bot(user["id"])
    return RedirectResponse("/", status_code=303)


@app.get("/logs", response_class=HTMLResponse)
def logs(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    log_lines = tail_log(user["id"])
    return HTMLResponse("<pre style='color:#0f0; background:#1e1e1e; padding:1rem; border-radius:8px;'>"
                        + "".join(log_lines) + "</pre>")
