import os, sys, threading, logging
from fastapi import FastAPI, Request, Form, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from manager import start_bot, stop_bot, auto_resume, tail_log
from db import init_db, add_user, get_users, get_user, set_active, decrypt_key
from cryptography.fernet import Fernet

# Allow imports from root project folder if needed
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=os.environ.get("SESSION_KEY", "change_this_secret"))
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize DB on startup
init_db()
threading.Thread(target=auto_resume, daemon=True).start()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# -----------------------------
# Helpers
# -----------------------------
def get_current_user(request: Request):
    uid = request.session.get("user_id")
    if uid:
        user = get_user(uid)
        if user:
            return user
    return None


def require_login(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    return user


# -----------------------------
# Routes
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=303)
    # Only show current user's bot
    users = [user]
    return templates.TemplateResponse("index.html", {"request": request, "users": users})


# ----- Register -----
@app.get("/register", response_class=HTMLResponse)
def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})


@app.post("/register")
def register(
    request: Request,
    username: str = Form(...),
    name: str = Form(...),
    address: str = Form(...),
    private_key: str = Form(...)
):
    # Encrypt key with Fernet
    add_user(name=name, address=address, private_key=private_key)
    # Auto-login after registration
    user = get_users()[-1]  # last added user
    request.session["user_id"] = user["id"]
    return RedirectResponse("/", status_code=303)


# ----- Login -----
@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/login")
def login(request: Request, username: str = Form(...), private_key: str = Form(...)):
    # Find user by username (or bot name) and check private key
    users = get_users()
    for u in users:
        try:
            dec_key = decrypt_key(u["encrypted_key"])
        except Exception:
            continue
        if u["name"] == username and dec_key == private_key:
            request.session["user_id"] = u["id"]
            return RedirectResponse("/", status_code=303)
    return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid credentials"})


# ----- Logout -----
@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


# ----- Bot control -----
@app.get("/start/{uid}")
def start(uid: int, request: Request):
    user = get_current_user(request)
    if not user or user["id"] != uid:
        return PlainTextResponse("Unauthorized", status_code=403)
    start_bot(uid)
    return RedirectResponse("/", status_code=303)


@app.get("/stop/{uid}")
def stop(uid: int, request: Request):
    user = get_current_user(request)
    if not user or user["id"] != uid:
        return PlainTextResponse("Unauthorized", status_code=403)
    stop_bot(uid)
    return RedirectResponse("/", status_code=303)


@app.get("/logs/{uid}")
def logs(uid: int, request: Request):
    user = get_current_user(request)
    if not user or user["id"] != uid:
        return PlainTextResponse("Unauthorized", status_code=403)
    logs = tail_log(uid)
    return PlainTextResponse("".join(logs))


# ----- Status API -----
@app.get("/status")
def status(request: Request):
    user = get_current_user(request)
    if not user:
        return PlainTextResponse("Unauthorized", status_code=403)
    return {"user": user}
