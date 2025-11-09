import os, sys, threading, time
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

# Allow imports from project root
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from manager import (
    init_db, add_user, get_users, get_user, get_user_by_username,
    verify_user, decrypt_key, start_bot, stop_bot, auto_resume, tail_log, LOG_DIR
)

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=os.environ.get("SESSION_SECRET", "supersecret"))

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize database
init_db()

# Resume active bots on startup
@app.on_event("startup")
def startup_event():
    threading.Thread(target=auto_resume, daemon=True).start()


# -----------------
# Auth utilities
# -----------------
def get_current_user(request: Request):
    username = request.session.get("username")
    if not username:
        return None
    return get_user_by_username(username)


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
    if not verify_user(username, password):
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
    if get_user_by_username(username):
        return templates.TemplateResponse("register.html", {"request": request, "error": "Username already exists"})

    # Add user properly according to manager.py
    add_user(username=username, password=password, name=name, address=address, private_key=private_key)
    request.session["username"] = username
    return RedirectResponse("/", status_code=303)


# -----------------
# Bot actions
# -----------------
@app.api_route("/start/{uid}", methods=["GET", "POST"])
def start(uid: int, request: Request):
    user = get_current_user(request)
    if not user or user["id"] != uid:
        return RedirectResponse("/login", status_code=303)
    start_bot(uid)
    return RedirectResponse("/", status_code=303)


@app.api_route("/stop/{uid}", methods=["GET", "POST"])
def stop(uid: int, request: Request):
    user = get_current_user(request)
    if not user or user["id"] != uid:
        return RedirectResponse("/login", status_code=303)
    stop_bot(uid)
    return RedirectResponse("/", status_code=303)


@app.get("/logs/{uid}", response_class=HTMLResponse)
def logs(uid: int, request: Request):
    user = get_current_user(request)
    if not user or user["id"] != uid:
        return RedirectResponse("/login", status_code=303)
    log_lines = tail_log(uid)
    return HTMLResponse("<pre style='color:#0f0; background:#1e1e1e; padding:1rem; border-radius:8px;'>"
                        + "".join(log_lines) + "</pre>")


@app.get("/stream_logs/{uid}")
def stream_logs(uid: int):
    def log_generator():
        path = os.path.join(LOG_DIR, f"user_{uid}.log")
        if not os.path.exists(path):
            open(path, "w").close()  # create empty log
        with open(path, "r") as f:
            f.seek(0, os.SEEK_END)  # go to end of file
            while True:
                line = f.readline()
                if line:
                    yield f"data: {line.rstrip()}\n\n"
                else:
                    time.sleep(0.5)

    return StreamingResponse(log_generator(), media_type="text/event-stream")
