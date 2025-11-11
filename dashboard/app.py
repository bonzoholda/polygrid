import os, sys, threading, time, json
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from manager import (
    init_db,
    get_user_by_username,
    verify_user,
    add_user,
    start_bot,
    stop_bot,
    auto_resume,
    LOG_DIR,
    record_bot_stat,
    get_bot_stat,
)
from core.portfolio import fetch_portfolio

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=os.environ.get("SESSION_SECRET", "supersecret"))
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------- Initialize DB and resume bots ----------
init_db()

@app.on_event("startup")
def startup_event():
    threading.Thread(target=auto_resume, daemon=True).start()

# ---------- Auth utilities ----------
def get_current_user(request: Request):
    username = request.session.get("username")
    if not username:
        return None
    return get_user_by_username(username)

# ---------- Routes ----------
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
    add_user(username=username, password=password, name=name, address=address, private_key=private_key)
    request.session["username"] = username
    return RedirectResponse("/", status_code=303)

# ---------- Bot control ----------
@app.get("/start/{uid}", response_class=HTMLResponse)
def choose_strategy_page(uid: int, request: Request):
    """Show a simple page to choose between Grid DCA and Asset Balancer."""
    user = get_current_user(request)
    if not user or user["id"] != uid:
        return RedirectResponse("/login", status_code=303)
    return templates.TemplateResponse("choose_strategy.html", {"request": request, "user": user})

@app.post("/start/{uid}")
def start(uid: int, request: Request, strategy: str = Form("grid_dca")):
    """Start the bot with a selected strategy."""
    user = get_current_user(request)
    if not user or user["id"] != uid:
        return RedirectResponse("/login", status_code=303)
    start_bot(uid, strategy)
    return RedirectResponse("/", status_code=303)

@app.api_route("/stop/{uid}", methods=["GET", "POST"])
def stop(uid: int, request: Request):
    user = get_current_user(request)
    if not user or user["id"] != uid:
        return RedirectResponse("/login", status_code=303)
    stop_bot(uid)
    return RedirectResponse("/", status_code=303)

# ---------- Stream logs ----------
@app.get("/stream_logs/{uid}")
def stream_logs(uid: int):
    def log_generator():
        path = os.path.join(LOG_DIR, f"user_{uid}.log")
        if not os.path.exists(path):
            open(path, "w").close()
        with open(path, "r") as f:
            f.seek(0, os.SEEK_END)
            while True:
                line = f.readline()
                if line:
                    yield f"data: {line.rstrip()}\n\n"
                else:
                    time.sleep(0.5)
    return StreamingResponse(log_generator(), media_type="text/event-stream")

# ---------- Portfolio tracking ----------
@app.get("/api/portfolio/{uid}")
def get_portfolio(request: Request):
    user = get_current_user(request)
    if not user:
        return {"error": "Unauthorized"}
    uid = user["id"]
    try:
        data = fetch_portfolio(uid)
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute("SELECT init_portfolio_value, start_timestamp FROM bots WHERE id=?", (uid,))
        row = c.fetchone()
        conn.close()
        if row:
            data["init_portfolio_value"] = row[0]
            data["start_timestamp"] = row[1]        
        return data
    except Exception as e:
        return {"error": str(e)}


# ---------- Bot statistics (initial value + runtime + growth) ----------
@app.post("/api/botstat/{uid}")
def start_bot_tracking(uid: int, request: Request):
    """Record the initial portfolio and timestamp when bot starts."""
    user = get_current_user(request)
    if not user or user["id"] != uid:
        return {"error": "Unauthorized"}
    return record_bot_stat(uid)

@app.get("/api/botstat/{uid}")
def get_bot_tracking(uid: int, request: Request):
    """Return live bot runtime and portfolio growth."""
    user = get_current_user(request)
    if not user or user["id"] != uid:
        return {"error": "Unauthorized"}
    return get_bot_stat(uid)

