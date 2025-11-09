import sys, os, threading, logging
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Import bot manager utilities
from manager import init_db, add_user, get_users, start_bot, stop_bot, auto_resume, tail_log

# ------------------------------
# Paths & Initialization
# ------------------------------
BASE_DIR = os.path.dirname(__file__)
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = FastAPI(title="DEX Bot Dashboard")
templates = Jinja2Templates(directory=TEMPLATES_DIR)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Initialize database on startup
init_db()

# ------------------------------
# Startup hook: auto-resume bots
# ------------------------------
@app.on_event("startup")
def startup_event():
    logging.info("🚀 FastAPI app started. Resuming active bots...")
    threading.Thread(target=auto_resume, daemon=True).start()

# ------------------------------
# Routes
# ------------------------------
@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    users = get_users()
    return templates.TemplateResponse("index.html", {"request": request, "users": users})


@app.post("/register")
def register(name: str = Form(...), address: str = Form(...), private_key: str = Form(...)):
    add_user(name, address, private_key)
    return RedirectResponse("/", status_code=303)


@app.get("/start/{uid}")
def start(uid: int):
    start_bot(uid)
    return RedirectResponse("/", status_code=303)


@app.get("/stop/{uid}")
def stop(uid: int):
    stop_bot(uid)
    return RedirectResponse("/", status_code=303)


@app.get("/logs/{uid}")
def get_logs(uid: int):
    logs = tail_log(uid)
    return PlainTextResponse("".join(logs))


@app.get("/status")
def get_status():
    users = get_users()
    return {"users": users}
