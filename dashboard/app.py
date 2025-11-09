from fastapi import FastAPI, Request, Form, Depends, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from manager import init_db, add_user, get_user_by_username, get_user, start_bot, stop_bot, tail_log, verify_user
import threading, uuid

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
init_db()

sessions = {}  # session_id -> username

# ---------- Helpers ----------
def get_current_user(request: Request):
    session_id = request.cookies.get("session_id")
    if session_id and session_id in sessions:
        return get_user_by_username(sessions[session_id])
    return None

# ---------- Login ----------
@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
def login(request: Request, response: Response, username: str = Form(...), password: str = Form(...)):
    if verify_user(username, password):
        session_id = str(uuid.uuid4())
        sessions[session_id] = username
        response = RedirectResponse("/", status_code=303)
        response.set_cookie(key="session_id", value=session_id)
        return response
    return RedirectResponse("/login", status_code=303)

@app.get("/logout")
def logout(request: Request, response: Response):
    session_id = request.cookies.get("session_id")
    if session_id in sessions:
        sessions.pop(session_id)
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie("session_id")
    return response

# ---------- Dashboard ----------
@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request, user=Depends(get_current_user)):
    if not user:
        return RedirectResponse("/login")
    return templates.TemplateResponse("index.html", {"request": request, "user": user})

@app.get("/start")
def start(user=Depends(get_current_user)):
    if not user:
        return RedirectResponse("/login")
    start_bot(user["id"])
    return RedirectResponse("/", status_code=303)

@app.get("/stop")
def stop(user=Depends(get_current_user)):
    if not user:
        return RedirectResponse("/login")
    stop_bot(user["id"])
    return RedirectResponse("/", status_code=303)

@app.get("/logs")
def logs(user=Depends(get_current_user)):
    if not user:
        return RedirectResponse("/login")
    logs = tail_log(user["id"])
    return HTMLResponse("<pre>" + "".join(logs) + "</pre>")
