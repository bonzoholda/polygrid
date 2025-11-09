import os, sys, threading, time, json
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from main import bot_state
from manager import init_db, get_user_by_username, verify_user, add_user, start_bot, stop_bot, auto_resume

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=os.environ.get("SESSION_SECRET","supersecret"))
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize DB and resume bots
init_db()
@app.on_event("startup")
def startup_event():
    threading.Thread(target=auto_resume, daemon=True).start()

# Auth utils
def get_current_user(request: Request):
    username = request.session.get("username")
    if not username:
        return None
    return get_user_by_username(username)

# Routes
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
        return templates.TemplateResponse("login.html", {"request": request, "error":"Invalid credentials"})
    request.session["username"] = username
    return RedirectResponse("/", status_code=303)

@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=303)

# Bot actions
@app.api_route("/start/{uid}", methods=["GET","POST"])
def start(uid: int, request: Request):
    user = get_current_user(request)
    if not user or user["id"] != uid:
        return RedirectResponse("/login", status_code=303)
    start_bot(uid)
    return RedirectResponse("/", status_code=303)

@app.api_route("/stop/{uid}", methods=["GET","POST"])
def stop(uid: int, request: Request):
    user = get_current_user(request)
    if not user or user["id"] != uid:
        return RedirectResponse("/login", status_code=303)
    stop_bot(uid)
    return RedirectResponse("/", status_code=303)

# SSE endpoint: live bot_state
@app.get("/stream_logs/{uid}")
def stream_logs(uid: int):
    def event_generator():
        while True:
            yield f"data: {json.dumps({ \
                'usdt_balance': bot_state['usdt_balance'], \
                'ai_signal': bot_state['ai_signal'], \
                'confidence': bot_state['confidence'], \
                'rsi': bot_state['rsi'], \
                'momentum': bot_state['momentum'], \
                'log': '\n'.join(bot_state['logs']) \
            })}\n\n"
            time.sleep(1)
    return StreamingResponse(event_generator(), media_type="text/event-stream")
