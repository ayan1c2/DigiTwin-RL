# uvicorn app.main:app --reload --port 5000
# app/main.py
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from pathlib import Path
import pickle
import numpy as np
import uvicorn

from .core import ACTIONS as CORE_ACTIONS, User, clip_user_state, who_compliance, encode_state

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
#MODEL_PATH = ROOT_DIR / "models" / "best_q_agent.pkl" 
MODEL_PATH = ROOT_DIR / "models" / "best_greedy_policy.pkl"

app = FastAPI(title="Digital Twin Q-Learning Recommender")

# Templates
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

# CORS (local dev; tighten for prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ------------- Model Loading -------------
class QAgentWrapper:
    def __init__(self):
        self.Q = None
        self.ACTIONS = CORE_ACTIONS
        self.n_states = 16
        self.n_actions = len(self.ACTIONS)

    def load(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(f"Model pickle not found at {path}. Train & save the model first.")
        with open(path, "rb") as f:
            payload = pickle.load(f)
        # Preferred: use saved ACTIONS from training to ensure encoding order matches
        self.ACTIONS = payload.get("ACTIONS", CORE_ACTIONS)
        self.Q = payload["Q"]
        self.n_states = payload.get("n_states", 16)
        self.n_actions = payload.get("n_actions", len(self.ACTIONS))
        if self.Q.shape != (self.n_states, self.n_actions):
            raise ValueError("Loaded Q table shape does not match n_states/n_actions.")
        print(f"[Model] Loaded Q-table with shape {self.Q.shape}. Actions: {self.ACTIONS}")

    def recommend_topk(self, state: int, k: int = 3):
        if self.Q is None:
            # Fallback: random if model not loaded (should not happen)
            idxs = np.random.choice(self.n_actions, size=min(k, self.n_actions), replace=False)
            return [int(i) for i in idxs]
        order = np.argsort(-self.Q[state])
        return [int(i) for i in order[:k]]

agent = QAgentWrapper()
agent.load(MODEL_PATH)

# ------------- Schemas -------------
class UserInput(BaseModel):
    sleepHours: float
    activityMinutes: int
    dietFv: int
    waterGlasses: int
    habit: int   # 1 = abstain; 0 = used tobacco/alcohol

# ------------- Routes -------------

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    # Serve the Bootstrap page (uses Fetch to call /recommend)
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/recommend")
def recommend(input_data: UserInput):
    # Build user snapshot from form
    u = User(
        user_id=0, age=30, sex="M", bmi=25.0, work_schedule="9-5",
        smoker=(input_data.habit == 0), baseline_activity=20,
        stress_base=5.0, adherence=0.8, dropout_chance_weekly=0.0, realign_chance_weekly=0.0,
        sleep_hours=input_data.sleepHours,
        activity_min=input_data.activityMinutes,
        diet_fv_g=input_data.dietFv,
        abstain=input_data.habit,
        stress=5.0,
        water_glasses=input_data.waterGlasses
    )
    clip_user_state(u)
    flags = who_compliance(u)
    state = encode_state(flags)

    topk_idx = agent.recommend_topk(state, k=3)
    # IMPORTANT: map indices to the agent's ACTIONS (from the pickle)
    actions_labels = [agent.ACTIONS[i] for i in topk_idx]

    return JSONResponse({
        "state": state,
        "compliance": flags,
        "recommendations": actions_labels,
        "topk_indices": topk_idx
    })

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=5000, reload=True)
