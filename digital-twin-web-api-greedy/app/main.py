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

# Prefer the new greedy filename; fall back to the earlier one if present
MODEL_CANDIDATES = [
    #ROOT_DIR / "models" / "greedy_policy.pkl",
    ROOT_DIR / "models" / "best_greedy_policy.pkl",
]

app = FastAPI(title="Digital Twin Greedy Recommender")

# Templates
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

# CORS (local dev; tighten for prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ------------- Greedy Policy Loader -------------
class GreedyPolicyWrapper:
    """
    Loads a greedy-policy PKL produced by the simulation.
    Expects payload fields:
      - ACTIONS (list[str])
      - n_states, n_actions
      - diagnostics.state_action_reward_avg : (n_states x n_actions) array
    Recommends actions by sorting the avg immediate r_eff for the current state.
    """
    def __init__(self):
        self.avg = None              # (n_states, n_actions) float
        self.ACTIONS = CORE_ACTIONS  # will be overwritten by PKL if present
        self.n_states = 16
        self.n_actions = len(self.ACTIONS)
        self.loaded_path = None

    def load(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(f"Model pickle not found at {path}. Run the greedy trainer to create it.")
        with open(path, "rb") as f:
            payload = pickle.load(f)

        self.ACTIONS = payload.get("ACTIONS", CORE_ACTIONS)
        self.n_states = payload.get("n_states", 16)
        self.n_actions = payload.get("n_actions", len(self.ACTIONS))

        diag = payload.get("diagnostics", {})
        avg = diag.get("state_action_reward_avg", None)
        if avg is None:
            raise ValueError("Greedy PKL missing diagnostics.state_action_reward_avg.")
        self.avg = np.array(avg, dtype=float)

        if self.avg.shape != (self.n_states, self.n_actions):
            raise ValueError(f"Avg table shape {self.avg.shape} does not match "
                             f"(n_states={self.n_states}, n_actions={self.n_actions}).")
        self.loaded_path = str(path)
        print(f"[Model] Loaded greedy policy from {path}. Avg table shape: {self.avg.shape}. Actions: {self.ACTIONS}")

    def load_first_existing(self, candidates):
        last_err = None
        for p in candidates:
            try:
                self.load(p)
                return
            except Exception as e:
                last_err = e
        # If none load, surface last error
        raise last_err if last_err else FileNotFoundError("No model candidates found.")

    def recommend_topk(self, state: int, k: int = 3):
        if self.avg is None:
            # Fallback: random if model not loaded
            idxs = np.random.choice(self.n_actions, size=min(k, self.n_actions), replace=False)
            return [int(i) for i in idxs]
        row = self.avg[state]
        # If the row is all equal (e.g., zeros), fall back to a stable order by index
        if np.allclose(row, row[0]):
            order = np.arange(self.n_actions)
        else:
            order = np.argsort(-row)
        return [int(i) for i in order[:k]]


agent = GreedyPolicyWrapper()
agent.load_first_existing(MODEL_CANDIDATES)

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
    actions_labels = [agent.ACTIONS[i] for i in topk_idx]

    return JSONResponse({
        "state": state,
        "compliance": flags,
        "recommendations": actions_labels,
        "topk_indices": topk_idx,
        "model_path": agent.loaded_path,
        "policy": "greedy_avg_immediate_reward"
    })

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=5000, reload=True)

