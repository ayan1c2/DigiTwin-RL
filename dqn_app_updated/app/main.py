# uvicorn app.main:app --reload --port 5000
# app/main.py
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from pathlib import Path
import pickle
import numpy as np
import uvicorn

from .core import (
    ACTIONS as CORE_ACTIONS,
    User,
    clip_user_state,
    who_compliance,
    encode_state_discrete,
    encode_state_rich,
)

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent

# DQN pickle (numpy weights) produced by training script
MODEL_PATH = ROOT_DIR / "models" / "best_dqn_agent.pkl"

app = FastAPI(title="Digital Twin DQN Recommender")

templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# -------------------- DQN Inference --------------------

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)

class DQNAgentWrapper:
    """
    Loads a lightweight DQN from a pickle and produces top-k recommendations.

    Expected pickle formats (any one of these is accepted):

    (A) {'ACTIONS': [...], 'state_dim': 26, 'layers': [{'W':..., 'b':...}, ...]}
        where the last layer outputs Q-values for |ACTIONS|.

    (B) {'ACTIONS': [...], 'state_dim': 26, 'weights': [(W1,b1), (W2,b2), ...]}

    (C) {'ACTIONS': [...], 'state_dim': 26, 'W1':..., 'b1':..., 'W2':..., 'b2':...}
    """
    def __init__(self):
        self.ACTIONS = CORE_ACTIONS
        self.state_dim = 26
        self.weights = []  # list of (W, b)

    def load(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(
                f"DQN model pickle not found at {path}. "
                "Train & save best_dqn_agent.pkl under /models."
            )
        with open(path, "rb") as f:
            payload = pickle.load(f)

        self.ACTIONS = payload.get("ACTIONS", CORE_ACTIONS)
        self.state_dim = int(payload.get("state_dim", 26))

        if "layers" in payload:
            self.weights = [(np.asarray(L["W"], dtype=np.float32), np.asarray(L["b"], dtype=np.float32))
                            for L in payload["layers"]]
        elif "weights" in payload:
            self.weights = [(np.asarray(W, dtype=np.float32), np.asarray(b, dtype=np.float32))
                            for (W, b) in payload["weights"]]
        elif all(k in payload for k in ["W1","b1","W2","b2"]):
            self.weights = [
                (np.asarray(payload["W1"], dtype=np.float32), np.asarray(payload["b1"], dtype=np.float32)),
                (np.asarray(payload["W2"], dtype=np.float32), np.asarray(payload["b2"], dtype=np.float32)),
            ]
        else:
            raise ValueError("Unrecognized DQN pickle format. See DQNAgentWrapper docstring.")

        # Basic shape checks
        if not self.weights:
            raise ValueError("No weights found in DQN pickle.")
        in_dim = self.weights[0][0].shape[0]
        out_dim = self.weights[-1][0].shape[1]
        if in_dim != self.state_dim:
            raise ValueError(f"Model expects state_dim={in_dim}, but config says {self.state_dim}.")
        if out_dim != len(self.ACTIONS):
            raise ValueError(f"Model outputs {out_dim} actions, but ACTIONS has {len(self.ACTIONS)}.")

        print(f"[Model] Loaded DQN with {len(self.weights)} layers. state_dim={self.state_dim}, actions={self.ACTIONS}")

    def q_values(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(1, -1)
        if x.shape[1] != self.state_dim:
            raise ValueError(f"Expected state dim {self.state_dim}, got {x.shape[1]}")
        h = x
        for li, (W, b) in enumerate(self.weights):
            h = h @ W + b
            if li < len(self.weights) - 1:
                h = _relu(h)
        return h.flatten()

    def recommend_topk(self, x: np.ndarray, k: int = 3):
        q = self.q_values(x)
        order = np.argsort(-q)
        idxs = [int(i) for i in order[: min(k, len(order))]]
        return idxs, [self.ACTIONS[i] for i in idxs], [float(q[i]) for i in idxs]

agent = DQNAgentWrapper()
agent.load(MODEL_PATH)

# -------------------- Schemas --------------------

class UserInput(BaseModel):
    # required fields (as before)
    sleepHours: float = Field(..., ge=0, le=24)
    activityMinutes: int = Field(..., ge=0, le=360)
    dietFv: int = Field(..., ge=0, le=2000)
    waterGlasses: int = Field(..., ge=0, le=30)
    habit: int = Field(..., ge=0, le=1)   # 1 = abstain; 0 = used tobacco/alcohol

    # optional enrichment (supports richer state)
    age: int | None = Field(None, ge=18, le=100)
    sex: str | None = None
    bmi: float | None = Field(None, ge=12, le=60)
    schedule: str | None = None
    adherence: float | None = Field(None, ge=0, le=1)
    stress: float | None = Field(None, ge=0, le=10)
    mindfulnessMinutes: int | None = Field(None, ge=0, le=240)
    screenHours: float | None = Field(None, ge=0, le=24)
    socialMinutes: int | None = Field(None, ge=0, le=600)

    # optional environment (for context-aware action effects)
    airAQI: float | None = Field(None, ge=0, le=500)
    noiseDb: float | None = Field(None, ge=0, le=120)
    lightLux: float | None = Field(None, ge=0, le=500)
    ecoAnxiety: float | None = Field(None, ge=0, le=10)
    heatwave: int | None = Field(None, ge=0, le=1)
    flood: int | None = Field(None, ge=0, le=1)

# -------------------- Routes --------------------

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/recommend")
def recommend(input_data: UserInput):
    u = User(
        user_id=0,
        age=int(input_data.age) if input_data.age is not None else 30,
        sex=(input_data.sex or "M"),
        bmi=float(input_data.bmi) if input_data.bmi is not None else 25.0,
        work_schedule=(input_data.schedule or "9-5"),
        smoker=(input_data.habit == 0),
        adherence=float(input_data.adherence) if input_data.adherence is not None else 0.8,

        sleep_hours=float(input_data.sleepHours),
        activity_min=int(input_data.activityMinutes),
        diet_fv_g=int(input_data.dietFv),
        abstain=int(input_data.habit),
        stress=float(input_data.stress) if input_data.stress is not None else 5.0,
        mindfulness_min=int(input_data.mindfulnessMinutes) if input_data.mindfulnessMinutes is not None else 10,
        water_glasses=int(input_data.waterGlasses),
        screen_hours=float(input_data.screenHours) if input_data.screenHours is not None else 2.0,
        social_min=int(input_data.socialMinutes) if input_data.socialMinutes is not None else 15,

        air_aqi=float(input_data.airAQI) if input_data.airAQI is not None else 30.0,
        noise_db=float(input_data.noiseDb) if input_data.noiseDb is not None else 50.0,
        light_lux=float(input_data.lightLux) if input_data.lightLux is not None else 3.0,
        eco_anxiety=float(input_data.ecoAnxiety) if input_data.ecoAnxiety is not None else 4.0,
        heatwave=int(input_data.heatwave) if input_data.heatwave is not None else 0,
        flood=int(input_data.flood) if input_data.flood is not None else 0,
    )

    clip_user_state(u)

    flags = who_compliance(u)
    state_discrete = encode_state_discrete(flags)
    x = encode_state_rich(u)

    idxs, labels, qvals = agent.recommend_topk(x, k=3)

    return JSONResponse({
        "state_discrete": int(state_discrete),
        "state_dim": int(x.shape[0]),
        "compliance": flags,
        "recommendations": labels,
        "topk_indices": idxs,
        "topk_qvalues": qvals
    })

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=5000, reload=True)
