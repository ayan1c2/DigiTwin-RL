"""
Q-Learning Only: Digital Twin for Behavioral Optimization (Enhanced)
--------------------------------------------------------------------
- Synthetic population (N users) with demographics
- 365-day simulation of daily actions & outcomes
- WHO-aligned constraints via reward/penalty + TSP-style shaping
- Q-learning agent
- Hyperparameter search over learning rate (alpha)
- Persist best model to pickle
- Persist full per-day dataset (CSV/optional Parquet) with proper ISO dates
- Customizable users/days via CLI args
- NEW: WHO-guided mental health + extra lifestyle parameters
    * Stress threshold (<= 5), daily mindfulness target (>= 10 min)
    * Screen time cap (<= 2 hours), social interaction minimum (>= 15 min)

Usage:
  python digital_twin_qlearning_q_only_extras.py --users 50 --days 180 --start-date 2024-01-01 --out-data data/sim_dataset.csv
"""

import argparse
import json
import math
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ---------------------------
# Reproducibility
# ---------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ---------------------------
# Problem Setup
# ---------------------------

ACTIONS = [
    "sleep_early",          # target 7-9h
    "walk_30",              # 30 min activity
    "eat_veg_400",          # 400g fruits+veg
    "no_tobacco_alcohol",   # abstain
    "meditate_10",          # stress reduction + mindfulness
    "drink_water_8",        # hydration
    "digital_detox_30",     # reduce screen time
    "social_15"             # increase social minutes
]

BEHAVIORS = ["Sleep", "Activity", "Diet", "Habit", "Stress", "Hydration", "Screen", "Social"]

WHO = {
    "sleep_min": 7.0,
    "sleep_max": 9.0,
    "activity_min": 30,       # minutes/day
    "diet_fv_min": 400,       # grams/day
    "habit_abstain": 1,       # 1=abstain, 0=not
    "water_glasses": 8,
    # NEW mental health & lifestyle
    "stress_max": 5.0,        # perceived stress should be moderate or lower
    "mindfulness_minutes": 10,# target mindfulness/relaxation minutes per day
    "screen_max_hours": 2.0,  # target daily screen time cap outside work
    "social_min_minutes": 15  # target minimum social interaction
}

# Penalties (lambda_j). Tuneable.
LAMBDA = {
    "sleep": 4.0,
    "activity": 3.5,
    "diet": 3.0,
    "habit": 5.0,
    "water": 1.5,
    "mental": 3.5,   # NEW
    "screen": 1.0,   # NEW
    "social": 1.0    # NEW
}

# Reward scale factors
REWARD_SCALE = {
    "sleep": 2.5,
    "activity": 2.0,
    "diet": 2.0,
    "habit": 3.0,
    "stress": 1.5,
    "hydration": 1.0,
    "mental": 2.0,   # NEW
    "screen": 0.6,   # NEW
    "social": 0.8    # NEW
}

# State = 4-bit vector of previous-day WHO compliance (Sleep, Activity, Diet, Habit)
# Keep state compact; new targets affect reward but not state size to avoid explosion.
def encode_state(compliance_flags: Dict[str, int]) -> int:
    bits = [
        compliance_flags.get("sleep", 0),
        compliance_flags.get("activity", 0),
        compliance_flags.get("diet", 0),
        compliance_flags.get("habit", 0),
    ]
    return bits[0] + (bits[1] << 1) + (bits[2] << 2) + (bits[3] << 3)

def decode_state(idx: int) -> Dict[str, int]:
    return {
        "sleep":   (idx & 1),
        "activity":(idx >> 1) & 1,
        "diet":    (idx >> 2) & 1,
        "habit":   (idx >> 3) & 1
    }

# ----------------------------------------------------
# Digital Twin User & Environment (stochastic dynamics)
# ----------------------------------------------------

@dataclass
class User:
    user_id: int
    age: int
    sex: str               # 'F'/'M'
    bmi: float
    work_schedule: str     # '9-5' or 'shift'
    smoker: bool
    baseline_activity: int # minutes/day typical
    stress_base: float     # 0-10 perceived stress baseline
    adherence: float       # 0-1 action adherence propensity
    dropout_chance_weekly: float  # chance to disengage in a week (0-1)
    realign_chance_weekly: float  # chance to re-engage (0-1)
    # Internal states evolve daily:
    sleep_hours: float = 6.5
    activity_min: int = 10
    diet_fv_g: int = 150
    abstain: int = 0
    stress: float = 5.0
    water_glasses: int = 4
    # NEW lifestyle states
    screen_time_hours: float = 3.5    # non-work leisure screen time
    social_minutes: int = 10          # daily minutes
    mindfulness_min: int = 0          # minutes meditated today
    dropped_out: bool = False

def generate_population(n_users=25) -> List[User]:
    users = []
    for uid in range(n_users):
        age = int(np.clip(np.random.normal(40, 12), 18, 75))
        sex = np.random.choice(['F','M'])
        bmi = float(np.clip(np.random.normal(26.0, 4.5), 18.5, 40.0))
        work_schedule = np.random.choice(['9-5','shift'], p=[0.7,0.3])
        smoker = bool(np.random.rand() < 0.25)
        baseline_activity = int(max(0, np.random.normal(20 if smoker else 30, 15)))
        stress_base = float(np.clip(np.random.normal(5.0, 2.0), 0, 10))
        adherence = float(np.clip(np.random.beta(2,2), 0.05, 0.95))
        dropout_chance_weekly = float(np.clip(np.random.beta(1.5, 12), 0.0, 0.25))
        realign_chance_weekly = float(np.clip(np.random.beta(1.5, 6), 0.05, 0.7))
        # NEW lifestyle baselines
        screen_time_hours = float(np.clip(np.random.normal(3.5, 1.0), 0.0, 8.0))
        social_minutes = int(np.clip(np.random.normal(12, 8), 0, 120))
        users.append(User(
            user_id=uid, age=age, sex=sex, bmi=bmi, work_schedule=work_schedule,
            smoker=smoker, baseline_activity=baseline_activity, stress_base=stress_base,
            adherence=adherence, dropout_chance_weekly=dropout_chance_weekly,
            realign_chance_weekly=realign_chance_weekly,
            screen_time_hours=screen_time_hours, social_minutes=social_minutes
        ))
    return users

def tsp_edge_weight(user: User, i: str, j: str) -> float:
    reward = 0.0
    penalty = 0.0
    if i == "Sleep" and j == "Activity":
        reward += 3.0
    if i == "Activity" and j == "Diet":
        reward += 2.0
    if j == "Habit" and user.smoker:
        penalty += 3.0
    if user.work_schedule == 'shift' and j == "Sleep":
        penalty += 2.0
    if j == "Hydration":
        reward += 1.0
    # NEW: small shaping—screen→sleep and social→stress improvements
    if i == "Screen" and j == "Sleep":
        reward += 1.0
    if i == "Social" and j == "Stress":
        reward += 1.0
    lam = 1.0
    return -reward + lam * penalty

def action_effects(user: User, action: str) -> Dict[str, float]:
    adhere = (np.random.rand() < user.adherence)
    noise = lambda s: np.random.normal(0, s)
    delta = dict(sleep=0.0, activity=0.0, diet=0.0, habit=0.0, stress=0.0, water=0.0,
                 screen=-0.0, social=0.0, mindful=0.0)
    if not adhere:
        # Non-adherence typically increases stress a bit
        delta["stress"] += noise(0.4)
        # and may increase screen time slightly and reduce social
        delta["screen"] += abs(noise(0.2))
        delta["social"] += -abs(noise(3.0))
        return delta

    if action == "sleep_early":
        delta["sleep"] += 1.5 + noise(0.5)
        delta["stress"] += -0.4 + noise(0.3)
        delta["screen"] += -0.3 + noise(0.2)
    elif action == "walk_30":
        delta["activity"] += 30 + int(np.random.normal(0, 8))
        delta["stress"] += -0.3 + noise(0.3)
        delta["sleep"] += 0.2 + noise(0.2)
        delta["social"] += 2 + int(np.random.normal(0, 2))
    elif action == "eat_veg_400":
        delta["diet"] += 300 + int(np.random.normal(0, 80))
        delta["stress"] += -0.1 + noise(0.2)
    elif action == "no_tobacco_alcohol":
        delta["habit"] += 1
        delta["stress"] += -0.2 + noise(0.3)
    elif action == "meditate_10":
        delta["stress"] += -0.8 + noise(0.3)
        delta["sleep"] += 0.2 + noise(0.2)
        delta["mindful"] += 10
    elif action == "drink_water_8":
        delta["water"] += 4 + int(np.random.normal(0, 2))
    elif action == "digital_detox_30":
        delta["screen"] += -0.5 + noise(0.2)  # ~30 min reduction
        delta["sleep"] += 0.1 + noise(0.2)
        delta["stress"] += -0.1 + noise(0.2)
    elif action == "social_15":
        delta["social"] += 15 + int(np.random.normal(0, 4))
        delta["stress"] += -0.2 + noise(0.2)
    return delta

def clip_user_state(u: User):
    u.sleep_hours = float(np.clip(u.sleep_hours, 3.0, 11.0))
    u.activity_min = int(np.clip(u.activity_min, 0, 180))
    u.diet_fv_g = int(np.clip(u.diet_fv_g, 0, 1200))
    u.abstain = int(np.clip(u.abstain, 0, 1))
    u.stress = float(np.clip(u.stress, 0.0, 10.0))
    u.water_glasses = int(np.clip(u.water_glasses, 0, 16))
    u.screen_time_hours = float(np.clip(u.screen_time_hours, 0.0, 12.0))
    u.social_minutes = int(np.clip(u.social_minutes, 0, 240))
    u.mindfulness_min = int(np.clip(u.mindfulness_min, 0, 120))

def apply_daily_drift(u: User):
    u.activity_min = int(max(0, u.activity_min - 10 + np.random.randint(-5, 6)))
    u.diet_fv_g = int(max(0, u.diet_fv_g - 80 + int(np.random.normal(0, 40))))
    u.sleep_hours += np.random.normal(-0.2, 0.4)
    u.abstain = 0 if (np.random.rand() < (0.15 if u.smoker else 0.05)) else u.abstain
    u.stress = float(np.clip(u.stress + np.random.normal(0.1, 0.5) + (0.3 if u.work_schedule=='shift' else 0.0), 0, 10))
    u.water_glasses = max(0, u.water_glasses - 2 + int(np.random.normal(0, 1)))
    # NEW drifts
    u.screen_time_hours = float(np.clip(u.screen_time_hours + np.random.normal(0.2, 0.4), 0.0, 12.0))
    u.social_minutes = int(np.clip(u.social_minutes + int(np.random.normal(-2, 6)), 0, 240))
    # Reset mindfulness each day (minutes practiced today)
    u.mindfulness_min = 0

def who_compliance(u: User) -> Dict[str, int]:
    return {
        "sleep": int(WHO["sleep_min"] <= u.sleep_hours <= WHO["sleep_max"]),
        "activity": int(u.activity_min >= WHO["activity_min"]),
        "diet": int(u.diet_fv_g >= WHO["diet_fv_min"]),
        "habit": int(u.abstain >= WHO["habit_abstain"]),
        "water": int(u.water_glasses >= WHO["water_glasses"]),
        # NEW flags
        "mental_health": int((u.stress <= WHO["stress_max"]) or (u.mindfulness_min >= WHO["mindfulness_minutes"])),
        "screen": int(u.screen_time_hours <= WHO["screen_max_hours"]),
        "social": int(u.social_minutes >= WHO["social_min_minutes"])
    }

def reward_and_penalty(u: User, action: str) -> Tuple[float, float, Dict[str,int]]:
    comp = who_compliance(u)
    reward = (
        REWARD_SCALE["sleep"] * comp["sleep"] +
        REWARD_SCALE["activity"] * comp["activity"] +
        REWARD_SCALE["diet"] * comp["diet"] +
        REWARD_SCALE["habit"] * comp["habit"] +
        REWARD_SCALE["stress"] * (1.0 - u.stress/10.0) +
        REWARD_SCALE["hydration"] * comp["water"] +
        REWARD_SCALE["mental"] * comp["mental_health"] +
        REWARD_SCALE["screen"] * comp["screen"] +
        REWARD_SCALE["social"] * comp["social"]
    )
    penalty = (
    LAMBDA["sleep"] * (1 - comp["sleep"])
    + LAMBDA["activity"] * (1 - comp["activity"])
    + LAMBDA["diet"] * (1 - comp["diet"])
    + LAMBDA["habit"] * (1 - comp["habit"])
    + LAMBDA["water"] * (1 - comp["water"])
    + LAMBDA["mental"] * (1 - comp["mental_health"])
    + LAMBDA["screen"] * (1 - comp["screen"])
    + LAMBDA["social"] * (1 - comp["social"])
    )
    deficits = {
        "Sleep": WHO["sleep_min"] - u.sleep_hours if u.sleep_hours < WHO["sleep_min"] else 0,
        "Activity": WHO["activity_min"] - u.activity_min if u.activity_min < WHO["activity_min"] else 0,
        "Diet": WHO["diet_fv_min"] - u.diet_fv_g if u.diet_fv_g < WHO["diet_fv_min"] else 0,
        "Habit": 1 - u.abstain,
        "Stress": u.stress/10.0,
        "Hydration": WHO["water_glasses"] - u.water_glasses if u.water_glasses < WHO["water_glasses"] else 0,
        "Screen": max(0.0, u.screen_time_hours - WHO["screen_max_hours"]),
        "Social": max(0, WHO["social_min_minutes"] - u.social_minutes)
    }
    dominant_from = max(deficits, key=deficits.get)
    action_map = {
        "sleep_early": "Sleep",
        "walk_30": "Activity",
        "eat_veg_400": "Diet",
        "no_tobacco_alcohol": "Habit",
        "meditate_10": "Stress",
        "drink_water_8": "Hydration",
        "digital_detox_30": "Screen",
        "social_15": "Social"
    }
    dominant_to = action_map[action]
    w_ij = tsp_edge_weight(u, dominant_from, dominant_to)
    edge_reward = max(-10.0, min(10.0, 10.0 - w_ij))
    shaped = reward + 0.25 * edge_reward
    return shaped, penalty, comp

def misreport(value: float, scale: float=0.1) -> float:
    return float(value + np.random.normal(0, scale * max(1.0, abs(value))))

# ---------------------------
# Q-Learning Agent
# ---------------------------

class QLearningAgent:
    def __init__(self, n_states=16, n_actions=None, alpha=0.2, gamma=0.95, epsilon=0.12):
        self.n_states = n_states
        self.n_actions = len(ACTIONS) if n_actions is None else n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((n_states, self.n_actions), dtype=float)

    def select_action(self, state: int) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def update(self, s, a, r_eff, s_next):
        td_target = r_eff + self.gamma * np.max(self.Q[s_next])
        self.Q[s, a] += self.alpha * (td_target - self.Q[s, a])

    def recommend_topk(self, state: int, k: int = 3) -> List[int]:
        order = np.argsort(-self.Q[state])
        return [int(i) for i in order[:k]]

# ---------------------------
# Simulation Orchestrator
# ---------------------------

@dataclass
class SimulationConfig:
    days: int = 365
    users: int = 25
    alpha: float = 0.25
    gamma: float = 0.97
    epsilon: float = 0.12

class DigitalTwinSim:
    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg
        self.population = generate_population(cfg.users)
        self.agent = QLearningAgent(alpha=cfg.alpha, gamma=cfg.gamma, epsilon=cfg.epsilon)
        self.logs: List[Dict] = []

    def maybe_dropout_or_realign(self, u: User, day: int):
        # Weekly (optional): keep users engaged for fairness in tuning (disable dropouts)
        pass

    def step_user(self, u: User, action: str, day: int, date: pd.Timestamp):
        delta = action_effects(u, action)
        u.sleep_hours += delta["sleep"]
        u.activity_min += int(delta["activity"])
        u.diet_fv_g += int(delta["diet"])
        u.abstain = max(u.abstain, int(delta["habit"]))
        u.stress += delta["stress"]
        u.water_glasses += int(delta["water"])
        # NEW lifestyle deltas
        u.screen_time_hours += float(delta["screen"])
        u.social_minutes += int(delta["social"])
        u.mindfulness_min += int(delta["mindful"])

        clip_user_state(u)
        shaped_reward, penalty, comp = reward_and_penalty(u, action)
        observed = misreport(shaped_reward - penalty, scale=0.15)
        apply_daily_drift(u)
        clip_user_state(u)

        # Log a row with ISO date (YYYY-MM-DD)
        self.logs.append({
            "date": date.strftime("%Y-%m-%d"),
            "day": day,
            "user_id": u.user_id,
            "action": action,
            "reward": shaped_reward,
            "penalty": penalty,
            "observed": observed,
            # raw/continuous states for analysis
            "sleep_hours": u.sleep_hours,
            "activity_min": u.activity_min,
            "diet_fv_g": u.diet_fv_g,
            "abstain": u.abstain,
            "stress_level": u.stress,
            "water_glasses": u.water_glasses,
            "screen_time_hours": u.screen_time_hours,
            "social_minutes": u.social_minutes,
            "mindfulness_min": u.mindfulness_min,
            # compliance flags
            **comp
        })
        return shaped_reward, penalty, comp, observed

    def simulate(self, start_date: str = "2023-01-01") -> float:
        """Run one full simulation; return mean cumulative effective reward per user, logging all steps."""
        cum = np.zeros(len(self.population), dtype=float)
        dates = pd.date_range(start=start_date, periods=self.cfg.days, freq="D")
        for day, date in enumerate(dates, start=1):
            for u in self.population:
                state = encode_state(who_compliance(u))
                a_idx = self.agent.select_action(state)
                r, pen, comp, obs = self.step_user(u, ACTIONS[a_idx], day, date)
                r_eff = r - pen
                next_state = encode_state(who_compliance(u))
                self.agent.update(state, a_idx, r_eff, next_state)
                cum[u.user_id] += r_eff
        return float(cum.mean())

    def save_logs(self, out_path: str = "data/sim_dataset.csv", save_parquet: bool = False):
        """Persist the full dataset to CSV (and optionally Parquet)"""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.logs)
        df.to_csv(out_path, index=False)
        print(f"Saved dataset → {out_path}")
        if save_parquet:
            pq_path = out_path.with_suffix(".parquet")
            df.to_parquet(pq_path, index=False)
            print(f"Saved dataset (Parquet) → {pq_path}")

# ---------------------------
# Hyperparameter Tuning & Save
# ---------------------------

def tune_and_train(days=365, users=25, gamma=0.97, epsilon=0.12,
                   alpha_grid=(0.05, 0.1, 0.15, 0.2, 0.25, 0.3),
                   seeds=(42, 1337, 12345)) -> Dict:
    """
    Grid-search alphas; for each alpha, average performance over multiple seeds.
    Return dict with best alpha, agent, and results.
    """
    results = []
    best_alpha = None
    best_score = -1e18
    best_agent_Q = None

    for alpha in alpha_grid:
        scores = []
        for s in seeds:
            random.seed(s); np.random.seed(s)
            cfg = SimulationConfig(days=days, users=users, alpha=alpha, gamma=gamma, epsilon=epsilon)
            sim = DigitalTwinSim(cfg)
            score = sim.simulate()
            scores.append(score)
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
        results.append({"alpha": alpha, "mean_score": mean_score, "std_score": std_score})

        if mean_score > best_score:
            best_score = mean_score
            best_alpha = alpha
            # retrain a fresh agent on canonical seed to persist
            random.seed(SEED); np.random.seed(SEED)
            cfg_best = SimulationConfig(days=days, users=users, alpha=best_alpha, gamma=gamma, epsilon=epsilon)
            sim_best = DigitalTwinSim(cfg_best)
            _ = sim_best.simulate()
            best_agent_Q = sim_best.agent.Q.copy()

    return {
        "grid_results": results,
        "best_alpha": best_alpha,
        "best_score": best_score,
        "best_Q": best_agent_Q
    }

def save_best_model(best_Q: np.ndarray, meta: Dict, outdir="models", filename="best_q_agent.pkl"):
    out_dir = Path(outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save pickle with Q and minimal config
    payload = {
        "ACTIONS": ACTIONS,
        "Q": best_Q,
        "n_states": 16,
        "n_actions": len(ACTIONS),
        "meta": meta
    }
    with open(out_dir / filename, "wb") as f:
        pickle.dump(payload, f)
    # Save a small JSON sidecar for quick inspection
    with open(out_dir / "best_q_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved best Q-learning model → {out_dir / filename}")
    print(f"Saved metadata              → {out_dir / 'best_q_meta.json'}")

# ---------------------------
# Helper: Load dataset for analysis
# ---------------------------

def load_dataset(path: str) -> pd.DataFrame:
    """Load the saved dataset with parsed dates."""
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p, parse_dates=["date"])
    return df

# ---------------------------
# Entrypoint
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Digital Twin Q-learning with dataset logging (enhanced)")
    parser.add_argument("--days", type=int, default=365, help="Number of simulation days")
    parser.add_argument("--users", type=int, default=25, help="Number of users in the synthetic population")
    parser.add_argument("--gamma", type=float, default=0.97, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=0.12, help="Exploration rate")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3],
                        help="Alpha grid for tuning")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 42, 77], help="Seeds for tuning runs")
    parser.add_argument("--model-outdir", type=str, default="models", help="Directory to write model artifacts")
    parser.add_argument("--model-filename", type=str, default="best_q_agent.pkl", help="Pickle filename for best model")
    parser.add_argument("--start-date", type=str, default="2023-01-01", help="ISO start date for the dataset")
    parser.add_argument("--out-data", type=str, default="data/sim_dataset.csv", help="Path to save the dataset CSV/Parquet")
    parser.add_argument("--save-parquet", action="store_true", help="Also save dataset as Parquet")
    args = parser.parse_args()

    DAYS = int(args.days)
    USERS = int(args.users)
    GAMMA = float(args.gamma)
    EPSILON = float(args.epsilon)
    ALPHAS = tuple(float(a) for a in args.alphas)
    SEEDS = tuple(int(s) for s in args.seeds)

    print("Running alpha grid-search...")
    out = tune_and_train(days=DAYS, users=USERS, gamma=GAMMA, epsilon=EPSILON,
                         alpha_grid=ALPHAS, seeds=SEEDS)

    print("\n=== Alpha Search Results (mean cumulative reward per user) ===")
    df = pd.DataFrame(out["grid_results"]).sort_values("alpha")
    try:
        print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    except Exception:
        # Fallback if float_format is not available in user env
        print(df.to_string(index=False))

    meta = {
        "days": DAYS,
        "users": USERS,
        "gamma": GAMMA,
        "epsilon": EPSILON,
        "alpha_grid": ALPHAS,
        "seeds": SEEDS,
        "best_alpha": out["best_alpha"],
        "best_mean_score": out["best_score"]
    }

    save_best_model(out["best_Q"], meta, outdir=args.model_outdir, filename=args.model_filename)

    # Final simulation with best alpha to generate dataset
    print("\nRunning final simulation with best alpha and saving dataset...")
    cfg_final = SimulationConfig(days=DAYS, users=USERS, alpha=out["best_alpha"], gamma=GAMMA, epsilon=EPSILON)
    sim_final = DigitalTwinSim(cfg_final)
    _ = sim_final.simulate(start_date=args.start_date)
    sim_final.save_logs(out_path=args.out_data, save_parquet=args.save_parquet)

    print("\nDone.")

if __name__ == "__main__":
    main()
