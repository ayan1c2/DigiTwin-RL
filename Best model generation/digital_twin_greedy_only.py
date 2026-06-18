"""
Greedy Policy (No Q-Learning): Digital Twin for Behavioral Optimization
-----------------------------------------------------------------------
- Synthetic population (N users) with demographics
- 365-day simulation of daily actions & outcomes
- WHO-aligned constraints via reward/penalty + TSP-style shaping
- **Greedy agent** chooses the action that maximizes *expected immediate* r_eff
- Hyperparameter search over samples_per_action (Monte Carlo lookahead)
- Persist the selected greedy policy config to pickle

Usage:
  python digital_twin_greedy_policy.py
"""

import math
import random
import json
import pickle
import copy
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from pathlib import Path

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
    "meditate_10",          # stress reduction
    "drink_water_8"         # hydration
]

BEHAVIORS = ["Sleep", "Activity", "Diet", "Habit", "Stress", "Hydration"]

WHO = {
    "sleep_min": 7.0,
    "sleep_max": 9.0,
    "activity_min": 30,      # minutes/day
    "diet_fv_min": 400,      # grams/day
    "habit_abstain": 1,      # 1=abstain, 0=not
    "water_glasses": 8
}

# Penalties (lambda_j). Tuneable.
LAMBDA = {
    "sleep": 4.0,
    "activity": 3.5,
    "diet": 3.0,
    "habit": 5.0,
    "water": 1.5
}

# Reward scale factors
REWARD_SCALE = {
    "sleep": 2.5,
    "activity": 2.0,
    "diet": 2.0,
    "habit": 3.0,
    "stress": 1.5,
    "hydration": 1.0
}

# State = 4-bit vector of previous-day WHO compliance (Sleep, Activity, Diet, Habit)
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
        users.append(User(
            user_id=uid, age=age, sex=sex, bmi=bmi, work_schedule=work_schedule,
            smoker=smoker, baseline_activity=baseline_activity, stress_base=stress_base,
            adherence=adherence, dropout_chance_weekly=dropout_chance_weekly,
            realign_chance_weekly=realign_chance_weekly
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
    lam = 1.0
    return -reward + lam * penalty


def action_effects(user: User, action: str) -> Dict[str, float]:
    adhere = (np.random.rand() < user.adherence)
    noise = lambda s: np.random.normal(0, s)
    delta = dict(sleep=0.0, activity=0.0, diet=0.0, habit=0.0, stress=0.0, water=0.0)
    if not adhere:
        delta["stress"] += noise(0.4)
        return delta
    if action == "sleep_early":
        delta["sleep"] += 1.5 + noise(0.5)
        delta["stress"] += -0.4 + noise(0.3)
    elif action == "walk_30":
        delta["activity"] += 30 + int(np.random.normal(0, 8))
        delta["stress"] += -0.3 + noise(0.3)
        delta["sleep"] += 0.2 + noise(0.2)
    elif action == "eat_veg_400":
        delta["diet"] += 300 + int(np.random.normal(0, 80))
        delta["stress"] += -0.1 + noise(0.2)
    elif action == "no_tobacco_alcohol":
        delta["habit"] += 1
        delta["stress"] += -0.2 + noise(0.3)
    elif action == "meditate_10":
        delta["stress"] += -0.8 + noise(0.3)
        delta["sleep"] += 0.2 + noise(0.2)
    elif action == "drink_water_8":
        delta["water"] += 4 + int(np.random.normal(0, 2))
    return delta


def clip_user_state(u: User):
    u.sleep_hours = float(np.clip(u.sleep_hours, 3.0, 11.0))
    u.activity_min = int(np.clip(u.activity_min, 0, 180))
    u.diet_fv_g = int(np.clip(u.diet_fv_g, 0, 1200))
    u.abstain = int(np.clip(u.abstain, 0, 1))
    u.stress = float(np.clip(u.stress, 0.0, 10.0))
    u.water_glasses = int(np.clip(u.water_glasses, 0, 16))


def apply_daily_drift(u: User):
    u.activity_min = int(max(0, u.activity_min - 10 + np.random.randint(-5, 6)))
    u.diet_fv_g = int(max(0, u.diet_fv_g - 80 + int(np.random.normal(0, 40))))
    u.sleep_hours += np.random.normal(-0.2, 0.4)
    u.abstain = 0 if (np.random.rand() < (0.15 if u.smoker else 0.05)) else u.abstain
    u.stress = float(np.clip(u.stress + np.random.normal(0.1, 0.5) + (0.3 if u.work_schedule=='shift' else 0.0), 0, 10))
    u.water_glasses = max(0, u.water_glasses - 2 + int(np.random.normal(0, 1)))


def who_compliance(u: User) -> Dict[str, int]:
    return {
        "sleep": int(WHO["sleep_min"] <= u.sleep_hours <= WHO["sleep_max"]),
        "activity": int(u.activity_min >= WHO["activity_min"]),
        "diet": int(u.diet_fv_g >= WHO["diet_fv_min"]),
        "habit": int(u.abstain >= WHO["habit_abstain"]),
        "water": int(u.water_glasses >= WHO["water_glasses"])
    }


def reward_and_penalty(u: User, action: str) -> Tuple[float, float, Dict[str,int]]:
    comp = who_compliance(u)
    reward = (
        REWARD_SCALE["sleep"] * comp["sleep"] +
        REWARD_SCALE["activity"] * comp["activity"] +
        REWARD_SCALE["diet"] * comp["diet"] +
        REWARD_SCALE["habit"] * comp["habit"] +
        REWARD_SCALE["stress"] * (1.0 - u.stress/10.0) +
        REWARD_SCALE["hydration"] * comp["water"]
    )
    penalty = (
        LAMBDA["sleep"] * (1 - comp["sleep"]) +
        LAMBDA["activity"] * (1 - comp["activity"]) +
        LAMBDA["diet"] * (1 - comp["diet"]) +
        LAMBDA["habit"] * (1 - comp["habit"]) +
        LAMBDA["water"] * (1 - comp["water"]) 
    )
    deficits = {
        "Sleep": WHO["sleep_min"] - u.sleep_hours if u.sleep_hours < WHO["sleep_min"] else 0,
        "Activity": WHO["activity_min"] - u.activity_min if u.activity_min < WHO["activity_min"] else 0,
        "Diet": WHO["diet_fv_min"] - u.diet_fv_g if u.diet_fv_g < WHO["diet_fv_min"] else 0,
        "Habit": 1 - u.abstain,
        "Stress": u.stress/10.0,
        "Hydration": WHO["water_glasses"] - u.water_glasses if u.water_glasses < WHO["water_glasses"] else 0
    }
    dominant_from = max(deficits, key=deficits.get)
    action_map = {
        "sleep_early": "Sleep",
        "walk_30": "Activity",
        "eat_veg_400": "Diet",
        "no_tobacco_alcohol": "Habit",
        "meditate_10": "Stress",
        "drink_water_8": "Hydration"
    }
    dominant_to = action_map[action]
    w_ij = tsp_edge_weight(u, dominant_from, dominant_to)
    edge_reward = max(-10.0, min(10.0, 10.0 - w_ij))
    shaped = reward + 0.25 * edge_reward
    return shaped, penalty, comp


def misreport(value: float, scale: float=0.1) -> float:
    return float(value + np.random.normal(0, scale * max(1.0, abs(value))))

# ---------------------------
# Greedy Agent (myopic, Monte Carlo lookahead)
# ---------------------------

class GreedyAgent:
    def __init__(self, n_states=16, n_actions=None, samples_per_action: int = 5, random_tie_break: bool = True):
        self.n_states = n_states
        self.n_actions = len(ACTIONS) if n_actions is None else n_actions
        self.samples_per_action = int(max(1, samples_per_action))
        self.random_tie_break = bool(random_tie_break)
        # For diagnostics / saving: track observed r_eff by (state, action)
        self.r_stats_sum = np.zeros((self.n_states, self.n_actions), dtype=float)
        self.r_stats_n = np.zeros((self.n_states, self.n_actions), dtype=int)

    def _one_step_r_eff(self, u: User, a_idx: int) -> float:
        """Return immediate r_eff if we apply ACTIONS[a_idx] once to a *copy* of user.
        This mirrors step_user's reward timing (before drift).
        """
        u2 = copy.deepcopy(u)
        action = ACTIONS[a_idx]
        delta = action_effects(u2, action)
        u2.sleep_hours += delta["sleep"]
        u2.activity_min += int(delta["activity"])
        u2.diet_fv_g += int(delta["diet"])
        u2.abstain = max(u2.abstain, int(delta["habit"]))
        u2.stress += delta["stress"]
        u2.water_glasses += int(delta["water"])
        clip_user_state(u2)
        shaped_reward, penalty, _ = reward_and_penalty(u2, action)
        r_eff = shaped_reward - penalty
        return float(r_eff)

    def _expected_r_eff(self, u: User, a_idx: int) -> float:
        s = 0.0
        for _ in range(self.samples_per_action):
            s += self._one_step_r_eff(u, a_idx)
        return s / self.samples_per_action

    def select_action(self, u: User) -> int:
        """Greedy choice: pick action with highest estimated immediate r_eff."""
        estimates = [self._expected_r_eff(u, a) for a in range(self.n_actions)]
        max_val = max(estimates)
        # Tie-breaker
        best_idxs = [i for i, v in enumerate(estimates) if np.isclose(v, max_val)]
        if self.random_tie_break and len(best_idxs) > 1:
            choice = int(np.random.choice(best_idxs))
        else:
            choice = int(best_idxs[0])
        return choice

    def record(self, state: int, a_idx: int, r_eff: float):
        self.r_stats_sum[state, a_idx] += r_eff
        self.r_stats_n[state, a_idx] += 1

    def get_stats(self) -> Dict[str, np.ndarray]:
        with np.errstate(divide='ignore', invalid='ignore'):
            avg = np.divide(self.r_stats_sum, self.r_stats_n, out=np.zeros_like(self.r_stats_sum), where=self.r_stats_n>0)
        return {"sum": self.r_stats_sum.copy(), "n": self.r_stats_n.copy(), "avg": avg}

# ---------------------------
# Simulation Orchestrator
# ---------------------------

@dataclass
class SimulationConfig:
    days: int = 365
    users: int = 25
    samples_per_action: int = 5


class DigitalTwinGreedySim:
    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg
        self.population = generate_population(cfg.users)
        self.agent = GreedyAgent(samples_per_action=cfg.samples_per_action)

    def maybe_dropout_or_realign(self, u: User, day: int):
        # Weekly (optional): keep users engaged for fairness in tuning (disable dropouts)
        pass

    def step_user(self, u: User, action: str):
        delta = action_effects(u, action)
        u.sleep_hours += delta["sleep"]
        u.activity_min += int(delta["activity"])
        u.diet_fv_g += int(delta["diet"])
        u.abstain = max(u.abstain, int(delta["habit"]))
        u.stress += delta["stress"]
        u.water_glasses += int(delta["water"])
        clip_user_state(u)
        shaped_reward, penalty, comp = reward_and_penalty(u, action)
        observed = misreport(shaped_reward - penalty, scale=0.15)
        apply_daily_drift(u)
        clip_user_state(u)
        return shaped_reward, penalty, comp, observed

    def simulate(self) -> float:
        """Run one full simulation; return mean cumulative effective reward per user."""
        cum = np.zeros(len(self.population), dtype=float)
        for day in range(self.cfg.days):
            for u in self.population:
                state = encode_state(who_compliance(u))
                a_idx = self.agent.select_action(u)
                r, pen, comp, obs = self.step_user(u, ACTIONS[a_idx])
                r_eff = r - pen
                self.agent.record(state, a_idx, r_eff)
                cum[u.user_id] += r_eff
        return float(cum.mean())

# ---------------------------
# Hyperparameter Tuning & Save
# ---------------------------

def tune_and_run(days=365, users=25,
                 samples_grid=(1, 3, 5, 8, 10),
                 seeds=(42, 1337, 12345)) -> Dict:
    """
    Grid-search samples_per_action; for each value, average performance over multiple seeds.
    Return dict with best setting and diagnostics (agent stats for the best run).
    """
    results = []
    best_k = None
    best_score = -1e18
    best_stats = None

    for k in samples_grid:
        scores = []
        for s in seeds:
            random.seed(s); np.random.seed(s)
            cfg = SimulationConfig(days=days, users=users, samples_per_action=k)
            sim = DigitalTwinGreedySim(cfg)
            score = sim.simulate()
            scores.append(score)
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
        results.append({"samples": int(k), "mean_score": mean_score, "std_score": std_score})

        if mean_score > best_score:
            best_score = mean_score
            best_k = int(k)
            # retrain a fresh agent on canonical seed to persist stats
            random.seed(SEED); np.random.seed(SEED)
            cfg_best = SimulationConfig(days=days, users=users, samples_per_action=best_k)
            sim_best = DigitalTwinGreedySim(cfg_best)
            _ = sim_best.simulate()
            best_stats = sim_best.agent.get_stats()

    return {
        "grid_results": results,
        "best_samples": best_k,
        "best_score": best_score,
        "best_stats": best_stats,
    }


def save_greedy_model(best_samples: int, stats: Dict[str, np.ndarray], outdir="models", filename="best_greedy_policy.pkl"):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    payload = {
        "policy": "greedy_immediate_r_eff",  # documentation string
        "ACTIONS": ACTIONS,
        "n_states": 16,
        "n_actions": len(ACTIONS),
        "params": {"samples_per_action": int(best_samples)},
        "diagnostics": {
            # Aggregate stats so downstream code can inspect empirical payoffs
            "state_action_reward_sum": stats["sum"],
            "state_action_counts": stats["n"],
            "state_action_reward_avg": stats["avg"],
        },
    }
    with open(Path(outdir) / filename, "wb") as f:
        pickle.dump(payload, f)
    # Save a small JSON sidecar for quick inspection
    with open(Path(outdir) / "best_greedy_meta.json", "w") as f:
        json.dump({
            "policy": payload["policy"],
            "ACTIONS": payload["ACTIONS"],
            "params": payload["params"],
        }, f, indent=2)
    print(f"\nSaved greedy policy → {Path(outdir) / filename}")
    print(f"Saved metadata      → {Path(outdir) / 'best_greedy_meta.json'}")

# ---------------------------
# Entrypoint
# ---------------------------

if __name__ == "__main__":
    DAYS = 365
    USERS = 25
    SAMPLES = (1, 3, 5, 8, 10)
    SEEDS = (11, 42, 77)  # keep small for speed; increase for robustness

    print("Running samples_per_action grid-search...")
    out = tune_and_run(days=DAYS, users=USERS, samples_grid=SAMPLES, seeds=SEEDS)

    print("\n=== Samples-per-Action Search Results (mean cumulative reward per user) ===")
    df = pd.DataFrame(out["grid_results"]).sort_values("samples")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    meta = {
        "days": DAYS,
        "users": USERS,
        "samples_grid": tuple(int(k) for k in SAMPLES),
        "seeds": SEEDS,
        "best_samples": int(out["best_samples"]),
        "best_mean_score": out["best_score"],
    }

    save_greedy_model(out["best_samples"], out["best_stats"], outdir="models", filename="best_greedy_policy.pkl")

    print("\nDone.")
