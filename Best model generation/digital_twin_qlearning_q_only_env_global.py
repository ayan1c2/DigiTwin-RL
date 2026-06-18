"""
Q-Learning Only: Digital Twin for Behavioral Optimization (Env-Enhanced)
------------------------------------------------------------------------
- Synthetic population (N users) with demographics
- 365-day simulation of daily actions & outcomes
- WHO-aligned constraints via reward/penalty + TSP-style shaping
- Q-learning agent
- Hyperparameter search over learning rate (alpha)
- Persist best model to pickle
- Persist full per-day dataset (CSV/optional Parquet) with proper ISO dates
- Customizable users/days via CLI args
- WHO-guided mental health + extra lifestyle parameters
    * Stress threshold (<= 5), daily mindfulness target (>= 10 min)
    * Screen time cap (<= 2 hours), social interaction minimum (>= 15 min)
- NEW: Eco-anxiety and environmental factors (air/noise/light, heatwave, flood)

Usage:
  python digital_twin_qlearning_q_only_env_global.py --users 50 --days 180 --start-date 2024-01-01 --out-data data/sim_dataset.csv
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
    "walk_30",              # 30 min activity (outdoor when weather good)
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
    # Mental health & lifestyle
    "stress_max": 5.0,        # perceived stress should be moderate or lower
    "mindfulness_minutes": 10,# target mindfulness/relaxation minutes per day
    "screen_max_hours": 2.0,  # target daily screen time cap outside work
    "social_min_minutes": 15  # target minimum social interaction
}

# Environmental thresholds (simplified WHO-aligned proxies)
WHO_ENV = {
    "air_quality_max": 50,     # AQI safe
    "noise_max": 55,           # dB daytime
    "light_max": 5,            # lux at night for sleep environments
    "eco_anxiety_max": 5       # target keep <= moderate
}

# Penalties (lambda_j). Tuneable.
LAMBDA = {
    "sleep": 4.0,
    "activity": 3.5,
    "diet": 3.0,
    "habit": 5.0,
    "water": 1.5,
    "mental": 3.5,
    "screen": 1.0,
    "social": 1.0,
    # Environment
    "eco": 2.5,
    "air": 1.5,
    "noise": 1.2,
    "light": 1.2,
    "heatwave": 1.8,
    "flood": 2.0
}

# Reward scale factors
REWARD_SCALE = {
    "sleep": 2.5,
    "activity": 2.0,
    "diet": 2.0,
    "habit": 3.0,
    "stress": 1.5,
    "hydration": 1.0,
    "mental": 2.0,
    "screen": 0.6,
    "social": 0.8,
    # Environment
    "eco": 1.2,
    "air": 0.6,
    "noise": 0.4,
    "light": 0.4,
    "heatwave": 0.5,
    "flood": 0.6
}

# State = 4-bit vector of previous-day WHO compliance (Sleep, Activity, Diet, Habit)
# Keep state compact; new targets affect reward but not state size.
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
    # Lifestyle states
    screen_time_hours: float = 3.5    # non-work leisure screen time
    social_minutes: int = 10          # daily minutes
    mindfulness_min: int = 0          # minutes meditated today
    # Eco-anxiety
    eco_anxiety: float = 3.0
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
        screen_time_hours = float(np.clip(np.random.normal(3.5, 1.0), 0.0, 8.0))
        social_minutes = int(np.clip(np.random.normal(12, 8), 0, 120))
        eco_anxiety = float(np.clip(np.random.normal(3.0, 1.2), 0, 10))
        users.append(User(
            user_id=uid, age=age, sex=sex, bmi=bmi, work_schedule=work_schedule,
            smoker=smoker, baseline_activity=baseline_activity, stress_base=stress_base,
            adherence=adherence, dropout_chance_weekly=dropout_chance_weekly,
            realign_chance_weekly=realign_chance_weekly,
            screen_time_hours=screen_time_hours, social_minutes=social_minutes,
            eco_anxiety=eco_anxiety
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
    if i == "Screen" and j == "Sleep":
        reward += 1.0
    if i == "Social" and j == "Stress":
        reward += 1.0
    lam = 1.0
    return -reward + lam * penalty

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
    u.eco_anxiety = float(np.clip(u.eco_anxiety, 0.0, 10.0))

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
# Global Environment
# ---------------------------

@dataclass
class Environment:
    air_quality_index: int = 55     # AQI
    noise_db: float = 58.0          # dB
    light_pollution_lux: float = 8.0# lux (night)
    heatwave: int = 0               # 1 if heatwave today
    flood: int = 0                  # 1 if flood today

    def daily_update(self):
        # stochastic drift
        self.air_quality_index = int(np.clip(self.air_quality_index + np.random.normal(0, 6), 0, 300))
        self.noise_db = float(np.clip(self.noise_db + np.random.normal(0, 3), 30, 100))
        self.light_pollution_lux = float(np.clip(self.light_pollution_lux + np.random.normal(0, 1.2), 0, 50))

        # rare events
        self.heatwave = 1 if np.random.rand() < 0.02 else 0
        self.flood = 1 if np.random.rand() < 0.005 else 0

env = Environment()

def action_effects(user: User, action: str, env: Environment) -> Dict[str, float]:
    adhere = (np.random.rand() < user.adherence)
    noise = lambda s: np.random.normal(0, s)
    delta = dict(sleep=0.0, activity=0.0, diet=0.0, habit=0.0, stress=0.0, water=0.0,
                 screen=-0.0, social=0.0, mindful=0.0, eco=0.0)
    if not adhere:
        delta["stress"] += noise(0.4)
        delta["screen"] += abs(noise(0.2))
        delta["social"] += -abs(noise(3.0))
        delta["eco"] += abs(noise(0.2))  # non-adherence can heighten anxiety
        return delta

    # Environmental modifiers
    bad_air = env.air_quality_index > WHO_ENV["air_quality_max"]
    bad_noise = env.noise_db > WHO_ENV["noise_max"]
    bad_light = env.light_pollution_lux > WHO_ENV["light_max"]
    extreme = env.heatwave or env.flood

    if action == "sleep_early":
        delta["sleep"] += 1.5 + noise(0.5)
        delta["stress"] += -0.4 + noise(0.3)
        delta["screen"] += -0.3 + noise(0.2)
        if bad_light:
            delta["sleep"] += -0.3  # light pollution harms sleep quality
            delta["eco"] += 0.1
    elif action == "walk_30":
        delta["activity"] += 30 + int(np.random.normal(0, 8))
        # Outdoor benefits depend on environment
        if not (bad_air or extreme):
            delta["stress"] += -0.5 + noise(0.25)
            delta["eco"] += -0.3
            delta["sleep"] += 0.2 + noise(0.2)
        else:
            # exercising in poor conditions may backfire a bit
            delta["stress"] += 0.1 + noise(0.2)
            delta["eco"] += 0.2
        delta["social"] += 2 + int(np.random.normal(0, 2))
    elif action == "eat_veg_400":
        delta["diet"] += 300 + int(np.random.normal(0, 80))
        delta["stress"] += -0.1 + noise(0.2)
    elif action == "no_tobacco_alcohol":
        delta["habit"] += 1
        delta["stress"] += -0.2 + noise(0.3)
        delta["eco"] += -0.05
    elif action == "meditate_10":
        delta["stress"] += -0.9 + noise(0.3)
        delta["sleep"] += 0.2 + noise(0.2)
        delta["mindful"] += 10
        delta["eco"] += -0.4
    elif action == "drink_water_8":
        delta["water"] += 4 + int(np.random.normal(0, 2))
    elif action == "digital_detox_30":
        delta["screen"] += -0.5 + noise(0.2)
        delta["sleep"] += 0.1 + noise(0.2)
        delta["stress"] += -0.1 + noise(0.2)
        delta["eco"] += -0.1
    elif action == "social_15":
        delta["social"] += 15 + int(np.random.normal(0, 4))
        delta["stress"] += -0.25 + noise(0.2)
        delta["eco"] += -0.15
    return delta

def apply_daily_drift(user: User, env: Environment):
    user.activity_min = int(max(0, user.activity_min - 10 + np.random.randint(-5, 6)))
    user.diet_fv_g = int(max(0, user.diet_fv_g - 80 + int(np.random.normal(0, 40))))
    user.sleep_hours += np.random.normal(-0.2, 0.4)
    user.abstain = 0 if (np.random.rand() < (0.15 if user.smoker else 0.05)) else user.abstain
    # Stress & eco-anxiety drift influenced by environment
    stress_drift = np.random.normal(0.1, 0.5) + (0.3 if user.work_schedule=='shift' else 0.0)
    eco_drift = 0.0
    if env.air_quality_index > WHO_ENV["air_quality_max"]: eco_drift += 0.15
    if env.noise_db > WHO_ENV["noise_max"]: eco_drift += 0.1
    if env.light_pollution_lux > WHO_ENV["light_max"]: eco_drift += 0.05
    if env.heatwave: eco_drift += 0.3; stress_drift += 0.2
    if env.flood: eco_drift += 0.5; stress_drift += 0.3
    user.stress = float(np.clip(user.stress + stress_drift, 0, 10))
    user.eco_anxiety = float(np.clip(user.eco_anxiety + eco_drift + np.random.normal(0, 0.2), 0, 10))
    user.water_glasses = max(0, user.water_glasses - 2 + int(np.random.normal(0, 1)))
    user.screen_time_hours = float(np.clip(user.screen_time_hours + np.random.normal(0.2, 0.4), 0.0, 12.0))
    user.social_minutes = int(np.clip(user.social_minutes + int(np.random.normal(-2, 6)), 0, 240))
    user.mindfulness_min = 0

def who_compliance(u: User, env: Environment) -> Dict[str, int]:
    return {
        "sleep": int(WHO["sleep_min"] <= u.sleep_hours <= WHO["sleep_max"]),
        "activity": int(u.activity_min >= WHO["activity_min"]),
        "diet": int(u.diet_fv_g >= WHO["diet_fv_min"]),
        "habit": int(u.abstain >= WHO["habit_abstain"]),
        "water": int(u.water_glasses >= WHO["water_glasses"]),
        "mental_health": int((u.stress <= WHO["stress_max"]) or (u.mindfulness_min >= WHO["mindfulness_minutes"])),
        "screen": int(u.screen_time_hours <= WHO["screen_max_hours"]),
        "social": int(u.social_minutes >= WHO["social_min_minutes"]),
        # Environmental compliance
        "eco_safe": int(u.eco_anxiety <= WHO_ENV["eco_anxiety_max"]),
        "air_quality": int(env.air_quality_index <= WHO_ENV["air_quality_max"]),
        "noise": int(env.noise_db <= WHO_ENV["noise_max"]),
        "light": int(env.light_pollution_lux <= WHO_ENV["light_max"]),
        "heatwave_safe": int(env.heatwave == 0),
        "flood_safe": int(env.flood == 0)
    }

def reward_and_penalty(u: User, action: str, env: Environment) -> Tuple[float, float, Dict[str,int]]:
    comp = who_compliance(u, env)
    reward = (
        REWARD_SCALE["sleep"] * comp["sleep"]
        + REWARD_SCALE["activity"] * comp["activity"]
        + REWARD_SCALE["diet"] * comp["diet"]
        + REWARD_SCALE["habit"] * comp["habit"]
        + REWARD_SCALE["stress"] * (1.0 - u.stress/10.0)
        + REWARD_SCALE["hydration"] * comp["water"]
        + REWARD_SCALE["mental"] * comp["mental_health"]
        + REWARD_SCALE["screen"] * comp["screen"]
        + REWARD_SCALE["social"] * comp["social"]
        + REWARD_SCALE["eco"] * comp["eco_safe"]
        + REWARD_SCALE["air"] * comp["air_quality"]
        + REWARD_SCALE["noise"] * comp["noise"]
        + REWARD_SCALE["light"] * comp["light"]
        + REWARD_SCALE["heatwave"] * comp["heatwave_safe"]
        + REWARD_SCALE["flood"] * comp["flood_safe"]
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
        + LAMBDA["eco"] * (1 - comp["eco_safe"])
        + LAMBDA["air"] * (1 - comp["air_quality"])
        + LAMBDA["noise"] * (1 - comp["noise"])
        + LAMBDA["light"] * (1 - comp["light"])
        + LAMBDA["heatwave"] * (1 - comp["heatwave_safe"])
        + LAMBDA["flood"] * (1 - comp["flood_safe"])
    )
    # Deficits for shaping origin
    deficits = {
        "Sleep": max(0.0, WHO["sleep_min"] - u.sleep_hours),
        "Activity": max(0, WHO["activity_min"] - u.activity_min),
        "Diet": max(0, WHO["diet_fv_min"] - u.diet_fv_g),
        "Habit": max(0, 1 - u.abstain),
        "Stress": u.stress/10.0,
        "Hydration": max(0, WHO["water_glasses"] - u.water_glasses),
        "Eco": max(0.0, u.eco_anxiety - WHO_ENV["eco_anxiety_max"]),
        "Air": max(0, env.air_quality_index - WHO_ENV["air_quality_max"]),
        "Noise": max(0.0, env.noise_db - WHO_ENV["noise_max"]),
        "Light": max(0.0, env.light_pollution_lux - WHO_ENV["light_max"]),
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

    def step_user(self, u: User, action: str, day: int, date: pd.Timestamp):
        delta = action_effects(u, action, env)
        u.sleep_hours += delta["sleep"]
        u.activity_min += int(delta["activity"])
        u.diet_fv_g += int(delta["diet"])
        u.abstain = max(u.abstain, int(delta["habit"]))
        u.stress += delta["stress"]
        u.water_glasses += int(delta["water"])
        u.screen_time_hours += float(delta["screen"])
        u.social_minutes += int(delta["social"])
        u.mindfulness_min += int(delta["mindful"])
        u.eco_anxiety += float(delta["eco"])

        clip_user_state(u)
        shaped_reward, penalty, comp = reward_and_penalty(u, action, env)
        observed = misreport(shaped_reward - penalty, scale=0.15)
        apply_daily_drift(u, env)
        clip_user_state(u)

        self.logs.append({
            "date": date.strftime("%Y-%m-%d"),
            "day": day,
            "user_id": u.user_id,
            "action": action,
            "reward": shaped_reward,
            "penalty": penalty,
            "observed": observed,
            # raw states
            "sleep_hours": u.sleep_hours,
            "activity_min": u.activity_min,
            "diet_fv_g": u.diet_fv_g,
            "abstain": u.abstain,
            "stress_level": u.stress,
            "water_glasses": u.water_glasses,
            "screen_time_hours": u.screen_time_hours,
            "social_minutes": u.social_minutes,
            "mindfulness_min": u.mindfulness_min,
            "eco_anxiety": u.eco_anxiety,
            # global environment snapshot
            "air_quality_index": env.air_quality_index,
            "noise_db": env.noise_db,
            "light_pollution_lux": env.light_pollution_lux,
            "heatwave": env.heatwave,
            "flood": env.flood,
            # compliance flags
            **comp
        })
        return shaped_reward, penalty, comp, observed

    def simulate(self, start_date: str = "2023-01-01") -> float:
        cum = np.zeros(len(self.population), dtype=float)
        dates = pd.date_range(start=start_date, periods=self.cfg.days, freq="D")
        for day, date in enumerate(dates, start=1):
            env.daily_update()
            for u in self.population:
                state = encode_state(who_compliance(u, env))
                a_idx = self.agent.select_action(state)
                r, pen, comp, obs = self.step_user(u, ACTIONS[a_idx], day, date)
                r_eff = r - pen
                next_state = encode_state(who_compliance(u, env))
                self.agent.update(state, a_idx, r_eff, next_state)
                cum[u.user_id] += r_eff
        return float(cum.mean())

    def save_logs(self, out_path: str = "data/sim_dataset_env_global.csv", save_parquet: bool = False):
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
# Hyperparameter Tuning & Entrypoint
# ---------------------------

def tune_and_train(days=365, users=25, gamma=0.97, epsilon=0.12,
                   alpha_grid=(0.05, 0.1, 0.15, 0.2, 0.25, 0.3),
                   seeds=(42, 1337, 12345)) -> Dict:
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

def save_best_model(best_Q: np.ndarray, meta: Dict, outdir="models", filename="best_q_agent_env_global.pkl"):
    out_dir = Path(outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"ACTIONS": ACTIONS, "Q": best_Q, "n_states": 16, "n_actions": len(ACTIONS), "meta": meta}
    with open(out_dir / filename, "wb") as f:
        pickle.dump(payload, f)
    with open(out_dir / "best_q_meta_env_global.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved best Q-learning model → {out_dir / filename}")
    print(f"Saved metadata              → {out_dir / 'best_q_meta_env_global.json'}")

def main():
    parser = argparse.ArgumentParser(description="Digital Twin Q-learning with global environmental factors")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--users", type=int, default=25)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--epsilon", type=float, default=0.12)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3])
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 42, 77])
    parser.add_argument("--model-outdir", type=str, default="models")
    parser.add_argument("--model-filename", type=str, default="best_q_agent_env_global.pkl")
    parser.add_argument("--start-date", type=str, default="2023-01-01")
    parser.add_argument("--out-data", type=str, default="data/sim_dataset_env_global.csv")
    parser.add_argument("--save-parquet", action="store_true")
    args = parser.parse_args()

    DAYS = int(args.days); USERS = int(args.users)
    GAMMA = float(args.gamma); EPSILON = float(args.epsilon)
    ALPHAS = tuple(float(a) for a in args.alphas); SEEDS = tuple(int(s) for s in args.seeds)

    print("Running alpha grid-search...")
    out = tune_and_train(days=DAYS, users=USERS, gamma=GAMMA, epsilon=EPSILON, alpha_grid=ALPHAS, seeds=SEEDS)

    print("\n=== Alpha Search Results (mean cumulative reward per user) ===")
    df = pd.DataFrame(out["grid_results"]).sort_values("alpha")
    try:
        print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    except Exception:
        print(df.to_string(index=False))

    meta = {
        "days": DAYS, "users": USERS, "gamma": GAMMA, "epsilon": EPSILON,
        "alpha_grid": ALPHAS, "seeds": SEEDS,
        "best_alpha": out["best_alpha"], "best_mean_score": out["best_score"]
    }
    save_best_model(out["best_Q"], meta, outdir=args.model_outdir, filename=args.model_filename)

    print("\nRunning final simulation with best alpha and saving dataset...")
    cfg_final = SimulationConfig(days=DAYS, users=USERS, alpha=out["best_alpha"], gamma=GAMMA, epsilon=EPSILON)
    sim_final = DigitalTwinSim(cfg_final)
    _ = sim_final.simulate(start_date=args.start_date)
    sim_final.save_logs(out_path=args.out_data, save_parquet=args.save_parquet)
    print("\nDone.")

if __name__ == "__main__":
    main()
