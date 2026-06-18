"""
Digital Twin + Constraint-Aware Q-Learning for Behavioral Optimization
----------------------------------------------------------------------
Implements:
- Synthetic population (N users) with demographics
- 365-day simulation of daily actions & outcomes with misreporting/feedback loops
- WHO-aligned hard/soft constraints via penalties
- TSP-style transition shaping (dynamic edge weights)
- Agents: QLearning, SARSA, Greedy, HillClimbing, SimulatedAnnealing (SA), GeneticAlgorithm (GA), DynamicProgramming (DP)
- Evaluation metrics, Welch's t-test (Q vs Others), plots
- Recommendation table for next-day actions per user
- Export of publication-ready CSVs

Usage:
  python digital_twin_qlearning.py

Author: (Ayan)
"""

import math
import random
import json
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------
# Seed for reproducibility
# ---------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ---------------------------
# Digital Twin — Definitions
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

# State = 4-bit vector of previous-day WHO compliance (Sleep, Activity, Diet, Habit) -> integer [0..15]
def encode_state(compliance_flags: Dict[str, int]) -> int:
    bits = [
        compliance_flags.get("sleep", 0),
        compliance_flags.get("activity", 0),
        compliance_flags.get("diet", 0),
        compliance_flags.get("habit", 0),
    ]
    idx = bits[0] + (bits[1] << 1) + (bits[2] << 2) + (bits[3] << 3)
    return idx

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
    print("generate_population------------------------------------->")
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

# TSP-like dynamic edge weights between behaviors (contextual difficulty):
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

# --------------------------------
# Agents: Q-Learning & Baselines
# --------------------------------

class QLearningAgent:
    def __init__(self, n_states=16, n_actions=None, alpha=0.2, gamma=0.95, epsilon=0.1):
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

class GreedyAgent:
    def select_action(self, state: int, user_snapshot: User) -> int:
        est_rewards = []
        for ai, a in enumerate(ACTIONS):
            u = User(**vars(user_snapshot))
            delta = action_effects(u, a)
            u.sleep_hours += delta["sleep"]
            u.activity_min += int(delta["activity"])
            u.diet_fv_g += int(delta["diet"])
            u.abstain = max(u.abstain, int(delta["habit"]))
            u.stress += delta["stress"]
            u.water_glasses += int(delta["water"])
            clip_user_state(u)
            shaped, pen, _ = reward_and_penalty(u, a)
            est_rewards.append(shaped - pen)
        return int(np.argmax(est_rewards))

class HillClimbingAgent:
    def __init__(self, neighborhood=2):
        self.current_action = np.random.randint(len(ACTIONS))
        self.neighborhood = neighborhood

    def select_action(self, state: int, user_snapshot: User) -> int:
        base_action = self.current_action
        def eval_action(aidx):
            u = User(**vars(user_snapshot))
            delta = action_effects(u, ACTIONS[aidx])
            u.sleep_hours += delta["sleep"]
            u.activity_min += int(delta["activity"])
            u.diet_fv_g += int(delta["diet"])
            u.abstain = max(u.abstain, int(delta["habit"]))
            u.stress += delta["stress"]
            u.water_glasses += int(delta["water"])
            clip_user_state(u)
            shaped, pen, _ = reward_and_penalty(u, ACTIONS[aidx])
            return shaped - pen
        best = eval_action(base_action)
        best_a = base_action
        for _ in range(self.neighborhood):
            cand = np.random.randint(len(ACTIONS))
            val = eval_action(cand)
            if val > best:
                best, best_a = val, cand
        self.current_action = best_a
        return best_a

# ===============================
# Extra Baselines for Comparison
# ===============================

class SARSAAgent:
    def __init__(self, n_states=16, n_actions=None, alpha=0.2, gamma=0.95, epsilon=0.1):
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

    def update(self, s: int, a: int, r_eff: float, s_next: int, a_next: int):
        td_target = r_eff + self.gamma * self.Q[s_next, a_next]
        self.Q[s, a] += self.alpha * (td_target - self.Q[s, a])

    def recommend_topk(self, state: int, k: int = 3):
        order = np.argsort(-self.Q[state])
        return [int(i) for i in order[:k]]

class SimulatedAnnealingAgent:
    def __init__(self, n_actions=None, start_temp=1.0, cooling=0.995):
        self.n_actions = len(ACTIONS) if n_actions is None else n_actions
        self.current_action = np.random.randint(self.n_actions)
        self.T = start_temp
        self.cooling = cooling

    def _eval_action_once(self, user_snapshot, aidx: int) -> float:
        u = User(**vars(user_snapshot))
        delta = action_effects(u, ACTIONS[aidx])
        u.sleep_hours += delta["sleep"]; u.activity_min += int(delta["activity"])
        u.diet_fv_g += int(delta["diet"]); u.abstain = max(u.abstain, int(delta["habit"]))
        u.stress += delta["stress"]; u.water_glasses += int(delta["water"])
        clip_user_state(u)
        shaped, pen, _ = reward_and_penalty(u, ACTIONS[aidx])
        return float(shaped - pen)

    def select_action(self, state: int, user_snapshot: User) -> int:
        proposal = np.random.randint(self.n_actions)
        score_curr = self._eval_action_once(user_snapshot, self.current_action)
        score_prop = self._eval_action_once(user_snapshot, proposal)
        if score_prop > score_curr:
            self.current_action = proposal
        else:
            if np.random.rand() < np.exp((score_prop - score_curr) / max(1e-8, self.T)):
                self.current_action = proposal
        self.T *= self.cooling
        return int(self.current_action)
'''
class GeneticAlgorithmAgent:
    def __init__(self, n_actions=None, horizon=7, pop_size=24, generations=12, mutation_rate=0.1, elite_frac=0.25):
        self.n_actions = len(ACTIONS) if n_actions is None else n_actions
        self.horizon = max(2, int(horizon))
        self.pop_size = max(4, int(pop_size))
        self.generations = max(1, int(generations))
        self.mutation_rate = float(mutation_rate)
        self.elite = max(1, int(elite_frac * self.pop_size))
        self.population = [np.random.randint(self.n_actions, size=self.horizon) for _ in range(self.pop_size)]
        self.best_plan = random.choice(self.population)

    def _fitness(self, chromo: np.ndarray, user_snapshot: User) -> float:
        u = User(**vars(user_snapshot))
        total = 0.0
        for ai in chromo:
            delta = action_effects(u, ACTIONS[int(ai)])
            u.sleep_hours += delta["sleep"]; u.activity_min += int(delta["activity"])
            u.diet_fv_g += int(delta["diet"]); u.abstain = max(u.abstain, int(delta["habit"]))
            u.stress += delta["stress"]; u.water_glasses += int(delta["water"])
            clip_user_state(u)
            shaped, pen, _ = reward_and_penalty(u, ACTIONS[int(ai)])
            total += float(shaped - pen)
            apply_daily_drift(u); clip_user_state(u)
        return total

    def _evolve_once(self, user_snapshot: User):
        fitness = np.array([self._fitness(ch, user_snapshot) for ch in self.population])
        idx = np.argsort(-fitness)
        elites = [self.population[i] for i in idx[:self.elite]]
        parents = [self.population[i] for i in idx[: max(self.elite, self.pop_size // 2)]]
        children = elites.copy()
        while len(children) < self.pop_size:
            p1, p2 = random.sample(parents, 2)
            cx = np.random.randint(1, self.horizon)
            child = np.concatenate([p1[:cx], p2[cx:]])
            for i in range(self.horizon):
                if np.random.rand() < self.mutation_rate:
                    child[i] = np.random.randint(self.n_actions)
            children.append(child)
        self.population = children
        best_idx = int(np.argmax([self._fitness(ch, user_snapshot) for ch in self.population]))
        self.best_plan = self.population[best_idx]

    def select_action(self, state: int, user_snapshot: User) -> int:
        for _ in range(self.generations):
            self._evolve_once(user_snapshot)
        return int(self.best_plan[0])

class DPAgent:
    def __init__(self, n_actions=None, horizon=7, gamma=0.95):
        self.n_actions = len(ACTIONS) if n_actions is None else n_actions
        self.horizon = max(1, int(horizon))
        self.gamma = float(gamma)

    def _dp_plan(self, user_snapshot: User):
        memo = {}
        def key_for(u: User, day_left: int):
            return (
                day_left,
                round(u.sleep_hours, 1),
                int(u.activity_min),
                int(u.diet_fv_g),
                int(u.abstain),
                round(u.stress, 1),
                int(u.water_glasses),
            )
        def recurse(day_left: int, u: User):
            k = key_for(u, day_left)
            if k in memo: return memo[k]
            if day_left == 0: return (0.0, [])
            best_val, best_seq = -1e12, []
            for ai in range(self.n_actions):
                u2 = User(**vars(u))
                delta = action_effects(u2, ACTIONS[ai])
                u2.sleep_hours += delta["sleep"]; u2.activity_min += int(delta["activity"])
                u2.diet_fv_g += int(delta["diet"]); u2.abstain = max(u2.abstain, int(delta["habit"]))
                u2.stress += delta["stress"]; u2.water_glasses += int(delta["water"])
                clip_user_state(u2)
                shaped, pen, _ = reward_and_penalty(u2, ACTIONS[ai])
                r_eff = float(shaped - pen)
                apply_daily_drift(u2); clip_user_state(u2)
                v_next, seq_next = recurse(day_left - 1, u2)
                total = r_eff + self.gamma * v_next
                if total > best_val:
                    best_val = total
                    best_seq = [ai] + seq_next
            memo[k] = (best_val, best_seq)
            return memo[k]
        _, seq = recurse(self.horizon, User(**vars(user_snapshot)))
        return seq

    def select_action(self, state: int, user_snapshot: User) -> int:
        seq = self._dp_plan(user_snapshot)
        return int(seq[0]) if seq else np.random.randint(self.n_actions)
'''
# ---------------------------
# Simulation Orchestrator
# ---------------------------

@dataclass
class SimulationConfig:
    days: int = 365
    users: int = 25
    alpha: float = 0.2
    gamma: float = 0.95
    epsilon: float = 0.15

@dataclass
class EpisodeLog:
    records: List[Dict] = field(default_factory=list)
    def append(self, d: Dict):
        self.records.append(d)
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)

class DigitalTwinSim:
    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg
        self.population = generate_population(cfg.users)

        # Agents
        self.q_agent = QLearningAgent(alpha=cfg.alpha, gamma=cfg.gamma, epsilon=cfg.epsilon)
        self.greedy = GreedyAgent()
        self.hc = HillClimbingAgent()
        self.sarsa = SARSAAgent(alpha=cfg.alpha, gamma=cfg.gamma, epsilon=cfg.epsilon)
        self.sa = SimulatedAnnealingAgent()
        #self.ga = GeneticAlgorithmAgent()
        #self.dp = DPAgent()
        print("1------------------------------------->")
        # Logs
        self.log_q = EpisodeLog()
        self.log_g = EpisodeLog()
        self.log_h = EpisodeLog()
        self.log_sarsa = EpisodeLog()
        self.log_sa = EpisodeLog()
        #self.log_ga = EpisodeLog()
        #self.log_dp = EpisodeLog()
        print("n------------------------------------->")
    def maybe_dropout_or_realign(self, u: User, day: int):
        if day % 7 == 0 and day > 0:
            if not u.dropped_out and (np.random.rand() < u.dropout_chance_weekly):
                u.dropped_out = True
            elif u.dropped_out and (np.random.rand() < u.realign_chance_weekly):
                u.dropped_out = False

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

    def simulate(self):
        print("simulate------------------------------------->")
        n_days = self.cfg.days
        for day in range(n_days):
            for u in self.population:
                self.maybe_dropout_or_realign(u, day)
                prev_flags = who_compliance(u)
                state = encode_state(prev_flags)

                if u.dropped_out:
                    chosen_q = np.random.randint(len(ACTIONS))
                    chosen_g = np.random.randint(len(ACTIONS))
                    chosen_h = np.random.randint(len(ACTIONS))
                    chosen_sarsa = np.random.randint(len(ACTIONS))
                    chosen_sa = np.random.randint(len(ACTIONS))
                    #chosen_ga = np.random.randint(len(ACTIONS))
                    #chosen_dp = np.random.randint(len(ACTIONS))
                else:
                    chosen_q = self.q_agent.select_action(state)
                    chosen_g = self.greedy.select_action(state, User(**vars(u)))
                    chosen_h = self.hc.select_action(state, User(**vars(u)))
                    chosen_sarsa = self.sarsa.select_action(state)
                    chosen_sa = self.sa.select_action(state, User(**vars(u)))
                    #chosen_ga = self.ga.select_action(state, User(**vars(u)))
                    #chosen_dp = self.dp.select_action(state, User(**vars(u)))

                # Q-Learning (updates)
                print("simulate---------------1---------------------->")
                uq = User(**vars(u))
                r_q, pen_q, comp_q, obs_q = self.step_user(uq, ACTIONS[chosen_q])
                next_state_q = encode_state(who_compliance(uq))
                r_eff_q = r_q - pen_q
                self.q_agent.update(state, chosen_q, r_eff_q, next_state_q)
                self.log_q.append({
                    "day": day, "user_id": u.user_id, "action": ACTIONS[chosen_q],
                    "reward": r_q, "penalty": pen_q, "reward_eff": r_eff_q,
                    "observed_signal": obs_q, **{f"comp_{k}": int(v) for k,v in who_compliance(uq).items()},
                    "dropped_out": int(u.dropped_out), "agent": "Q"
                })

                # Greedy
                print("simulate---------------2---------------------->")
                ug = User(**vars(u))
                r_g, pen_g, comp_g, obs_g = self.step_user(ug, ACTIONS[chosen_g])
                self.log_g.append({
                    "day": day, "user_id": u.user_id, "action": ACTIONS[chosen_g],
                    "reward": r_g, "penalty": pen_g, "reward_eff": r_g - pen_g,
                    "observed_signal": obs_g, **{f"comp_{k}": int(v) for k,v in who_compliance(ug).items()},
                    "dropped_out": int(u.dropped_out), "agent": "Greedy"
                })

                # Hill Climbing
                print("simulate---------------3---------------------->")
                uh = User(**vars(u))
                r_h, pen_h, comp_h, obs_h = self.step_user(uh, ACTIONS[chosen_h])
                self.log_h.append({
                    "day": day, "user_id": u.user_id, "action": ACTIONS[chosen_h],
                    "reward": r_h, "penalty": pen_h, "reward_eff": r_h - pen_h,
                    "observed_signal": obs_h, **{f"comp_{k}": int(v) for k,v in who_compliance(uh).items()},
                    "dropped_out": int(u.dropped_out), "agent": "HC"
                })

                # SARSA (on-policy)
                print("simulate---------------4---------------------->")
                us = User(**vars(u))
                r_s, pen_s, comp_s, obs_s = self.step_user(us, ACTIONS[chosen_sarsa])
                next_state_s = encode_state(who_compliance(us))
                a_next = np.random.randint(len(ACTIONS)) if u.dropped_out else self.sarsa.select_action(next_state_s)
                r_eff_s = r_s - pen_s
                self.sarsa.update(state, chosen_sarsa, r_eff_s, next_state_s, a_next)
                self.log_sarsa.append({
                    "day": day, "user_id": u.user_id, "action": ACTIONS[chosen_sarsa],
                    "reward": r_s, "penalty": pen_s, "reward_eff": r_eff_s,
                    "observed_signal": obs_s, **{f"comp_{k}": int(v) for k,v in who_compliance(us).items()},
                    "dropped_out": int(u.dropped_out), "agent": "SARSA"
                })
                '''
                # Simulated Annealing
                print("simulate---------------5---------------------->")
                usa = User(**vars(u))
                r_sa, pen_sa, comp_sa, obs_sa = self.step_user(usa, ACTIONS[chosen_sa])
                self.log_sa.append({
                    "day": day, "user_id": u.user_id, "action": ACTIONS[chosen_sa],
                    "reward": r_sa, "penalty": pen_sa, "reward_eff": r_sa - pen_sa,
                    "observed_signal": obs_sa, **{f"comp_{k}": int(v) for k,v in who_compliance(usa).items()},
                    "dropped_out": int(u.dropped_out), "agent": "SA"
                })

                # Genetic Algorithm
                print("simulate---------------6---------------------->")
                uga = User(**vars(u))
                r_ga, pen_ga, comp_ga, obs_ga = self.step_user(uga, ACTIONS[chosen_ga])
                self.log_ga.append({
                    "day": day, "user_id": u.user_id, "action": ACTIONS[chosen_ga],
                    "reward": r_ga, "penalty": pen_ga, "reward_eff": r_ga - pen_ga,
                    "observed_signal": obs_ga, **{f"comp_{k}": int(v) for k,v in who_compliance(uga).items()},
                    "dropped_out": int(u.dropped_out), "agent": "GA"
                })

                # Dynamic Programming
                print("simulate---------------7---------------------->")
                udp = User(**vars(u))
                r_dp, pen_dp, comp_dp, obs_dp = self.step_user(udp, ACTIONS[chosen_dp])
                self.log_dp.append({
                    "day": day, "user_id": u.user_id, "action": ACTIONS[chosen_dp],
                    "reward": r_dp, "penalty": pen_dp, "reward_eff": r_dp - pen_dp,
                    "observed_signal": obs_dp, **{f"comp_{k}": int(v) for k,v in who_compliance(udp).items()},
                    "dropped_out": int(u.dropped_out), "agent": "DP"
                })
                '''
                # Drive the real twin using Q-agent action
                _ = self.step_user(u, ACTIONS[chosen_q])

    # ---------------
    # Analysis / I/O
    # ---------------

    def _weekly_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        print("_weekly_stats------------------------------------->")
        tmp = df.copy()
        tmp["week"] = (tmp["day"] // 7) + 1
        grp = tmp.groupby(["agent", "week"])["reward_eff"].agg(["mean","std","count"]).reset_index()
        return grp

    @staticmethod
    def welch_t_test(sample_a: np.ndarray, sample_b: np.ndarray) -> Tuple[float, float]:
        """Returns (t_stat, p_value one-sided: A > B)."""
        print("welch_t_test------------------------------------->")
        a = np.asarray(sample_a, dtype=float)
        b = np.asarray(sample_b, dtype=float)
        ma, mb = a.mean(), b.mean()
        va, vb = a.var(ddof=1), b.var(ddof=1)
        na, nb = len(a), len(b)
        num = ma - mb
        den = math.sqrt(va/na + vb/nb + 1e-12)
        t = num / (den if den>0 else 1e-9)
        # Normal approximation to one-sided p-value
        p = 0.5 * (1 - math.erf(t / math.sqrt(2)))
        return float(t), float(p)

    def make_recommendations(self) -> pd.DataFrame:
        print("make_recommendations------------------------------------->")
        recs = []
        for u in self.population:
            flags = who_compliance(u)
            s = encode_state(flags)
            topk = self.q_agent.recommend_topk(s, k=3)
            recs.append({
                "user_id": u.user_id,
                "age": u.age,
                "sex": u.sex,
                "bmi": round(u.bmi,1),
                "work": u.work_schedule,
                "smoker": int(u.smoker),
                "adherence": round(u.adherence, 2),
                "top1": ACTIONS[topk[0]],
                "top2": ACTIONS[topk[1]] if len(topk)>1 else None,
                "top3": ACTIONS[topk[2]] if len(topk)>2 else None
            })
        return pd.DataFrame(recs)

    def export_population(self, path: Path):
        print("export_population------------------------------------->")
        rows = []
        for u in self.population:
            rows.append({
                "user_id": u.user_id, "age": u.age, "sex": u.sex, "bmi": u.bmi,
                "work_schedule": u.work_schedule, "smoker": int(u.smoker),
                "baseline_activity": u.baseline_activity, "stress_base": u.stress_base,
                "adherence": u.adherence, "dropout_chance_weekly": u.dropout_chance_weekly,
                "realign_chance_weekly": u.realign_chance_weekly
            })
        pd.DataFrame(rows).to_csv(path, index=False)

    def run_and_report(self, outdir="outputs"):
        print("run_and_report------------------------------------->")
        Path(outdir).mkdir(parents=True, exist_ok=True)
        self.export_population(Path(outdir) / "population_demographics.csv")

        self.simulate()

        # Gather logs
        df_q = self.log_q.to_dataframe()
        df_g = self.log_g.to_dataframe()
        df_h = self.log_h.to_dataframe()
        df_sarsa = self.log_sarsa.to_dataframe()
        df_sa = self.log_sa.to_dataframe()
        #df_ga = self.log_ga.to_dataframe()
        #df_dp = self.log_dp.to_dataframe()

        #df_all = pd.concat([df_q, df_g, df_h, df_sarsa, df_sa, df_ga, df_dp], ignore_index=True)
        df_all = pd.concat([df_q, df_g, df_h, df_sarsa, df_sa], ignore_index=True)
        df_all.to_csv(Path(outdir) / "simulation_log.csv", index=False)

        # Weekly stats + plot
        weekly = self._weekly_stats(df_all)
        weekly.to_csv(Path(outdir) / "weekly_reward_stats.csv", index=False)

        plt.figure()
        #for agent in ["Q","SARSA","Greedy","HC","SA","GA","DP"]:
        for agent in ["Q","SARSA","Greedy","HC","SA"]:
            sub = weekly[weekly["agent"]==agent].sort_values("week")
            if len(sub)==0:
                continue
            x = sub["week"].values
            y = sub["mean"].values
            sd = sub["std"].fillna(0).values
            plt.plot(x, y, label=agent)
            plt.fill_between(x, y - sd, y + sd, alpha=0.2)
        plt.xlabel("Week")
        plt.ylabel("Weekly mean effective reward")
        plt.title("Weekly Reward Evolution (mean ± std)")
        plt.legend(ncols=2)
        plt.tight_layout()
        plt.savefig(Path(outdir) / "weekly_reward_curves.png", dpi=200)
        plt.close()

        # Cumulative per user + export
        cum_by_user = df_all.groupby(["agent","user_id"])["reward_eff"].sum().reset_index()
        cum_by_user.to_csv(Path(outdir) / "cumulative_rewards_by_user.csv", index=False)

        # Welch t-tests: Q vs everyone
        q_vals = cum_by_user[cum_by_user["agent"]=="Q"]["reward_eff"].values
        tests = {}
        #for other in ["Greedy","HC","SA","GA","SARSA","DP"]:
        for other in ["Greedy","HC","SA","SARSA"]:
            o_vals = cum_by_user[cum_by_user["agent"]==other]["reward_eff"].values
            if len(o_vals)==0: 
                continue
            t_stat, p_val = self.welch_t_test(q_vals, o_vals)  # one-sided (Q > Other)
            tests[f"Q_vs_{other}"] = {"t_stat": float(t_stat), "p_one_sided": float(p_val)}

        with open(Path(outdir) / "ttest_results.json", "w") as f:
            json.dump(tests, f, indent=2)

        # Recommendations from Q
        recs = self.make_recommendations()
        recs.to_csv(Path(outdir) / "recommendations_next_day.csv", index=False)

        # Console summary
        def fmt(mean, std): return f"{mean:.2f} ± {std:.2f}"
        summary = df_all.groupby("agent")["reward_eff"].agg(["mean","std"]).reset_index()
        print("\n=== Summary (All Days × Users) ===")
        for _, row in summary.iterrows():
            print(f"{row['agent']:>6}: mean±std reward_eff = {fmt(row['mean'], row['std'])}")

        print("\n=== Welch's t-test (one-sided, H1: Q > Other) ===")
        for k,v in tests.items():
            print(f"{k}: t = {v['t_stat']:.3f}, p = {v['p_one_sided']:.4f}")

        print(f"\nArtifacts saved to: {Path(outdir).resolve()}")
        print(" - population_demographics.csv")
        print(" - simulation_log.csv")
        print(" - weekly_reward_stats.csv")
        print(" - weekly_reward_curves.png")
        print(" - cumulative_rewards_by_user.csv")
        print(" - ttest_results.json")
        print(" - recommendations_next_day.csv")

# ---------------------------
# Entrypoint
# ---------------------------

if __name__ == "__main__":
    cfg = SimulationConfig(days=365, users=50, alpha=0.2, gamma=0.97, epsilon=0.12)
    sim = DigitalTwinSim(cfg)
    sim.run_and_report(outdir="outputs")
