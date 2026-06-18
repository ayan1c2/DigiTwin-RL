"""
RL Actors Analysis (standalone)
-------------------------------
Reads the digital twin CSVs and runs Q-Learning, SARSA, Greedy, Hill Climbing.
Generates comparative analysis, execution time, and output artifacts similar
to the original single-file simulator.

Inputs (from --indir, default: outputs_gen):
- population_demographics.csv
- initial_states.csv
- metadata.json  (optional, for reference)

Outputs (in --outdir, default: outputs):
- simulation_log.csv
- weekly_reward_stats.csv
- weekly_reward_curves.png
- cumulative_rewards_by_user.csv
- ttest_results.json
- recommendations_next_day.csv
- timings.json

Usage:
  python rl_actors_analysis.py --indir outputs_gen --days 365 --outdir outputs

This file is derived from the original single-file simulator, restricted to
four actors: Q-Learning, SARSA, Greedy, Hill Climbing.
"""

import math
import json
import time
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Canonical definitions
# ---------------------------

ACTIONS = [
    "sleep_early",
    "walk_30",
    "eat_veg_400",
    "no_tobacco_alcohol",
    "meditate_10",
    "drink_water_8"
]

WHO = {
    "sleep_min": 7.0,
    "sleep_max": 9.0,
    "activity_min": 30,
    "diet_fv_min": 400,
    "habit_abstain": 1,
    "water_glasses": 8
}

LAMBDA = {
    "sleep": 4.0,
    "activity": 3.5,
    "diet": 3.0,
    "habit": 5.0,
    "water": 1.5
}

REWARD_SCALE = {
    "sleep": 2.5,
    "activity": 2.0,
    "diet": 2.0,
    "habit": 3.0,
    "stress": 1.5,
    "hydration": 1.0
}

# ---------------------------
# Utilities
# ---------------------------

@dataclass
class User:
    user_id: int
    age: int
    sex: str
    bmi: float
    work_schedule: str
    smoker: bool
    baseline_activity: int
    stress_base: float
    adherence: float
    dropout_chance_weekly: float
    realign_chance_weekly: float
    sleep_hours: float
    activity_min: int
    diet_fv_g: int
    abstain: int
    stress: float
    water_glasses: int
    dropped_out: bool = False

def encode_state(compliance_flags: Dict[str, int]) -> int:
    bits = [
        compliance_flags.get("sleep", 0),
        compliance_flags.get("activity", 0),
        compliance_flags.get("diet", 0),
        compliance_flags.get("habit", 0),
    ]
    idx = bits[0] + (bits[1] << 1) + (bits[2] << 2) + (bits[3] << 3)
    return idx

def who_compliance(u: User) -> Dict[str, int]:
    return {
        "sleep": int(WHO["sleep_min"] <= u.sleep_hours <= WHO["sleep_max"]),
        "activity": int(u.activity_min >= WHO["activity_min"]),
        "diet": int(u.diet_fv_g >= WHO["diet_fv_min"]),
        "habit": int(u.abstain >= WHO["habit_abstain"]),
        "water": int(u.water_glasses >= WHO["water_glasses"])
    }

def clip_user_state(u: User):
    u.sleep_hours = float(np.clip(u.sleep_hours, 3.0, 11.0))
    u.activity_min = int(np.clip(u.activity_min, 0, 180))
    u.diet_fv_g = int(np.clip(u.diet_fv_g, 0, 1200))
    u.abstain = int(np.clip(u.abstain, 0, 1))
    u.stress = float(np.clip(u.stress, 0.0, 10.0))
    u.water_glasses = int(np.clip(u.water_glasses, 0, 16))

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
    return -reward + penalty

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
    edge_reward = np.clip(10.0 - w_ij, -10.0, 10.0)
    shaped = reward + 0.25 * edge_reward
    return shaped, penalty, comp

def apply_daily_drift(u: User):
    u.activity_min = int(max(0, u.activity_min - 10 + np.random.randint(-5, 6)))
    u.diet_fv_g = int(max(0, u.diet_fv_g - 80 + int(np.random.normal(0, 40))))
    u.sleep_hours += np.random.normal(-0.2, 0.4)
    u.abstain = 0 if (np.random.rand() < (0.15 if u.smoker else 0.05)) else u.abstain
    u.stress = float(np.clip(u.stress + np.random.normal(0.1, 0.5) + (0.3 if u.work_schedule=='shift' else 0.0), 0, 10))
    u.water_glasses = max(0, u.water_glasses - 2 + int(np.random.normal(0, 1)))

def misreport(value: float, scale: float=0.15) -> float:
    return float(value + np.random.normal(0, scale * max(1.0, abs(value))))

# ---------------------------
# Agents (Q, SARSA, Greedy, Hill Climbing)
# ---------------------------
'''
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

    def recommend_topk(self, state: int, k: int = 3):
        order = np.argsort(-self.Q[state])
        return [int(i) for i in order[:k]]
'''

# Import the tuned agent
from tuned_qlearning_agent import TunedQLearningAgent, TQLConfig

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

class GreedyAgent:
    def select_action(self, state: int, user_snapshot: User) -> int:
        est_rewards = []
        for ai, a in enumerate(ACTIONS):
            u = User(**vars(user_snapshot))
            d = action_effects(u, a)
            u.sleep_hours += d["sleep"]; u.activity_min += int(d["activity"])
            u.diet_fv_g += int(d["diet"]); u.abstain = max(u.abstain, int(d["habit"]))
            u.stress += d["stress"]; u.water_glasses += int(d["water"])
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
            d = action_effects(u, ACTIONS[aidx])
            u.sleep_hours += d["sleep"]; u.activity_min += int(d["activity"])
            u.diet_fv_g += int(d["diet"]); u.abstain = max(u.abstain, int(d["habit"]))
            u.stress += d["stress"]; u.water_glasses += int(d["water"])
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

# ---------------------------
# Simulation & Logging
# ---------------------------

@dataclass
class SimulationConfig:
    days: int = 365
    alpha: float = 0.25
    gamma: float = 0.97
    epsilon: float = 0.12

@dataclass
class EpisodeLog:
    records: List[Dict] = field(default_factory=list)
    def append(self, d: Dict):
        self.records.append(d)
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)

class DigitalTwinAnalysis:
    def __init__(self, population_df: pd.DataFrame, initial_df: pd.DataFrame, cfg: SimulationConfig):
        pop = population_df.set_index("user_id")
        init = initial_df.set_index("user_id")
        users = []
        for uid in sorted(pop.index.intersection(init.index)):
            p = pop.loc[uid]; s = init.loc[uid]
            users.append(User(
                user_id=int(uid),
                age=int(p["age"]), sex=str(p["sex"]), bmi=float(p["bmi"]),
                work_schedule=str(p["work_schedule"]), smoker=bool(int(p["smoker"])),
                baseline_activity=int(p["baseline_activity"]), stress_base=float(p["stress_base"]),
                adherence=float(p["adherence"]),
                dropout_chance_weekly=float(p["dropout_chance_weekly"]),
                realign_chance_weekly=float(p["realign_chance_weekly"]),
                sleep_hours=float(s["sleep_hours"]), activity_min=int(s["activity_min"]),
                diet_fv_g=int(s["diet_fv_g"]), abstain=int(s["abstain"]),
                stress=float(s["stress"]), water_glasses=int(s["water_glasses"])
            ))
        self.population: List[User] = users
        self.cfg = cfg

        # Agents
        self.q = TunedQLearningAgent(TQLConfig(n_states=16, n_actions=len(ACTIONS)))
        self.sarsa = SARSAAgent(alpha=cfg.alpha, gamma=cfg.gamma, epsilon=cfg.epsilon)
        self.greedy = GreedyAgent()
        self.hc = HillClimbingAgent()

        # Logs
        self.log_q = EpisodeLog()
        self.log_sarsa = EpisodeLog()
        self.log_g = EpisodeLog()
        self.log_hc = EpisodeLog()
        
        # Timings
        self.timings = {"Tuned-Q": 0.0, "SARSA": 0.0, "Greedy": 0.0, "HC": 0.0, "overall_wall_seconds": 0.0}

    def maybe_dropout_or_realign(self, u: User, day: int):
        if day % 7 == 0 and day > 0:
            if not u.dropped_out and (np.random.rand() < u.dropout_chance_weekly):
                u.dropped_out = True
            elif u.dropped_out and (np.random.rand() < u.realign_chance_weekly):
                u.dropped_out = False

    def step_user(self, u: User, action: str):
        d = action_effects(u, action)
        u.sleep_hours += d["sleep"]; u.activity_min += int(d["activity"])
        u.diet_fv_g += int(d["diet"]); u.abstain = max(u.abstain, int(d["habit"]))
        u.stress += d["stress"]; u.water_glasses += int(d["water"])
        clip_user_state(u)
        shaped, pen, comp = reward_and_penalty(u, action)
        observed = misreport(shaped - pen, scale=0.15)
        apply_daily_drift(u); clip_user_state(u)
        return shaped, pen, comp, observed

    def simulate(self):
        wall_start = time.perf_counter()
        for day in range(self.cfg.days):
            for u in self.population:
                self.maybe_dropout_or_realign(u, day)
                prev_flags = who_compliance(u)
                state = encode_state(prev_flags)

                if u.dropped_out:
                    chosen_q = np.random.randint(len(ACTIONS))
                    chosen_s = np.random.randint(len(ACTIONS))
                    chosen_g = np.random.randint(len(ACTIONS))
                    chosen_h = np.random.randint(len(ACTIONS))
                else:
                    t0 = time.perf_counter()
                    chosen_q = self.q.select_action(state)
                    r_q, pen_q, _, obs_q = self.step_user(User(**vars(u)), ACTIONS[chosen_q])
                    next_state_q = encode_state(who_compliance(User(**vars(u))))
                    self.timings["Tuned-Q"] += time.perf_counter() - t0

                    t0 = time.perf_counter()
                    chosen_s = self.sarsa.select_action(state)
                    r_s, pen_s, _, obs_s = self.step_user(User(**vars(u)), ACTIONS[chosen_s])
                    next_state_s = encode_state(who_compliance(User(**vars(u))))
                    a_next = self.sarsa.select_action(next_state_s)
                    self.timings["SARSA"] += time.perf_counter() - t0

                    t0 = time.perf_counter()
                    chosen_g = self.greedy.select_action(state, User(**vars(u)))
                    self.timings["Greedy"] += time.perf_counter() - t0

                    t0 = time.perf_counter()
                    chosen_h = self.hc.select_action(state, User(**vars(u)))
                    self.timings["HC"] += time.perf_counter() - t0

                # Q (learn on copy, log), then apply Q to real twin
                uq = User(**vars(u))
                r_q, pen_q, comp_q, obs_q = self.step_user(uq, ACTIONS[chosen_q])
                next_state_q = encode_state(who_compliance(uq))
                r_eff_q = r_q - pen_q
                self.q.update(state, chosen_q, r_eff_q, next_state_q)
                self.log_q.append({
                    "day": day, "user_id": u.user_id, "agent": "Q", "action": ACTIONS[chosen_q],
                    "reward": r_q, "penalty": pen_q, "reward_eff": r_eff_q, "observed_signal": obs_q,
                    **{f"comp_{k}": int(v) for k,v in who_compliance(uq).items()},
                    "dropped_out": int(u.dropped_out)
                })

                # SARSA
                us = User(**vars(u))
                r_s, pen_s, comp_s, obs_s = self.step_user(us, ACTIONS[chosen_s])
                next_state_s = encode_state(who_compliance(us))
                a_next = np.random.randint(len(ACTIONS)) if u.dropped_out else self.sarsa.select_action(next_state_s)
                r_eff_s = r_s - pen_s
                self.sarsa.update(state, chosen_s, r_eff_s, next_state_s, a_next)
                self.log_sarsa.append({
                    "day": day, "user_id": u.user_id, "agent": "SARSA", "action": ACTIONS[chosen_s],
                    "reward": r_s, "penalty": pen_s, "reward_eff": r_eff_s, "observed_signal": obs_s,
                    **{f"comp_{k}": int(v) for k,v in who_compliance(us).items()},
                    "dropped_out": int(u.dropped_out)
                })

                # Greedy
                ug = User(**vars(u))
                r_g, pen_g, comp_g, obs_g = self.step_user(ug, ACTIONS[chosen_g])
                self.log_g.append({
                    "day": day, "user_id": u.user_id, "agent": "Greedy", "action": ACTIONS[chosen_g],
                    "reward": r_g, "penalty": pen_g, "reward_eff": r_g - pen_g, "observed_signal": obs_g,
                    **{f"comp_{k}": int(v) for k,v in who_compliance(ug).items()},
                    "dropped_out": int(u.dropped_out)
                })

                # Hill Climbing
                uh = User(**vars(u))
                r_h, pen_h, comp_h, obs_h = self.step_user(uh, ACTIONS[chosen_h])
                self.log_hc.append({
                    "day": day, "user_id": u.user_id, "agent": "HC", "action": ACTIONS[chosen_h],
                    "reward": r_h, "penalty": pen_h, "reward_eff": r_h - pen_h, "observed_signal": obs_h,
                    **{f"comp_{k}": int(v) for k,v in who_compliance(uh).items()},
                    "dropped_out": int(u.dropped_out)
                })

                # Apply the Q-agent action to the *real* twin for closed-loop simulation
                _ = self.step_user(u, ACTIONS[chosen_q])

        self.timings["overall_wall_seconds"] = time.perf_counter() - wall_start

    # ---------------
    # Analysis / I/O
    # ---------------

    def _weekly_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        tmp = df.copy()
        tmp["week"] = (tmp["day"] // 7) + 1
        return tmp.groupby(["agent", "week"])["reward_eff"].agg(["mean","std","count"]).reset_index()

    @staticmethod
    def welch_t_test(sample_a: np.ndarray, sample_b: np.ndarray) -> Tuple[float, float]:
        a = np.asarray(sample_a, dtype=float)
        b = np.asarray(sample_b, dtype=float)
        ma, mb = a.mean(), b.mean()
        va, vb = a.var(ddof=1), b.var(ddof=1)
        na, nb = len(a), len(b)
        num = ma - mb
        den = math.sqrt(va/na + vb/nb + 1e-12)
        t = num / (den if den>0 else 1e-9)
        # one-sided p-value for H1: A > B via normal approx
        p = 0.5 * (1 - math.erf(t / math.sqrt(2)))
        return float(t), float(p)

    def make_recommendations(self) -> pd.DataFrame:
        recs = []
        for u in self.population:
            flags = who_compliance(u)
            s = encode_state(flags)
            topk = self.q.recommend_topk(s, k=3)
            recs.append({
                "user_id": u.user_id,
                "age": u.age,
                "sex": u.sex,
                "bmi": round(u.bmi,1),
                "work": u.work_schedule,
                "smoker": int(u.smoker),
                "adherence": round(u.adherence, 2),
                "top1": ACTIONS[topk[0]] if len(topk)>0 else None,
                "top2": ACTIONS[topk[1]] if len(topk)>1 else None,
                "top3": ACTIONS[topk[2]] if len(topk)>2 else None
            })
        return pd.DataFrame(recs)

    def run_and_report(self, outdir: Path):
        outdir.mkdir(parents=True, exist_ok=True)

        # Simulate & collect logs
        self.simulate()

        df_q = self.log_q.to_dataframe()
        df_sarsa = self.log_sarsa.to_dataframe()
        df_g = self.log_g.to_dataframe()
        df_hc = self.log_hc.to_dataframe()

        df_all = pd.concat([df_q, df_sarsa, df_g, df_hc], ignore_index=True)
        df_all.to_csv(outdir / "simulation_log.csv", index=False)

        weekly = self._weekly_stats(df_all)
        weekly.to_csv(outdir / "weekly_reward_stats.csv", index=False)

        # Plot weekly mean ± std for all four agents
        plt.figure()
        for agent in ["Q","SARSA","Greedy","HC"]:
            sub = weekly[weekly["agent"]==agent].sort_values("week")
            if len(sub)==0: 
                continue
            x = sub["week"].values; y = sub["mean"].values; sd = sub["std"].fillna(0).values
            plt.plot(x, y, label=agent)
            plt.fill_between(x, y - sd, y + sd, alpha=0.2)
        plt.xlabel("Week")
        plt.ylabel("Weekly mean effective reward")
        plt.title("Weekly Reward Evolution (mean ± std)")
        plt.legend(ncols=2)
        plt.tight_layout()
        plt.savefig(outdir / "weekly_reward_curves.png", dpi=200)
        plt.close()

        cum_by_user = df_all.groupby(["agent","user_id"])["reward_eff"].sum().reset_index()
        cum_by_user.to_csv(outdir / "cumulative_rewards_by_user.csv", index=False)

        # Welch t-tests: Q vs others
        results = {}
        q_vals = cum_by_user[cum_by_user["agent"]=="Q"]["reward_eff"].values
        for other in ["Greedy","HC","SARSA"]:
            o_vals = cum_by_user[cum_by_user["agent"]==other]["reward_eff"].values
            if len(q_vals)>0 and len(o_vals)>0:
                t, p = self.welch_t_test(q_vals, o_vals)
                results[f"Q_vs_{other}"] = {"t_stat": float(t), "p_one_sided": float(p)}
        with open(outdir / "ttest_results.json", "w") as f:
            json.dump(results, f, indent=2)

        recs = self.make_recommendations()
        recs.to_csv(outdir / "recommendations_next_day.csv", index=False)

        # Timings
        with open(outdir / "timings.json", "w") as f:
            json.dump(self.timings, f, indent=2)

        # Console summary
        def fmt(mean, std): return f"{mean:.2f} ± {std:.2f}"
        summary = df_all.groupby("agent")["reward_eff"].agg(["mean","std"]).reset_index()
        print("\n=== Summary (All Days × Users) ===")
        for _, row in summary.iterrows():
            print(f"{row['agent']:>6}: mean±std reward_eff = {fmt(row['mean'], row['std'])}")
        print("\n=== Welch's t-test (one-sided, H1: Q > Other) ===")
        for k,v in results.items():
            print(f"{k}: t = {v['t_stat']:.3f}, p = {v['p_one_sided']:.4f}")
        print("\n=== Timings (seconds, cumulative selection/update) ===")
        for k,v in self.timings.items():
            print(f"{k}: {v:.4f}")
        print(f"\nArtifacts saved to: {outdir.resolve()}")

def load_inputs(indir: Path):
    pop = pd.read_csv(indir / "population_demographics.csv")
    init = pd.read_csv(indir / "initial_states.csv")
    # metadata is optional
    meta = None
    meta_path = indir / "metadata.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
    return pop, init, meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=str, default="outputs_gen")
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--gamma", type=float, default=0.97)
    ap.add_argument("--epsilon", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=42, help="global RNG seed for reproducibility")
    ap.add_argument("--outdir", type=str, default="outputs")
    args = ap.parse_args()

    # Reproducibility
    np.random.seed(args.seed)

    indir = Path(args.indir)
    outdir = Path(args.outdir)

    pop, init, meta = load_inputs(indir)
    cfg = SimulationConfig(days=args.days, alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon)

    runner = DigitalTwinAnalysis(pop, init, cfg)
    runner.run_and_report(outdir)

if __name__ == "__main__":
    main()
