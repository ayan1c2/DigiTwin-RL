"""
Digital Twin Data Generator (standalone)
---------------------------------------
Creates the input CSVs for the RL analysis pipeline.

Outputs (in --outdir, default: outputs_gen):
- population_demographics.csv
- initial_states.csv
- metadata.json  (seed, actions, WHO thresholds, etc.)

Usage:
  python digital_twin_data_generator.py --users 25 --seed 42 --outdir outputs_gen
  python digital_twin_data_generator.py --users 50 --seed 42 --outdir outputs_gen
  python digital_twin_data_generator.py --users 75 --seed 42 --outdir outputs_gen
  python digital_twin_data_generator.py --users 100 --seed 42 --outdir outputs_gen

This file is derived from the original single-file simulator, split so that
RL actors (Q-learning, SARSA, Hill Climbing, Greedy) can read the CSVs to
perform recommendations and analysis separately.
"""

import json
import argparse
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path

# ---------------------------
# Canonical definitions
# ---------------------------

ACTIONS = [
    "sleep_early",          # target 7-9h
    "walk_30",              # 30 min activity
    "eat_veg_400",          # 400g fruits+veg
    "no_tobacco_alcohol",   # abstain
    "meditate_10",          # stress reduction
    "drink_water_8"         # hydration
]

WHO = {
    "sleep_min": 7.0,
    "sleep_max": 9.0,
    "activity_min": 30,      # minutes/day
    "diet_fv_min": 400,      # grams/day
    "habit_abstain": 1,      # 1=abstain, 0=not
    "water_glasses": 8
}

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

    # Initial internal states (one row per user in initial_states.csv)
    sleep_hours: float = 6.5
    activity_min: int = 10
    diet_fv_g: int = 150
    abstain: int = 0
    stress: float = 5.0
    water_glasses: int = 4

def generate_population(n_users=50, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    users = []
    for uid in range(n_users):
        age = int(np.clip(rng.normal(40, 12), 18, 75))
        sex = rng.choice(['F','M'])
        bmi = float(np.clip(rng.normal(26.0, 4.5), 18.5, 40.0))
        work_schedule = rng.choice(['9-5','shift'], p=[0.7,0.3])
        smoker = bool(rng.random() < 0.25)
        baseline_activity = int(max(0, rng.normal(20 if smoker else 30, 15)))
        stress_base = float(np.clip(rng.normal(5.0, 2.0), 0, 10))
        adherence = float(np.clip(rng.beta(2,2), 0.05, 0.95))
        dropout_chance_weekly = float(np.clip(rng.beta(1.5, 12), 0.0, 0.25))
        realign_chance_weekly = float(np.clip(rng.beta(1.5, 6), 0.05, 0.7))
        users.append(User(
            user_id=uid, age=age, sex=sex, bmi=bmi, work_schedule=work_schedule,
            smoker=smoker, baseline_activity=baseline_activity, stress_base=stress_base,
            adherence=adherence, dropout_chance_weekly=dropout_chance_weekly,
            realign_chance_weekly=realign_chance_weekly
        ))
    return users

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="outputs_gen")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    users = generate_population(args.users, rng=rng)

    # Export population demographics
    pop_rows = []
    for u in users:
        pop_rows.append({
            "user_id": u.user_id, "age": u.age, "sex": u.sex, "bmi": u.bmi,
            "work_schedule": u.work_schedule, "smoker": int(u.smoker),
            "baseline_activity": u.baseline_activity, "stress_base": u.stress_base,
            "adherence": u.adherence, "dropout_chance_weekly": u.dropout_chance_weekly,
            "realign_chance_weekly": u.realign_chance_weekly
        })
    pd.DataFrame(pop_rows).to_csv(outdir / "population_demographics.csv", index=False)

    # Export initial internal states (one row per user)
    init_rows = []
    for u in users:
        init_rows.append({
            "user_id": u.user_id,
            "sleep_hours": u.sleep_hours,
            "activity_min": u.activity_min,
            "diet_fv_g": u.diet_fv_g,
            "abstain": u.abstain,
            "stress": u.stress,
            "water_glasses": u.water_glasses
        })
    pd.DataFrame(init_rows).to_csv(outdir / "initial_states.csv", index=False)

    # Metadata (for reproducibility & documentation)
    meta = {
        "seed": args.seed,
        "actions": ACTIONS,
        "WHO": WHO,
        "notes": "Use rl_actors_analysis.py to read these CSVs and run RL actors."
    }
    with open(outdir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved: {outdir.resolve()}")
    print(" - population_demographics.csv")
    print(" - initial_states.csv")
    print(" - metadata.json")

if __name__ == "__main__":
    main()
