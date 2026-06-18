# app/core.py
from dataclasses import dataclass
from typing import Dict
import numpy as np

ACTIONS = [
    "sleep_early",
    "walk_30",
    "eat_veg_400",
    "no_tobacco_alcohol",
    "meditate_10",
    "drink_water_8",
]

WHO = {
    "sleep_min": 7.0,
    "sleep_max": 9.0,
    "activity_min": 30,   # minutes/day
    "diet_fv_min": 400,   # grams/day
    "habit_abstain": 1,   # 1=abstain, 0=not
    "water_glasses": 8
}

@dataclass
class User:
    user_id: int = 0
    age: int = 30
    sex: str = "M"
    bmi: float = 25.0
    work_schedule: str = "9-5"
    smoker: bool = False
    baseline_activity: int = 20
    stress_base: float = 5.0
    adherence: float = 0.8
    dropout_chance_weekly: float = 0.0
    realign_chance_weekly: float = 0.0
    # live snapshot (from form)
    sleep_hours: float = 7.0
    activity_min: int = 30
    diet_fv_g: int = 400
    abstain: int = 1
    stress: float = 5.0
    water_glasses: int = 8

def clip_user_state(u: User):
    u.sleep_hours = float(np.clip(u.sleep_hours, 3.0, 11.0))
    u.activity_min = int(np.clip(u.activity_min, 0, 180))
    u.diet_fv_g = int(np.clip(u.diet_fv_g, 0, 1200))
    u.abstain = int(np.clip(u.abstain, 0, 1))
    u.stress = float(np.clip(u.stress, 0.0, 10.0))
    u.water_glasses = int(np.clip(u.water_glasses, 0, 16))

def who_compliance(u: User) -> Dict[str, int]:
    return {
        "sleep": int(WHO["sleep_min"] <= u.sleep_hours <= WHO["sleep_max"]),
        "activity": int(u.activity_min >= WHO["activity_min"]),
        "diet": int(u.diet_fv_g >= WHO["diet_fv_min"]),
        "habit": int(u.abstain >= WHO["habit_abstain"]),
        "water": int(u.water_glasses >= WHO["water_glasses"]),
    }

def encode_state(flags: Dict[str, int]) -> int:
    # state := [sleep, activity, diet, habit] -> 4-bit int (0..15)
    bits = [flags.get("sleep",0), flags.get("activity",0), flags.get("diet",0), flags.get("habit",0)]
    return bits[0] + (bits[1] << 1) + (bits[2] << 2) + (bits[3] << 3)
