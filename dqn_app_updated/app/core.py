# app/core.py
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

# ---------------------------
# Problem Setup (WHO-aligned proxies)
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

WHO_ENV = {
    "air_quality_max": 50,     # AQI safe
    "noise_max": 55,           # dB daytime
    "light_max": 5,            # lux at night for sleep environments
    "eco_anxiety_max": 5       # target keep <= moderate
}

# Reward scale factors (used by training; UI uses for display)
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
    "eco": 1.2,
    "air": 0.6,
    "noise": 0.4,
    "light": 0.4,
    "heatwave": 0.5,
    "flood": 0.6
}

# Penalties (lambda_j). Tuneable in training.
LAMBDA = {
    "sleep": 4.0,
    "activity": 3.5,
    "diet": 3.0,
    "habit": 5.0,
    "water": 1.5,
    "mental": 3.5,
    "screen": 1.0,
    "social": 1.0,
    "eco": 2.5,
    "air": 1.5,
    "noise": 1.2,
    "light": 1.2,
    "heatwave": 1.8,
    "flood": 2.0
}

# ---------------------------
# Data model
# ---------------------------

@dataclass
class User:
    # identity/profile
    user_id: int = 0
    age: int = 30
    sex: str = "M"                      # "M" or "F"
    bmi: float = 25.0
    work_schedule: str = "9-5"          # "9-5" or "shift"
    smoker: bool = False

    # engagement parameters (optional; used in training)
    adherence: float = 0.8
    dropout_chance_weekly: float = 0.0
    realign_chance_weekly: float = 0.0

    # live snapshot (from form) - behaviors
    sleep_hours: float = 7.0
    activity_min: int = 30
    diet_fv_g: int = 400
    abstain: int = 1
    stress: float = 5.0
    mindfulness_min: int = 10
    water_glasses: int = 8
    screen_hours: float = 2.0
    social_min: int = 15

    # environment (optional, defaults used if not provided)
    air_aqi: float = 30.0
    noise_db: float = 50.0
    light_lux: float = 3.0
    eco_anxiety: float = 4.0
    heatwave: int = 0
    flood: int = 0


# ---------------------------
# Normalization helpers (simple, stable ranges)
# ---------------------------

def _norm(x: float, lo: float, hi: float) -> float:
    x = float(np.clip(x, lo, hi))
    if hi == lo:
        return 0.0
    return (x - lo) / (hi - lo)

def clip_user_state(u: User):
    # behaviors
    u.sleep_hours = float(np.clip(u.sleep_hours, 0.0, 24.0))
    u.activity_min = int(np.clip(u.activity_min, 0, 360))
    u.diet_fv_g = int(np.clip(u.diet_fv_g, 0, 2000))
    u.abstain = int(np.clip(u.abstain, 0, 1))
    u.stress = float(np.clip(u.stress, 0.0, 10.0))
    u.mindfulness_min = int(np.clip(u.mindfulness_min, 0, 240))
    u.water_glasses = int(np.clip(u.water_glasses, 0, 30))
    u.screen_hours = float(np.clip(u.screen_hours, 0.0, 24.0))
    u.social_min = int(np.clip(u.social_min, 0, 600))

    # profile
    u.age = int(np.clip(u.age, 18, 100))
    u.bmi = float(np.clip(u.bmi, 12.0, 60.0))
    u.adherence = float(np.clip(u.adherence, 0.0, 1.0))

    # env
    u.air_aqi = float(np.clip(u.air_aqi, 0.0, 500.0))
    u.noise_db = float(np.clip(u.noise_db, 0.0, 120.0))
    u.light_lux = float(np.clip(u.light_lux, 0.0, 500.0))
    u.eco_anxiety = float(np.clip(u.eco_anxiety, 0.0, 10.0))
    u.heatwave = int(np.clip(u.heatwave, 0, 1))
    u.flood = int(np.clip(u.flood, 0, 1))


# ---------------------------
# Compliance + state encoding
# ---------------------------

def who_compliance(u: User) -> Dict[str, int]:
    """Binary compliance flags used for monitoring (and discrete state encoding)."""
    return {
        "sleep": int(WHO["sleep_min"] <= u.sleep_hours <= WHO["sleep_max"]),
        "activity": int(u.activity_min >= WHO["activity_min"]),
        "diet": int(u.diet_fv_g >= WHO["diet_fv_min"]),
        "habit": int(u.abstain >= WHO["habit_abstain"]),
        "mental": int((u.stress <= WHO["stress_max"]) and (u.mindfulness_min >= WHO["mindfulness_minutes"])),
        "water": int(u.water_glasses >= WHO["water_glasses"]),
        "screen": int(u.screen_hours <= WHO["screen_max_hours"]),
        "social": int(u.social_min >= WHO["social_min_minutes"]),
    }

def encode_state_discrete(flags: Dict[str, int]) -> int:
    """8-bit compliance state -> int in [0,255]."""
    order = ["sleep","activity","diet","habit","mental","water","screen","social"]
    s = 0
    for i, k in enumerate(order):
        s |= (int(flags.get(k, 0)) & 1) << i
    return s

def encode_state_rich(u: User) -> np.ndarray:
    """
    26D continuous feature vector used by DQN/TD.
    Layout:
    - 8 compliance bits
    - 8 normalized behavioral measures
    - 6 normalized env measures
    - 4 normalized/profile measures
    Total = 26
    """
    flags = who_compliance(u)
    bits = np.array([flags[k] for k in ["sleep","activity","diet","habit","mental","water","screen","social"]], dtype=np.float32)

    beh = np.array([
        _norm(u.sleep_hours, 0.0, 12.0),
        _norm(u.activity_min, 0.0, 180.0),
        _norm(u.diet_fv_g, 0.0, 1200.0),
        float(u.abstain),
        _norm(u.stress, 0.0, 10.0),
        _norm(u.mindfulness_min, 0.0, 60.0),
        _norm(u.water_glasses, 0.0, 16.0),
        _norm(u.screen_hours, 0.0, 8.0),
    ], dtype=np.float32)

    env = np.array([
        _norm(u.air_aqi, 0.0, 150.0),
        _norm(u.noise_db, 0.0, 90.0),
        _norm(u.light_lux, 0.0, 50.0),
        _norm(u.eco_anxiety, 0.0, 10.0),
        float(u.heatwave),
        float(u.flood),
    ], dtype=np.float32)

    prof = np.array([
        _norm(u.age, 18.0, 80.0),
        1.0 if str(u.sex).upper().startswith("M") else 0.0,
        _norm(u.bmi, 18.5, 40.0),
        float(u.adherence),
    ], dtype=np.float32)

    x = np.concatenate([bits, beh, env, prof], axis=0)
    assert x.shape[0] == 26, f"Expected 26 features, got {x.shape[0]}"
    return x
