"""
Digital Twin Q-Learning Grid Search Runner
------------------------------------------
This script reads the digital twin Q-learning environment (q_only version)
and performs a grid search over hyperparameters (alpha, gamma, epsilon).
It reports results and saves the best Q-table + metadata.

Usage:
  python digital_twin_qlearning_gridsearch.py --days 365 --users 25
"""

import argparse
import random
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path

# import core simulation from q_only version
import digital_twin_qlearning_q_only as core

def full_grid_search(days=365, users=25,
                     alphas=(0.05,0.1,0.15,0.2,0.25,0.3),
                     gammas=(0.90,0.95,0.97,0.99),
                     epsilons=(0.05,0.1,0.12,0.2),
                     seeds=(42,1337,77)) -> dict:
    results = []
    best = {"score": -1e18}
    for alpha in alphas:
        for gamma in gammas:
            for eps in epsilons:
                scores = []
                for s in seeds:
                    random.seed(s); np.random.seed(s)
                    cfg = core.SimulationConfig(days=days, users=users, alpha=alpha, gamma=gamma, epsilon=eps)
                    sim = core.DigitalTwinSim(cfg)
                    score = sim.simulate()
                    scores.append(score)
                mean_score = float(np.mean(scores))
                std_score = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
                rec = {"alpha": alpha, "gamma": gamma, "epsilon": eps,
                       "mean_score": mean_score, "std_score": std_score}
                results.append(rec)
                if mean_score > best["score"]:
                    best = {"alpha": alpha, "gamma": gamma, "epsilon": eps,
                            "score": mean_score, "std": std_score,
                            "Q": sim.agent.Q.copy()}
    return {"grid_results": results, "best": best}

def save_best(best: dict, outdir="models", filename="best_q_grid.pkl"):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    payload = {
        "ACTIONS": core.ACTIONS,
        "Q": best["Q"],
        "params": {"alpha": best["alpha"], "gamma": best["gamma"], "epsilon": best["epsilon"]}
    }
    with open(Path(outdir)/filename, "wb") as f:
        pickle.dump(payload, f)
    with open(Path(outdir)/"best_q_grid_meta.json", "w") as f:
        json.dump({"best": best}, f, indent=2)
    print(f"Saved best Q-agent to {Path(outdir)/filename}")
    print(f"Metadata saved to {Path(outdir)/'best_q_grid_meta.json'}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--users", type=int, default=25)
    args = ap.parse_args()

    out = full_grid_search(days=args.days, users=args.users)

    df = pd.DataFrame(out["grid_results"]).sort_values(["mean_score"], ascending=False)
    print("\n=== Grid Search Results (Top 10 configs) ===")
    print(df.head(10).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\nBest configuration:")
    print(out["best"])

    save_best(out["best"])

if __name__ == "__main__":
    main()
