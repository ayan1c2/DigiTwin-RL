
#############################
'''
| Metric                       | Measure                                                                                                                                               | Target   | Interpretation                   |
| :--------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------- | :------- | :------------------------------- |
| **KL Divergence**            | Compares the shape of simulated vs. real variable distributions. A value `< 0.1` means the generated data distributions resemble real-world patterns. | `< 0.1`  | Low divergence → realistic shape |
| **Wasserstein Distance**     | Measures how much the simulated samples “shift” compared to real data. Smaller values imply closer alignment.                                         | `< 0.1`  | Low drift → stable generation    |
| **Correlation Similarity**   | Compares pairwise correlations between features in both datasets using Pearson similarity.                                                            | `> 0.8`  | Preserved internal structure     |
| **Overall Similarity Score** | Composite realism index combining all metrics to evaluate synthetic fidelity.                                                                         | `> 0.75` | High fidelity synthetic data     |
'''
######################


import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, entropy, wasserstein_distance, pearsonr
import seaborn as sns
import matplotlib.pyplot as plt

# Read dataset
data = pd.read_csv("./outputs_gen/initial_states.csv")

# Keep only numeric columns (avoid constant or categorical problems)
data = data.select_dtypes(include=[np.number]).dropna(axis=1, how='all')

# ---- 2. Generate Simulated "Real" Baseline ----
np.random.seed(42)
real_data = pd.DataFrame({
    'sleep_hours': np.random.normal(7, 1, len(data)),
    'activity_min': np.random.normal(30, 10, len(data)),
    'diet_fv_g': np.random.normal(200, 50, len(data)),
    'abstain': np.random.choice([0, 1], len(data)),
    'stress': np.random.normal(5, 1.5, len(data)),
    'water_glasses': np.random.normal(6, 2, len(data))
})

# Restrict to overlapping numeric columns
common_cols = list(set(real_data.columns).intersection(data.columns))
real_data = real_data[common_cols]
data = data[common_cols]

# ---- 3. Statistical Distribution Metrics ----
eps = 1e-9
metrics = []

for col in common_cols:
    p_hist, bins = np.histogram(real_data[col], bins=20, density=True)
    q_hist, _ = np.histogram(data[col], bins=bins, density=True)
    p_hist = p_hist + eps
    q_hist = q_hist + eps
    p_hist /= p_hist.sum()
    q_hist /= q_hist.sum()

    kl = entropy(p_hist, q_hist)
    wd = wasserstein_distance(real_data[col], data[col])
    metrics.append({'Variable': col, 'KL Divergence': kl, 'Wasserstein Distance': wd})

# ---- 4. Correlation Similarity ----
def upper_triangle_flatten(corr):
    mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
    vals = corr.where(mask).stack().values
    return vals[~np.isnan(vals)]

corr_real = real_data.corr()
corr_synth = data.corr()

common_cols = corr_real.columns.intersection(corr_synth.columns)
corr_real = corr_real.loc[common_cols, common_cols]
corr_synth = corr_synth.loc[common_cols, common_cols]

corr_real_flat = upper_triangle_flatten(corr_real)
corr_synth_flat = upper_triangle_flatten(corr_synth)

if len(corr_real_flat) > 0 and len(corr_synth_flat) > 0:
    min_len = min(len(corr_real_flat), len(corr_synth_flat))
    corr_similarity = np.corrcoef(
        corr_real_flat[:min_len],
        corr_synth_flat[:min_len]
    )[0, 1]
else:
    corr_similarity = np.nan

# ---- 5. Overall Similarity Score ----
kl_mean = np.nanmean([m['KL Divergence'] for m in metrics])
wd_mean = np.nanmean([m['Wasserstein Distance'] for m in metrics])
if np.isnan(corr_similarity):
    overall_similarity = np.nan
else:
    overall_similarity = np.clip((1 - kl_mean) * (1 - wd_mean) * corr_similarity, 0, 1)

# ---- 6. Results Table ----
report = pd.DataFrame({
    'Metric': ['KL Divergence', 'Wasserstein Distance', 'Correlation Similarity', 'Overall Similarity Score'],
    'Measure': ['Distribution shape similarity', 'Shift between real and synthetic samples',
                'Structure of variable relationships', 'Combined realism index'],
    'Target': ['< 0.1', '< 0.1', '> 0.8', '> 0.75'],
    'Value': [round(kl_mean, 3), round(wd_mean, 3),
              round(corr_similarity, 3) if not np.isnan(corr_similarity) else 'N/A',
              round(overall_similarity, 3) if not np.isnan(overall_similarity) else 'N/A'],
    'Interpretation': [
        'Realistic variable distributions' if kl_mean < 0.1 else 'Significant divergence',
        'Minimal distribution drift' if wd_mean < 0.1 else 'Potential distribution drift',
        'Preserved relationships' if (not np.isnan(corr_similarity) and corr_similarity > 0.8)
            else 'Weak or undefined structure',
        'High-fidelity synthetic data' if (not np.isnan(overall_similarity) and overall_similarity > 0.75)
            else 'Low realism'
    ]
})

# ---- 7. Visualization ----
plt.figure(figsize=(8, 5))
sns.barplot(x='Metric', y='Value', data=report[report['Value'] != 'N/A'])
plt.title("Synthetic Data Statistical Fidelity")
plt.ylim(0, 1)
plt.axhline(0.75, color='gray', linestyle='--', label='Realism Threshold')
plt.legend()
plt.tight_layout()
plt.savefig("realism_report.png", dpi=300)
plt.show()

# ---- 8. Display Results ----
print("\n=== Statistical Realism Report ===")
print(report.to_string(index=False))
