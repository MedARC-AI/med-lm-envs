#!/usr/bin/env python3
"""Plot MedCaseReasoning reward curve from W&B exported CSV."""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Plasma colormap colors
PLASMA_COLORS = ['#0d0887', '#7e03a8', '#cc4778', '#f89540', '#f0f921']

# Style settings for publication
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (8, 5),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Load data
csv_path = Path("/admin/home/nikhil/med-lm-envs/wandb_export_2026-01-29T03_37_48.290-05_00.csv")
df = pd.read_csv(csv_path)

# Clean column names
df.columns = ['Step', 'reward_mean', 'reward_min', 'reward_max']

# Create plot
fig, ax = plt.subplots(figsize=(8, 5))

# Plot raw data with low alpha (plasma colors - same as other datasets)
ax.plot(df['Step'], df['reward_mean'], '-', color=PLASMA_COLORS[1],  # 7e03a8 purple
        linewidth=0.8, alpha=0.3, label='_nolegend_')

# Add smoothed trend line (exponential moving average)
span = 20  # smoothing window
df['smoothed'] = df['reward_mean'].ewm(span=span).mean()
ax.plot(df['Step'], df['smoothed'], '-', color=PLASMA_COLORS[3],  # f89540 orange
        linewidth=2.5, label='MedCaseReasoning (smoothed)')

# Calculate improvement for summary
start_reward = df['smoothed'].iloc[:10].mean()
end_reward = df['smoothed'].iloc[-10:].mean()
improvement = end_reward - start_reward

ax.set_title('MedCaseReasoning Training Reward')

ax.set_xlabel('Training Step')
ax.set_ylabel('Mean Reward')
ax.legend(loc='lower right', framealpha=0.9)
ax.set_ylim(0.0, 0.4)
ax.set_xlim(0, 335)

plt.tight_layout()

# Save
output_dir = Path("/admin/home/nikhil/med-lm-envs/plots")
output_dir.mkdir(exist_ok=True)
plt.savefig(output_dir / 'medcasereasoning_reward.png')
plt.savefig(output_dir / 'medcasereasoning_reward.pdf')
print(f"Saved to {output_dir / 'medcasereasoning_reward.png'}")
print(f"Saved to {output_dir / 'medcasereasoning_reward.pdf'}")

# Print summary
print(f"\nReward Summary:")
print(f"  Start (smoothed): {start_reward:.3f}")
print(f"  End (smoothed):   {end_reward:.3f}")
print(f"  Improvement:      +{improvement:.3f}")
print(f"  Raw min:          {df['reward_mean'].min():.3f}")
print(f"  Raw max:          {df['reward_mean'].max():.3f}")
