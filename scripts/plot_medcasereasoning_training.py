#!/usr/bin/env python3
"""Plot MedCaseReasoning training accuracy from W&B exported CSV."""

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
csv_path = Path("/admin/home/nikhil/med-lm-envs/wandb_export_2026-01-29T03_37_39.838-05_00.csv")
df = pd.read_csv(csv_path)

# Clean column names
df.columns = ['Step', 'pass@1', 'pass@1_min', 'pass@1_max']

# Create plot
fig, ax = plt.subplots(figsize=(8, 5))

# Plot main line (plasma colors - same as other datasets)
ax.plot(df['Step'], df['pass@1'], 'o-', color=PLASMA_COLORS[1],  # 7e03a8 purple
        linewidth=2, markersize=5, alpha=0.8, label='MedCaseReasoning')

# Add trend line (moving average)
window = 3
df['smoothed'] = df['pass@1'].rolling(window=window, center=True).mean()
ax.plot(df['Step'], df['smoothed'], '--', color=PLASMA_COLORS[3],  # f89540 orange
        linewidth=2, alpha=0.8, label=f'Smoothed (window={window})')

# For summary stats
start_acc = df['pass@1'].iloc[0]
end_acc = df['pass@1'].iloc[-1]
improvement = end_acc - start_acc

ax.set_xlabel('Training Step')
ax.set_ylabel('Pass@1 Accuracy')
ax.set_title('MedCaseReasoning Test Accuracy')
ax.legend(loc='lower right', framealpha=0.9)
ax.set_ylim(0.1, 0.4)
ax.set_xlim(-10, 350)

ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

plt.tight_layout()

# Save
output_dir = Path("/admin/home/nikhil/med-lm-envs/plots")
output_dir.mkdir(exist_ok=True)
plt.savefig(output_dir / 'medcasereasoning_training.png')
plt.savefig(output_dir / 'medcasereasoning_training.pdf')
print(f"Saved to {output_dir / 'medcasereasoning_training.png'}")
print(f"Saved to {output_dir / 'medcasereasoning_training.pdf'}")

# Print summary
print(f"\nTraining Summary:")
print(f"  Start accuracy: {start_acc:.1%}")
print(f"  End accuracy:   {end_acc:.1%}")
print(f"  Improvement:    +{improvement:.1%}")
print(f"  Steps:          {df['Step'].iloc[-1]}")
