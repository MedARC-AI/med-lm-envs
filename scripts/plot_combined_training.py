#!/usr/bin/env python3
"""
Combined plot showing test accuracy and training reward for all three datasets.
"""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Plasma colormap colors - distinct colors for each dataset
COLORS = {
    'medcalc': '#0d0887',      # Dark blue/purple
    'medmcqa': '#7e03a8',      # Purple
    'medcase': '#cc4778',      # Pink/magenta
}

LABELS = {
    'medcalc': 'MedCalc-Bench-Verified',
    'medmcqa': 'MedMCQA',
    'medcase': 'MedCaseReasoning',
}

# Style settings for publication
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# CSV file paths
TEST_CSVS = {
    'medcalc': "/admin/home/nikhil/med-lm-envs/wandb_export_2026-01-29T01_13_09.334-05_00.csv",
    'medmcqa': "/admin/home/nikhil/med-lm-envs/wandb_export_2026-01-29T02_22_42.275-05_00.csv",
    'medcase': "/admin/home/nikhil/med-lm-envs/wandb_export_2026-01-29T03_37_39.838-05_00.csv",
}

REWARD_CSVS = {
    'medcalc': "/admin/home/nikhil/med-lm-envs/wandb_export_2026-01-29T01_22_36.871-05_00.csv",
    'medmcqa': "/admin/home/nikhil/med-lm-envs/wandb_export_2026-01-29T02_22_54.385-05_00.csv",
    'medcase': "/admin/home/nikhil/med-lm-envs/wandb_export_2026-01-29T03_37_48.290-05_00.csv",
}


def load_test_data(csv_path: str) -> pd.DataFrame:
    """Load test accuracy data."""
    df = pd.read_csv(csv_path)
    df.columns = ['Step', 'pass@1', 'pass@1_min', 'pass@1_max']
    return df


def load_reward_data(csv_path: str) -> pd.DataFrame:
    """Load reward data."""
    df = pd.read_csv(csv_path)
    df.columns = ['Step', 'reward_mean', 'reward_min', 'reward_max']
    return df


def main():
    # Create figure with 2x3 grid of subplots
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    datasets = ['medcalc', 'medmcqa', 'medcase']
    
    # Y-axis limits for each dataset (test accuracy)
    test_ylims = {
        'medcalc': (0.3, 0.7),
        'medmcqa': (0.5, 0.65),
        'medcase': (0.1, 0.4),
    }
    
    # Y-axis limits for each dataset (reward)
    reward_ylims = {
        'medcalc': (0.2, 0.9),
        'medmcqa': (0.4, 0.9),
        'medcase': (0.0, 0.4),
    }
    
    # =========================================
    # Row 1: Test Accuracy plots
    # =========================================
    for col, dataset_key in enumerate(datasets):
        ax = axes[0, col]
        df = load_test_data(TEST_CSVS[dataset_key])
        color = COLORS[dataset_key]
        label = LABELS[dataset_key]
        
        # Plot data points
        ax.plot(df['Step'], df['pass@1'], 'o-', color=color, 
                linewidth=2, markersize=4, alpha=0.5, label='Data')
        
        # Add smoothed trend
        window = 3
        df['smoothed'] = df['pass@1'].rolling(window=window, center=True).mean()
        ax.plot(df['Step'], df['smoothed'], '-', color=color, 
                linewidth=3, alpha=1.0, label=f'Smoothed (w={window})')
        
        ax.set_xlabel('Training Step')
        if col == 0:
            ax.set_ylabel('Pass@1 Accuracy')
        ax.set_title(f'{label}')
        ax.set_ylim(test_ylims[dataset_key])
        ax.set_xlim(-10, 350)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
    
    # =========================================
    # Row 2: Training Reward plots
    # =========================================
    for col, dataset_key in enumerate(datasets):
        ax = axes[1, col]
        df = load_reward_data(REWARD_CSVS[dataset_key])
        color = COLORS[dataset_key]
        label = LABELS[dataset_key]
        
        # Plot raw data with low alpha
        ax.plot(df['Step'], df['reward_mean'], '-', color=color, 
                linewidth=0.5, alpha=0.3, label='Raw')
        
        # Add smoothed trend line (EMA)
        span = 20
        df['smoothed'] = df['reward_mean'].ewm(span=span).mean()
        ax.plot(df['Step'], df['smoothed'], '-', color=color, 
                linewidth=2.5, alpha=1.0, label=f'Smoothed (EMA={span})')
        
        ax.set_xlabel('Training Step')
        if col == 0:
            ax.set_ylabel('Mean Reward')
        ax.set_ylim(reward_ylims[dataset_key])
        ax.set_xlim(0, 335)
        ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
    
    # Add row labels
    fig.text(0.02, 0.75, 'Test Accuracy', va='center', rotation='vertical', fontsize=12, fontweight='bold')
    fig.text(0.02, 0.3, 'Training Reward', va='center', rotation='vertical', fontsize=12, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(left=0.08)
    
    # Save
    output_dir = Path("/admin/home/nikhil/med-lm-envs/plots")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'combined_training.png')
    plt.savefig(output_dir / 'combined_training.pdf')
    print(f"Saved to {output_dir / 'combined_training.png'}")
    print(f"Saved to {output_dir / 'combined_training.pdf'}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    
    for dataset_key in ['medcalc', 'medmcqa', 'medcase']:
        test_df = load_test_data(TEST_CSVS[dataset_key])
        reward_df = load_reward_data(REWARD_CSVS[dataset_key])
        
        start_acc = test_df['pass@1'].iloc[0]
        end_acc = test_df['pass@1'].iloc[-1]
        
        reward_df['smoothed'] = reward_df['reward_mean'].ewm(span=20).mean()
        start_reward = reward_df['smoothed'].iloc[:10].mean()
        end_reward = reward_df['smoothed'].iloc[-10:].mean()
        
        print(f"\n{LABELS[dataset_key]}:")
        print(f"  Test Accuracy: {start_acc:.1%} → {end_acc:.1%} (+{end_acc - start_acc:.1%})")
        print(f"  Reward:        {start_reward:.3f} → {end_reward:.3f} (+{end_reward - start_reward:.3f})")


if __name__ == "__main__":
    main()
