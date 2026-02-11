#!/usr/bin/env python3
"""
7-panel plot for combined medical RL training.
Row 1: 3 test accuracy plots (MedMCQA, MedCalc, MedCase)
Row 2: 4 training reward plots (MedMCQA, MedCalc, MedCase, Mean)
"""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Colors for each dataset
COLORS = {
    'medmcqa': '#0d0887',      # Dark blue/purple
    'medcalc': '#7e03a8',      # Purple
    'medcase': '#cc4778',      # Pink/magenta
    'mean': '#f89540',         # Orange
}

LABELS = {
    'medmcqa': 'MedMCQA',
    'medcalc': 'MedCalc-Bench',
    'medcase': 'MedCaseReasoning',
    'mean': 'Mean Reward',
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

# CSV file paths for combined training
TEST_CSVS = {
    'medmcqa': "/admin/home/nikhil/med-lm-envs/wandb_export_2026-01-29T08_59_05.095-05_00.csv",
    'medcalc': "/admin/home/nikhil/med-lm-envs/wandb_export_2026-01-29T08_59_30.758-05_00.csv",
    'medcase': "/admin/home/nikhil/med-lm-envs/wandb_export_2026-01-29T08_59_51.236-05_00.csv",
}

REWARD_CSVS = {
    'medmcqa': "/admin/home/nikhil/med-lm-envs/wandb_export_2026-01-29T09_00_00.134-05_00.csv",
    'medcase': "/admin/home/nikhil/med-lm-envs/wandb_export_2026-01-29T09_00_08.253-05_00.csv",
    'medcalc': "/admin/home/nikhil/med-lm-envs/wandb_export_2026-01-29T09_00_16.389-05_00.csv",
    'mean': "/admin/home/nikhil/med-lm-envs/wandb_export_2026-01-29T09_00_21.894-05_00.csv",
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
    # Create figure with custom layout: 3 on top, 4 on bottom
    fig = plt.figure(figsize=(16, 8))
    
    # Create grid spec for custom layout
    gs = fig.add_gridspec(2, 12, hspace=0.35, wspace=0.5)
    
    # Row 1: 3 test accuracy plots (each takes 4 columns, with gaps)
    ax_test = [
        fig.add_subplot(gs[0, 0:4]),   # MedMCQA
        fig.add_subplot(gs[0, 4:8]),   # MedCalc
        fig.add_subplot(gs[0, 8:12]),  # MedCase
    ]
    
    # Row 2: 4 training reward plots (each takes 3 columns)
    ax_reward = [
        fig.add_subplot(gs[1, 0:3]),   # MedMCQA reward
        fig.add_subplot(gs[1, 3:6]),   # MedCalc reward
        fig.add_subplot(gs[1, 6:9]),   # MedCase reward
        fig.add_subplot(gs[1, 9:12]),  # Mean reward
    ]
    
    test_datasets = ['medmcqa', 'medcalc', 'medcase']
    reward_datasets = ['medmcqa', 'medcalc', 'medcase', 'mean']
    
    # Y-axis limits for test accuracy
    test_ylims = {
        'medmcqa': (0.55, 0.65),
        'medcalc': (0.35, 0.75),
        'medcase': (0.15, 0.40),
    }
    
    # Y-axis limits for reward
    reward_ylims = {
        'medmcqa': (0.4, 0.9),
        'medcalc': (0.2, 0.9),
        'medcase': (0.0, 0.5),
        'mean': (0.3, 0.7),
    }
    
    # =========================================
    # Row 1: Test Accuracy plots
    # =========================================
    for idx, dataset_key in enumerate(test_datasets):
        ax = ax_test[idx]
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
        if idx == 0:
            ax.set_ylabel('Pass@1 Accuracy')
        ax.set_title(f'{label}')
        ax.set_ylim(test_ylims[dataset_key])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
    
    # =========================================
    # Row 2: Training Reward plots
    # =========================================
    for idx, dataset_key in enumerate(reward_datasets):
        ax = ax_reward[idx]
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
        if idx == 0:
            ax.set_ylabel('Mean Reward')
        ax.set_title(f'{label}')
        ax.set_ylim(reward_ylims[dataset_key])
        ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
    
    # Adjust layout
    plt.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.08)
    
    # Save
    output_dir = Path("/admin/home/nikhil/med-lm-envs/plots")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'combined_7panel.png')
    plt.savefig(output_dir / 'combined_7panel.pdf')
    print(f"Saved to {output_dir / 'combined_7panel.png'}")
    print(f"Saved to {output_dir / 'combined_7panel.pdf'}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Combined Training Summary Statistics")
    print("="*60)
    
    for dataset_key in test_datasets:
        test_df = load_test_data(TEST_CSVS[dataset_key])
        start_acc = test_df['pass@1'].iloc[0]
        end_acc = test_df['pass@1'].iloc[-1]
        print(f"\n{LABELS[dataset_key]}:")
        print(f"  Test Accuracy: {start_acc:.1%} → {end_acc:.1%} (+{end_acc - start_acc:.1%})")
    
    print("\nReward Summary:")
    for dataset_key in reward_datasets:
        reward_df = load_reward_data(REWARD_CSVS[dataset_key])
        reward_df['smoothed'] = reward_df['reward_mean'].ewm(span=20).mean()
        start_reward = reward_df['smoothed'].iloc[:10].mean()
        end_reward = reward_df['smoothed'].iloc[-10:].mean()
        print(f"  {LABELS[dataset_key]}: {start_reward:.3f} → {end_reward:.3f} (+{end_reward - start_reward:.3f})")


if __name__ == "__main__":
    main()
