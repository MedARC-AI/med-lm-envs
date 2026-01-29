#!/usr/bin/env python3
"""
Generate publication-quality plots from RL training results.
Fetches data from W&B and creates matplotlib figures.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Try to import wandb, fall back to manual data if not available

import wandb
WANDB_AVAILABLE = True


# Style settings for publication
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (8, 5),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Color palette (colorblind-friendly)
COLORS = {
    'medmcqa': '#2196F3',      # Blue
    'medcalc': '#4CAF50',      # Green  
    'medcase': '#FF9800',      # Orange
    'combined': '#9C27B0',     # Purple
}

LABELS = {
    'medmcqa': 'MedMCQA',
    'medcalc': 'MedCalc-Bench',
    'medcase': 'MedCaseReasoning',
}


def fetch_wandb_data(project: str, run_name: str | None = None) -> pd.DataFrame:
    """Fetch training data from W&B."""
    if not WANDB_AVAILABLE:
        raise ImportError("wandb not installed. Install with: pip install wandb")
    
    api = wandb.Api()
    runs = api.runs(project)
    
    if run_name:
        runs = [r for r in runs if run_name in r.name]
    
    all_data = []
    for run in runs:
        history = run.history()
        history['run_name'] = run.name
        all_data.append(history)
    
    return pd.concat(all_data, ignore_index=True)


def fetch_wandb_run(
    entity: str,
    project: str, 
    run_id: str,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """
    Fetch specific metrics from a W&B run.
    
    Args:
        entity: W&B entity (username or team)
        project: W&B project name
        run_id: The run ID (from the URL, e.g., 'abc123xy')
        metrics: List of metric names to fetch. If None, fetches all.
    
    Returns:
        DataFrame with step and requested metrics
    """
    if not WANDB_AVAILABLE:
        raise ImportError("wandb not installed. Install with: pip install wandb")
    
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    
    # Get history with specific keys if provided
    if metrics:
        # Always include _step
        keys = ['_step'] + [m for m in metrics if m != '_step']
        history = run.history(keys=keys)
    else:
        history = run.history()
    
    return history


def fetch_training_data_from_wandb(
    entity: str,
    project: str,
    run_id: str,
    reward_key: str = "train/mean_reward",
    accuracy_keys: dict[str, str] | None = None,
) -> tuple[dict, dict]:
    """
    Fetch reward and accuracy data from a W&B run.
    
    Args:
        entity: W&B entity
        project: W&B project  
        run_id: Run ID
        reward_key: Key for reward metric
        accuracy_keys: Dict mapping env name to accuracy metric key
                      e.g., {'medmcqa': 'eval/medmcqa/pass@1'}
    
    Returns:
        Tuple of (accuracy_data, reward_data) ready for plotting
    """
    if not WANDB_AVAILABLE:
        raise ImportError("wandb not installed. Install with: pip install wandb")
    
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    
    # Fetch all history
    history = run.history()
    
    # Process reward data
    reward_df = history[['_step', reward_key]].dropna()
    reward_data = {
        'combined': {
            'steps': reward_df['_step'].tolist(),
            'reward': reward_df[reward_key].tolist(),
            'label': 'Combined Training',
        }
    }
    
    # Process accuracy data if keys provided
    accuracy_data = {}
    if accuracy_keys:
        for env_name, key in accuracy_keys.items():
            if key in history.columns:
                acc_df = history[['_step', key]].dropna()
                accuracy_data[env_name] = {
                    'steps': acc_df['_step'].tolist(),
                    'values': acc_df[key].tolist(),
                }
    
    return accuracy_data, reward_data


def list_wandb_runs(entity: str, project: str, limit: int = 20):
    """List recent runs in a W&B project."""
    if not WANDB_AVAILABLE:
        raise ImportError("wandb not installed. Install with: pip install wandb")
    
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", per_page=limit)
    
    print(f"\nRecent runs in {entity}/{project}:")
    print("-" * 80)
    for run in runs:
        print(f"  ID: {run.id}")
        print(f"  Name: {run.name}")
        print(f"  State: {run.state}")
        print(f"  Created: {run.created_at}")
        print(f"  URL: {run.url}")
        print("-" * 80)


def list_wandb_metrics(entity: str, project: str, run_id: str):
    """List available metrics in a W&B run."""
    if not WANDB_AVAILABLE:
        raise ImportError("wandb not installed. Install with: pip install wandb")
    
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    history = run.history()
    
    print(f"\nAvailable metrics in run {run_id}:")
    print("-" * 40)
    for col in sorted(history.columns):
        non_null = history[col].notna().sum()
        print(f"  {col}: {non_null} values")


def plot_from_wandb(
    entity: str,
    project: str,
    run_id: str,
    output_dir: Path,
    reward_key: str = "train/mean_reward",
    accuracy_keys: dict[str, str] | None = None,
):
    """
    Fetch data from W&B and generate plots.
    
    Example:
        plot_from_wandb(
            entity="your-username",
            project="medical-rl",
            run_id="abc123xy",
            output_dir=Path("plots"),
            reward_key="train/mean_reward",
            accuracy_keys={
                'medmcqa': 'eval/medmcqa/pass@1',
                'medcalc': 'eval/medcalc_bench/pass@1',
                'medcase': 'eval/medcasereasoning/pass@1',
            }
        )
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    accuracy_data, reward_data = fetch_training_data_from_wandb(
        entity, project, run_id, reward_key, accuracy_keys
    )
    
    if accuracy_data:
        plot_training_curves(
            accuracy_data,
            output_dir / 'training_curves_wandb.png',
            title='Medical RL Training Progress',
            ylabel='Pass@1 Accuracy'
        )
    
    if reward_data:
        plot_reward_curves(
            reward_data,
            output_dir / 'reward_curves_wandb.png',
            title='Training Reward'
        )
    
    if accuracy_data and reward_data:
        plot_multi_panel(
            accuracy_data,
            reward_data,
            output_dir / 'training_combined_wandb.png',
            title='Medical RL Training'
        )
    
    print(f"\nPlots saved to {output_dir}")


def plot_training_curves(
    data: dict[str, dict],
    output_path: Path,
    title: str = "RL Training Progress",
    ylabel: str = "Pass@1 Accuracy",
):
    """
    Plot training curves for multiple environments.
    
    Args:
        data: Dict of {env_name: {'steps': [...], 'values': [...], 'label': str}}
        output_path: Where to save the figure
        title: Plot title
        ylabel: Y-axis label
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for env_name, env_data in data.items():
        steps = env_data['steps']
        values = env_data['values']
        label = env_data.get('label', LABELS.get(env_name, env_name))
        color = COLORS.get(env_name, None)
        
        ax.plot(steps, values, 'o-', label=label, color=color, 
                linewidth=2, markersize=6, alpha=0.8)
        
        # Add error bars if provided
        if 'std' in env_data:
            ax.fill_between(steps, 
                           np.array(values) - np.array(env_data['std']),
                           np.array(values) + np.array(env_data['std']),
                           alpha=0.2, color=color)
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='best', framealpha=0.9)
    ax.set_ylim(0, 1)
    
    # Add minor gridlines
    ax.grid(True, which='major', linestyle='-', alpha=0.3)
    ax.grid(True, which='minor', linestyle=':', alpha=0.2)
    ax.minorticks_on()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.savefig(output_path.with_suffix('.pdf'))  # Also save PDF for paper
    print(f"Saved plot to {output_path} and {output_path.with_suffix('.pdf')}")
    plt.close()


def plot_comparison_bar(
    data: dict[str, dict],
    output_path: Path,
    title: str = "Model Comparison",
):
    """
    Create a grouped bar chart comparing base model vs trained model.
    
    Args:
        data: Dict of {env_name: {'base': float, 'trained': float}}
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    envs = list(data.keys())
    x = np.arange(len(envs))
    width = 0.35
    
    base_values = [data[env]['base'] for env in envs]
    trained_values = [data[env]['trained'] for env in envs]
    
    bars1 = ax.bar(x - width/2, base_values, width, label='Base Model', 
                   color='#9E9E9E', alpha=0.8)
    bars2 = ax.bar(x + width/2, trained_values, width, label='RL-Trained',
                   color='#2196F3', alpha=0.8)
    
    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1%}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    add_labels(bars1)
    add_labels(bars2)
    
    ax.set_ylabel('Pass@1 Accuracy')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(env, env) for env in envs])
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.savefig(output_path.with_suffix('.pdf'))
    print(f"Saved plot to {output_path} and {output_path.with_suffix('.pdf')}")
    plt.close()


def plot_reward_curves(
    data: dict[str, dict],
    output_path: Path,
    title: str = "Training Reward",
):
    """
    Plot reward curves during training.
    
    Args:
        data: Dict of {env_name: {'steps': [...], 'reward': [...], 'reward_std': [...]}}
        output_path: Where to save the figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for env_name, env_data in data.items():
        steps = env_data['steps']
        rewards = env_data['reward']
        label = env_data.get('label', LABELS.get(env_name, env_name))
        color = COLORS.get(env_name, None)
        
        ax.plot(steps, rewards, '-', label=label, color=color, 
                linewidth=2, alpha=0.8)
        
        # Add shading for std if provided
        if 'reward_std' in env_data:
            ax.fill_between(steps, 
                           np.array(rewards) - np.array(env_data['reward_std']),
                           np.array(rewards) + np.array(env_data['reward_std']),
                           alpha=0.2, color=color)
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Mean Reward')
    ax.set_title(title)
    ax.legend(loc='best', framealpha=0.9)
    
    # Add minor gridlines
    ax.grid(True, which='major', linestyle='-', alpha=0.3)
    ax.grid(True, which='minor', linestyle=':', alpha=0.2)
    ax.minorticks_on()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.savefig(output_path.with_suffix('.pdf'))
    print(f"Saved plot to {output_path} and {output_path.with_suffix('.pdf')}")
    plt.close()


def plot_multi_panel(
    accuracy_data: dict[str, dict],
    reward_data: dict[str, dict],
    output_path: Path,
    title: str = "Medical RL Training",
):
    """
    Create a two-panel figure with accuracy and reward curves.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # Left panel: Accuracy
    for env_name, env_data in accuracy_data.items():
        steps = env_data['steps']
        values = env_data['values']
        label = env_data.get('label', LABELS.get(env_name, env_name))
        color = COLORS.get(env_name, None)
        
        ax1.plot(steps, values, 'o-', label=label, color=color, 
                linewidth=2, markersize=5, alpha=0.8)
        
        if 'std' in env_data:
            ax1.fill_between(steps, 
                           np.array(values) - np.array(env_data['std']),
                           np.array(values) + np.array(env_data['std']),
                           alpha=0.15, color=color)
    
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Pass@1 Accuracy')
    ax1.set_title('(a) Evaluation Accuracy')
    ax1.legend(loc='lower right', framealpha=0.9)
    ax1.set_ylim(0, 1)
    ax1.grid(True, which='major', linestyle='-', alpha=0.3)
    
    # Right panel: Reward
    for env_name, env_data in reward_data.items():
        steps = env_data['steps']
        rewards = env_data['reward']
        label = env_data.get('label', LABELS.get(env_name, env_name))
        color = COLORS.get(env_name, None)
        
        ax2.plot(steps, rewards, '-', label=label, color=color, 
                linewidth=2, alpha=0.8)
        
        if 'reward_std' in env_data:
            ax2.fill_between(steps, 
                           np.array(rewards) - np.array(env_data['reward_std']),
                           np.array(rewards) + np.array(env_data['reward_std']),
                           alpha=0.15, color=color)
    
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Mean Reward')
    ax2.set_title('(b) Training Reward')
    ax2.legend(loc='lower right', framealpha=0.9)
    ax2.grid(True, which='major', linestyle='-', alpha=0.3)
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.savefig(output_path.with_suffix('.pdf'))
    print(f"Saved plot to {output_path} and {output_path.with_suffix('.pdf')}")
    plt.close()

    # Individual plots
    plot_training_curves(
        training_data,
        output_dir / 'training_curves.png',
        title='Medical RL Training Progress',
        ylabel='Pass@1 Accuracy'
    )
    
    plot_reward_curves(
        reward_data,
        output_dir / 'reward_curves.png',
        title='Training Reward'
    )
    
    # Combined multi-panel figure (good for papers)
    plot_multi_panel(
        training_data,
        reward_data,
        output_dir / 'training_combined.png',
        title='Medical RL Training'
    )
    
    # Comparison bar chart
    comparison_data = {
        'medmcqa': {'base': 0.59, 'trained': 0.70},
        'medcalc': {'base': 0.42, 'trained': 0.59},
        'medcase': {'base': 0.19, 'trained': 0.31},
    }
    
    plot_comparison_bar(
        comparison_data,
        output_dir / 'model_comparison.png',
        title='Base Model vs RL-Trained Model'
    )
    
    print(f"\nExample plots created in {output_dir}")
    print("Replace placeholder data with actual W&B data for final plots.")


def export_wandb_to_csv(project: str, output_path: Path):
    """Export W&B data to CSV for manual plotting."""
    if not WANDB_AVAILABLE:
        print("wandb not installed. Install with: pip install wandb")
        return
    
    api = wandb.Api()
    runs = api.runs(project)
    
    for run in runs:
        history = run.history()
        csv_path = output_path / f"{run.name.replace('/', '_')}.csv"
        history.to_csv(csv_path, index=False)
        print(f"Exported {run.name} to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate publication-quality plots")
    parser.add_argument("--output-dir", type=Path, default=Path("plots"),
                       help="Output directory for plots")
    parser.add_argument("--example", action="store_true",
                       help="Generate example plots with placeholder data")
    
    # W&B options
    parser.add_argument("--wandb-entity", type=str, default=None,
                       help="W&B entity (username or team)")
    parser.add_argument("--wandb-project", type=str, default=None,
                       help="W&B project name")
    parser.add_argument("--wandb-run", type=str, default=None,
                       help="W&B run ID")
    parser.add_argument("--list-runs", action="store_true",
                       help="List runs in W&B project")
    parser.add_argument("--list-metrics", action="store_true",
                       help="List metrics in a W&B run")
    parser.add_argument("--export-csv", action="store_true",
                       help="Export W&B data to CSV")
    
    # Metric keys
    parser.add_argument("--reward-key", type=str, default="train/mean_reward",
                       help="W&B key for reward metric")
    
    args = parser.parse_args()
    
    if args.example:
        create_example_plots(args.output_dir)
    
    elif args.list_runs:
        if not args.wandb_entity or not args.wandb_project:
            print("Error: --wandb-entity and --wandb-project required")
        else:
            list_wandb_runs(args.wandb_entity, args.wandb_project)
    
    elif args.list_metrics:
        if not args.wandb_entity or not args.wandb_project or not args.wandb_run:
            print("Error: --wandb-entity, --wandb-project, and --wandb-run required")
        else:
            list_wandb_metrics(args.wandb_entity, args.wandb_project, args.wandb_run)
    
    elif args.export_csv:
        if not args.wandb_entity or not args.wandb_project or not args.wandb_run:
            print("Error: --wandb-entity, --wandb-project, and --wandb-run required")
        else:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            df = fetch_wandb_run(args.wandb_entity, args.wandb_project, args.wandb_run)
            csv_path = args.output_dir / f"{args.wandb_run}.csv"
            df.to_csv(csv_path, index=False)
            print(f"Exported to {csv_path}")
    
    elif args.wandb_run:
        if not args.wandb_entity or not args.wandb_project:
            print("Error: --wandb-entity and --wandb-project required")
        else:
            # Default accuracy keys - customize these for your runs
            accuracy_keys = {
                'medmcqa': 'eval/medmcqa/pass@1',
                'medcalc': 'eval/medcalc_bench/pass@1', 
                'medcase': 'eval/medcasereasoning/pass@1',
            }
            plot_from_wandb(
                args.wandb_entity,
                args.wandb_project,
                args.wandb_run,
                args.output_dir,
                args.reward_key,
                accuracy_keys,
            )
    
    else:
        print("""Usage:
  # Generate example plots with placeholder data
  python plot_training_results.py --example
  
  # List runs in a W&B project
  python plot_training_results.py --list-runs --wandb-entity USER --wandb-project PROJECT
  
  # List metrics available in a run
  python plot_training_results.py --list-metrics --wandb-entity USER --wandb-project PROJECT --wandb-run RUN_ID
  
  # Export run data to CSV
  python plot_training_results.py --export-csv --wandb-entity USER --wandb-project PROJECT --wandb-run RUN_ID
  
  # Generate plots from W&B run
  python plot_training_results.py --wandb-entity USER --wandb-project PROJECT --wandb-run RUN_ID
""")
