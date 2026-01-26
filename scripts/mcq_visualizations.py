#!/usr/bin/env python3
"""
MCQ Analysis Visualizations

Generates paper-ready visualizations from MCQ analysis outputs.
Uses viridis colormap for consistency with blog style.

Usage:
    python scripts/mcq_visualizations.py \
        --analysis-dir ./analysis_output \
        --output-dir ./figures
"""

import argparse
import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# Publication style settings
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# Color scheme
PERF_CMAP = 'viridis'
DIVERGE_CMAP = 'RdBu_r'


def clean_model_name(name: str) -> str:
    """Clean model name for display."""
    name = re.sub(r'-\d{8}-\d{6}$', '', name)
    return name.replace('_', '-')


def extract_model_size(name: str) -> str:
    """Extract model size category.

    Categories:
    - small: <10B parameters
    - medium: 10-30B parameters
    - large: >30B parameters or frontier models
    """
    name_lower = name.lower()

    # Known frontier/large models without size in name
    LARGE_MODELS = {
        'gpt-5', 'gpt_5', 'gpt-oss-120b', 'grok-4', 'sonnet-4', 'opus-4',
        'claude-4', 'gemini-2', 'baichuan-m2', 'minimax-m2', 'intellect3'
    }

    # Known medium models without clear size
    MEDIUM_MODELS = {
        'gpt-oss-20b', 'magistral-small'
    }

    # Check known model patterns first
    for pattern in LARGE_MODELS:
        if pattern in name_lower:
            return 'large'

    for pattern in MEDIUM_MODELS:
        if pattern in name_lower:
            return 'medium'

    # Try to extract size from "Xb" pattern
    match = re.search(r'(\d+\.?\d*)b', name_lower)
    if match:
        size = float(match.group(1))
        if size < 10:
            return 'small'
        elif size < 30:
            return 'medium'
        return 'large'

    # Default to large for unknown models (frontier models often lack size in name)
    return 'large'


def to_percent(series: pd.Series) -> pd.Series:
    """Convert 0-1 values to percentages if needed."""
    if series.dropna().empty:
        return series
    if series.max() <= 1.5:
        return series * 100
    return series


def load_model_metrics(analysis_dir: Path) -> pd.DataFrame:
    """Load all-models metrics."""
    path = analysis_dir / 'all_models_metrics.csv'
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_csv(path)
    df['clean_model'] = df['model'].apply(clean_model_name)
    for col in ['variation_rate', 'overall_accuracy', 'semantic_consistency_rate', 'mean_win_rate', 'weighted_mean_win_rate']:
        if col in df.columns:
            df[f'{col}_pct'] = to_percent(df[col])
    return df


def load_benchmark_metrics(analysis_dir: Path) -> pd.DataFrame:
    """Load per-benchmark metrics from all models."""
    files = list(analysis_dir.glob("**/*_benchmark_metrics.csv"))
    if not files:
        raise FileNotFoundError(f"No benchmark metrics found in {analysis_dir}")

    dfs = []
    for fpath in files:
        df = pd.read_csv(fpath)
        match = re.match(r"(.+)_benchmark_metrics$", fpath.stem)
        model_name = match.group(1) if match else fpath.parent.name
        df["model"] = model_name
        df["clean_model"] = clean_model_name(model_name)
        for col in ['variation_rate', 'accuracy', 'semantic_consistency_rate']:
            if col in df.columns:
                df[f'{col}_pct'] = to_percent(df[col])
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def create_model_heatmap(df: pd.DataFrame, output_dir: Path):
    """Create model performance heatmap."""
    print("Creating model performance heatmap...")

    # Aggregate by model
    agg_cols = ['variation_rate_pct', 'overall_accuracy_pct', 'semantic_consistency_rate_pct']
    available = [c for c in agg_cols if c in df.columns]
    if not available:
        print("  Skipping - no data")
        return

    model_stats = df.groupby('clean_model')[available].mean().reset_index()
    model_stats = model_stats.sort_values(available[1] if len(available) > 1 else available[0], ascending=False)

    labels = {
        'overall_accuracy_pct': 'Accuracy\n(Higher=Better)',
        'variation_rate_pct': 'Variation Rate\n(Higher=More Sensitive)',
        'semantic_consistency_rate_pct': 'Semantic Consistency\n(Higher=Better)',
    }

    heatmap_data = model_stats[available].values
    fig, ax = plt.subplots(figsize=(10, max(8, len(model_stats) * 0.3)))

    sns.heatmap(
        heatmap_data, annot=True, fmt='.1f', cmap=PERF_CMAP,
        cbar_kws={'label': 'Percentage (%)'}, linewidths=1, linecolor='white',
        ax=ax, vmin=0, vmax=100,
        xticklabels=[labels.get(c, c) for c in available],
        yticklabels=model_stats['clean_model'].values
    )

    ax.set_title('Model Performance Summary\n(Yellow = Higher Value)', fontweight='bold', pad=15)
    ax.set_ylabel('Model', fontweight='bold')
    ax.tick_params(axis='y', labelsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'model_heatmap.png', bbox_inches='tight')
    plt.savefig(output_dir / 'model_heatmap.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved with {len(model_stats)} models")


def create_model_ranking(df: pd.DataFrame, output_dir: Path):
    """Create model ranking by accuracy."""
    print("Creating model ranking plot...")

    if 'overall_accuracy_pct' not in df.columns:
        print("  Skipping - no accuracy data")
        return

    model_stats = df.groupby('clean_model').agg({
        'overall_accuracy_pct': 'mean',
        'variation_rate_pct': 'mean' if 'variation_rate_pct' in df.columns else 'first',
    }).reset_index()
    model_stats = model_stats.sort_values('overall_accuracy_pct', ascending=True)

    fig, ax = plt.subplots(figsize=(12, max(8, len(model_stats) * 0.25)))

    colors = plt.cm.viridis(model_stats['overall_accuracy_pct'] / 100.0)
    ax.barh(range(len(model_stats)), model_stats['overall_accuracy_pct'], color=colors, edgecolor='black', linewidth=0.3)

    ax.set_yticks(range(len(model_stats)))
    ax.set_yticklabels(model_stats['clean_model'], fontsize=8)
    ax.set_xlabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Models Ranked by Accuracy', fontweight='bold', pad=15)

    mean_acc = model_stats['overall_accuracy_pct'].mean()
    ax.axvline(mean_acc, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_acc:.1f}%')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(0, 100)

    plt.tight_layout()
    plt.savefig(output_dir / 'model_ranking.png', bbox_inches='tight')
    plt.savefig(output_dir / 'model_ranking.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved with {len(model_stats)} models")


def create_variation_ranking(df: pd.DataFrame, output_dir: Path):
    """Create model ranking by variation rate."""
    print("Creating variation ranking plot...")

    if 'variation_rate_pct' not in df.columns:
        print("  Skipping - no variation data")
        return

    model_stats = df.dropna(subset=['variation_rate_pct']).groupby('clean_model').agg({
        'variation_rate_pct': 'mean',
        'overall_accuracy_pct': 'mean' if 'overall_accuracy_pct' in df.columns else 'first',
    }).reset_index()
    model_stats = model_stats.sort_values('variation_rate_pct', ascending=True)

    if model_stats.empty:
        print("  Skipping - no data after filtering")
        return

    fig, ax = plt.subplots(figsize=(12, max(8, len(model_stats) * 0.25)))

    colors = plt.cm.viridis(model_stats['overall_accuracy_pct'] / 100.0) if 'overall_accuracy_pct' in model_stats else 'steelblue'
    ax.barh(range(len(model_stats)), model_stats['variation_rate_pct'], color=colors, edgecolor='black', linewidth=0.3)

    ax.set_yticks(range(len(model_stats)))
    ax.set_yticklabels(model_stats['clean_model'], fontsize=8)
    ax.set_xlabel('Variation Rate (%)', fontweight='bold')
    ax.set_title('Models Ranked by Variation Rate\n(Lower = More Robust)', fontweight='bold', pad=15)

    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=100))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Accuracy (%)', pad=0.02)
    cbar.ax.tick_params(labelsize=9)

    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(0, 100)

    plt.tight_layout()
    plt.savefig(output_dir / 'variation_ranking.png', bbox_inches='tight')
    plt.savefig(output_dir / 'variation_ranking.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved with {len(model_stats)} models")


def create_scatter_plot(df: pd.DataFrame, output_dir: Path):
    """Create accuracy vs semantic consistency scatter with model labels."""
    print("Creating scatter plot...")

    required = ['overall_accuracy_pct', 'semantic_consistency_rate_pct']
    if not all(c in df.columns for c in required):
        print("  Skipping - missing columns")
        return

    extra_cols = ['weighted_mean_win_rate_pct'] if 'weighted_mean_win_rate_pct' in df.columns else []
    model_stats = df.dropna(subset=required).groupby('clean_model')[required + extra_cols].mean().reset_index()
    model_stats['size'] = model_stats['clean_model'].apply(extract_model_size)

    if model_stats.empty:
        print("  Skipping - no data")
        return

    fig, ax = plt.subplots(figsize=(16, 14))
    ax.set_facecolor('#fafafa')

    markers = {'small': 'o', 'medium': 's', 'large': '^'}
    sizes = {'small': 180, 'medium': 220, 'large': 280}

    color_col = 'weighted_mean_win_rate_pct' if 'weighted_mean_win_rate_pct' in model_stats.columns else 'overall_accuracy_pct'
    vmin, vmax = model_stats[color_col].min() - 2, model_stats[color_col].max() + 2

    for size_cat, marker in markers.items():
        mask = model_stats['size'] == size_cat
        if mask.sum() == 0:
            continue
        subset = model_stats[mask]
        ax.scatter(
            subset['semantic_consistency_rate_pct'],
            subset['overall_accuracy_pct'],
            c=subset[color_col],
            cmap=PERF_CMAP,
            s=sizes[size_cat],
            alpha=0.85,
            marker=marker,
            edgecolors='black',
            linewidth=1.2,
            label=size_cat.title(),
            vmin=vmin,
            vmax=vmax,
            zorder=3
        )

    # Add model name labels with smart positioning
    from adjustText import adjust_text
    texts = []
    for _, row in model_stats.iterrows():
        # Shorten long model names
        label = row['clean_model']
        if len(label) > 20:
            label = label[:18] + '...'
        texts.append(ax.annotate(
            label,
            (row['semantic_consistency_rate_pct'], row['overall_accuracy_pct']),
            fontsize=7,
            alpha=0.9,
            zorder=4
        ))

    # Try to use adjustText if available, otherwise use basic offset
    try:
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5, lw=0.5),
                    expand_points=(1.5, 1.5), force_text=(0.5, 0.5))
    except:
        # Fallback: simple offset
        for text in texts:
            text.set_position((text.get_position()[0] + 0.5, text.get_position()[1] + 0.5))

    sm = plt.cm.ScalarMappable(cmap=PERF_CMAP, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.02)
    cbar.set_label('Win Rate (%)' if 'win_rate' in color_col else 'Accuracy (%)', fontweight='bold', fontsize=11)

    ax.set_xlabel('Semantic Consistency (%) — Higher = Better', fontweight='bold', fontsize=12)
    ax.set_ylabel('Accuracy (%) — Higher = Better', fontweight='bold', fontsize=12)
    ax.set_title('Model Performance: Accuracy vs Semantic Consistency\n(Each point is a model, colored by win rate)',
                 fontweight='bold', pad=20, fontsize=14)

    # Clean up legend - only show unique entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), ['Small (<10B)', 'Medium (10-30B)', 'Large (>30B)'],
              loc='lower left', framealpha=0.9, fontsize=10)

    ax.grid(True, alpha=0.4, linestyle='--', zorder=1)
    ax.set_axisbelow(True)

    # Add quadrant labels
    x_mid = model_stats['semantic_consistency_rate_pct'].median()
    y_mid = model_stats['overall_accuracy_pct'].median()
    ax.axvline(x_mid, color='gray', linestyle=':', alpha=0.5, zorder=2)
    ax.axhline(y_mid, color='gray', linestyle=':', alpha=0.5, zorder=2)

    plt.tight_layout()
    plt.savefig(output_dir / 'scatter_accuracy_consistency.png', bbox_inches='tight', dpi=300)
    plt.savefig(output_dir / 'scatter_accuracy_consistency.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved with {len(model_stats)} models")


def create_violin_plots(df_bench: pd.DataFrame, output_dir: Path):
    """Create violin distribution plots."""
    print("Creating violin plots...")

    if 'variation_rate_pct' not in df_bench.columns:
        print("  Skipping - no variation data")
        return

    df = df_bench.dropna(subset=['variation_rate_pct'])
    if df.empty:
        print("  Skipping - no data")
        return

    model_order = df.groupby('clean_model')['variation_rate_pct'].median().sort_values().index
    height = max(10, len(model_order) * 0.3)

    # Variation rate violin
    fig, ax = plt.subplots(figsize=(12, height))
    sns.violinplot(
        data=df, y='clean_model', x='variation_rate_pct', ax=ax,
        palette='viridis', inner='box', orient='h', order=model_order,
        hue='clean_model', legend=False
    )
    ax.set_xlabel('Variation Rate (%) Across Benchmarks', fontweight='bold')
    ax.set_ylabel('Model', fontweight='bold')
    ax.set_title('Variation Rate Distribution\n(Higher = More Answer-Order Sensitive)', fontweight='bold', pad=15)
    ax.set_xlim(0, 100)
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(output_dir / 'violin_variation.png', bbox_inches='tight')
    plt.savefig(output_dir / 'violin_variation.pdf', bbox_inches='tight')
    plt.close()

    # Accuracy violin
    if 'accuracy_pct' in df.columns:
        fig, ax = plt.subplots(figsize=(12, height))
        sns.violinplot(
            data=df, y='clean_model', x='accuracy_pct', ax=ax,
            palette='viridis', inner='box', orient='h', order=model_order,
            hue='clean_model', legend=False
        )
        ax.set_xlabel('Accuracy (%) Across Benchmarks', fontweight='bold')
        ax.set_ylabel('Model', fontweight='bold')
        ax.set_title('Accuracy Distribution', fontweight='bold', pad=15)
        ax.set_xlim(0, 100)
        ax.tick_params(axis='y', labelsize=8)
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(output_dir / 'violin_accuracy.png', bbox_inches='tight')
        plt.savefig(output_dir / 'violin_accuracy.pdf', bbox_inches='tight')
        plt.close()

    print(f"  Saved violin plots for {len(model_order)} models")


def create_benchmark_heatmap(df_bench: pd.DataFrame, output_dir: Path):
    """Create benchmark performance heatmap."""
    print("Creating benchmark heatmap...")

    available = [c for c in ['variation_rate_pct', 'accuracy_pct', 'semantic_consistency_rate_pct'] if c in df_bench.columns]
    if not available:
        print("  Skipping - no data")
        return

    bench_stats = df_bench.groupby('benchmark')[available].mean().reset_index()
    bench_stats = bench_stats.sort_values(available[0], ascending=True).head(30)
    bench_stats['clean_bench'] = bench_stats['benchmark'].str.replace('_', ' ').str.replace('-', ' ').str.title()

    labels = {
        'variation_rate_pct': 'Variation\n(Higher=Sensitive)',
        'accuracy_pct': 'Accuracy\n(Higher=Better)',
        'semantic_consistency_rate_pct': 'Consistency\n(Higher=Better)',
    }

    heatmap_data = bench_stats[available].values
    fig, ax = plt.subplots(figsize=(10, max(8, len(bench_stats) * 0.35)))

    sns.heatmap(
        heatmap_data, annot=True, fmt='.1f', cmap=PERF_CMAP,
        cbar_kws={'label': 'Percentage (%)'}, linewidths=1, linecolor='white',
        ax=ax, vmin=0, vmax=100,
        xticklabels=[labels.get(c, c) for c in available],
        yticklabels=bench_stats['clean_bench'].values
    )

    ax.set_title('Benchmark Performance Summary\n(Averaged Across Models)', fontweight='bold', pad=15)
    ax.set_ylabel('Benchmark', fontweight='bold')
    ax.tick_params(axis='y', labelsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'benchmark_heatmap.png', bbox_inches='tight')
    plt.savefig(output_dir / 'benchmark_heatmap.pdf', bbox_inches='tight')
    plt.close()
    print(f"  Saved with {len(bench_stats)} benchmarks")


def create_correlation_heatmap(df: pd.DataFrame, output_dir: Path):
    """Create metric correlation heatmap."""
    print("Creating correlation heatmap...")

    metrics = [c for c in ['weighted_mean_win_rate_pct', 'overall_accuracy_pct', 'semantic_consistency_rate_pct', 'variation_rate_pct'] if c in df.columns]
    if len(metrics) < 2:
        print("  Skipping - not enough metrics")
        return

    corr = df[metrics].corr()

    labels = {
        'weighted_mean_win_rate_pct': 'Win Rate',
        'overall_accuracy_pct': 'Accuracy',
        'semantic_consistency_rate_pct': 'Consistency',
        'variation_rate_pct': 'Variation',
    }

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        corr, annot=True, fmt='.3f', cmap=DIVERGE_CMAP,
        center=0, vmin=-1, vmax=1,
        cbar_kws={'label': 'Correlation'},
        linewidths=2, linecolor='white', ax=ax, square=True,
        annot_kws={'fontsize': 12, 'fontweight': 'bold'}
    )

    ax.set_xticklabels([labels.get(c, c) for c in metrics], rotation=45, ha='right')
    ax.set_yticklabels([labels.get(c, c) for c in metrics], rotation=0)
    ax.set_title('Metric Correlations\n(Red = Negative, Blue = Positive)', fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_heatmap.png', bbox_inches='tight')
    plt.savefig(output_dir / 'correlation_heatmap.pdf', bbox_inches='tight')
    plt.close()
    print("  Saved correlation heatmap")


def create_summary_table(df: pd.DataFrame, output_dir: Path):
    """Create summary statistics visualization."""
    print("Creating summary table...")

    stats = {
        'Total Models': df['clean_model'].nunique(),
        'Avg Accuracy (%)': df['overall_accuracy_pct'].mean() if 'overall_accuracy_pct' in df.columns else None,
        'Avg Variation (%)': df['variation_rate_pct'].mean() if 'variation_rate_pct' in df.columns else None,
        'Avg Consistency (%)': df['semantic_consistency_rate_pct'].mean() if 'semantic_consistency_rate_pct' in df.columns else None,
    }

    fig = plt.figure(figsize=(14, 10))

    # Summary box
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.axis('off')
    text = "ANALYSIS SUMMARY\n\n"
    for k, v in stats.items():
        if v is not None:
            text += f"{k}: {v:.1f}\n" if isinstance(v, float) else f"{k}: {v}\n"
    ax1.text(0.5, 0.5, text, ha='center', va='center', fontsize=14,
             bbox=dict(boxstyle='round', facecolor='#FDE724', alpha=0.7, pad=1.5))

    # Distribution histograms
    if 'variation_rate_pct' in df.columns:
        ax2 = fig.add_subplot(2, 2, 2)
        df['variation_rate_pct'].hist(bins=20, ax=ax2, color='steelblue', edgecolor='black', alpha=0.7)
        ax2.axvline(df['variation_rate_pct'].mean(), color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Variation Rate (%)', fontweight='bold')
        ax2.set_ylabel('Count', fontweight='bold')
        ax2.set_title('Variation Rate Distribution', fontweight='bold')

    if 'overall_accuracy_pct' in df.columns:
        ax3 = fig.add_subplot(2, 2, 3)
        df['overall_accuracy_pct'].hist(bins=20, ax=ax3, color='forestgreen', edgecolor='black', alpha=0.7)
        ax3.axvline(df['overall_accuracy_pct'].mean(), color='red', linestyle='--', linewidth=2)
        ax3.set_xlabel('Accuracy (%)', fontweight='bold')
        ax3.set_ylabel('Count', fontweight='bold')
        ax3.set_title('Accuracy Distribution', fontweight='bold')

    # Key findings
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    findings = f"""
KEY FINDINGS

- {stats.get('Avg Variation (%)', 0):.0f}% average variation rate
  indicates answer-order sensitivity

- {stats.get('Avg Consistency (%)', 0):.0f}% semantic consistency
  suggests content-based selection

- Evaluation protocols should use
  multiple random seeds for reliability
"""
    ax4.text(0.5, 0.5, findings, ha='center', va='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='#B8DE6F', alpha=0.8, pad=1.5),
             family='monospace', linespacing=1.6)

    plt.suptitle('MedMarks MCQ Analysis Summary', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / 'summary.png', bbox_inches='tight')
    plt.savefig(output_dir / 'summary.pdf', bbox_inches='tight')
    plt.close()
    print("  Saved summary visualization")


def create_pdf_report(output_dir: Path, df: pd.DataFrame):
    """Create combined PDF report."""
    print("Creating PDF report...")

    figures = [
        'summary.png',
        'model_heatmap.png',
        'model_ranking.png',
        'variation_ranking.png',
        'scatter_accuracy_consistency.png',
        'benchmark_heatmap.png',
        'correlation_heatmap.png',
    ]

    pdf_path = output_dir / 'mcq_analysis_report.pdf'

    with PdfPages(pdf_path) as pdf:
        # Title page
        fig = plt.figure(figsize=(8.5, 11))
        plt.axis('off')
        plt.text(0.5, 0.6, 'MedMarks MCQ Analysis Report', ha='center', fontsize=24, fontweight='bold')
        plt.text(0.5, 0.5, f'Models: {df["clean_model"].nunique()}', ha='center', fontsize=14)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Figures
        for fname in figures:
            fpath = output_dir / fname
            if not fpath.exists():
                continue
            fig = plt.figure(figsize=(11, 8.5))
            plt.axis('off')
            img = plt.imread(fpath)
            plt.imshow(img)
            plt.title(fname.replace('.png', '').replace('_', ' ').title(), fontsize=12)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

    print(f"  Saved to {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate MCQ analysis visualizations")
    parser.add_argument("--analysis-dir", type=str, required=True, help="Directory with analysis outputs")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for figures")

    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("MCQ Analysis Visualizations")
    print("="*60)

    # Load data
    print("\nLoading data...")
    try:
        df_models = load_model_metrics(analysis_dir)
        print(f"  Loaded {df_models['clean_model'].nunique()} models")
    except FileNotFoundError as e:
        print(f"  {e}")
        df_models = pd.DataFrame()

    try:
        df_bench = load_benchmark_metrics(analysis_dir)
        print(f"  Loaded {df_bench['benchmark'].nunique()} benchmarks")
    except FileNotFoundError as e:
        print(f"  {e}")
        df_bench = pd.DataFrame()

    if df_models.empty and df_bench.empty:
        print("\nNo data to visualize!")
        return

    # Generate visualizations
    print("\nGenerating visualizations...")

    if not df_models.empty:
        create_model_heatmap(df_models, output_dir)
        create_model_ranking(df_models, output_dir)
        create_variation_ranking(df_models, output_dir)
        create_scatter_plot(df_models, output_dir)
        create_correlation_heatmap(df_models, output_dir)
        create_summary_table(df_models, output_dir)

    if not df_bench.empty:
        create_violin_plots(df_bench, output_dir)
        create_benchmark_heatmap(df_bench, output_dir)

    # PDF report
    if not df_models.empty:
        create_pdf_report(output_dir, df_models)

    print("\n" + "="*60)
    print("Visualizations complete!")
    print("="*60)
    print(f"\nSaved to: {output_dir}")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
