"""
Plotting Utilities for Experiments
==================================
Generate visualizations for all experiments.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10


def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: pd.DatetimeIndex,
    title: str,
    save_path: Path,
    show_last_n_days: int = 14
) -> None:
    """
    Plot predictions vs actual values.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        dates: Datetime index
        title: Plot title
        save_path: Where to save the plot
        show_last_n_days: Number of days to show (for clarity)
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Full series (sampled for visibility)
    ax1 = axes[0]
    sample_step = max(1, len(y_true) // 1000)  # Sample for performance
    ax1.plot(dates[::sample_step], y_true[::sample_step], 'b-', alpha=0.7, label='Actual', linewidth=0.8)
    ax1.plot(dates[::sample_step], y_pred[::sample_step], 'r-', alpha=0.7, label='Predicted', linewidth=0.8)
    ax1.set_title(f'{title} - Full Test Period')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Last N days zoomed
    ax2 = axes[1]
    n_hours = show_last_n_days * 24
    ax2.plot(dates[-n_hours:], y_true[-n_hours:], 'b-', alpha=0.8, label='Actual', linewidth=1.2)
    ax2.plot(dates[-n_hours:], y_pred[-n_hours:], 'r--', alpha=0.8, label='Predicted', linewidth=1.2)
    ax2.set_title(f'{title} - Last {show_last_n_days} Days')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Value')
    ax2.legend()
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    save_path: Path
) -> None:
    """
    Plot error distribution histogram and box plot.
    """
    errors = y_pred - y_true
    abs_errors = np.abs(errors)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Error histogram
    ax1 = axes[0, 0]
    ax1.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    ax1.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero')
    ax1.axvline(np.mean(errors), color='g', linestyle='-', linewidth=2, label=f'Mean: {np.mean(errors):.2f}')
    ax1.set_title('Error Distribution (Pred - Actual)')
    ax1.set_xlabel('Error')
    ax1.set_ylabel('Frequency')
    ax1.legend()

    # Absolute error histogram
    ax2 = axes[0, 1]
    ax2.hist(abs_errors, bins=50, edgecolor='black', alpha=0.7, color='orange')
    ax2.axvline(np.mean(abs_errors), color='r', linestyle='-', linewidth=2, label=f'MAE: {np.mean(abs_errors):.2f}')
    ax2.axvline(np.median(abs_errors), color='g', linestyle='--', linewidth=2, label=f'Median: {np.median(abs_errors):.2f}')
    ax2.set_title('Absolute Error Distribution')
    ax2.set_xlabel('Absolute Error')
    ax2.set_ylabel('Frequency')
    ax2.legend()

    # Box plot of errors by percentile
    ax3 = axes[1, 0]
    bp = ax3.boxplot([errors], vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    ax3.set_title('Error Box Plot')
    ax3.set_ylabel('Error')

    # Q-Q plot
    ax4 = axes[1, 1]
    sorted_errors = np.sort(errors)
    n = len(sorted_errors)
    theoretical = np.linspace(-3, 3, n)
    ax4.scatter(theoretical, sorted_errors, alpha=0.5, s=1)
    ax4.plot([-3, 3], [-3 * np.std(errors), 3 * np.std(errors)], 'r--', linewidth=2)
    ax4.set_title('Q-Q Plot (vs Normal)')
    ax4.set_xlabel('Theoretical Quantiles')
    ax4.set_ylabel('Sample Quantiles')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_hourly_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    hours: np.ndarray,
    title: str,
    save_path: Path
) -> None:
    """
    Plot error metrics by hour of day.
    """
    errors = y_pred - y_true
    abs_errors = np.abs(errors)

    hourly_mae = []
    hourly_mbe = []
    hourly_std = []

    for h in range(24):
        mask = hours == h
        if mask.sum() > 0:
            hourly_mae.append(np.mean(abs_errors[mask]))
            hourly_mbe.append(np.mean(errors[mask]))
            hourly_std.append(np.std(errors[mask]))
        else:
            hourly_mae.append(0)
            hourly_mbe.append(0)
            hourly_std.append(0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # MAE by hour
    ax1 = axes[0]
    bars1 = ax1.bar(range(24), hourly_mae, color='steelblue', edgecolor='black')
    ax1.set_title('MAE by Hour')
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('MAE')
    ax1.set_xticks(range(0, 24, 2))

    # MBE by hour (bias)
    ax2 = axes[1]
    colors = ['green' if x >= 0 else 'red' for x in hourly_mbe]
    bars2 = ax2.bar(range(24), hourly_mbe, color=colors, edgecolor='black')
    ax2.axhline(0, color='black', linestyle='-', linewidth=1)
    ax2.set_title('Mean Bias Error by Hour')
    ax2.set_xlabel('Hour')
    ax2.set_ylabel('Bias (Pred - Actual)')
    ax2.set_xticks(range(0, 24, 2))

    # Std by hour
    ax3 = axes[2]
    bars3 = ax3.bar(range(24), hourly_std, color='orange', edgecolor='black')
    ax3.set_title('Error Std Dev by Hour')
    ax3.set_xlabel('Hour')
    ax3.set_ylabel('Std Dev')
    ax3.set_xticks(range(0, 24, 2))

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_feature_importance(
    importance_df: pd.DataFrame,
    title: str,
    save_path: Path,
    top_n: int = 20
) -> None:
    """
    Plot feature importance bar chart.
    """
    df = importance_df.head(top_n).copy()
    df = df.sort_values('importance', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(df)))
    bars = ax.barh(df['feature'], df['importance'], color=colors, edgecolor='black')

    ax.set_xlabel('Importance')
    ax.set_title(title)

    # Add value labels
    for bar, val in zip(bars, df['importance']):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_history(
    train_metrics: List[float],
    val_metrics: List[float],
    metric_name: str,
    title: str,
    save_path: Path
) -> None:
    """
    Plot training history (loss/metric over iterations).
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    iterations = range(1, len(train_metrics) + 1)
    ax.plot(iterations, train_metrics, 'b-', label='Train', linewidth=1.5)
    ax.plot(iterations, val_metrics, 'r-', label='Validation', linewidth=1.5)

    ax.set_xlabel('Iteration')
    ax.set_ylabel(metric_name)
    ax.set_title(title)
    ax.legend()

    # Mark best validation
    best_idx = np.argmin(val_metrics)
    ax.axvline(best_idx + 1, color='g', linestyle='--', alpha=0.7, label=f'Best: {best_idx + 1}')
    ax.scatter([best_idx + 1], [val_metrics[best_idx]], color='g', s=100, zorder=5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_metrics_comparison(
    results: Dict[str, Dict[str, float]],
    metrics_to_plot: List[str],
    title: str,
    save_path: Path
) -> None:
    """
    Plot comparison of metrics across experiments.
    """
    n_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))

    if n_metrics == 1:
        axes = [axes]

    experiments = list(results.keys())
    x = np.arange(len(experiments))

    for ax, metric in zip(axes, metrics_to_plot):
        values = [results[exp].get(metric, 0) for exp in experiments]

        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(values)))
        bars = ax.bar(x, values, color=colors, edgecolor='black')

        ax.set_ylabel(metric)
        ax.set_title(f'{metric} Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(experiments, rotation=45, ha='right')

        # Add value labels
        for bar, val in zip(bars, values):
            suffix = '%' if metric in ['MAPE', 'sMAPE'] else ''
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.2f}{suffix}', ha='center', va='bottom', fontsize=9)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_data_size_impact(
    results: Dict[str, Dict[str, float]],
    save_path: Path,
    model_type: str = 'Consumption'
) -> None:
    """
    Plot how training data size affects performance.
    """
    # Order by data size
    size_order = ['1_year', '2_years', '3_years', '4_years', 'full']
    ordered_results = {k: results[k] for k in size_order if k in results}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    experiments = list(ordered_results.keys())
    x = range(len(experiments))

    metrics = ['MAE', 'sMAPE', 'RMSE', 'R2']
    colors = ['steelblue', 'coral', 'seagreen', 'mediumpurple']

    for ax, metric, color in zip(axes.flat, metrics, colors):
        values = [ordered_results[exp].get(metric, 0) for exp in experiments]

        ax.plot(x, values, 'o-', color=color, linewidth=2, markersize=10)
        ax.fill_between(x, values, alpha=0.3, color=color)

        ax.set_xticks(x)
        ax.set_xticklabels(experiments)
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} vs Training Data Size')

        # Add value labels
        for i, val in enumerate(values):
            suffix = '%' if metric in ['MAPE', 'sMAPE'] else ''
            ax.annotate(f'{val:.2f}{suffix}', (i, val), textcoords="offset points",
                       xytext=(0, 10), ha='center', fontsize=9)

    plt.suptitle(f'{model_type} Model: Impact of Training Data Size', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_scatter_actual_vs_pred(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    save_path: Path
) -> None:
    """
    Scatter plot of actual vs predicted values.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Sample for performance if too many points
    n = len(y_true)
    if n > 5000:
        idx = np.random.choice(n, 5000, replace=False)
        y_true_plot = y_true[idx]
        y_pred_plot = y_pred[idx]
    else:
        y_true_plot = y_true
        y_pred_plot = y_pred

    ax.scatter(y_true_plot, y_pred_plot, alpha=0.3, s=10)

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(title)
    ax.legend()

    # Add R² annotation
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
            fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_summary_dashboard(
    experiment_name: str,
    metrics: Dict[str, float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: pd.DatetimeIndex,
    hours: np.ndarray,
    save_path: Path,
    feature_importance: Optional[pd.DataFrame] = None
) -> None:
    """
    Create a comprehensive summary dashboard for an experiment.
    """
    n_cols = 3 if feature_importance is not None else 2
    fig = plt.figure(figsize=(6 * n_cols, 12))

    # 1. Predictions vs Actual (last 7 days)
    ax1 = fig.add_subplot(2, n_cols, 1)
    n_hours = 7 * 24
    ax1.plot(dates[-n_hours:], y_true[-n_hours:], 'b-', label='Actual', linewidth=1)
    ax1.plot(dates[-n_hours:], y_pred[-n_hours:], 'r--', label='Predicted', linewidth=1)
    ax1.set_title('Last 7 Days: Actual vs Predicted')
    ax1.legend()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 2. Error distribution
    ax2 = fig.add_subplot(2, n_cols, 2)
    errors = y_pred - y_true
    ax2.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    ax2.axvline(0, color='r', linestyle='--', linewidth=2)
    ax2.set_title('Error Distribution')
    ax2.set_xlabel('Error')

    # 3. Feature importance (if provided)
    if feature_importance is not None:
        ax3 = fig.add_subplot(2, n_cols, 3)
        top_n = min(15, len(feature_importance))
        df = feature_importance.head(top_n).sort_values('importance', ascending=True)
        ax3.barh(df['feature'], df['importance'], color='steelblue', edgecolor='black')
        ax3.set_title('Top Feature Importance')
        ax3.set_xlabel('Importance')

    # 4. Hourly MAE
    ax4 = fig.add_subplot(2, n_cols, n_cols + 1)
    abs_errors = np.abs(errors)
    hourly_mae = [np.mean(abs_errors[hours == h]) if (hours == h).sum() > 0 else 0 for h in range(24)]
    ax4.bar(range(24), hourly_mae, color='coral', edgecolor='black')
    ax4.set_title('MAE by Hour')
    ax4.set_xlabel('Hour')
    ax4.set_ylabel('MAE')

    # 5. Scatter plot
    ax5 = fig.add_subplot(2, n_cols, n_cols + 2)
    sample_idx = np.random.choice(len(y_true), min(2000, len(y_true)), replace=False)
    ax5.scatter(y_true[sample_idx], y_pred[sample_idx], alpha=0.3, s=5)
    min_val, max_val = y_true.min(), y_true.max()
    ax5.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax5.set_title(f'Actual vs Predicted (R²={metrics.get("R2", 0):.4f})')
    ax5.set_xlabel('Actual')
    ax5.set_ylabel('Predicted')

    # 6. Metrics table
    ax6 = fig.add_subplot(2, n_cols, 2 * n_cols) if n_cols == 3 else None
    if ax6:
        ax6.axis('off')
        table_data = [[k, f'{v:.4f}' + ('%' if k in ['MAPE', 'sMAPE'] else '')]
                     for k, v in list(metrics.items())[:10]]
        table = ax6.table(cellText=table_data, colLabels=['Metric', 'Value'],
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax6.set_title('Metrics Summary')

    plt.suptitle(f'Experiment: {experiment_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
