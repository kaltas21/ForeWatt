"""
ForeWatt Experiments Package
============================
Comprehensive experiment suite for consumption and price forecasting models.

Modules:
- config: Central configuration for all experiments
- metrics: Comprehensive error metrics (MAE, RMSE, MAPE, sMAPE, MASE, etc.)
- plotting: Visualization utilities for results
- baselines: Naive and simple baseline models for comparison
- run_consumption_experiments: Consumption model experiment runner
- run_price_experiments: Price model experiment runner
- run_all_experiments: Main entry point for all experiments

Usage:
    # Run all experiments
    python -m experiments.run_all_experiments

    # Run only consumption experiments
    python -m experiments.run_consumption_experiments

    # Run only price experiments
    python -m experiments.run_price_experiments
"""

from experiments.config import (
    PROJECT_ROOT, DATA_PATH, RESULTS_DIR, PLOTS_DIR, LOGS_DIR,
    RUN_TIMESTAMP
)
from experiments.metrics import calculate_all_metrics, format_metrics_table
from experiments.plotting import (
    plot_predictions_vs_actual,
    plot_error_distribution,
    plot_feature_importance,
    plot_metrics_comparison
)

__all__ = [
    'PROJECT_ROOT', 'DATA_PATH', 'RESULTS_DIR', 'PLOTS_DIR', 'LOGS_DIR',
    'RUN_TIMESTAMP',
    'calculate_all_metrics', 'format_metrics_table',
    'plot_predictions_vs_actual', 'plot_error_distribution',
    'plot_feature_importance', 'plot_metrics_comparison',
]
