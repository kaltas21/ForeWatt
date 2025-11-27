"""
ForeWatt Dashboard Utilities
Centralized utilities for data loading, model operations, metrics, and plotting.
"""

from .config import *
from .data_loader import *
from .model_loader import *
from .model_loader_v2 import *
from .metrics import *
from .plotting import *
from .checkpoint_loader import *

__all__ = [
    # Config
    'PROJECT_ROOT', 'DATA_DIR', 'GOLD_DIR', 'MASTER_DATA',
    'MLRUNS_DIR', 'BASELINE_MODELS', 'DEEP_LEARNING_MODELS',
    'METRICS_CONFIG', 'PLOT_CONFIG', 'COLORS', 'VALIDATION_CONFIG',
    'TARGET_VARIABLE', 'PAGE_CONFIG',

    # Data loader
    'load_master_data', 'get_data_summary', 'split_train_test',
    'get_recent_data', 'get_feature_groups', 'get_date_range_options',
    'apply_date_filter', 'get_hourly_patterns', 'get_daily_patterns',

    # Model loader (MLflow) - Legacy
    'get_mlflow_experiments', 'get_experiment_runs', 'load_best_model',
    'get_model_metrics', 'make_forecast',
    'load_all_model_metrics', 'simulate_predictions',

    # Model loader V2 (New Experiments) - Primary
    'load_all_metrics', 'get_best_models_per_type', 'load_model_config',
    'get_model_summary', 'get_model_comparison_df', 'load_feature_importance',
    'get_available_models',  # V2 version overrides old MLflow version

    # Checkpoint loader (Deep Learning)
    'get_available_dl_runs', 'load_checkpoint_metrics', 'extract_training_curve_from_metadata',
    'get_model_config_summary',

    # Metrics
    'calculate_mae', 'calculate_rmse', 'calculate_mape', 'calculate_smape',
    'calculate_mase', 'calculate_r2', 'calculate_all_metrics',
    'calculate_horizon_metrics', 'calculate_residuals', 'calculate_coverage',
    'compare_models', 'calculate_skill_score', 'format_metric_value',
    'rolling_metric', 'calculate_directional_accuracy', 'calculate_error_percentiles',

    # Plotting
    'create_forecast_plot', 'create_horizon_performance_plot',
    'create_metrics_comparison', 'create_residual_plot',
    'create_time_series_plot', 'create_correlation_heatmap',
    'create_feature_importance_plot', 'create_box_plot',
    'create_hourly_pattern_plot', 'create_scatter_plot', 'create_gauge_chart',
    'create_split_visualization', 'create_learning_curve_plot',
    'create_error_analysis_plot'
]
