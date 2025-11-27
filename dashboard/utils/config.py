"""
ForeWatt Dashboard Configuration
Centralized configuration for paths, constants, and settings.
"""
import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
GOLD_DIR = DATA_DIR / "gold"
MASTER_DATA = GOLD_DIR / "master" / "master_v2_fundamental.csv"

# Model paths
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
MLARTIFACTS_DIR = PROJECT_ROOT / "mlartifacts"
LIGHTNING_LOGS_DIR = PROJECT_ROOT / "lightning_logs"

# Reports
REPORTS_DIR = PROJECT_ROOT / "reports"
BASELINE_REPORTS = REPORTS_DIR / "baseline" / "grid_search"
NEW_EXPERIMENT_DIR = REPORTS_DIR / "new_experiment"
NEW_BASELINE_DIR = NEW_EXPERIMENT_DIR / "baseline"
NEW_DEEPLEARNING_DIR = NEW_EXPERIMENT_DIR / "deeplearning"

# Forecasting constants
FORECAST_HORIZON = 24  # 1-24 hours ahead
TARGET_VARIABLE = "consumption"  # MWh
FREQUENCY = "H"  # Hourly data

# Model metadata
BASELINE_MODELS = {
    # "LightGBM": {
    #     "experiment_name": "ForeWatt-Baseline-LightGBM",
    #     "description": "Best performing model (MASE: 0.224)",
    #     "type": "baseline",
    #     "color": "#1f77b4"
    # },
    "CatBoost": {
        "experiment_name": "ForeWatt-Baseline-CatBoost",
        "description": "Gradient boosting with categorical handling",
        "type": "baseline",
        "color": "#ff7f0e"
    },
    "XGBoost": {
        "experiment_name": "ForeWatt-Baseline-XGBoost",
        "description": "Industry-standard gradient boosting",
        "type": "baseline",
        "color": "#2ca02c"
    },
    "Prophet": {
        "experiment_name": "ForeWatt-Baseline-Prophet",
        "description": "Interpretable time series model",
        "type": "baseline",
        "color": "#d62728"
    },
    "SARIMAX": {
        "experiment_name": "ForeWatt-Baseline-SARIMAX",
        "description": "Seasonal ARIMA with exogenous variables",
        "type": "baseline",
        "color": "#1f77b4"
    }
}

DEEP_LEARNING_MODELS = {
    "N-HiTS": {
        "description": "Neural Hierarchical Interpolation for Time Series",
        "type": "deep_learning",
        "color": "#9467bd"
    },
    "TFT": {
        "description": "Temporal Fusion Transformer",
        "type": "deep_learning",
        "color": "#8c564b"
    },
    "PatchTST": {
        "description": "Patched Time Series Transformer",
        "type": "deep_learning",
        "color": "#e377c2"
    }
}

# Metrics to display
METRICS_CONFIG = {
    "MAE": {"name": "Mean Absolute Error", "unit": "MWh", "lower_is_better": True},
    "RMSE": {"name": "Root Mean Squared Error", "unit": "MWh", "lower_is_better": True},
    "MASE": {"name": "Mean Absolute Scaled Error", "unit": "", "lower_is_better": True},
    "sMAPE": {"name": "Symmetric Mean Absolute Percentage Error", "unit": "%", "lower_is_better": True},
    "R2": {"name": "R² Score", "unit": "", "lower_is_better": False}
}

# Feature groups
FEATURE_GROUPS = {
    "Lag Features": ["consumption_lag_*", "temperature_*_lag_*", "price_*_lag_*"],
    "Rolling Features": ["consumption_roll_*", "temperature_*_roll_*"],
    "Calendar Features": ["hour", "day_*", "month", "holiday_*", "is_weekend"],
    "Weather Features": ["temperature_*", "humidity_*", "wind_*", "precipitation_*", "hdd_*", "cdd_*"],
    "Price Features": ["price_ptf*", "price_smf*", "price_idm*"],
    "External Features": ["usd_try", "eur_try", "gold_*", "m2_*", "tufe_*"]
}

# Streamlit page config
PAGE_CONFIG = {
    "page_title": "ForeWatt Dashboard",
    "page_icon": "⚡",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Plotting config
PLOT_CONFIG = {
    "height": 500,
    "template": "plotly_white",
    "font_family": "Inter, sans-serif"
}

# Color schemes
COLORS = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "success": "#2ca02c",
    "warning": "#ff9800",
    "error": "#d62728",
    "info": "#17a2b8",
    "background": "#f8f9fa",
    "text": "#212529"
}

# Validation periods (Train: 3 years, Val: 1 year, Test: 1 year)
VALIDATION_CONFIG = {
    "train_start": "2020-01-01",
    "train_end": "2022-12-31",
    "val_start": "2023-01-01",
    "val_end": "2023-12-31",
    "test_start": "2024-01-01",
    "test_end": "2024-12-31"
}
