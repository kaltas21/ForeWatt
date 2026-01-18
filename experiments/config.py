"""
Experiment Configuration
========================
Central configuration for all experiments.
"""

from pathlib import Path
from datetime import datetime

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / 'data' / 'gold' / 'master' / 'master_v2_fundamental.parquet'
EXPERIMENTS_DIR = PROJECT_ROOT / 'experiments'
RESULTS_DIR = EXPERIMENTS_DIR / 'results'
PLOTS_DIR = EXPERIMENTS_DIR / 'plots'
LOGS_DIR = EXPERIMENTS_DIR / 'logs'

# Create timestamp for this run
RUN_TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

# Test split configuration
# IMPORTANT: Must match original model test_start for fair comparison
TEST_START = '2024-06-01'  # Same as original price_train.py (17 months test)
VAL_RATIO = 0.15  # 15% of training for validation

# =====================================================================
# CONSUMPTION MODEL CONFIGURATION
# =====================================================================

# Full feature set for consumption (23 features)
CONSUMPTION_FEATURES_ALL = [
    'consumption_lag_24h', 'consumption_lag_48h', 'consumption_lag_168h',
    'consumption_rolling_mean_24h', 'consumption_rolling_std_24h',
    'temp_national', 'humidity_national', 'HDD', 'CDD', 'heat_index',
    'is_hot', 'is_cold', 'temp_lag_24h',
    'hour_sin', 'hour_cos', 'dow_sin_x', 'dow_cos_x',
    'month_sin', 'month_cos', 'is_weekend_x',
    'is_holiday_day', 'is_holiday_hour', 'price_ptf_lag_24h',
]

# Feature groups for ablation
CONSUMPTION_FEATURE_GROUPS = {
    'lag_only': [
        'consumption_lag_24h', 'consumption_lag_48h', 'consumption_lag_168h',
        'consumption_rolling_mean_24h', 'consumption_rolling_std_24h',
    ],
    'weather_only': [
        'temp_national', 'humidity_national', 'HDD', 'CDD', 'heat_index',
        'is_hot', 'is_cold', 'temp_lag_24h',
    ],
    'calendar_only': [
        'hour_sin', 'hour_cos', 'dow_sin_x', 'dow_cos_x',
        'month_sin', 'month_cos', 'is_weekend_x',
        'is_holiday_day', 'is_holiday_hour',
    ],
    'lag_weather': [
        'consumption_lag_24h', 'consumption_lag_48h', 'consumption_lag_168h',
        'consumption_rolling_mean_24h', 'consumption_rolling_std_24h',
        'temp_national', 'humidity_national', 'HDD', 'CDD', 'heat_index',
        'is_hot', 'is_cold', 'temp_lag_24h',
    ],
    'lag_calendar': [
        'consumption_lag_24h', 'consumption_lag_48h', 'consumption_lag_168h',
        'consumption_rolling_mean_24h', 'consumption_rolling_std_24h',
        'hour_sin', 'hour_cos', 'dow_sin_x', 'dow_cos_x',
        'month_sin', 'month_cos', 'is_weekend_x',
        'is_holiday_day', 'is_holiday_hour',
    ],
    'all': CONSUMPTION_FEATURES_ALL,
}

# CatBoost hyperparameters for consumption
CONSUMPTION_CATBOOST_PARAMS = {
    'iterations': 1000,
    'depth': 5,
    'learning_rate': 0.03,
    'l2_leaf_reg': 15.0,
    'border_count': 128,
    'random_seed': 42,
    'loss_function': 'RMSE',
    'eval_metric': 'MAE',
    'early_stopping_rounds': 50,
    'subsample': 0.8,
    'rsm': 0.8,
}

# =====================================================================
# PRICE MODEL CONFIGURATION
# =====================================================================

# Base feature set for price (21 features)
PRICE_FEATURES_BASE = [
    'hour_sin', 'hour_cos', 'dow_sin_x', 'dow_cos_x', 'is_weekend_x',
    'price_ptf_rolling_std_24h', 'price_ptf_rolling_mean_24h',
    'price_ptf_rolling_min_24h', 'price_ptf_rolling_max_24h',
    'price_ptf_lag_24h', 'price_ptf_lag_168h',
    'thermal_gap', 'thermal_gap_lag_24h',
    'renewable_saturation', 'spark_spread_proxy_lag_24h',
    'system_short_signal', 'load_factor', 'consumption_forecast',
    'reserve_margin_ratio', 'price_volatility_lag24h', 'realtime_premium_lag24h',
]

# Feature groups for price ablation
PRICE_FEATURE_GROUPS = {
    'price_lags_only': [
        'price_ptf_rolling_std_24h', 'price_ptf_rolling_mean_24h',
        'price_ptf_rolling_min_24h', 'price_ptf_rolling_max_24h',
        'price_ptf_lag_24h', 'price_ptf_lag_168h',
        'price_volatility_lag24h', 'realtime_premium_lag24h',
    ],
    'market_signals_only': [
        'thermal_gap', 'thermal_gap_lag_24h',
        'renewable_saturation', 'spark_spread_proxy_lag_24h',
        'system_short_signal', 'load_factor', 'consumption_forecast',
        'reserve_margin_ratio',
    ],
    'calendar_only': [
        'hour_sin', 'hour_cos', 'dow_sin_x', 'dow_cos_x', 'is_weekend_x',
    ],
    'price_market': [
        'price_ptf_rolling_std_24h', 'price_ptf_rolling_mean_24h',
        'price_ptf_rolling_min_24h', 'price_ptf_rolling_max_24h',
        'price_ptf_lag_24h', 'price_ptf_lag_168h',
        'price_volatility_lag24h', 'realtime_premium_lag24h',
        'thermal_gap', 'thermal_gap_lag_24h',
        'renewable_saturation', 'spark_spread_proxy_lag_24h',
        'system_short_signal', 'load_factor', 'consumption_forecast',
        'reserve_margin_ratio',
    ],
    'all': PRICE_FEATURES_BASE,
}

# CatBoost hyperparameters for price
PRICE_CATBOOST_PARAMS = {
    'iterations': 2000,
    'depth': 8,
    'learning_rate': 0.02,
    'l2_leaf_reg': 3,
    'random_seed': 42,
    'loss_function': 'MAE',
    'verbose': False,
    'early_stopping_rounds': 100,
}

# LightGBM hyperparameters for price
PRICE_LIGHTGBM_PARAMS = {
    'objective': 'mae',
    'n_estimators': 2000,
    'max_depth': 8,
    'learning_rate': 0.02,
    'num_leaves': 127,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'random_state': 42,
    'verbosity': -1,
}

# Ensemble weights
CATBOOST_WEIGHT = 0.658
LIGHTGBM_WEIGHT = 0.413

# =====================================================================
# DATA SIZE EXPERIMENTS
# =====================================================================

# Different training periods to test
# Note: train_end is implicitly TEST_START (2024-06-01) minus finetune period
DATA_SIZE_EXPERIMENTS = {
    '1_year': {'start': '2023-01-01', 'end': None},  # 1 year before finetune
    '2_years': {'start': '2022-01-01', 'end': None},  # 2 years
    '3_years': {'start': '2021-01-01', 'end': None},  # 3 years
    '4_years': {'start': '2020-01-01', 'end': None},  # 4 years (full data)
    'full': {'start': None, 'end': None},  # Use all available data
}

# =====================================================================
# METRICS TO TRACK
# =====================================================================

METRICS_LIST = ['MAE', 'RMSE', 'MAPE', 'sMAPE', 'MASE', 'MBE', 'R2']
