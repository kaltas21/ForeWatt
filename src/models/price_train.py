"""
Price Forecasting Model Training - CHybrid V14
===============================================
Isolated training script for price forecasting using CatBoost+LightGBM Ensemble.

This is an isolated, self-contained training script that can be run independently.
The trained models are saved to: ForeWatt/models/price/

Features:
- Transfer Learning (Base + Fine-tune)
- CatBoost + LightGBM Ensemble
- Profile Evolution Features
- Context-Aware KNN Error Correction

Usage:
    python src/models/price_train.py

Author: ForeWatt Team
Date: December 2025
"""

import sys
import os
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / 'data' / 'gold' / 'master' / 'master_v2_fundamental.parquet'
MODELS_OUTPUT_DIR = PROJECT_ROOT / 'models' / 'price'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Ensemble weights
CATBOOST_WEIGHT = 0.658
LIGHTGBM_WEIGHT = 0.413

# KNN Parameters
KNN_K = 5
KNN_LOOKBACK_DAYS = 45

# Context features for similarity
CONTEXT_FEATURES = ['load_factor', 'renewable_saturation', 'thermal_gap']

# Problem hours (morning ramp)
PROBLEM_HOURS = [9, 10]

# Base feature set (21 features)
BASE_FEATURES = [
    'hour_sin', 'hour_cos', 'dow_sin_x', 'dow_cos_x', 'is_weekend_x',
    'price_ptf_rolling_std_24h', 'price_ptf_rolling_mean_24h',
    'price_ptf_rolling_min_24h', 'price_ptf_rolling_max_24h',
    'price_ptf_lag_24h', 'price_ptf_lag_168h',
    'thermal_gap', 'thermal_gap_lag_24h',
    'renewable_saturation', 'spark_spread_proxy_lag_24h',
    'system_short_signal', 'load_factor', 'consumption_forecast',
    'reserve_margin_ratio', 'price_volatility_lag24h', 'realtime_premium_lag24h',
]


# =============================================================================
# PROFILE FEATURES
# =============================================================================

def create_enhanced_profile_features(df: pd.DataFrame, price_col: str = 'price_real') -> Tuple[pd.DataFrame, List[str]]:
    """Create enhanced Profile Evolution Features including Solar Profile."""
    df = df.copy()
    new_features = []

    df['hour'] = df.index.hour

    # V10 PRICE PROFILE FEATURES
    df['daily_avg_price'] = df[price_col].shift(1).rolling(24, min_periods=12).mean()
    df['hourly_ratio'] = (df[price_col].shift(1) / df['daily_avg_price'].shift(1)).clip(0.2, 5.0)
    new_features.append('hourly_ratio')

    profile_14d_list = []
    profile_28d_list = []

    for hour in range(24):
        hour_mask = df['hour'] == hour
        hour_ratios = df.loc[hour_mask, 'hourly_ratio']
        p14 = hour_ratios.rolling(14, min_periods=7).mean().shift(1)
        p28 = hour_ratios.rolling(28, min_periods=14).mean().shift(1)
        profile_14d_list.append(p14)
        profile_28d_list.append(p28)

    df['profile_14d'] = pd.concat(profile_14d_list).sort_index()
    df['profile_28d'] = pd.concat(profile_28d_list).sort_index()
    new_features.extend(['profile_14d', 'profile_28d'])

    df['profile_momentum'] = df['profile_14d'] - df['profile_28d']
    new_features.append('profile_momentum')

    df['daily_avg_momentum'] = df['daily_avg_price'] - df['daily_avg_price'].shift(24)
    new_features.append('daily_avg_momentum')

    # V11 SOLAR PROFILE FEATURES
    if 'renewable_saturation' in df.columns and 'load_factor' in df.columns:
        load = df['load_factor'].clip(lower=0.1)
        df['solar_ratio'] = (df['renewable_saturation'].shift(1) / load.shift(1)).clip(0, 5)
        new_features.append('solar_ratio')

        solar_14d_list = []
        solar_28d_list = []

        for hour in range(24):
            hour_mask = df['hour'] == hour
            hour_solar = df.loc[hour_mask, 'solar_ratio']
            s14 = hour_solar.rolling(14, min_periods=7).mean().shift(1)
            s28 = hour_solar.rolling(28, min_periods=14).mean().shift(1)
            solar_14d_list.append(s14)
            solar_28d_list.append(s28)

        df['solar_profile_14d'] = pd.concat(solar_14d_list).sort_index()
        df['solar_profile_28d'] = pd.concat(solar_28d_list).sort_index()
        new_features.extend(['solar_profile_14d', 'solar_profile_28d'])

        df['solar_momentum'] = df['solar_profile_14d'] - df['solar_profile_28d']
        new_features.append('solar_momentum')

    if 'solar_momentum' in df.columns:
        df['price_solar_interaction'] = df['profile_14d'] * df['solar_momentum']
        new_features.append('price_solar_interaction')

    for feat in new_features:
        if feat in df.columns and df[feat].isna().any():
            median_val = df[feat].median()
            df[feat] = df[feat].fillna(median_val)

    return df, new_features


# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

def load_data() -> pd.DataFrame:
    """Load master dataset."""
    logger.info(f"Loading data from: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)

    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date' in df.columns:
            df = df.set_index('date')
        elif 'datetime' in df.columns:
            df = df.set_index('datetime')

    if 'price_real' not in df.columns:
        df['price_real'] = df['price']

    logger.info(f"Loaded data: {df.shape}")
    return df


def prepare_data(df: pd.DataFrame, features: List[str],
                 test_start: str = '2024-06-01',
                 finetune_months: int = 6) -> Dict:
    """Prepare data with splits."""
    available = [f for f in features if f in df.columns]

    X = df[available].copy()
    y = df['price_real'].copy()

    X_with_price = X.copy()
    X_with_price['price_real'] = y
    X_with_price, profile_features = create_enhanced_profile_features(X_with_price, 'price_real')
    X = X_with_price.drop(columns=['price_real'])

    X['hour'] = X.index.hour

    mask = ~(X.isna().any(axis=1) | y.isna())
    X, y = X[mask], y[mask]
    hours = X['hour']

    tz = X.index.tz
    test_start_dt = pd.Timestamp(test_start, tz=tz)
    finetune_start_dt = test_start_dt - pd.DateOffset(months=finetune_months)

    base_mask = X.index < finetune_start_dt
    finetune_mask = (X.index >= finetune_start_dt) & (X.index < test_start_dt)
    test_mask = X.index >= test_start_dt

    return {
        'X_base': X[base_mask],
        'y_base': y[base_mask],
        'X_finetune': X[finetune_mask],
        'y_finetune': y[finetune_mask],
        'X_test': X[test_mask],
        'y_test': y[test_mask],
        'hours_base': hours[base_mask],
        'hours_finetune': hours[finetune_mask],
        'hours_test': hours[test_mask],
        'features': list(X.columns),
    }


# =============================================================================
# METRICS
# =============================================================================

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Calculate metrics."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae = np.mean(np.abs(y_true - y_pred))
    smape = 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))
    bias = np.mean(y_pred - y_true)

    return {'mae': mae, 'smape': smape, 'bias': bias}


# =============================================================================
# TRANSFER LEARNING TRAINERS
# =============================================================================

def train_catboost_transfer(data: Dict) -> Tuple[object, np.ndarray, np.ndarray]:
    """Train CatBoost with transfer learning."""
    from catboost import CatBoostRegressor

    logger.info("\n  Training CatBoost Transfer Learning...")

    X_base, y_base = data['X_base'], data['y_base']
    split_idx = int(len(X_base) * 0.85)

    base_model = CatBoostRegressor(
        loss_function='MAE',
        iterations=2000,
        depth=8,
        learning_rate=0.02,
        l2_leaf_reg=3,
        random_state=42,
        verbose=False,
        early_stopping_rounds=100,
    )
    base_model.fit(
        X_base.iloc[:split_idx], y_base.iloc[:split_idx],
        eval_set=(X_base.iloc[split_idx:], y_base.iloc[split_idx:]),
        verbose=False
    )
    logger.info(f"    Base: {base_model.tree_count_} trees")

    X_ft, y_ft = data['X_finetune'], data['y_finetune']
    split_idx = int(len(X_ft) * 0.8)

    finetune_model = CatBoostRegressor(
        loss_function='MAE',
        iterations=500,
        depth=8,
        learning_rate=0.005,
        l2_leaf_reg=3,
        random_state=42,
        verbose=False,
        early_stopping_rounds=50,
    )
    finetune_model.fit(
        X_ft.iloc[:split_idx], y_ft.iloc[:split_idx],
        eval_set=(X_ft.iloc[split_idx:], y_ft.iloc[split_idx:]),
        init_model=base_model,
        verbose=False
    )
    logger.info(f"    Fine-tuned: {finetune_model.tree_count_} trees")

    finetune_pred = finetune_model.predict(data['X_finetune'])
    test_pred = finetune_model.predict(data['X_test'])

    return finetune_model, finetune_pred, test_pred


def train_lightgbm_transfer(data: Dict) -> Tuple[object, np.ndarray, np.ndarray]:
    """Train LightGBM with transfer learning."""
    import lightgbm as lgb

    logger.info("\n  Training LightGBM Transfer Learning...")

    X_base, y_base = data['X_base'], data['y_base']
    split_idx = int(len(X_base) * 0.85)

    base_model = lgb.LGBMRegressor(
        objective='mae',
        n_estimators=2000,
        max_depth=8,
        learning_rate=0.02,
        num_leaves=127,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        verbosity=-1,
    )
    base_model.fit(
        X_base.iloc[:split_idx], y_base.iloc[:split_idx],
        eval_set=[(X_base.iloc[split_idx:], y_base.iloc[split_idx:])],
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    logger.info(f"    Base: {base_model.n_estimators_} trees")

    X_ft, y_ft = data['X_finetune'], data['y_finetune']
    split_idx = int(len(X_ft) * 0.8)

    finetune_model = lgb.LGBMRegressor(
        objective='mae',
        n_estimators=500,
        max_depth=8,
        learning_rate=0.005,
        num_leaves=127,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        verbosity=-1,
    )
    finetune_model.fit(
        X_ft.iloc[:split_idx], y_ft.iloc[:split_idx],
        eval_set=[(X_ft.iloc[split_idx:], y_ft.iloc[split_idx:])],
        callbacks=[lgb.early_stopping(50, verbose=False)],
        init_model=base_model.booster_
    )
    logger.info(f"    Fine-tuned: {finetune_model.n_estimators_} trees")

    finetune_pred = finetune_model.predict(data['X_finetune'])
    test_pred = finetune_model.predict(data['X_test'])

    return finetune_model, finetune_pred, test_pred


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train_price_model():
    """Train price forecasting ensemble model."""
    MODELS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("="*70)
    logger.info("PRICE MODEL TRAINING - CHybrid V14")
    logger.info("CatBoost + LightGBM Ensemble with Transfer Learning")
    logger.info("="*70)

    # Load data
    df = load_data()

    # Prepare data
    data = prepare_data(df, BASE_FEATURES)

    logger.info(f"\n  DATA SPLITS:")
    logger.info(f"    Base:      {len(data['X_base']):,} rows")
    logger.info(f"    Fine-tune: {len(data['X_finetune']):,} rows")
    logger.info(f"    Test:      {len(data['X_test']):,} rows")

    # Train ensemble
    logger.info("\n" + "="*70)
    logger.info("TRAINING ENSEMBLE")
    logger.info("="*70)

    catboost_model, cat_ft_pred, cat_test_pred = train_catboost_transfer(data)
    lightgbm_model, lgb_ft_pred, lgb_test_pred = train_lightgbm_transfer(data)

    # Calculate ensemble weights
    total_weight = CATBOOST_WEIGHT + LIGHTGBM_WEIGHT
    w_cat = CATBOOST_WEIGHT / total_weight
    w_lgb = LIGHTGBM_WEIGHT / total_weight

    logger.info(f"\n  Ensemble weights: CatBoost={w_cat:.3f}, LightGBM={w_lgb:.3f}")

    # Ensemble predictions
    raw_test_pred = w_cat * cat_test_pred + w_lgb * lgb_test_pred

    # Evaluate
    test_metrics = evaluate(data['y_test'].values, raw_test_pred)
    logger.info(f"\n  TEST METRICS:")
    logger.info(f"    MAE:   {test_metrics['mae']:.2f}")
    logger.info(f"    sMAPE: {test_metrics['smape']:.2f}%")

    # Save models
    logger.info("\n" + "="*70)
    logger.info("SAVING MODELS")
    logger.info("="*70)

    # Save CatBoost model
    catboost_path = MODELS_OUTPUT_DIR / 'catboost_v14.cbm'
    catboost_model.save_model(str(catboost_path))
    logger.info(f"  Saved CatBoost: {catboost_path}")

    # Save LightGBM model
    lightgbm_path = MODELS_OUTPUT_DIR / 'lightgbm_v14.txt'
    lightgbm_model.booster_.save_model(str(lightgbm_path))
    logger.info(f"  Saved LightGBM: {lightgbm_path}")

    # Save feature list
    feature_list = data['features']
    features_path = MODELS_OUTPUT_DIR / 'features.json'
    with open(features_path, 'w') as f:
        json.dump({
            'features': feature_list,
            'base_features': BASE_FEATURES,
            'n_features': len(feature_list),
        }, f, indent=2)
    logger.info(f"  Saved features: {features_path}")

    # Save ensemble config
    config_path = MODELS_OUTPUT_DIR / 'ensemble_config.json'
    with open(config_path, 'w') as f:
        json.dump({
            'catboost_weight': w_cat,
            'lightgbm_weight': w_lgb,
            'test_smape': test_metrics['smape'],
            'test_mae': test_metrics['mae'],
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2)
    logger.info(f"  Saved config: {config_path}")

    logger.info(f"\n{'='*70}")
    logger.info(f"TRAINING COMPLETE")
    logger.info(f"Models saved to: {MODELS_OUTPUT_DIR}")
    logger.info(f"Final sMAPE: {test_metrics['smape']:.2f}%")
    logger.info(f"{'='*70}")

    return {
        'catboost_model': catboost_model,
        'lightgbm_model': lightgbm_model,
        'test_metrics': test_metrics,
        'features': feature_list,
    }


if __name__ == "__main__":
    train_price_model()
