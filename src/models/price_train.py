"""
Price Forecasting Model Training - CHybrid V14
===============================================
Isolated training script for price forecasting using CatBoost+LightGBM Ensemble
with Context-Aware KNN Error Correction.

This is an isolated, self-contained training script that can be run independently.
The trained models are saved to: ForeWatt/models/price/

Features:
- Transfer Learning (Base + Fine-tune)
- CatBoost + LightGBM Ensemble
- Profile Evolution Features
- Simple AEC (Adaptive Error Correction)
- Context-Aware KNN Error Correction
- Hybrid Correction (50% Simple + 50% KNN)

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
from sklearn.model_selection import TimeSeriesSplit

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

# V13 baseline (Oracle floor)
BASELINE_SMAPE = 11.89
ORACLE_FLOOR = 11.75

# Problem hours (morning ramp)
PROBLEM_HOURS = [9, 10]

# Ensemble weights
CATBOOST_WEIGHT = 0.658
LIGHTGBM_WEIGHT = 0.413

# KNN Parameters
KNN_K = 5
KNN_LOOKBACK_DAYS = 45

# Cross-Validation Parameters
CV_N_SPLITS = 5  # Number of time-series CV folds

# Context features for similarity
CONTEXT_FEATURES = ['load_factor', 'renewable_saturation', 'thermal_gap']

# V13 Hourly AEC Parameters (optimized)
HOURLY_AEC_PARAMS = {
    0: {'lookback': 14, 'damping': 0.5},
    1: {'lookback': 14, 'damping': 0.5},
    2: {'lookback': 21, 'damping': 0.5},
    3: {'lookback': 7, 'damping': 0.5},
    4: {'lookback': 21, 'damping': 0.5},
    5: {'lookback': 7, 'damping': 0.7},
    6: {'lookback': 21, 'damping': 0.5},
    7: {'lookback': 21, 'damping': 0.5},
    8: {'lookback': 21, 'damping': 0.5},
    9: {'lookback': 7, 'damping': 0.5},
    10: {'lookback': 7, 'damping': 0.5},
    11: {'lookback': 5, 'damping': 0.7},
    12: {'lookback': 7, 'damping': 0.5},
    13: {'lookback': 7, 'damping': 0.5},
    14: {'lookback': 21, 'damping': 0.6},
    15: {'lookback': 7, 'damping': 0.5},
    16: {'lookback': 5, 'damping': 0.5},
    17: {'lookback': 7, 'damping': 0.5},
    18: {'lookback': 7, 'damping': 0.5},
    19: {'lookback': 21, 'damping': 0.5},
    20: {'lookback': 14, 'damping': 0.5},
    21: {'lookback': 21, 'damping': 0.7},
    22: {'lookback': 7, 'damping': 0.5},
    23: {'lookback': 7, 'damping': 0.5},
}

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


def evaluate_with_breakdown(y_true: np.ndarray, y_pred: np.ndarray, hours: np.ndarray) -> Dict:
    """Evaluate with hour breakdown."""
    global_metrics = evaluate(y_true, y_pred)

    problem_mask = np.isin(hours, PROBLEM_HOURS)
    problem_metrics = evaluate(y_true[problem_mask], y_pred[problem_mask]) if problem_mask.sum() > 0 else {}
    problem_metrics['count'] = int(problem_mask.sum())

    other_mask = ~problem_mask
    other_metrics = evaluate(y_true[other_mask], y_pred[other_mask]) if other_mask.sum() > 0 else {}
    other_metrics['count'] = int(other_mask.sum())

    return {
        'global': global_metrics,
        'hours_9_10': problem_metrics,
        'other': other_metrics,
    }


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
# CROSS-VALIDATION TRAINING
# =============================================================================

def train_with_cv(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> Dict:
    """
    Train models using TimeSeriesSplit cross-validation for better generalization.

    Returns CV scores and final models trained on all data.
    """
    from catboost import CatBoostRegressor
    import lightgbm as lgb

    logger.info(f"\n  Training with {n_splits}-fold TimeSeriesSplit CV...")

    tscv = TimeSeriesSplit(n_splits=n_splits)

    cv_scores_cat = []
    cv_scores_lgb = []
    cv_scores_ensemble = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # CatBoost
        cat_model = CatBoostRegressor(
            loss_function='MAE',
            iterations=1500,
            depth=7,  # Reduced from 8 for less overfitting
            learning_rate=0.02,
            l2_leaf_reg=5,  # Increased regularization
            random_state=42,
            verbose=False,
            early_stopping_rounds=100,
        )
        cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
        cat_pred = cat_model.predict(X_val)
        cat_smape = 100 * np.mean(2 * np.abs(y_val - cat_pred) / (np.abs(y_val) + np.abs(cat_pred) + 1e-8))
        cv_scores_cat.append(cat_smape)

        # LightGBM
        lgb_model = lgb.LGBMRegressor(
            objective='mae',
            n_estimators=1500,
            max_depth=7,  # Reduced from 8
            learning_rate=0.02,
            num_leaves=63,  # Reduced from 127
            subsample=0.8,  # More regularization
            colsample_bytree=0.8,
            reg_lambda=5.0,  # L2 regularization
            random_state=42,
            verbosity=-1,
        )
        lgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )
        lgb_pred = lgb_model.predict(X_val)
        lgb_smape = 100 * np.mean(2 * np.abs(y_val - lgb_pred) / (np.abs(y_val) + np.abs(lgb_pred) + 1e-8))
        cv_scores_lgb.append(lgb_smape)

        # Ensemble
        total_w = CATBOOST_WEIGHT + LIGHTGBM_WEIGHT
        ens_pred = (CATBOOST_WEIGHT * cat_pred + LIGHTGBM_WEIGHT * lgb_pred) / total_w
        ens_smape = 100 * np.mean(2 * np.abs(y_val - ens_pred) / (np.abs(y_val) + np.abs(ens_pred) + 1e-8))
        cv_scores_ensemble.append(ens_smape)

        logger.info(f"    Fold {fold+1}: CatBoost={cat_smape:.2f}%, LightGBM={lgb_smape:.2f}%, Ensemble={ens_smape:.2f}%")

    logger.info(f"\n  CV Results:")
    logger.info(f"    CatBoost:  {np.mean(cv_scores_cat):.2f}% (+/- {np.std(cv_scores_cat):.2f}%)")
    logger.info(f"    LightGBM:  {np.mean(cv_scores_lgb):.2f}% (+/- {np.std(cv_scores_lgb):.2f}%)")
    logger.info(f"    Ensemble:  {np.mean(cv_scores_ensemble):.2f}% (+/- {np.std(cv_scores_ensemble):.2f}%)")

    return {
        'cv_scores_catboost': cv_scores_cat,
        'cv_scores_lightgbm': cv_scores_lgb,
        'cv_scores_ensemble': cv_scores_ensemble,
        'mean_catboost': np.mean(cv_scores_cat),
        'mean_lightgbm': np.mean(cv_scores_lgb),
        'mean_ensemble': np.mean(cv_scores_ensemble),
        'std_ensemble': np.std(cv_scores_ensemble),
    }


# =============================================================================
# SIMPLE AEC (V13 Baseline)
# =============================================================================

def apply_simple_aec(df_preds: pd.DataFrame, hourly_params: Dict) -> np.ndarray:
    """Apply V13-style hourly Adaptive Error Correction."""
    df = df_preds.copy().sort_values('datetime').reset_index(drop=True)
    df['error'] = df['y_raw'] - df['y_true']
    df['y_corrected'] = df['y_raw'].copy()

    for hour in range(24):
        hour_mask = df['hour'] == hour
        if hour_mask.sum() == 0:
            continue

        params = hourly_params.get(hour, {'lookback': 7, 'damping': 0.5})
        lookback = params['lookback']
        damping = params['damping']

        hour_df = df[hour_mask].copy().reset_index(drop=True)
        errors = hour_df['error'].values
        raw = hour_df['y_raw'].values

        corrections = np.zeros(len(errors))
        for i in range(1, len(errors)):
            start_idx = max(0, i - lookback)
            past_errors = errors[start_idx:i]
            if len(past_errors) > 0:
                corrections[i] = damping * np.mean(past_errors)

        df.loc[hour_mask, 'y_corrected'] = raw - corrections

    return df['y_corrected'].values


# =============================================================================
# KNN CONTEXT-AWARE ERROR CORRECTION
# =============================================================================

def apply_knn_correction(df_preds: pd.DataFrame, X_context: pd.DataFrame,
                         scaler: StandardScaler, context_features: List[str],
                         k: int = 5, lookback_days: int = 45, damping: float = 0.8) -> np.ndarray:
    """
    Apply Context-Aware KNN Error Correction.

    Uses similar historical hours (by context features) to estimate bias,
    rather than simple time-based rolling averages.
    """
    df = df_preds.copy().sort_values('datetime').reset_index(drop=True)
    df['error'] = df['y_raw'] - df['y_true']

    # Extract and normalize context features
    available_features = [f for f in context_features if f in X_context.columns]
    context_data = X_context[available_features].fillna(0).values
    context_normalized = scaler.transform(context_data)

    n = len(df)
    corrections = np.zeros(n)

    # Process by hour for efficiency
    for hour in range(24):
        hour_mask = (df['hour'] == hour).values
        hour_indices = np.where(hour_mask)[0]

        if len(hour_indices) < k + 1:
            continue

        for i, idx in enumerate(hour_indices):
            if i < k:
                # Not enough history
                continue

            # Define lookback window (same hour, past lookback_days)
            lookback_hours = lookback_days  # For hourly data, 1 day = 1 sample per hour
            start_i = max(0, i - lookback_hours)
            history_indices = hour_indices[start_i:i]

            if len(history_indices) < 2:
                continue

            # Build KNN index on history
            history_contexts = context_normalized[history_indices]
            current_context = context_normalized[idx].reshape(1, -1)

            # Use KNN to find similar days
            k_actual = min(k, len(history_indices))
            knn = NearestNeighbors(n_neighbors=k_actual, metric='euclidean')
            knn.fit(history_contexts)

            distances, neighbor_idx = knn.kneighbors(current_context)
            distances = distances[0]
            neighbor_idx = neighbor_idx[0]

            # Get errors from neighbors
            neighbor_original_idx = history_indices[neighbor_idx]
            neighbor_errors = df.loc[neighbor_original_idx, 'error'].values

            # Weight by inverse distance (with epsilon to avoid div by zero)
            epsilon = 1e-6
            weights = 1.0 / (distances + epsilon)
            weights = weights / weights.sum()

            # Weighted bias
            weighted_bias = np.sum(weights * neighbor_errors)
            corrections[idx] = damping * weighted_bias

    return df['y_raw'].values - corrections


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train_price_model():
    """Train price forecasting ensemble model with KNN Error Correction."""
    MODELS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("="*70)
    logger.info("PRICE MODEL TRAINING - CHybrid V14")
    logger.info("CatBoost + LightGBM Ensemble with Context-Aware KNN-EC")
    logger.info("="*70)

    # =========================================================================
    # STEP 0: CROSS-VALIDATION FOR ROBUST ESTIMATION
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 0: CROSS-VALIDATION (TimeSeriesSplit)")
    logger.info("="*70)

    df = load_data()
    data = prepare_data(df, BASE_FEATURES)

    # Run CV on base+finetune data to get robust error estimates
    X_cv = pd.concat([data['X_base'], data['X_finetune']])
    y_cv = pd.concat([data['y_base'], data['y_finetune']])
    cv_results = train_with_cv(X_cv, y_cv, n_splits=CV_N_SPLITS)

    # =========================================================================
    # STEP 1: LOAD DATA AND TRAIN ENSEMBLE
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 1: TRAINING ENSEMBLE MODELS (Transfer Learning)")
    logger.info("="*70)

    logger.info(f"\n  DATA SPLITS:")
    logger.info(f"    Base:      {len(data['X_base']):,} rows")
    logger.info(f"    Fine-tune: {len(data['X_finetune']):,} rows")
    logger.info(f"    Test:      {len(data['X_test']):,} rows")

    catboost_model, cat_ft_pred, cat_test_pred = train_catboost_transfer(data)
    lightgbm_model, lgb_ft_pred, lgb_test_pred = train_lightgbm_transfer(data)

    # Calculate ensemble weights
    total_weight = CATBOOST_WEIGHT + LIGHTGBM_WEIGHT
    w_cat = CATBOOST_WEIGHT / total_weight
    w_lgb = LIGHTGBM_WEIGHT / total_weight

    logger.info(f"\n  Ensemble weights: CatBoost={w_cat:.3f}, LightGBM={w_lgb:.3f}")

    raw_ft_pred = w_cat * cat_ft_pred + w_lgb * lgb_ft_pred
    raw_test_pred = w_cat * cat_test_pred + w_lgb * lgb_test_pred

    # =========================================================================
    # STEP 2: PREPARE DATAFRAMES FOR CORRECTION
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 2: PREPARING DATA FOR KNN CORRECTION")
    logger.info("="*70)

    df_ft = pd.DataFrame({
        'datetime': data['X_finetune'].index,
        'hour': data['hours_finetune'].values,
        'y_true': data['y_finetune'].values,
        'y_raw': raw_ft_pred,
    })

    df_test = pd.DataFrame({
        'datetime': data['X_test'].index,
        'hour': data['hours_test'].values,
        'y_true': data['y_test'].values,
        'y_raw': raw_test_pred,
    })

    logger.info(f"  Test: {len(df_test):,} rows ({df_test['datetime'].min()} to {df_test['datetime'].max()})")

    # =========================================================================
    # STEP 3: FIT SCALER ON FINE-TUNE SET
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 3: FITTING CONTEXT SCALER")
    logger.info("="*70)

    available_context = [f for f in CONTEXT_FEATURES if f in data['X_finetune'].columns]
    logger.info(f"  Context features: {available_context}")

    scaler = StandardScaler()
    scaler.fit(data['X_finetune'][available_context].fillna(0))
    logger.info(f"  Scaler fitted on Fine-Tune set ({len(data['X_finetune']):,} rows)")

    # =========================================================================
    # STEP 4: RUN CORRECTION EXPERIMENTS
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 4: RUNNING CORRECTION EXPERIMENTS")
    logger.info("="*70)

    results = {}

    # Config A: V13 Simple AEC (Baseline)
    logger.info("\n  Config A: V13 Simple AEC (Baseline)...")
    simple_aec_pred = apply_simple_aec(df_test, HOURLY_AEC_PARAMS)
    simple_metrics = evaluate_with_breakdown(
        data['y_test'].values, simple_aec_pred, data['hours_test'].values
    )
    results['A_Simple_AEC'] = {'pred': simple_aec_pred, 'metrics': simple_metrics}
    logger.info(f"    Global sMAPE: {simple_metrics['global']['smape']:.2f}%")

    # Config B: Pure KNN-EC
    logger.info("\n  Config B: Pure KNN-EC (K=5, 45d, d=0.8)...")
    knn_pred = apply_knn_correction(
        df_test, data['X_test'].reset_index(drop=True),
        scaler, available_context,
        k=KNN_K, lookback_days=KNN_LOOKBACK_DAYS, damping=0.8
    )
    knn_metrics = evaluate_with_breakdown(
        data['y_test'].values, knn_pred, data['hours_test'].values
    )
    results['B_KNN_EC'] = {'pred': knn_pred, 'metrics': knn_metrics}
    logger.info(f"    Global sMAPE: {knn_metrics['global']['smape']:.2f}%")

    # Config C: Hybrid (50% Simple + 50% KNN)
    logger.info("\n  Config C: Hybrid (50% Simple + 50% KNN)...")
    hybrid_pred = 0.5 * simple_aec_pred + 0.5 * knn_pred
    hybrid_metrics = evaluate_with_breakdown(
        data['y_test'].values, hybrid_pred, data['hours_test'].values
    )
    results['C_Hybrid'] = {'pred': hybrid_pred, 'metrics': hybrid_metrics}
    logger.info(f"    Global sMAPE: {hybrid_metrics['global']['smape']:.2f}%")

    # Config F: Raw (no correction)
    raw_metrics = evaluate_with_breakdown(
        data['y_test'].values, raw_test_pred, data['hours_test'].values
    )
    results['F_Raw'] = {'pred': raw_test_pred, 'metrics': raw_metrics}
    logger.info(f"\n  Config F: Raw (no correction): {raw_metrics['global']['smape']:.2f}%")

    # =========================================================================
    # STEP 5: COMPARISON TABLE
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("COMPARISON TABLE")
    logger.info("="*70)

    logger.info(f"\n{'Method':<20} {'Global':>10} {'H9-10':>10} {'H9-10 Bias':>12} {'Beat 11.89%?':>14}")
    logger.info("-"*70)

    comparison = []
    for name, res in sorted(results.items()):
        m = res['metrics']
        beat = "YES" if m['global']['smape'] < BASELINE_SMAPE else "no"
        beat_oracle = " (ORACLE!)" if m['global']['smape'] < ORACLE_FLOOR else ""
        logger.info(f"{name:<20} {m['global']['smape']:>9.2f}% {m['hours_9_10']['smape']:>9.2f}% "
                    f"{m['hours_9_10']['bias']:>+11.2f} {beat:>14}{beat_oracle}")
        comparison.append({
            'Method': name,
            'Global_sMAPE': m['global']['smape'],
            'H910_sMAPE': m['hours_9_10']['smape'],
            'H910_Bias': m['hours_9_10']['bias'],
            'Beat_Baseline': str(m['global']['smape'] < BASELINE_SMAPE),
            'Beat_Oracle': str(m['global']['smape'] < ORACLE_FLOOR),
        })

    # =========================================================================
    # STEP 6: FIND BEST METHOD
    # =========================================================================
    best_method = min(comparison, key=lambda x: x['Global_sMAPE'])
    best_name = best_method['Method']
    best_pred = results[best_name]['pred']

    logger.info("\n" + "="*70)
    logger.info("FINAL RESULTS")
    logger.info("="*70)

    logger.info(f"\n  Best Method: {best_name}")
    logger.info(f"  Final sMAPE: {best_method['Global_sMAPE']:.2f}%")
    logger.info(f"  V13 Baseline: {BASELINE_SMAPE:.2f}%")
    logger.info(f"  Oracle Floor: {ORACLE_FLOOR:.2f}%")

    if best_method['Beat_Oracle'] == 'True':
        improvement = ORACLE_FLOOR - best_method['Global_sMAPE']
        logger.info(f"\n  BREAKTHROUGH! Beat Oracle Floor by {improvement:.2f}%!")
    elif best_method['Beat_Baseline'] == 'True':
        improvement = BASELINE_SMAPE - best_method['Global_sMAPE']
        logger.info(f"\n  SUCCESS! Beat V13 Baseline by {improvement:.2f}%!")

    # =========================================================================
    # STEP 7: SAVE MODELS
    # =========================================================================
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

    # Save ensemble config with AEC params
    config_path = MODELS_OUTPUT_DIR / 'ensemble_config.json'
    with open(config_path, 'w') as f:
        json.dump({
            'catboost_weight': w_cat,
            'lightgbm_weight': w_lgb,
            'best_method': best_name,
            'test_smape': best_method['Global_sMAPE'],
            'test_mae': results[best_name]['metrics']['global']['mae'],
            'hourly_aec_params': {str(k): v for k, v in HOURLY_AEC_PARAMS.items()},
            'knn_params': {
                'k': KNN_K,
                'lookback_days': KNN_LOOKBACK_DAYS,
                'context_features': available_context,
            },
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2)
    logger.info(f"  Saved config: {config_path}")

    # Save summary
    summary_path = MODELS_OUTPUT_DIR / 'summary.json'
    summary = {
        'timestamp': datetime.now().isoformat(),
        'baseline_smape': BASELINE_SMAPE,
        'oracle_floor': ORACLE_FLOOR,
        'final_smape': float(best_method['Global_sMAPE']),
        'best_method': best_name,
        'beat_oracle': best_method['Beat_Oracle'],
        'comparison': comparison,
        'cross_validation': {
            'n_splits': CV_N_SPLITS,
            'cv_mean_ensemble': float(cv_results['mean_ensemble']),
            'cv_std_ensemble': float(cv_results['std_ensemble']),
            'cv_scores': [float(s) for s in cv_results['cv_scores_ensemble']],
        },
        'knn_params': {
            'k': KNN_K,
            'lookback_days': KNN_LOOKBACK_DAYS,
            'context_features': available_context,
        },
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"  Saved summary: {summary_path}")

    logger.info(f"\n{'='*70}")
    logger.info(f"TRAINING COMPLETE")
    logger.info(f"Models saved to: {MODELS_OUTPUT_DIR}")
    logger.info(f"Final sMAPE: {best_method['Global_sMAPE']:.2f}%")
    logger.info(f"{'='*70}")

    return {
        'catboost_model': catboost_model,
        'lightgbm_model': lightgbm_model,
        'best_method': best_name,
        'test_smape': best_method['Global_sMAPE'],
        'features': feature_list,
        'comparison': comparison,
    }


if __name__ == "__main__":
    train_price_model()
