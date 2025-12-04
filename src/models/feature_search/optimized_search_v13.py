"""
Optimized Search V13 - Hourly-Dynamic Adaptive Error Correction
===============================================================
Context: V12 achieved 12.03% sMAPE with global AEC (7d/0.7).
         Global settings are suboptimal - Morning needs aggressive correction,
         Night suffers from over-correction.

Objective: Break 11.9% sMAPE by implementing Hourly-Dynamic AEC.

Strategy:
    1. Base Generator: V11/V12 Weighted Ensemble (CatBoost + LightGBM)
    2. Walk-Forward Optimization: Find best (lookback, damping) per hour on fine-tune set
    3. Apply per-hour parameters to test set

Author: ForeWatt Team
Date: December 2025
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# V12 baseline to beat
BASELINE_SMAPE = 12.03

# Problem hours
PROBLEM_HOURS = [9, 10]

# Ensemble weights from V11
CATBOOST_WEIGHT = 0.658
LIGHTGBM_WEIGHT = 0.413

# Grid search space
LOOKBACK_GRID = [3, 5, 7, 14, 21]
DAMPING_GRID = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Conservative defaults for low-bias hours
DEFAULT_LOOKBACK = 7
DEFAULT_DAMPING = 0.5
LOW_BIAS_THRESHOLD = 5.0  # TL - if abs(bias) < 5, use conservative settings


# =============================================================================
# BASE FEATURE SET
# =============================================================================

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
# ENHANCED PROFILE FEATURES
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
    path = PROJECT_ROOT / 'data' / 'gold' / 'master' / 'master_v2_fundamental.parquet'
    df = pd.read_parquet(path)

    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date' in df.columns:
            df = df.set_index('date')
        elif 'datetime' in df.columns:
            df = df.set_index('datetime')

    if 'price_real' not in df.columns:
        df['price_real'] = df['price']

    return df


def prepare_data(df: pd.DataFrame, features: List[str],
                 test_start: str = '2024-06-01',
                 finetune_months: int = 6) -> Dict:
    """Prepare data with splits for walk-forward optimization."""
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
    """Train CatBoost with transfer learning. Returns predictions for fine-tune AND test."""
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

    # Predict on both fine-tune (for optimization) and test (for evaluation)
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
# HOURLY-DYNAMIC AEC
# =============================================================================

def apply_aec_single_hour(errors: np.ndarray, lookback: int, damping: float) -> np.ndarray:
    """
    Apply AEC correction for a single hour's time series.
    Returns correction values (to be subtracted from raw predictions).
    """
    n = len(errors)
    corrections = np.zeros(n)

    for i in range(1, n):
        # Rolling mean of past errors (up to lookback)
        start_idx = max(0, i - lookback)
        past_errors = errors[start_idx:i]
        if len(past_errors) > 0:
            bias = np.mean(past_errors)
            corrections[i] = damping * bias

    return corrections


def optimize_hourly_aec(df_preds: pd.DataFrame, is_validation: bool = True) -> Dict:
    """
    Optimize AEC parameters per hour using grid search.

    Args:
        df_preds: DataFrame with ['datetime', 'hour', 'y_true', 'y_raw']
        is_validation: If True, use walk-forward (causal). If False, use full data (oracle).

    Returns:
        Dict with optimal parameters per hour
    """
    df = df_preds.copy().sort_values('datetime').reset_index(drop=True)
    df['error'] = df['y_raw'] - df['y_true']

    hourly_params = {}

    for hour in range(24):
        hour_mask = df['hour'] == hour
        hour_df = df[hour_mask].copy().reset_index(drop=True)

        if len(hour_df) < 10:
            hourly_params[hour] = {
                'lookback': DEFAULT_LOOKBACK,
                'damping': DEFAULT_DAMPING,
                'best_smape': np.nan,
                'raw_bias': np.nan,
            }
            continue

        errors = hour_df['error'].values
        y_true = hour_df['y_true'].values
        y_raw = hour_df['y_raw'].values
        raw_bias = np.mean(errors)

        # If bias is very low, use conservative defaults
        if abs(raw_bias) < LOW_BIAS_THRESHOLD:
            hourly_params[hour] = {
                'lookback': DEFAULT_LOOKBACK,
                'damping': DEFAULT_DAMPING,
                'best_smape': evaluate(y_true, y_raw)['smape'],
                'raw_bias': raw_bias,
                'note': 'low_bias_default',
            }
            continue

        # Grid search
        best_smape = float('inf')
        best_lookback = DEFAULT_LOOKBACK
        best_damping = DEFAULT_DAMPING

        for lookback in LOOKBACK_GRID:
            for damping in DAMPING_GRID:
                corrections = apply_aec_single_hour(errors, lookback, damping)
                y_corrected = y_raw - corrections
                smape = evaluate(y_true, y_corrected)['smape']

                if smape < best_smape:
                    best_smape = smape
                    best_lookback = lookback
                    best_damping = damping

        hourly_params[hour] = {
            'lookback': best_lookback,
            'damping': best_damping,
            'best_smape': best_smape,
            'raw_bias': raw_bias,
        }

    return hourly_params


def apply_hourly_aec(df_preds: pd.DataFrame, hourly_params: Dict) -> np.ndarray:
    """
    Apply hourly-specific AEC parameters.

    Args:
        df_preds: DataFrame with ['datetime', 'hour', 'y_true', 'y_raw']
        hourly_params: Dict mapping hour -> {lookback, damping}

    Returns:
        Array of corrected predictions
    """
    df = df_preds.copy().sort_values('datetime').reset_index(drop=True)
    df['error'] = df['y_raw'] - df['y_true']
    df['y_corrected'] = df['y_raw'].copy()

    for hour in range(24):
        hour_mask = df['hour'] == hour
        hour_indices = df[hour_mask].index.values

        if len(hour_indices) == 0:
            continue

        params = hourly_params.get(hour, {'lookback': DEFAULT_LOOKBACK, 'damping': DEFAULT_DAMPING})
        lookback = params['lookback']
        damping = params['damping']

        # Get hour-specific data
        hour_errors = df.loc[hour_mask, 'error'].values
        hour_raw = df.loc[hour_mask, 'y_raw'].values

        # Apply AEC
        corrections = apply_aec_single_hour(hour_errors, lookback, damping)
        corrected = hour_raw - corrections

        df.loc[hour_mask, 'y_corrected'] = corrected

    return df['y_corrected'].values


def apply_global_aec(df_preds: pd.DataFrame, lookback: int, damping: float) -> np.ndarray:
    """Apply global AEC (same params for all hours)."""
    hourly_params = {h: {'lookback': lookback, 'damping': damping} for h in range(24)}
    return apply_hourly_aec(df_preds, hourly_params)


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_v13_experiments():
    """Run V13 Hourly-Dynamic AEC experiments."""
    output_dir = PROJECT_ROOT / 'reports' / 'optimized_search_v13'
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*70)
    logger.info("OPTIMIZED SEARCH V13 - Hourly-Dynamic Adaptive Error Correction")
    logger.info(f"Baseline to beat: {BASELINE_SMAPE}% sMAPE")
    logger.info("="*70)

    # Load data
    df = load_data()
    logger.info(f"Loaded data: {df.shape}")

    # Prepare data
    data = prepare_data(df, BASE_FEATURES)

    logger.info(f"\n  DATA SPLITS:")
    logger.info(f"    Base:      {len(data['X_base']):,} rows")
    logger.info(f"    Fine-tune: {len(data['X_finetune']):,} rows")
    logger.info(f"    Test:      {len(data['X_test']):,} rows")

    # =========================================================================
    # STEP 1: TRAIN BASE ENSEMBLE
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 1: TRAINING BASE ENSEMBLE (CatBoost + LightGBM)")
    logger.info("="*70)

    # Train models
    _, cat_ft_pred, cat_test_pred = train_catboost_transfer(data)
    _, lgb_ft_pred, lgb_test_pred = train_lightgbm_transfer(data)

    # Create weighted ensemble
    total_weight = CATBOOST_WEIGHT + LIGHTGBM_WEIGHT
    w_cat = CATBOOST_WEIGHT / total_weight
    w_lgb = LIGHTGBM_WEIGHT / total_weight

    logger.info(f"\n  Ensemble weights: CatBoost={w_cat:.3f}, LightGBM={w_lgb:.3f}")

    # Fine-tune predictions (for walk-forward optimization)
    raw_ft_pred = w_cat * cat_ft_pred + w_lgb * lgb_ft_pred

    # Test predictions
    raw_test_pred = w_cat * cat_test_pred + w_lgb * lgb_test_pred

    # =========================================================================
    # STEP 2: PREPARE DATAFRAMES
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 2: PREPARING DATA FOR HOURLY AEC OPTIMIZATION")
    logger.info("="*70)

    # Fine-tune DataFrame (for parameter selection)
    df_ft = pd.DataFrame({
        'datetime': data['X_finetune'].index,
        'hour': data['hours_finetune'].values,
        'y_true': data['y_finetune'].values,
        'y_raw': raw_ft_pred,
    })

    # Test DataFrame (for evaluation)
    df_test = pd.DataFrame({
        'datetime': data['X_test'].index,
        'hour': data['hours_test'].values,
        'y_true': data['y_test'].values,
        'y_raw': raw_test_pred,
    })

    logger.info(f"  Fine-tune: {len(df_ft):,} rows ({df_ft['datetime'].min()} to {df_ft['datetime'].max()})")
    logger.info(f"  Test:      {len(df_test):,} rows ({df_test['datetime'].min()} to {df_test['datetime'].max()})")

    # =========================================================================
    # STEP 3: WALK-FORWARD PARAMETER OPTIMIZATION
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 3: WALK-FORWARD HOURLY PARAMETER OPTIMIZATION")
    logger.info("="*70)

    # Optimize on fine-tune set (causal/production way)
    logger.info("\n  Optimizing per-hour parameters on Fine-Tune Set...")
    hourly_params_wf = optimize_hourly_aec(df_ft, is_validation=True)

    # Also get oracle parameters (optimized on test set - theoretical limit)
    logger.info("  Computing Oracle parameters on Test Set (theoretical limit)...")
    hourly_params_oracle = optimize_hourly_aec(df_test, is_validation=False)

    # Print parameter map
    logger.info("\n" + "-"*70)
    logger.info("PER-HOUR PARAMETER MAP (Walk-Forward)")
    logger.info("-"*70)
    logger.info(f"{'Hour':<6} {'Lookback':>10} {'Damping':>10} {'Raw Bias':>12} {'Note':>15}")
    logger.info("-"*55)

    for hour in range(24):
        params = hourly_params_wf[hour]
        note = params.get('note', '')
        marker = " <--" if hour in PROBLEM_HOURS else ""
        logger.info(f"{hour:>4}   {params['lookback']:>10}d {params['damping']:>10.1f} "
                    f"{params['raw_bias']:>+11.2f} {note:>15}{marker}")

    # =========================================================================
    # STEP 4: RUN EXPERIMENTS
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 4: RUNNING EXPERIMENTS")
    logger.info("="*70)

    results = {}

    # Exp A: Global Baseline (V12 winner: 7d/0.7)
    logger.info("\n  Exp A: Global Baseline (7d/0.7)...")
    global_pred = apply_global_aec(df_test, lookback=7, damping=0.7)
    global_metrics = evaluate_with_breakdown(
        data['y_test'].values, global_pred, data['hours_test'].values
    )
    results['A_Global'] = {
        'pred': global_pred,
        'metrics': global_metrics,
        'params': 'Global 7d/0.7',
    }
    logger.info(f"    Global sMAPE: {global_metrics['global']['smape']:.2f}%")

    # Exp B: Hourly Static (Walk-Forward optimized)
    logger.info("\n  Exp B: Hourly Static (Walk-Forward optimized)...")
    hourly_wf_pred = apply_hourly_aec(df_test, hourly_params_wf)
    hourly_wf_metrics = evaluate_with_breakdown(
        data['y_test'].values, hourly_wf_pred, data['hours_test'].values
    )
    results['B_Hourly_WF'] = {
        'pred': hourly_wf_pred,
        'metrics': hourly_wf_metrics,
        'params': hourly_params_wf,
    }
    logger.info(f"    Global sMAPE: {hourly_wf_metrics['global']['smape']:.2f}%")

    # Exp C: Oracle Bound (Test-optimized - theoretical limit)
    logger.info("\n  Exp C: Oracle Bound (Test-optimized)...")
    oracle_pred = apply_hourly_aec(df_test, hourly_params_oracle)
    oracle_metrics = evaluate_with_breakdown(
        data['y_test'].values, oracle_pred, data['hours_test'].values
    )
    results['C_Oracle'] = {
        'pred': oracle_pred,
        'metrics': oracle_metrics,
        'params': hourly_params_oracle,
    }
    logger.info(f"    Global sMAPE: {oracle_metrics['global']['smape']:.2f}%")

    # Raw ensemble (no AEC)
    logger.info("\n  Exp D: Raw Ensemble (no AEC)...")
    raw_metrics = evaluate_with_breakdown(
        data['y_test'].values, raw_test_pred, data['hours_test'].values
    )
    results['D_Raw'] = {
        'pred': raw_test_pred,
        'metrics': raw_metrics,
        'params': 'No AEC',
    }
    logger.info(f"    Global sMAPE: {raw_metrics['global']['smape']:.2f}%")

    # =========================================================================
    # STEP 5: COMPARISON TABLE
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("COMPARISON TABLE")
    logger.info("="*70)

    logger.info(f"\n{'Method':<30} {'Global':>10} {'H9-10':>10} {'H9-10 Bias':>12} {'Beat 12.03%?':>14}")
    logger.info("-"*80)

    comparison = []
    for name, res in sorted(results.items()):
        m = res['metrics']
        beat = "YES" if m['global']['smape'] < BASELINE_SMAPE else "no"
        logger.info(f"{name:<30} {m['global']['smape']:>9.2f}% {m['hours_9_10']['smape']:>9.2f}% "
                    f"{m['hours_9_10']['bias']:>+11.2f} {beat:>14}")
        comparison.append({
            'Method': name,
            'Global_sMAPE': m['global']['smape'],
            'H910_sMAPE': m['hours_9_10']['smape'],
            'H910_Bias': m['hours_9_10']['bias'],
            'Beat_Baseline': m['global']['smape'] < BASELINE_SMAPE,
        })

    # =========================================================================
    # STEP 6: HOURLY BREAKDOWN
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("HOURLY BREAKDOWN: Global vs Hourly-Dynamic")
    logger.info("="*70)

    y_true = data['y_test'].values
    hours = data['hours_test'].values

    logger.info(f"\n{'Hour':<6} {'Global AEC':>12} {'Hourly AEC':>12} {'Oracle':>12} {'Δ (H-G)':>10} {'Params':>15}")
    logger.info("-"*75)

    hourly_gains = []
    for hour in range(24):
        hour_mask = hours == hour
        if hour_mask.sum() == 0:
            continue

        global_h = evaluate(y_true[hour_mask], global_pred[hour_mask])['smape']
        hourly_h = evaluate(y_true[hour_mask], hourly_wf_pred[hour_mask])['smape']
        oracle_h = evaluate(y_true[hour_mask], oracle_pred[hour_mask])['smape']

        delta = hourly_h - global_h
        params = hourly_params_wf[hour]
        param_str = f"{params['lookback']}d/{params['damping']:.1f}"
        marker = " <--" if hour in PROBLEM_HOURS else ""

        hourly_gains.append({'hour': hour, 'delta': delta})

        logger.info(f"{hour:>4}   {global_h:>11.2f}% {hourly_h:>11.2f}% {oracle_h:>11.2f}% "
                    f"{delta:>+9.2f}% {param_str:>15}{marker}")

    # =========================================================================
    # STEP 7: FINAL RESULTS
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("FINAL RESULTS")
    logger.info("="*70)

    best_method = min(comparison, key=lambda x: x['Global_sMAPE'])
    production_method = next(c for c in comparison if c['Method'] == 'B_Hourly_WF')

    logger.info(f"\n  Production Method (Hourly-WF): {production_method['Global_sMAPE']:.2f}%")
    logger.info(f"  Oracle Bound:                  {results['C_Oracle']['metrics']['global']['smape']:.2f}%")
    logger.info(f"  V12 Baseline:                  {BASELINE_SMAPE:.2f}%")

    if production_method['Beat_Baseline']:
        improvement = BASELINE_SMAPE - production_method['Global_sMAPE']
        logger.info(f"\n  SUCCESS! Beat {BASELINE_SMAPE}% by {improvement:.2f}%!")
    else:
        logger.info(f"\n  Did not beat baseline. Gap: {production_method['Global_sMAPE'] - BASELINE_SMAPE:.2f}%")

    # =========================================================================
    # JOURNEY SUMMARY
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("JOURNEY SUMMARY: V1 → V13")
    logger.info("="*70)

    journey = [
        ('N-HiTS Baseline', 16.01, 'Deep Learning'),
        ('V4', 14.29, 'Transfer Learning'),
        ('V5', 14.08, '6-month Fine-tune Window'),
        ('V10', 12.73, 'Profile Evolution Features'),
        ('V11', 12.31, 'CatBoost+LightGBM Ensemble'),
        ('V12', 12.03, 'Adaptive Error Correction'),
        ('V13', production_method['Global_sMAPE'], 'Hourly-Dynamic AEC'),
    ]

    logger.info(f"\n{'Version':<20} {'sMAPE':>10} {'Improvement':>12} {'Key Innovation':>35}")
    logger.info("-"*80)

    prev_smape = journey[0][1]
    for version, smape, innovation in journey:
        improvement = prev_smape - smape
        logger.info(f"{version:<20} {smape:>9.2f}% {improvement:>+11.2f}% {innovation:>35}")
        prev_smape = smape

    total_improvement = journey[0][1] - journey[-1][1]
    relative_improvement = 100 * total_improvement / journey[0][1]
    logger.info("-"*80)
    logger.info(f"{'TOTAL':<20} {journey[-1][1]:>9.2f}% {total_improvement:>+11.2f}% "
                f"({relative_improvement:.1f}% relative reduction)")

    logger.info(f"\n{'='*70}")
    logger.info(f"Final sMAPE: {production_method['Global_sMAPE']:.2f}%")
    logger.info(f"{'='*70}")

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    summary = {
        'timestamp': datetime.now().isoformat(),
        'baseline_smape': BASELINE_SMAPE,
        'final_smape': float(production_method['Global_sMAPE']),
        'oracle_smape': float(results['C_Oracle']['metrics']['global']['smape']),
        'hourly_params': {str(k): v for k, v in hourly_params_wf.items()},
        'comparison': comparison,
        'journey': journey,
    }

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Save production predictions
    pred_df = pd.DataFrame({
        'datetime': data['X_test'].index,
        'hour': data['hours_test'].values,
        'is_problem_hour': np.isin(data['hours_test'].values, PROBLEM_HOURS),
        'y_true': data['y_test'].values,
        'y_raw': raw_test_pred,
        'y_corrected_global': global_pred,
        'y_corrected_hourly': hourly_wf_pred,
        'y_corrected_oracle': oracle_pred,
    })
    pred_df.to_csv(output_dir / 'final_production_predictions.csv', index=False)
    pred_df.to_parquet(output_dir / 'final_production_predictions.parquet', index=False)

    # Save hourly parameters for production
    params_df = pd.DataFrame([
        {'hour': h, 'lookback': p['lookback'], 'damping': p['damping'], 'raw_bias': p['raw_bias']}
        for h, p in hourly_params_wf.items()
    ])
    params_df.to_csv(output_dir / 'hourly_aec_params.csv', index=False)

    logger.info(f"\nResults saved to: {output_dir}")

    return results


if __name__ == "__main__":
    run_v13_experiments()
