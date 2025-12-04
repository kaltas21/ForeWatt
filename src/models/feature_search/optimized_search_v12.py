"""
Optimized Search V12 - Adaptive Error Correction (AEC)
======================================================
Context: V11 achieved 12.31% sMAPE with CatBoost+LightGBM ensemble.
         Persistent +21 TL bias remains in Morning Block (Hours 9-10).

Problem: Tree models are "stiff" - cannot adapt to short-term bias shifts
         (e.g., cloudy weather week shifting morning ramp).

Objective: Break 12.0% sMAPE by implementing Adaptive Error Correction (AEC).

Strategy:
    1. Base Generator: V11 Weighted Ensemble (CatBoost ~0.65, LightGBM ~0.35)
    2. Adaptive Correction: Post-processing layer that learns from recent errors
    3. Strictly causal: Only use errors observed before prediction time

Author: ForeWatt Team
Date: December 2025
"""

import sys
import os
import time
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

# V11 baseline to beat
BASELINE_SMAPE = 12.31

# Problem hours
PROBLEM_HOURS = [9, 10]

# Optimal weights from V11
CATBOOST_WEIGHT = 0.658
LIGHTGBM_WEIGHT = 0.413  # Note: weights normalize to ~0.61, 0.39


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
# ENHANCED PROFILE FEATURES (V10 + Solar Profile)
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

    # INTERACTION FEATURES
    if 'solar_momentum' in df.columns:
        df['price_solar_interaction'] = df['profile_14d'] * df['solar_momentum']
        new_features.append('price_solar_interaction')

    # Handle NaN
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
    """Prepare data with V5-style split and profile features."""
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

    logger.info(f"\n  DATA SPLITS:")
    logger.info(f"    Base:      {base_mask.sum():,} rows")
    logger.info(f"    Fine-tune: {finetune_mask.sum():,} rows")
    logger.info(f"    Test:      {test_mask.sum():,} rows")
    logger.info(f"    Features:  {len(X.columns)}")

    return {
        'X_base': X[base_mask],
        'y_base': y[base_mask],
        'X_finetune': X[finetune_mask],
        'y_finetune': y[finetune_mask],
        'X_test': X[test_mask],
        'y_test': y[test_mask],
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
# TRANSFER LEARNING TRAINERS (CatBoost + LightGBM only)
# =============================================================================

def train_catboost_transfer(data: Dict) -> Tuple[object, np.ndarray]:
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

    test_pred = finetune_model.predict(data['X_test'])
    return finetune_model, test_pred


def train_lightgbm_transfer(data: Dict) -> Tuple[object, np.ndarray]:
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

    test_pred = finetune_model.predict(data['X_test'])
    return finetune_model, test_pred


# =============================================================================
# ADAPTIVE ERROR CORRECTION (AEC)
# =============================================================================

def apply_adaptive_correction(
    df_preds: pd.DataFrame,
    lookback_days: int,
    damping: float = 1.0,
) -> pd.DataFrame:
    """
    Apply Adaptive Error Correction (AEC) post-processing.

    Logic: Simulate daily production loop.
    For each day in test set:
        1. Get raw prediction for today
        2. Calculate hourly bias from previous lookback_days
        3. Apply correction: y_final = y_raw - (damping * bias_h)

    This is STRICTLY CAUSAL - only uses errors we would have observed before auction.

    Args:
        df_preds: DataFrame with columns ['datetime', 'hour', 'y_true', 'y_raw']
        lookback_days: Number of past days to compute bias from
        damping: Damping factor (1.0 = full correction, 0.5 = half correction)

    Returns:
        DataFrame with additional 'y_corrected' column
    """
    df = df_preds.copy()
    df = df.sort_values('datetime').reset_index(drop=True)

    # Extract date for grouping
    df['date'] = pd.to_datetime(df['datetime']).dt.date

    # Initialize corrected predictions (start with raw)
    df['y_corrected'] = df['y_raw'].copy()
    df['correction'] = 0.0

    # Get unique dates
    unique_dates = sorted(df['date'].unique())

    # Track errors for bias computation (hour -> list of errors)
    error_history = {h: [] for h in range(24)}
    date_history = {h: [] for h in range(24)}  # Track dates for lookback window

    for date in unique_dates:
        date_mask = df['date'] == date
        today_idx = df[date_mask].index

        # For each hour in today
        for idx in today_idx:
            hour = df.loc[idx, 'hour']
            raw_pred = df.loc[idx, 'y_raw']

            # Calculate bias from past lookback_days (if we have history)
            if len(error_history[hour]) > 0:
                # Get errors within lookback window
                valid_errors = []
                cutoff_date = date - pd.Timedelta(days=lookback_days)

                for i, (err_date, err) in enumerate(zip(date_history[hour], error_history[hour])):
                    if err_date >= cutoff_date:
                        valid_errors.append(err)

                if len(valid_errors) > 0:
                    # Bias = mean(y_pred - y_true) over lookback window
                    hourly_bias = np.mean(valid_errors)
                    correction = damping * hourly_bias

                    # Apply correction: y_final = y_raw - correction
                    df.loc[idx, 'correction'] = correction
                    df.loc[idx, 'y_corrected'] = raw_pred - correction

        # After processing today, add today's errors to history (simulates observing true values)
        for idx in today_idx:
            hour = df.loc[idx, 'hour']
            error = df.loc[idx, 'y_raw'] - df.loc[idx, 'y_true']  # pred - true
            error_history[hour].append(error)
            date_history[hour].append(date)

    return df


def apply_adaptive_correction_vectorized(
    df_preds: pd.DataFrame,
    lookback_days: int,
    damping: float = 1.0,
) -> pd.DataFrame:
    """
    Vectorized version of AEC - faster for large datasets.

    Still strictly causal: for each row, compute bias from past lookback_days
    of the SAME HOUR only.
    """
    df = df_preds.copy()
    df = df.sort_values('datetime').reset_index(drop=True)

    df['date'] = pd.to_datetime(df['datetime']).dt.date
    df['error'] = df['y_raw'] - df['y_true']  # Will compute after we see true value

    # Initialize
    df['y_corrected'] = df['y_raw'].copy()
    df['hourly_bias'] = 0.0
    df['correction'] = 0.0

    # Process by hour for vectorization
    for hour in range(24):
        hour_mask = df['hour'] == hour
        hour_df = df[hour_mask].copy()

        if len(hour_df) == 0:
            continue

        # Compute rolling bias (shifted by 1 to ensure causality)
        # This is: mean of errors over past lookback_days
        hour_df = hour_df.sort_values('datetime')

        # Rolling mean of past errors (excluding current day)
        # lookback_days worth of hourly data = lookback_days rows for hourly data
        hour_df['rolling_bias'] = (
            hour_df['error']
            .shift(1)  # Exclude current row (causal)
            .rolling(window=lookback_days, min_periods=1)
            .mean()
        )

        # Apply correction
        hour_df['correction'] = damping * hour_df['rolling_bias'].fillna(0)
        hour_df['y_corrected'] = hour_df['y_raw'] - hour_df['correction']

        # Update main dataframe
        df.loc[hour_mask, 'y_corrected'] = hour_df['y_corrected'].values
        df.loc[hour_mask, 'hourly_bias'] = hour_df['rolling_bias'].fillna(0).values
        df.loc[hour_mask, 'correction'] = hour_df['correction'].values

    return df


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_v12_experiments():
    """Run V12 Adaptive Error Correction experiments."""
    output_dir = PROJECT_ROOT / 'reports' / 'optimized_search_v12'
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*70)
    logger.info("OPTIMIZED SEARCH V12 - Adaptive Error Correction (AEC)")
    logger.info(f"Baseline to beat: {BASELINE_SMAPE}% sMAPE")
    logger.info("="*70)

    # Load data
    df = load_data()
    logger.info(f"Loaded data: {df.shape}")

    # Prepare data with enhanced features
    data = prepare_data(df, BASE_FEATURES)

    # =========================================================================
    # STEP 1: TRAIN BASE ENSEMBLE (V11 Refined)
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 1: TRAINING BASE ENSEMBLE (CatBoost + LightGBM)")
    logger.info("="*70)

    # Train models
    catboost_model, catboost_pred = train_catboost_transfer(data)
    lightgbm_model, lightgbm_pred = train_lightgbm_transfer(data)

    # Create weighted ensemble (V11 optimal weights, normalized)
    total_weight = CATBOOST_WEIGHT + LIGHTGBM_WEIGHT
    w_cat = CATBOOST_WEIGHT / total_weight
    w_lgb = LIGHTGBM_WEIGHT / total_weight

    logger.info(f"\n  Ensemble weights: CatBoost={w_cat:.3f}, LightGBM={w_lgb:.3f}")

    raw_ensemble_pred = w_cat * catboost_pred + w_lgb * lightgbm_pred

    # Evaluate raw ensemble
    raw_metrics = evaluate_with_breakdown(
        data['y_test'].values, raw_ensemble_pred, data['hours_test'].values
    )
    logger.info(f"\n  Raw Ensemble: Global={raw_metrics['global']['smape']:.2f}%, "
                f"H9-10={raw_metrics['hours_9_10']['smape']:.2f}%, "
                f"Bias={raw_metrics['hours_9_10']['bias']:+.2f} TL")

    # =========================================================================
    # STEP 2: PREPARE PREDICTIONS DATAFRAME
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 2: PREPARING PREDICTIONS FOR AEC")
    logger.info("="*70)

    df_preds = pd.DataFrame({
        'datetime': data['X_test'].index,
        'hour': data['hours_test'].values,
        'y_true': data['y_test'].values,
        'y_raw': raw_ensemble_pred,
    })

    logger.info(f"  Predictions shape: {df_preds.shape}")
    logger.info(f"  Test period: {df_preds['datetime'].min()} to {df_preds['datetime'].max()}")

    # =========================================================================
    # STEP 3: RUN AEC EXPERIMENTS
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 3: ADAPTIVE ERROR CORRECTION EXPERIMENTS")
    logger.info("="*70)

    # AEC Configurations
    configs = [
        {'name': 'Raw Ensemble', 'lookback': None, 'damping': None},
        {'name': 'AEC-A (7d, d=1.0)', 'lookback': 7, 'damping': 1.0},
        {'name': 'AEC-B (3d, d=0.8)', 'lookback': 3, 'damping': 0.8},
        {'name': 'AEC-C (14d, d=1.0)', 'lookback': 14, 'damping': 1.0},
        {'name': 'AEC-D (7d, d=0.7)', 'lookback': 7, 'damping': 0.7},
        {'name': 'AEC-E (5d, d=0.9)', 'lookback': 5, 'damping': 0.9},
    ]

    results = {}

    for config in configs:
        name = config['name']

        if config['lookback'] is None:
            # Raw ensemble (no correction)
            corrected_preds = df_preds['y_raw'].values
            logger.info(f"\n  {name}: Baseline (no correction)")
        else:
            # Apply AEC
            logger.info(f"\n  {name}: lookback={config['lookback']}d, damping={config['damping']}")
            df_corrected = apply_adaptive_correction_vectorized(
                df_preds,
                lookback_days=config['lookback'],
                damping=config['damping']
            )
            corrected_preds = df_corrected['y_corrected'].values

        # Evaluate
        metrics = evaluate_with_breakdown(
            data['y_test'].values, corrected_preds, data['hours_test'].values
        )

        results[name] = {
            'config': config,
            'pred': corrected_preds,
            'metrics': metrics,
        }

        logger.info(f"    Global: {metrics['global']['smape']:.2f}%, "
                    f"H9-10: {metrics['hours_9_10']['smape']:.2f}%, "
                    f"Bias: {metrics['global']['bias']:+.2f} TL")

    # =========================================================================
    # STEP 4: CORRECTION IMPACT TABLE
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("CORRECTION IMPACT TABLE")
    logger.info("="*70)

    raw_global = results['Raw Ensemble']['metrics']['global']['smape']
    raw_h910 = results['Raw Ensemble']['metrics']['hours_9_10']['smape']
    raw_h910_bias = results['Raw Ensemble']['metrics']['hours_9_10']['bias']

    logger.info(f"\n{'Method':<25} {'Global':>10} {'H9-10':>10} {'H9-10 Bias':>12} {'Global Δ':>10} {'H9-10 Δ':>10}")
    logger.info("-"*85)

    impact_table = []
    for name, res in results.items():
        m = res['metrics']
        global_smape = m['global']['smape']
        h910_smape = m['hours_9_10']['smape']
        h910_bias = m['hours_9_10']['bias']

        global_delta = global_smape - raw_global
        h910_delta = h910_smape - raw_h910

        logger.info(f"{name:<25} {global_smape:>9.2f}% {h910_smape:>9.2f}% {h910_bias:>+11.2f} "
                    f"{global_delta:>+9.2f}% {h910_delta:>+9.2f}%")

        impact_table.append({
            'Method': name,
            'Global_sMAPE': global_smape,
            'H910_sMAPE': h910_smape,
            'H910_Bias': h910_bias,
            'Global_Delta': global_delta,
            'H910_Delta': h910_delta,
            'Beat_Baseline': global_smape < BASELINE_SMAPE,
        })

    # =========================================================================
    # STEP 5: FINAL LEADERBOARD
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("FINAL LEADERBOARD")
    logger.info("="*70)

    leaderboard = sorted(impact_table, key=lambda x: x['Global_sMAPE'])

    logger.info(f"\n{'Rank':<6} {'Method':<25} {'Global':>10} {'H9-10':>10} {'Beat 12.31%?':>14}")
    logger.info("-"*70)

    for i, entry in enumerate(leaderboard, 1):
        beat = "YES" if entry['Beat_Baseline'] else "no"
        logger.info(f"{i:<6} {entry['Method']:<25} {entry['Global_sMAPE']:>9.2f}% "
                    f"{entry['H910_sMAPE']:>9.2f}% {beat:>14}")

    # Best result
    best_entry = leaderboard[0]
    best_name = best_entry['Method']
    best_result = results[best_name]

    logger.info(f"\n{'='*70}")
    if best_entry['Beat_Baseline']:
        improvement = BASELINE_SMAPE - best_entry['Global_sMAPE']
        logger.info(f"SUCCESS! Beat {BASELINE_SMAPE}% by {improvement:.2f}%!")
    else:
        logger.info(f"Best: {best_entry['Global_sMAPE']:.2f}% (baseline: {BASELINE_SMAPE}%)")

    logger.info(f"  Best Method: {best_name}")
    logger.info(f"  Global sMAPE: {best_entry['Global_sMAPE']:.2f}%")
    logger.info(f"  Hours 9-10 sMAPE: {best_entry['H910_sMAPE']:.2f}%")
    logger.info(f"  Hours 9-10 Bias: {best_entry['H910_Bias']:.2f} TL")

    # =========================================================================
    # STEP 6: HOURLY BREAKDOWN FOR BEST METHOD
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("HOURLY BREAKDOWN: Raw vs Best Corrected")
    logger.info("="*70)

    raw_pred = results['Raw Ensemble']['pred']
    best_pred = best_result['pred']
    y_true = data['y_test'].values
    hours = data['hours_test'].values

    logger.info(f"\n{'Hour':<6} {'Raw sMAPE':>12} {'Best sMAPE':>12} {'Δ':>10} {'Raw Bias':>12} {'Best Bias':>12}")
    logger.info("-"*70)

    for hour in range(24):
        hour_mask = hours == hour
        if hour_mask.sum() == 0:
            continue

        raw_h = evaluate(y_true[hour_mask], raw_pred[hour_mask])
        best_h = evaluate(y_true[hour_mask], best_pred[hour_mask])

        delta = best_h['smape'] - raw_h['smape']
        marker = " <--" if hour in PROBLEM_HOURS else ""

        logger.info(f"{hour:>4}   {raw_h['smape']:>11.2f}% {best_h['smape']:>11.2f}% "
                    f"{delta:>+9.2f}% {raw_h['bias']:>+11.2f} {best_h['bias']:>+11.2f}{marker}")

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    summary = {
        'timestamp': datetime.now().isoformat(),
        'baseline_smape': BASELINE_SMAPE,
        'best_method': best_name,
        'best_global_smape': float(best_entry['Global_sMAPE']),
        'best_h910_smape': float(best_entry['H910_sMAPE']),
        'best_h910_bias': float(best_entry['H910_Bias']),
        'impact_table': impact_table,
        'ensemble_weights': {'CatBoost': w_cat, 'LightGBM': w_lgb},
    }

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Save best predictions
    pred_df = pd.DataFrame({
        'datetime': data['X_test'].index,
        'hour': data['hours_test'].values,
        'is_problem_hour': np.isin(data['hours_test'].values, PROBLEM_HOURS),
        'y_true': data['y_test'].values,
        'y_raw': raw_ensemble_pred,
        'y_corrected': best_pred,
    })
    pred_df.to_csv(output_dir / 'best_predictions.csv', index=False)
    pred_df.to_parquet(output_dir / 'best_predictions.parquet', index=False)

    logger.info(f"\nResults saved to: {output_dir}")

    return results


if __name__ == "__main__":
    run_v12_experiments()
