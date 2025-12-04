"""
Optimized Search V9 - Seasonally-Aligned Transfer Learning
============================================================
Context: V5 (14.08% sMAPE) over-predicts Hours 09-10 by +26 TL.
         The Fine-Tune window (Jan-Jun 2024) has Winter/Spring physics
         but the Test set (Jun 2024+) has Summer physics.
         The model learns the wrong seasonal shape for morning hours.

Insight: Morning prices are HIGH in winter (no solar) but LOW in summer (solar ramp).
         We need to inject "Summer shape" into the fine-tuning data.

Strategy:
    - Component A (Recency): Last 2-3 months for price LEVEL
    - Component B (Seasonality): Same calendar months from PREVIOUS YEAR for SHAPE
    - Solar ramp features to help model identify the transition

Experiments:
    A. V5 Baseline: Last 6 months contiguous
    B. Seasonal Injection: Last 2mo + Last Summer (Jun-Aug 2023)
    C. Weighted Seasonal: Same as B with higher weight on recency

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
from sklearn.model_selection import KFold

warnings.filterwarnings('ignore')
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Baseline to beat
BASELINE_SMAPE = 14.08

# Problem hours
PROBLEM_HOURS = [9, 10]
SOLAR_RAMP_HOURS = [8, 9, 10, 11]


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
# SOLAR RAMP FEATURE ENGINEERING
# =============================================================================

def add_solar_ramp_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Add solar ramp features to help model identify morning transitions.

    Key features:
    - is_solar_ramp: Binary flag for hours 8-11
    - solar_ramp_intensity: renewable_saturation * is_solar_ramp
    - solar_impact: renewable_saturation * (1 + sin(hour))
    """
    df = df.copy()
    new_features = []

    # Hour as integer
    hour = df.index.hour
    df['hour'] = hour
    new_features.append('hour')

    # Month
    month = df.index.month
    df['month'] = month
    new_features.append('month')

    # Is solar ramp (hours 8-11)
    df['is_solar_ramp'] = hour.isin(SOLAR_RAMP_HOURS).astype(int)
    new_features.append('is_solar_ramp')

    # Solar ramp intensity
    if 'renewable_saturation' in df.columns:
        df['solar_ramp_intensity'] = df['renewable_saturation'] * df['is_solar_ramp']
        new_features.append('solar_ramp_intensity')

        # Solar impact: renewable_saturation * (1 + sin(hour_scaled))
        hour_rad = 2 * np.pi * hour / 24
        df['solar_impact'] = df['renewable_saturation'] * (1 + np.sin(hour_rad))
        new_features.append('solar_impact')

    # Month sin/cos for seasonality
    df['month_sin'] = np.sin(2 * np.pi * month / 12)
    df['month_cos'] = np.cos(2 * np.pi * month / 12)
    new_features.extend(['month_sin', 'month_cos'])

    # Is summer (Jun-Aug)
    df['is_summer'] = month.isin([6, 7, 8]).astype(int)
    new_features.append('is_summer')

    # Morning solar interaction: is_solar_ramp * is_summer
    df['summer_solar_ramp'] = df['is_solar_ramp'] * df['is_summer']
    new_features.append('summer_solar_ramp')

    logger.info(f"  Added {len(new_features)} solar ramp features")

    return df, new_features


# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

def load_data() -> pd.DataFrame:
    """Load master dataset with robust deflated prices."""
    path = PROJECT_ROOT / 'data' / 'gold' / 'master' / 'master_v2_fundamental.parquet'
    df = pd.read_parquet(path)

    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date' in df.columns:
            df = df.set_index('date')
        elif 'datetime' in df.columns:
            df = df.set_index('datetime')

    if 'price_real' not in df.columns:
        logger.warning("'price_real' not found - using 'price' column")
        df['price_real'] = df['price']

    return df


def get_smart_finetuning_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_start: str = '2024-06-01',
    recency_months: int = 2,
    seasonal_months: List[int] = [6, 7, 8],
    seasonal_year: int = 2023
) -> Dict:
    """
    Create seasonally-aligned fine-tuning dataset.

    Args:
        recency_months: How many recent months before test (price level)
        seasonal_months: Which calendar months from previous year (shape)
        seasonal_year: Which year to take seasonal data from
    """
    tz = X.index.tz
    test_start_dt = pd.Timestamp(test_start, tz=tz)

    # Base train: everything before seasonal_year end
    base_end_dt = pd.Timestamp(f'{seasonal_year}-12-31', tz=tz)

    # Component A (Recency): last N months before test
    recency_start = test_start_dt - pd.DateOffset(months=recency_months)

    # Component B (Seasonality): specific months from previous year
    seasonal_start = pd.Timestamp(f'{seasonal_year}-{min(seasonal_months):02d}-01', tz=tz)
    seasonal_end = pd.Timestamp(f'{seasonal_year}-{max(seasonal_months):02d}-28', tz=tz) + pd.DateOffset(days=3)

    # Create masks
    base_mask = X.index <= base_end_dt
    recency_mask = (X.index >= recency_start) & (X.index < test_start_dt)
    seasonal_mask = (X.index >= seasonal_start) & (X.index <= seasonal_end)
    test_mask = X.index >= test_start_dt

    # Combine fine-tune components
    finetune_mask = recency_mask | seasonal_mask

    logger.info(f"\n  SMART DATA SPLITS:")
    logger.info(f"    Base Train:     {base_mask.sum():,} rows (to {base_end_dt.date()})")
    logger.info(f"    Fine-Tune (A):  {recency_mask.sum():,} rows (Recency: {recency_start.date()} to {test_start_dt.date()})")
    logger.info(f"    Fine-Tune (B):  {seasonal_mask.sum():,} rows (Seasonal: {seasonal_year} months {seasonal_months})")
    logger.info(f"    Fine-Tune Total:{finetune_mask.sum():,} rows")
    logger.info(f"    Test:           {test_mask.sum():,} rows ({test_start_dt.date()} onwards)")

    # Log price levels to verify deflator is working
    if recency_mask.sum() > 0 and seasonal_mask.sum() > 0:
        recency_mean = y[recency_mask].mean()
        seasonal_mean = y[seasonal_mask].mean()
        test_mean = y[test_mask].mean()
        logger.info(f"\n  PRICE LEVELS (Deflated):")
        logger.info(f"    Recency (2024):  {recency_mean:.2f} TL/MWh")
        logger.info(f"    Seasonal (2023): {seasonal_mean:.2f} TL/MWh")
        logger.info(f"    Test:            {test_mean:.2f} TL/MWh")

    return {
        'base_mask': base_mask,
        'recency_mask': recency_mask,
        'seasonal_mask': seasonal_mask,
        'finetune_mask': finetune_mask,
        'test_mask': test_mask,
    }


# =============================================================================
# METRICS
# =============================================================================

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Calculate metrics including Bias."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae = np.mean(np.abs(y_true - y_pred))
    smape = 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))
    bias = np.mean(y_pred - y_true)

    return {'mae': mae, 'smape': smape, 'bias': bias}


def evaluate_with_hour_breakdown(y_true: np.ndarray, y_pred: np.ndarray, hours: np.ndarray) -> Dict:
    """Evaluate with specific breakdown for problem hours."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    hours = np.array(hours)

    global_metrics = evaluate(y_true, y_pred)

    # Problem hours (9, 10)
    problem_mask = np.isin(hours, PROBLEM_HOURS)
    problem_metrics = evaluate(y_true[problem_mask], y_pred[problem_mask]) if problem_mask.sum() > 0 else {}
    problem_metrics['count'] = problem_mask.sum()

    # Solar ramp hours (8-11)
    solar_mask = np.isin(hours, SOLAR_RAMP_HOURS)
    solar_metrics = evaluate(y_true[solar_mask], y_pred[solar_mask]) if solar_mask.sum() > 0 else {}
    solar_metrics['count'] = solar_mask.sum()

    # Other hours
    other_mask = ~solar_mask
    other_metrics = evaluate(y_true[other_mask], y_pred[other_mask]) if other_mask.sum() > 0 else {}
    other_metrics['count'] = other_mask.sum()

    return {
        'global': global_metrics,
        'hours_9_10': problem_metrics,
        'hours_8_11': solar_metrics,
        'other': other_metrics,
    }


# =============================================================================
# TRANSFER LEARNING TRAINER
# =============================================================================

def train_transfer_model(X_base, y_base, X_finetune, y_finetune, X_test, y_test,
                         sample_weights=None, config_name="default"):
    """
    Train transfer learning model with optional sample weights.
    """
    from catboost import CatBoostRegressor

    logger.info(f"\n  Training {config_name}...")

    # Step A: Base training
    split_idx = int(len(X_base) * 0.85)
    X_base_tr = X_base.iloc[:split_idx]
    X_base_vl = X_base.iloc[split_idx:]
    y_base_tr = y_base.iloc[:split_idx]
    y_base_vl = y_base.iloc[split_idx:]

    base_model = CatBoostRegressor(
        loss_function='MAE',
        iterations=2000,
        depth=10,
        learning_rate=0.02,
        l2_leaf_reg=3,
        random_state=42,
        verbose=False,
        early_stopping_rounds=100,
    )
    base_model.fit(X_base_tr, y_base_tr, eval_set=(X_base_vl, y_base_vl), verbose=False)
    logger.info(f"    Base model: {base_model.tree_count_} trees")

    # Step B: Fine-tune
    split_idx = int(len(X_finetune) * 0.8)

    fit_params = {
        'eval_set': (X_finetune.iloc[split_idx:], y_finetune.iloc[split_idx:]),
        'init_model': base_model,
        'verbose': False,
    }

    if sample_weights is not None:
        fit_params['sample_weight'] = sample_weights[:split_idx]

    finetune_model = CatBoostRegressor(
        loss_function='MAE',
        iterations=500,
        depth=10,
        learning_rate=0.005,
        l2_leaf_reg=3,
        random_state=42,
        verbose=False,
        early_stopping_rounds=50,
    )
    finetune_model.fit(X_finetune.iloc[:split_idx], y_finetune.iloc[:split_idx], **fit_params)
    logger.info(f"    Fine-tuned model: {finetune_model.tree_count_} trees")

    # Predictions
    test_pred = finetune_model.predict(X_test)

    return finetune_model, test_pred


# =============================================================================
# EXPERIMENT CONFIGS
# =============================================================================

def run_config_a_v5_baseline(df: pd.DataFrame, features: List[str]) -> Dict:
    """
    Config A: V5 Baseline - Last 6 months contiguous.
    """
    logger.info("\n" + "="*70)
    logger.info("CONFIG A: V5 BASELINE")
    logger.info("Last 6 months contiguous fine-tuning")
    logger.info("="*70)

    available = [f for f in features if f in df.columns]
    X = df[available].copy()
    y = df['price_real'].copy()

    # Add solar ramp features
    X, solar_features = add_solar_ramp_features(X)

    # Drop NaN
    mask = ~(X.isna().any(axis=1) | y.isna())
    X, y = X[mask], y[mask]
    hours = X.index.hour

    # Standard V5 split
    tz = X.index.tz
    test_start_dt = pd.Timestamp('2024-06-01', tz=tz)
    finetune_start_dt = test_start_dt - pd.DateOffset(months=6)

    base_mask = X.index < finetune_start_dt
    finetune_mask = (X.index >= finetune_start_dt) & (X.index < test_start_dt)
    test_mask = X.index >= test_start_dt

    logger.info(f"  Base: {base_mask.sum():,}, Fine-tune: {finetune_mask.sum():,}, Test: {test_mask.sum():,}")

    model, test_pred = train_transfer_model(
        X[base_mask], y[base_mask],
        X[finetune_mask], y[finetune_mask],
        X[test_mask], y[test_mask],
        config_name="V5_Baseline"
    )

    metrics = evaluate_with_hour_breakdown(y[test_mask].values, test_pred, hours[test_mask].values)

    logger.info(f"\n  RESULTS (V5 Baseline):")
    logger.info(f"    Global sMAPE:    {metrics['global']['smape']:.2f}%")
    logger.info(f"    Hours 9-10 sMAPE:{metrics['hours_9_10']['smape']:.2f}% (n={metrics['hours_9_10']['count']})")
    logger.info(f"    Hours 9-10 Bias: {metrics['hours_9_10']['bias']:+.2f} TL")
    logger.info(f"    Other sMAPE:     {metrics['other']['smape']:.2f}%")

    return {
        'config': 'A_V5_Baseline',
        'test_pred': test_pred,
        'metrics': metrics,
        'X_test': X[test_mask],
        'y_test': y[test_mask],
        'hours_test': hours[test_mask],
    }


def run_config_b_seasonal_injection(df: pd.DataFrame, features: List[str]) -> Dict:
    """
    Config B: Seasonal Injection - Last 2mo + Summer 2023.
    """
    logger.info("\n" + "="*70)
    logger.info("CONFIG B: SEASONAL INJECTION")
    logger.info("Last 2 months (Apr-May 2024) + Summer 2023 (Jun-Aug)")
    logger.info("="*70)

    available = [f for f in features if f in df.columns]
    X = df[available].copy()
    y = df['price_real'].copy()

    # Add solar ramp features
    X, solar_features = add_solar_ramp_features(X)

    # Drop NaN
    mask = ~(X.isna().any(axis=1) | y.isna())
    X, y = X[mask], y[mask]
    hours = X.index.hour

    # Smart split
    splits = get_smart_finetuning_data(
        X, y,
        test_start='2024-06-01',
        recency_months=2,  # Apr-May 2024
        seasonal_months=[6, 7, 8],  # Jun-Aug
        seasonal_year=2023
    )

    model, test_pred = train_transfer_model(
        X[splits['base_mask']], y[splits['base_mask']],
        X[splits['finetune_mask']], y[splits['finetune_mask']],
        X[splits['test_mask']], y[splits['test_mask']],
        config_name="Seasonal_Injection"
    )

    metrics = evaluate_with_hour_breakdown(
        y[splits['test_mask']].values, test_pred,
        hours[splits['test_mask']].values
    )

    logger.info(f"\n  RESULTS (Seasonal Injection):")
    logger.info(f"    Global sMAPE:    {metrics['global']['smape']:.2f}%")
    logger.info(f"    Hours 9-10 sMAPE:{metrics['hours_9_10']['smape']:.2f}% (n={metrics['hours_9_10']['count']})")
    logger.info(f"    Hours 9-10 Bias: {metrics['hours_9_10']['bias']:+.2f} TL")
    logger.info(f"    Other sMAPE:     {metrics['other']['smape']:.2f}%")

    return {
        'config': 'B_Seasonal_Injection',
        'test_pred': test_pred,
        'metrics': metrics,
        'X_test': X[splits['test_mask']],
        'y_test': y[splits['test_mask']],
        'hours_test': hours[splits['test_mask']],
    }


def run_config_c_weighted_seasonal(df: pd.DataFrame, features: List[str]) -> Dict:
    """
    Config C: Weighted Seasonal - Same as B but with higher weight on recency.
    """
    logger.info("\n" + "="*70)
    logger.info("CONFIG C: WEIGHTED SEASONAL")
    logger.info("Last 2 months (weight=2.0) + Summer 2023 (weight=1.0)")
    logger.info("="*70)

    available = [f for f in features if f in df.columns]
    X = df[available].copy()
    y = df['price_real'].copy()

    # Add solar ramp features
    X, solar_features = add_solar_ramp_features(X)

    # Drop NaN
    mask = ~(X.isna().any(axis=1) | y.isna())
    X, y = X[mask], y[mask]
    hours = X.index.hour

    # Smart split
    splits = get_smart_finetuning_data(
        X, y,
        test_start='2024-06-01',
        recency_months=2,
        seasonal_months=[6, 7, 8],
        seasonal_year=2023
    )

    # Create sample weights
    finetune_idx = X[splits['finetune_mask']].index
    recency_idx = X[splits['recency_mask']].index
    seasonal_idx = X[splits['seasonal_mask']].index

    weights = np.ones(len(finetune_idx))
    for i, idx in enumerate(finetune_idx):
        if idx in recency_idx:
            weights[i] = 2.0  # Higher weight for recency (price level)
        # Seasonal keeps weight 1.0

    logger.info(f"  Weights: Recency={2.0}, Seasonal={1.0}")
    logger.info(f"  Mean weight: {weights.mean():.2f}")

    model, test_pred = train_transfer_model(
        X[splits['base_mask']], y[splits['base_mask']],
        X[splits['finetune_mask']], y[splits['finetune_mask']],
        X[splits['test_mask']], y[splits['test_mask']],
        sample_weights=weights,
        config_name="Weighted_Seasonal"
    )

    metrics = evaluate_with_hour_breakdown(
        y[splits['test_mask']].values, test_pred,
        hours[splits['test_mask']].values
    )

    logger.info(f"\n  RESULTS (Weighted Seasonal):")
    logger.info(f"    Global sMAPE:    {metrics['global']['smape']:.2f}%")
    logger.info(f"    Hours 9-10 sMAPE:{metrics['hours_9_10']['smape']:.2f}% (n={metrics['hours_9_10']['count']})")
    logger.info(f"    Hours 9-10 Bias: {metrics['hours_9_10']['bias']:+.2f} TL")
    logger.info(f"    Other sMAPE:     {metrics['other']['smape']:.2f}%")

    return {
        'config': 'C_Weighted_Seasonal',
        'test_pred': test_pred,
        'metrics': metrics,
        'X_test': X[splits['test_mask']],
        'y_test': y[splits['test_mask']],
        'hours_test': hours[splits['test_mask']],
    }


def run_config_d_full_seasonal_blend(df: pd.DataFrame, features: List[str]) -> Dict:
    """
    Config D: Full Seasonal Blend - 3mo recency + Full year seasonal pattern.
    """
    logger.info("\n" + "="*70)
    logger.info("CONFIG D: FULL SEASONAL BLEND")
    logger.info("Last 3 months + Same months from 2023 (Mar-May 2023)")
    logger.info("="*70)

    available = [f for f in features if f in df.columns]
    X = df[available].copy()
    y = df['price_real'].copy()

    # Add solar ramp features
    X, solar_features = add_solar_ramp_features(X)

    # Drop NaN
    mask = ~(X.isna().any(axis=1) | y.isna())
    X, y = X[mask], y[mask]
    hours = X.index.hour

    # Smart split with same calendar months
    splits = get_smart_finetuning_data(
        X, y,
        test_start='2024-06-01',
        recency_months=3,  # Mar-May 2024
        seasonal_months=[3, 4, 5],  # Mar-May 2023 (same calendar)
        seasonal_year=2023
    )

    model, test_pred = train_transfer_model(
        X[splits['base_mask']], y[splits['base_mask']],
        X[splits['finetune_mask']], y[splits['finetune_mask']],
        X[splits['test_mask']], y[splits['test_mask']],
        config_name="Full_Seasonal_Blend"
    )

    metrics = evaluate_with_hour_breakdown(
        y[splits['test_mask']].values, test_pred,
        hours[splits['test_mask']].values
    )

    logger.info(f"\n  RESULTS (Full Seasonal Blend):")
    logger.info(f"    Global sMAPE:    {metrics['global']['smape']:.2f}%")
    logger.info(f"    Hours 9-10 sMAPE:{metrics['hours_9_10']['smape']:.2f}% (n={metrics['hours_9_10']['count']})")
    logger.info(f"    Hours 9-10 Bias: {metrics['hours_9_10']['bias']:+.2f} TL")
    logger.info(f"    Other sMAPE:     {metrics['other']['smape']:.2f}%")

    return {
        'config': 'D_Full_Seasonal_Blend',
        'test_pred': test_pred,
        'metrics': metrics,
        'X_test': X[splits['test_mask']],
        'y_test': y[splits['test_mask']],
        'hours_test': hours[splits['test_mask']],
    }


# =============================================================================
# MAIN EXPERIMENT LOOP
# =============================================================================

def run_v9_experiments():
    """Run all V9 experiments: Seasonally-Aligned Transfer Learning."""
    output_dir = PROJECT_ROOT / 'reports' / 'optimized_search_v9'
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*70)
    logger.info("OPTIMIZED SEARCH V9 - Seasonally-Aligned Transfer Learning")
    logger.info(f"Baseline to beat: {BASELINE_SMAPE}% sMAPE")
    logger.info(f"Problem hours: {PROBLEM_HOURS}")
    logger.info("="*70)

    # Load data
    df = load_data()
    logger.info(f"Loaded data: {df.shape}")
    logger.info(f"Date range: {df.index.min()} to {df.index.max()}")

    results = []

    # Run all configs
    results.append(run_config_a_v5_baseline(df, BASE_FEATURES))
    results.append(run_config_b_seasonal_injection(df, BASE_FEATURES))
    results.append(run_config_c_weighted_seasonal(df, BASE_FEATURES))
    results.append(run_config_d_full_seasonal_blend(df, BASE_FEATURES))

    # =========================================================================
    # FINAL COMPARISON
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("FINAL COMPARISON")
    logger.info("="*70)

    comparison = []
    for r in results:
        m = r['metrics']
        comparison.append({
            'Config': r['config'],
            'Global_sMAPE': m['global']['smape'],
            'H9_10_sMAPE': m['hours_9_10']['smape'],
            'H9_10_Bias': m['hours_9_10']['bias'],
            'Other_sMAPE': m['other']['smape'],
        })

    logger.info(f"\n{'Config':<30} {'Global':>10} {'H9-10':>10} {'Bias':>10} {'Other':>10} {'Beat?':>8}")
    logger.info("-"*85)

    for c in sorted(comparison, key=lambda x: x['Global_sMAPE']):
        beat = "YES" if c['Global_sMAPE'] < BASELINE_SMAPE else "no"
        logger.info(f"{c['Config']:<30} {c['Global_sMAPE']:>9.2f}% {c['H9_10_sMAPE']:>9.2f}% {c['H9_10_Bias']:>+9.2f} {c['Other_sMAPE']:>9.2f}% {beat:>8}")

    # Best result
    best_comp = min(comparison, key=lambda x: x['Global_sMAPE'])
    best_result = next(r for r in results if r['config'] == best_comp['Config'])

    logger.info(f"\n{'='*70}")
    if best_comp['Global_sMAPE'] < BASELINE_SMAPE:
        improvement = BASELINE_SMAPE - best_comp['Global_sMAPE']
        logger.info(f"SUCCESS! Beat {BASELINE_SMAPE}% by {improvement:.2f}%!")
    else:
        logger.info(f"Best: {best_comp['Global_sMAPE']:.2f}% (baseline: {BASELINE_SMAPE}%)")

    logger.info(f"  Best Config: {best_comp['Config']}")
    logger.info(f"  Global sMAPE: {best_comp['Global_sMAPE']:.2f}%")
    logger.info(f"  Hours 9-10 sMAPE: {best_comp['H9_10_sMAPE']:.2f}%")
    logger.info(f"  Hours 9-10 Bias: {best_comp['H9_10_Bias']:.2f} TL")

    # =========================================================================
    # HOUR 9-10 BIAS COMPARISON
    # =========================================================================
    logger.info(f"\n{'='*70}")
    logger.info("HOURS 9-10 BIAS COMPARISON")
    logger.info("="*70)
    logger.info(f"Target: Reduce from +26 TL to ~0 TL")

    for c in comparison:
        logger.info(f"  {c['Config']:<30}: Bias = {c['H9_10_Bias']:+.2f} TL")

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    summary = {
        'timestamp': datetime.now().isoformat(),
        'baseline_smape': BASELINE_SMAPE,
        'problem_hours': PROBLEM_HOURS,
        'best_config': best_comp['Config'],
        'best_global_smape': float(best_comp['Global_sMAPE']),
        'best_h910_smape': float(best_comp['H9_10_sMAPE']),
        'best_h910_bias': float(best_comp['H9_10_Bias']),
        'comparison': comparison,
    }

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Save best predictions
    pred_df = pd.DataFrame({
        'datetime': best_result['X_test'].index,
        'hour': best_result['hours_test'].values,
        'is_problem_hour': np.isin(best_result['hours_test'].values, PROBLEM_HOURS),
        'y_true': best_result['y_test'].values,
        'y_pred': best_result['test_pred'],
    })
    pred_df.to_csv(output_dir / 'best_predictions.csv', index=False)
    pred_df.to_parquet(output_dir / 'best_predictions.parquet', index=False)

    logger.info(f"\nResults saved to: {output_dir}")

    return results


if __name__ == "__main__":
    run_v9_experiments()
