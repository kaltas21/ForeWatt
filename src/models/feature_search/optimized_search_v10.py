"""
Optimized Search V10 - Profile Evolution Features (Shape vs Level Decoupling)
===============================================================================
Context: V5 (14.08% sMAPE) fails in Morning (Hours 9-10) because it learns
         Winter RELATIVE shape (High) and applies it to Summer (Low).
         V9 failed because historical 2023 data had wrong ABSOLUTE levels.

Insight: Decouple "Shape" from "Level" by:
    1. Hourly Ratio = price / daily_avg (removes level noise)
    2. Rolling Profile (14d, 28d) captures shape evolution
    3. Seasonal Shape Proxy = 2023 ratio * 2024 level (correct shape + level)

Key Concept:
    - "1.2" means "20% more expensive than average today"
    - This is invariant to whether the average is 2000 TL or 500 TL
    - Profile_28d will dynamically trend DOWN for Hour 09 as Spring→Summer

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
# PROFILE EVOLUTION FEATURE ENGINEERING
# =============================================================================

def create_profile_features(df: pd.DataFrame, price_col: str = 'price_real') -> Tuple[pd.DataFrame, List[str]]:
    """
    Create Profile Evolution Features that decouple Shape from Level.

    Step A: Daily Normalization
        - daily_avg_price: Rolling 24h mean of price
        - hourly_ratio: price / daily_avg (level-invariant shape)

    Step B: Profile Evolution (The Trend)
        - profile_14d: Rolling 14-day mean of hourly_ratio per hour
        - profile_28d: Rolling 28-day mean of hourly_ratio per hour

    Step C: Seasonal Proxy (The History)
        - seasonal_shape_proxy: hourly_ratio from 52 weeks ago * current daily_avg

    IMPORTANT: All features are shifted to prevent leakage.
    """
    logger.info("\n  Creating Profile Evolution Features...")

    df = df.copy()
    new_features = []

    # Ensure hour column
    df['hour'] = df.index.hour

    # =========================================================================
    # STEP A: Daily Normalization
    # =========================================================================
    logger.info("    Step A: Daily Normalization...")

    # Daily average price (rolling 24h mean, shifted to prevent leakage)
    df['daily_avg_price'] = df[price_col].shift(1).rolling(24, min_periods=12).mean()

    # Hourly ratio: price / daily_avg (clipped to prevent extremes)
    # SHIFTED by 1 to prevent leakage (we use yesterday's ratio)
    df['hourly_ratio'] = (df[price_col].shift(1) / df['daily_avg_price'].shift(1)).clip(0.2, 5.0)
    new_features.append('hourly_ratio')

    logger.info(f"      hourly_ratio range: [{df['hourly_ratio'].min():.2f}, {df['hourly_ratio'].max():.2f}]")

    # =========================================================================
    # STEP B: Profile Evolution (Rolling per Hour)
    # =========================================================================
    logger.info("    Step B: Profile Evolution (Rolling per Hour)...")

    # For each hour, calculate rolling mean of the ratio
    # This captures how the "shape" of that hour evolves over time

    # Sort by (hour, datetime) for proper groupby rolling
    df_sorted = df.sort_index()

    # 14-day profile: rolling mean of hourly_ratio for each hour
    # We need 14 observations of the same hour = 14 days
    profile_14d = []
    profile_28d = []

    for hour in range(24):
        hour_mask = df_sorted['hour'] == hour
        hour_ratios = df_sorted.loc[hour_mask, 'hourly_ratio']

        # Rolling mean (shifted by 1 day = 24 hours to prevent leakage)
        p14 = hour_ratios.rolling(14, min_periods=7).mean().shift(1)
        p28 = hour_ratios.rolling(28, min_periods=14).mean().shift(1)

        profile_14d.append(p14)
        profile_28d.append(p28)

    # Combine all hours back
    df['profile_14d'] = pd.concat(profile_14d).sort_index()
    df['profile_28d'] = pd.concat(profile_28d).sort_index()
    new_features.extend(['profile_14d', 'profile_28d'])

    logger.info(f"      profile_14d range: [{df['profile_14d'].min():.3f}, {df['profile_14d'].max():.3f}]")
    logger.info(f"      profile_28d range: [{df['profile_28d'].min():.3f}, {df['profile_28d'].max():.3f}]")

    # =========================================================================
    # STEP C: Seasonal Shape Proxy (2023 Shape + Current Level)
    # =========================================================================
    logger.info("    Step C: Seasonal Shape Proxy...")

    # Get hourly_ratio from 52 weeks ago (364 days = 364*24 hours)
    lag_hours = 364 * 24
    df['ratio_52w_ago'] = df['hourly_ratio'].shift(lag_hours)

    # Seasonal shape proxy = historical ratio * current daily average
    # This gives us "what the price SHOULD be based on last year's shape"
    df['seasonal_shape_proxy'] = df['ratio_52w_ago'] * df['daily_avg_price']
    new_features.extend(['ratio_52w_ago', 'seasonal_shape_proxy'])

    logger.info(f"      seasonal_shape_proxy coverage: {df['seasonal_shape_proxy'].notna().sum()} rows")

    # =========================================================================
    # ADDITIONAL PROFILE FEATURES
    # =========================================================================
    logger.info("    Step D: Additional Profile Features...")

    # Profile momentum: how fast is the shape changing?
    df['profile_momentum'] = df['profile_14d'] - df['profile_28d']
    new_features.append('profile_momentum')

    # Profile vs historical: is current shape above/below seasonal?
    df['profile_vs_seasonal'] = df['profile_14d'] - df['ratio_52w_ago']
    new_features.append('profile_vs_seasonal')

    # Daily average momentum
    df['daily_avg_momentum'] = df['daily_avg_price'] - df['daily_avg_price'].shift(24)
    new_features.append('daily_avg_momentum')

    # =========================================================================
    # HANDLE NaN
    # =========================================================================
    # Fill NaN with median for each feature
    for feat in new_features:
        if feat in df.columns and df[feat].isna().any():
            median_val = df[feat].median()
            df[feat] = df[feat].fillna(median_val)

    logger.info(f"    Created {len(new_features)} profile features")

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


def prepare_data_with_profiles(df: pd.DataFrame, features: List[str],
                               add_profiles: bool = True,
                               test_start: str = '2024-06-01',
                               finetune_months: int = 6) -> Dict:
    """
    Prepare data with V5-style split and optional profile features.
    """
    available = [f for f in features if f in df.columns]

    X = df[available].copy()
    y = df['price_real'].copy()

    # Add profile features BEFORE splitting (needs full history)
    profile_features = []
    if add_profiles:
        # Need to join price for profile calculation
        X_with_price = X.copy()
        X_with_price['price_real'] = y
        X_with_price, profile_features = create_profile_features(X_with_price, 'price_real')
        # Remove price_real from features
        X = X_with_price.drop(columns=['price_real'])

    # Add hour column
    X['hour'] = X.index.hour

    # Drop NaN
    mask = ~(X.isna().any(axis=1) | y.isna())
    X, y = X[mask], y[mask]
    hours = X['hour']

    # V5-style split
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
    logger.info(f"    Features:  {len(X.columns)} ({len(profile_features)} profile)")

    return {
        'X_base': X[base_mask],
        'y_base': y[base_mask],
        'X_finetune': X[finetune_mask],
        'y_finetune': y[finetune_mask],
        'X_test': X[test_mask],
        'y_test': y[test_mask],
        'hours_test': hours[test_mask],
        'features': list(X.columns),
        'profile_features': profile_features,
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

    # Other hours
    other_mask = ~problem_mask
    other_metrics = evaluate(y_true[other_mask], y_pred[other_mask]) if other_mask.sum() > 0 else {}
    other_metrics['count'] = other_mask.sum()

    return {
        'global': global_metrics,
        'hours_9_10': problem_metrics,
        'other': other_metrics,
    }


# =============================================================================
# TRANSFER LEARNING TRAINER
# =============================================================================

def train_v5_transfer_model(data: Dict, config_name: str = "default"):
    """Train V5-style transfer learning model."""
    from catboost import CatBoostRegressor

    logger.info(f"\n  Training {config_name}...")

    # Step A: Base training
    X_base = data['X_base']
    y_base = data['y_base']

    split_idx = int(len(X_base) * 0.85)

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
    base_model.fit(
        X_base.iloc[:split_idx], y_base.iloc[:split_idx],
        eval_set=(X_base.iloc[split_idx:], y_base.iloc[split_idx:]),
        verbose=False
    )
    logger.info(f"    Base model: {base_model.tree_count_} trees")

    # Step B: Fine-tune
    X_ft = data['X_finetune']
    y_ft = data['y_finetune']

    split_idx = int(len(X_ft) * 0.8)

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
    finetune_model.fit(
        X_ft.iloc[:split_idx], y_ft.iloc[:split_idx],
        eval_set=(X_ft.iloc[split_idx:], y_ft.iloc[split_idx:]),
        init_model=base_model,
        verbose=False
    )
    logger.info(f"    Fine-tuned model: {finetune_model.tree_count_} trees")

    # Predictions
    test_pred = finetune_model.predict(data['X_test'])

    # Evaluate
    metrics = evaluate_with_hour_breakdown(
        data['y_test'].values, test_pred, data['hours_test'].values
    )

    logger.info(f"    Global sMAPE:    {metrics['global']['smape']:.2f}%")
    logger.info(f"    Hours 9-10 sMAPE:{metrics['hours_9_10']['smape']:.2f}%")
    logger.info(f"    Hours 9-10 Bias: {metrics['hours_9_10']['bias']:+.2f} TL")

    return {
        'model': finetune_model,
        'test_pred': test_pred,
        'metrics': metrics,
    }


# =============================================================================
# EXPERIMENT CONFIGS
# =============================================================================

def run_config_a_baseline(df: pd.DataFrame) -> Dict:
    """Config A: V5 Baseline (no profile features)."""
    logger.info("\n" + "="*70)
    logger.info("CONFIG A: V5 BASELINE")
    logger.info("Standard feature set (no profile features)")
    logger.info("="*70)

    data = prepare_data_with_profiles(df, BASE_FEATURES, add_profiles=False)
    result = train_v5_transfer_model(data, "V5_Baseline")

    return {
        'config': 'A_V5_Baseline',
        'test_pred': result['test_pred'],
        'metrics': result['metrics'],
        'data': data,
    }


def run_config_b_profile_evolution(df: pd.DataFrame) -> Dict:
    """Config B: Add profile evolution features (14d, 28d rolling)."""
    logger.info("\n" + "="*70)
    logger.info("CONFIG B: PROFILE EVOLUTION")
    logger.info("Add profile_14d, profile_28d, profile_momentum")
    logger.info("="*70)

    # Start with base features
    features = BASE_FEATURES.copy()

    # Prepare with profiles
    data = prepare_data_with_profiles(df, features, add_profiles=True)

    # Keep only profile evolution features (not seasonal proxy)
    profile_cols = ['hourly_ratio', 'profile_14d', 'profile_28d', 'profile_momentum',
                    'daily_avg_price', 'daily_avg_momentum', 'hour']
    cols_to_keep = [c for c in data['X_base'].columns
                    if c in features or c in profile_cols]

    data['X_base'] = data['X_base'][cols_to_keep]
    data['X_finetune'] = data['X_finetune'][cols_to_keep]
    data['X_test'] = data['X_test'][cols_to_keep]

    logger.info(f"  Features kept: {len(cols_to_keep)}")

    result = train_v5_transfer_model(data, "Profile_Evolution")

    return {
        'config': 'B_Profile_Evolution',
        'test_pred': result['test_pred'],
        'metrics': result['metrics'],
        'data': data,
    }


def run_config_c_seasonal_proxy(df: pd.DataFrame) -> Dict:
    """Config C: Add seasonal shape proxy (2023 ratio * 2024 level)."""
    logger.info("\n" + "="*70)
    logger.info("CONFIG C: SEASONAL SHAPE PROXY")
    logger.info("Add ratio_52w_ago, seasonal_shape_proxy, profile_vs_seasonal")
    logger.info("="*70)

    features = BASE_FEATURES.copy()

    data = prepare_data_with_profiles(df, features, add_profiles=True)

    # Keep seasonal proxy features
    proxy_cols = ['ratio_52w_ago', 'seasonal_shape_proxy', 'profile_vs_seasonal',
                  'daily_avg_price', 'hour']
    cols_to_keep = [c for c in data['X_base'].columns
                    if c in features or c in proxy_cols]

    data['X_base'] = data['X_base'][cols_to_keep]
    data['X_finetune'] = data['X_finetune'][cols_to_keep]
    data['X_test'] = data['X_test'][cols_to_keep]

    logger.info(f"  Features kept: {len(cols_to_keep)}")

    result = train_v5_transfer_model(data, "Seasonal_Proxy")

    return {
        'config': 'C_Seasonal_Proxy',
        'test_pred': result['test_pred'],
        'metrics': result['metrics'],
        'data': data,
    }


def run_config_d_all_features(df: pd.DataFrame) -> Dict:
    """Config D: All profile + seasonal features combined."""
    logger.info("\n" + "="*70)
    logger.info("CONFIG D: ALL-IN (Profile + Seasonal)")
    logger.info("All profile evolution and seasonal proxy features")
    logger.info("="*70)

    features = BASE_FEATURES.copy()

    data = prepare_data_with_profiles(df, features, add_profiles=True)

    # Keep all profile features
    logger.info(f"  All features: {len(data['X_base'].columns)}")

    result = train_v5_transfer_model(data, "All_Features")

    return {
        'config': 'D_All_Features',
        'test_pred': result['test_pred'],
        'metrics': result['metrics'],
        'data': data,
    }


# =============================================================================
# MAIN EXPERIMENT LOOP
# =============================================================================

def run_v10_experiments():
    """Run all V10 experiments: Profile Evolution Features."""
    output_dir = PROJECT_ROOT / 'reports' / 'optimized_search_v10'
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*70)
    logger.info("OPTIMIZED SEARCH V10 - Profile Evolution Features")
    logger.info(f"Baseline to beat: {BASELINE_SMAPE}% sMAPE")
    logger.info(f"Problem hours: {PROBLEM_HOURS}")
    logger.info("="*70)

    # Load data
    df = load_data()
    logger.info(f"Loaded data: {df.shape}")
    logger.info(f"Date range: {df.index.min()} to {df.index.max()}")

    results = []

    # Run all configs
    results.append(run_config_a_baseline(df))
    results.append(run_config_b_profile_evolution(df))
    results.append(run_config_c_seasonal_proxy(df))
    results.append(run_config_d_all_features(df))

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

    logger.info(f"\n{'Config':<25} {'Global':>10} {'H9-10':>10} {'Bias':>10} {'Other':>10} {'Beat?':>8}")
    logger.info("-"*80)

    for c in sorted(comparison, key=lambda x: x['Global_sMAPE']):
        beat = "YES" if c['Global_sMAPE'] < BASELINE_SMAPE else "no"
        logger.info(f"{c['Config']:<25} {c['Global_sMAPE']:>9.2f}% {c['H9_10_sMAPE']:>9.2f}% {c['H9_10_Bias']:>+9.2f} {c['Other_sMAPE']:>9.2f}% {beat:>8}")

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
    # HOUR 9-10 BIAS IMPROVEMENT
    # =========================================================================
    logger.info(f"\n{'='*70}")
    logger.info("HOURS 9-10 BIAS ANALYSIS")
    logger.info("="*70)
    logger.info("Target: Reduce from +26 TL (V5) to <10 TL")

    baseline_bias = comparison[0]['H9_10_Bias']  # A is baseline
    for c in comparison:
        improvement = baseline_bias - c['H9_10_Bias']
        logger.info(f"  {c['Config']:<25}: Bias = {c['H9_10_Bias']:+.2f} TL (change: {improvement:+.2f})")

    # =========================================================================
    # FEATURE IMPORTANCE (Best Model)
    # =========================================================================
    if 'model' in best_result.get('result', {}):
        logger.info(f"\n{'='*70}")
        logger.info("FEATURE IMPORTANCE (Best Model)")
        logger.info("="*70)

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
        'datetime': best_result['data']['X_test'].index,
        'hour': best_result['data']['hours_test'].values,
        'is_problem_hour': np.isin(best_result['data']['hours_test'].values, PROBLEM_HOURS),
        'y_true': best_result['data']['y_test'].values,
        'y_pred': best_result['test_pred'],
    })
    pred_df.to_csv(output_dir / 'best_predictions.csv', index=False)
    pred_df.to_parquet(output_dir / 'best_predictions.parquet', index=False)

    logger.info(f"\nResults saved to: {output_dir}")

    return results


if __name__ == "__main__":
    run_v10_experiments()
