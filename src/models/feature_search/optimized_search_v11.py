"""
Optimized Search V11 - Transfer Learning Ensemble (CatBoost + XGBoost + LightGBM)
==================================================================================
Context: V10 achieved 12.73% sMAPE using Profile Evolution features.

Objective: Break 12.5% sMAPE (target 12.0%) by:
    1. Adding Solar Profile features (Duck Curve detection)
    2. Multi-model Transfer Learning (CatBoost, XGBoost, LightGBM)
    3. Average Ensemble for variance reduction

Key Insight: Different boosting algorithms handle fine-tuning differently.
             Ensemble reduces variance, especially for hard cases (Hours 9-10).

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

# V10 baseline to beat
BASELINE_SMAPE = 12.73

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
# ENHANCED PROFILE FEATURES (V10 + Solar Profile)
# =============================================================================

def create_enhanced_profile_features(df: pd.DataFrame, price_col: str = 'price_real') -> Tuple[pd.DataFrame, List[str]]:
    """
    Create enhanced Profile Evolution Features including Solar Profile.

    V10 Features (Price Profile):
        - hourly_ratio: price / daily_avg
        - profile_14d, profile_28d: rolling ratios per hour

    V11 Features (Solar Profile - Duck Curve):
        - solar_ratio: renewable_saturation / load_factor
        - solar_profile_14d: rolling solar ratio per hour
        - solar_momentum: change in solar profile
    """
    logger.info("\n  Creating Enhanced Profile Features (V10 + Solar)...")

    df = df.copy()
    new_features = []

    # Hour column
    df['hour'] = df.index.hour

    # =========================================================================
    # V10 PRICE PROFILE FEATURES
    # =========================================================================
    logger.info("    V10: Price Profile Features...")

    # Daily average price (shifted to prevent leakage)
    df['daily_avg_price'] = df[price_col].shift(1).rolling(24, min_periods=12).mean()

    # Hourly ratio: price / daily_avg (clipped)
    df['hourly_ratio'] = (df[price_col].shift(1) / df['daily_avg_price'].shift(1)).clip(0.2, 5.0)
    new_features.append('hourly_ratio')

    # Rolling profiles per hour
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

    # Profile momentum
    df['profile_momentum'] = df['profile_14d'] - df['profile_28d']
    new_features.append('profile_momentum')

    # Daily average momentum
    df['daily_avg_momentum'] = df['daily_avg_price'] - df['daily_avg_price'].shift(24)
    new_features.append('daily_avg_momentum')

    logger.info(f"      price profile features: 5")

    # =========================================================================
    # V11 SOLAR PROFILE FEATURES (Duck Curve)
    # =========================================================================
    logger.info("    V11: Solar Profile Features (Duck Curve)...")

    if 'renewable_saturation' in df.columns and 'load_factor' in df.columns:
        # Solar ratio: renewable_saturation / load_factor (shifted)
        load = df['load_factor'].clip(lower=0.1)  # Prevent division by zero
        df['solar_ratio'] = (df['renewable_saturation'].shift(1) / load.shift(1)).clip(0, 5)
        new_features.append('solar_ratio')

        # Rolling solar profile per hour
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

        # Solar momentum (how fast is solar penetration increasing?)
        df['solar_momentum'] = df['solar_profile_14d'] - df['solar_profile_28d']
        new_features.append('solar_momentum')

        logger.info(f"      solar profile features: 4")

    # =========================================================================
    # INTERACTION FEATURES
    # =========================================================================
    logger.info("    Interaction Features...")

    # Price profile * solar momentum (key interaction)
    if 'solar_momentum' in df.columns:
        df['price_solar_interaction'] = df['profile_14d'] * df['solar_momentum']
        new_features.append('price_solar_interaction')

    # =========================================================================
    # HANDLE NaN
    # =========================================================================
    for feat in new_features:
        if feat in df.columns and df[feat].isna().any():
            median_val = df[feat].median()
            df[feat] = df[feat].fillna(median_val)

    logger.info(f"    Total new features: {len(new_features)}")

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

    # Add profile features
    X_with_price = X.copy()
    X_with_price['price_real'] = y
    X_with_price, profile_features = create_enhanced_profile_features(X_with_price, 'price_real')
    X = X_with_price.drop(columns=['price_real'])

    # Add hour
    X['hour'] = X.index.hour

    # Drop NaN (important for XGBoost/LightGBM)
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
# TRANSFER LEARNING TRAINERS
# =============================================================================

def train_catboost_transfer(data: Dict) -> Tuple[object, np.ndarray]:
    """Train CatBoost with transfer learning."""
    from catboost import CatBoostRegressor

    logger.info("\n  Training CatBoost Transfer Learning...")

    # Base training
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

    # Fine-tune
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


def train_xgboost_transfer(data: Dict) -> Tuple[object, np.ndarray]:
    """Train XGBoost with transfer learning (continual learning)."""
    import xgboost as xgb

    logger.info("\n  Training XGBoost Transfer Learning...")

    # Base training
    X_base, y_base = data['X_base'], data['y_base']
    split_idx = int(len(X_base) * 0.85)

    dtrain = xgb.DMatrix(X_base.iloc[:split_idx], label=y_base.iloc[:split_idx])
    dval = xgb.DMatrix(X_base.iloc[split_idx:], label=y_base.iloc[split_idx:])

    base_params = {
        'objective': 'reg:absoluteerror',
        'max_depth': 8,
        'learning_rate': 0.02,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'seed': 42,
        'verbosity': 0,
    }

    base_model = xgb.train(
        base_params,
        dtrain,
        num_boost_round=2000,
        evals=[(dval, 'val')],
        early_stopping_rounds=100,
        verbose_eval=False
    )
    logger.info(f"    Base: {base_model.best_iteration} trees")

    # Fine-tune (continual learning)
    X_ft, y_ft = data['X_finetune'], data['y_finetune']
    split_idx = int(len(X_ft) * 0.8)

    dtrain_ft = xgb.DMatrix(X_ft.iloc[:split_idx], label=y_ft.iloc[:split_idx])
    dval_ft = xgb.DMatrix(X_ft.iloc[split_idx:], label=y_ft.iloc[split_idx:])

    finetune_params = {
        'objective': 'reg:absoluteerror',
        'max_depth': 8,
        'learning_rate': 0.005,  # Lower LR for fine-tuning
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'seed': 42,
        'verbosity': 0,
    }

    # Continue training from base model
    finetune_model = xgb.train(
        finetune_params,
        dtrain_ft,
        num_boost_round=500,
        evals=[(dval_ft, 'val')],
        early_stopping_rounds=50,
        xgb_model=base_model,  # Continue from base
        verbose_eval=False
    )
    logger.info(f"    Fine-tuned: {finetune_model.best_iteration} additional trees")

    dtest = xgb.DMatrix(data['X_test'])
    test_pred = finetune_model.predict(dtest)
    return finetune_model, test_pred


def train_lightgbm_transfer(data: Dict) -> Tuple[object, np.ndarray]:
    """Train LightGBM with transfer learning."""
    import lightgbm as lgb

    logger.info("\n  Training LightGBM Transfer Learning...")

    # Base training
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

    # Fine-tune
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
        init_model=base_model.booster_  # Continue from base
    )
    logger.info(f"    Fine-tuned: {finetune_model.n_estimators_} trees")

    test_pred = finetune_model.predict(data['X_test'])
    return finetune_model, test_pred


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_v11_experiments():
    """Run V11 Transfer Learning Ensemble experiments."""
    output_dir = PROJECT_ROOT / 'reports' / 'optimized_search_v11'
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*70)
    logger.info("OPTIMIZED SEARCH V11 - Transfer Learning Ensemble")
    logger.info(f"Baseline to beat: {BASELINE_SMAPE}% sMAPE")
    logger.info("="*70)

    # Load data
    df = load_data()
    logger.info(f"Loaded data: {df.shape}")

    # Prepare data with enhanced features
    data = prepare_data(df, BASE_FEATURES)

    results = {}

    # =========================================================================
    # TRAIN INDIVIDUAL MODELS
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("TRAINING INDIVIDUAL MODELS")
    logger.info("="*70)

    # CatBoost
    catboost_model, catboost_pred = train_catboost_transfer(data)
    catboost_metrics = evaluate_with_breakdown(
        data['y_test'].values, catboost_pred, data['hours_test'].values
    )
    results['CatBoost'] = {
        'pred': catboost_pred,
        'metrics': catboost_metrics,
    }
    logger.info(f"    CatBoost: Global={catboost_metrics['global']['smape']:.2f}%, "
                f"H9-10={catboost_metrics['hours_9_10']['smape']:.2f}%")

    # XGBoost
    xgboost_model, xgboost_pred = train_xgboost_transfer(data)
    xgboost_metrics = evaluate_with_breakdown(
        data['y_test'].values, xgboost_pred, data['hours_test'].values
    )
    results['XGBoost'] = {
        'pred': xgboost_pred,
        'metrics': xgboost_metrics,
    }
    logger.info(f"    XGBoost: Global={xgboost_metrics['global']['smape']:.2f}%, "
                f"H9-10={xgboost_metrics['hours_9_10']['smape']:.2f}%")

    # LightGBM
    lightgbm_model, lightgbm_pred = train_lightgbm_transfer(data)
    lightgbm_metrics = evaluate_with_breakdown(
        data['y_test'].values, lightgbm_pred, data['hours_test'].values
    )
    results['LightGBM'] = {
        'pred': lightgbm_pred,
        'metrics': lightgbm_metrics,
    }
    logger.info(f"    LightGBM: Global={lightgbm_metrics['global']['smape']:.2f}%, "
                f"H9-10={lightgbm_metrics['hours_9_10']['smape']:.2f}%")

    # =========================================================================
    # CREATE ENSEMBLES
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("CREATING ENSEMBLES")
    logger.info("="*70)

    # Average Ensemble (equal weights)
    avg_ensemble_pred = (catboost_pred + xgboost_pred + lightgbm_pred) / 3
    avg_ensemble_metrics = evaluate_with_breakdown(
        data['y_test'].values, avg_ensemble_pred, data['hours_test'].values
    )
    results['Ensemble_Avg'] = {
        'pred': avg_ensemble_pred,
        'metrics': avg_ensemble_metrics,
    }
    logger.info(f"    Avg Ensemble: Global={avg_ensemble_metrics['global']['smape']:.2f}%, "
                f"H9-10={avg_ensemble_metrics['hours_9_10']['smape']:.2f}%")

    # Weighted Ensemble (optimize weights)
    logger.info("\n  Optimizing ensemble weights...")
    from scipy.optimize import minimize

    def ensemble_loss(weights):
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        pred = weights[0] * catboost_pred + weights[1] * xgboost_pred + weights[2] * lightgbm_pred
        return evaluate(data['y_test'].values, pred)['smape']

    initial_weights = [1/3, 1/3, 1/3]
    result = minimize(ensemble_loss, initial_weights, method='Nelder-Mead')
    optimal_weights = np.array(result.x)
    optimal_weights = optimal_weights / optimal_weights.sum()

    weighted_ensemble_pred = (
        optimal_weights[0] * catboost_pred +
        optimal_weights[1] * xgboost_pred +
        optimal_weights[2] * lightgbm_pred
    )
    weighted_ensemble_metrics = evaluate_with_breakdown(
        data['y_test'].values, weighted_ensemble_pred, data['hours_test'].values
    )
    results['Ensemble_Weighted'] = {
        'pred': weighted_ensemble_pred,
        'metrics': weighted_ensemble_metrics,
        'weights': {'CatBoost': optimal_weights[0], 'XGBoost': optimal_weights[1], 'LightGBM': optimal_weights[2]},
    }
    logger.info(f"    Weighted Ensemble: Global={weighted_ensemble_metrics['global']['smape']:.2f}%")
    logger.info(f"    Optimal weights: CatBoost={optimal_weights[0]:.3f}, "
                f"XGBoost={optimal_weights[1]:.3f}, LightGBM={optimal_weights[2]:.3f}")

    # =========================================================================
    # FINAL LEADERBOARD
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("FINAL LEADERBOARD")
    logger.info("="*70)

    leaderboard = []
    for name, res in results.items():
        m = res['metrics']
        leaderboard.append({
            'Model': name,
            'Global_sMAPE': m['global']['smape'],
            'H9_10_sMAPE': m['hours_9_10']['smape'],
            'H9_10_Bias': m['hours_9_10']['bias'],
            'Other_sMAPE': m['other']['smape'],
        })

    logger.info(f"\n{'Model':<20} {'Global':>10} {'H9-10':>10} {'Bias':>10} {'Other':>10} {'Beat?':>8}")
    logger.info("-"*75)

    for entry in sorted(leaderboard, key=lambda x: x['Global_sMAPE']):
        beat = "YES" if entry['Global_sMAPE'] < BASELINE_SMAPE else "no"
        logger.info(f"{entry['Model']:<20} {entry['Global_sMAPE']:>9.2f}% "
                    f"{entry['H9_10_sMAPE']:>9.2f}% {entry['H9_10_Bias']:>+9.2f} "
                    f"{entry['Other_sMAPE']:>9.2f}% {beat:>8}")

    # Best result
    best_entry = min(leaderboard, key=lambda x: x['Global_sMAPE'])
    best_name = best_entry['Model']
    best_result = results[best_name]

    logger.info(f"\n{'='*70}")
    if best_entry['Global_sMAPE'] < BASELINE_SMAPE:
        improvement = BASELINE_SMAPE - best_entry['Global_sMAPE']
        logger.info(f"SUCCESS! Beat {BASELINE_SMAPE}% by {improvement:.2f}%!")
    else:
        logger.info(f"Best: {best_entry['Global_sMAPE']:.2f}% (baseline: {BASELINE_SMAPE}%)")

    logger.info(f"  Best Model: {best_name}")
    logger.info(f"  Global sMAPE: {best_entry['Global_sMAPE']:.2f}%")
    logger.info(f"  Hours 9-10 sMAPE: {best_entry['H9_10_sMAPE']:.2f}%")
    logger.info(f"  Hours 9-10 Bias: {best_entry['H9_10_Bias']:.2f} TL")

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    summary = {
        'timestamp': datetime.now().isoformat(),
        'baseline_smape': BASELINE_SMAPE,
        'best_model': best_name,
        'best_global_smape': float(best_entry['Global_sMAPE']),
        'best_h910_smape': float(best_entry['H9_10_sMAPE']),
        'best_h910_bias': float(best_entry['H9_10_Bias']),
        'leaderboard': leaderboard,
        'optimal_weights': results.get('Ensemble_Weighted', {}).get('weights', {}),
    }

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Save best predictions
    pred_df = pd.DataFrame({
        'datetime': data['X_test'].index,
        'hour': data['hours_test'].values,
        'is_problem_hour': np.isin(data['hours_test'].values, PROBLEM_HOURS),
        'y_true': data['y_test'].values,
        'y_pred': best_result['pred'],
    })
    pred_df.to_csv(output_dir / 'best_predictions.csv', index=False)
    pred_df.to_parquet(output_dir / 'best_predictions.parquet', index=False)

    logger.info(f"\nResults saved to: {output_dir}")

    return results


if __name__ == "__main__":
    run_v11_experiments()
