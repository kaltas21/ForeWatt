"""
Optimized Search V4 - Training Window Strategy for Concept Drift
=================================================================
Context: Electricity price forecasting model trained on 2020-2025 data.
Problem: Market structure changed in 2024 (prices dropped ~40%, volatility vanished).
         Model is "over-remembering" the 2021-2022 crisis spikes.

Objective: Determine optimal Training Window Strategy to beat 15.96% sMAPE baseline.

Strategies:
    A. Recent History (Hard Cut) - Train only on 2023+ data
    B. Time Decay (Soft Cut) - Full data with exponential sample weights
    C. Transfer Learning - Pre-train on full, fine-tune on recent

Author: ForeWatt Team
Date: December 2025
"""

import sys
import os
import gc
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Baseline from stacking ensemble (optimized_search.py)
BASELINE_SMAPE = 15.96

# =============================================================================
# FEATURE SET (Same as winning V1 model)
# =============================================================================

BEST_FEATURES = [
    'hour_sin', 'hour_cos', 'dow_sin_x', 'dow_cos_x', 'is_weekend_x',
    'price_ptf_rolling_std_24h', 'price_ptf_rolling_mean_24h',
    'price_ptf_rolling_min_24h', 'price_ptf_rolling_max_24h',
    'price_ptf_lag_24h', 'price_ptf_lag_168h',
    'thermal_gap', 'thermal_gap_lag_24h',
    'renewable_saturation', 'spark_spread_proxy_lag_24h',
    'system_short_signal', 'load_factor', 'consumption_forecast',
    'reserve_margin_ratio', 'price_volatility_lag24h', 'realtime_premium_lag24h',
]


def load_data() -> pd.DataFrame:
    """Load master dataset with robust deflated prices."""
    path = PROJECT_ROOT / 'data' / 'gold' / 'master' / 'master_v2_fundamental.parquet'
    df = pd.read_parquet(path)

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date' in df.columns:
            df = df.set_index('date')
        elif 'datetime' in df.columns:
            df = df.set_index('datetime')

    # Check for price_real (robust deflated prices)
    if 'price_real' not in df.columns:
        logger.warning("'price_real' not found - using 'price' column")
        df['price_real'] = df['price']

    return df


def add_feature_interactions(df: pd.DataFrame, features: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Add interaction features for top predictors."""
    df = df.copy()

    interactions = [
        ('price_ptf_lag_24h', 'thermal_gap'),
        ('price_ptf_rolling_std_24h', 'renewable_saturation'),
        ('load_factor', 'reserve_margin_ratio'),
        ('hour_sin', 'price_ptf_lag_24h'),
        ('system_short_signal', 'thermal_gap'),
    ]

    new_features = []
    for f1, f2 in interactions:
        if f1 in df.columns and f2 in df.columns:
            name = f'{f1}_x_{f2}'
            df[name] = df[f1] * df[f2]
            new_features.append(name)

    return df, new_features


def prepare_base_data(df: pd.DataFrame, features: List[str],
                      test_start: str = '2024-06-01') -> Dict:
    """
    Prepare data with FIXED test set for fair comparison.

    Test set: 2024-06-01 onwards (stable market regime)
    Everything before: available for training strategies
    """
    available = [f for f in features if f in df.columns]

    X = df[available].copy()
    y = df['price_real'].copy()

    # Add interactions
    X, interaction_features = add_feature_interactions(X, available)
    all_features = available + interaction_features

    # Drop NaN
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]

    # Ensure timezone-aware comparison
    if X.index.tz is not None:
        test_start_dt = pd.Timestamp(test_start, tz=X.index.tz)
    else:
        test_start_dt = pd.Timestamp(test_start)

    # Fixed test split
    test_mask = X.index >= test_start_dt

    X_pretrain = X[~test_mask]
    y_pretrain = y[~test_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    logger.info(f"Full pretrain data: {len(X_pretrain)} rows ({X_pretrain.index.min()} to {X_pretrain.index.max()})")
    logger.info(f"Test data: {len(X_test)} rows ({X_test.index.min()} to {X_test.index.max()})")

    return {
        'X_pretrain': X_pretrain,
        'y_pretrain': y_pretrain,
        'X_test': X_test,
        'y_test': y_test,
        'features': all_features,
    }


# =============================================================================
# METRICS
# =============================================================================

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Calculate metrics including Bias (Mean Signed Error)."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae = np.mean(np.abs(y_true - y_pred))
    smape = 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))
    bias = np.mean(y_pred - y_true)  # Positive = over-predicting

    return {'mae': mae, 'smape': smape, 'bias': bias}


# =============================================================================
# MODEL TRAINING FUNCTIONS
# =============================================================================

def train_catboost(X_train, y_train, X_val, y_val, config, sample_weight=None, init_model=None):
    """Train CatBoost with optional sample weights and init_model for fine-tuning."""
    from catboost import CatBoostRegressor

    model = CatBoostRegressor(
        loss_function='MAE',
        random_state=42,
        verbose=False,
        early_stopping_rounds=100,
        **config
    )

    fit_params = {
        'eval_set': (X_val, y_val),
        'verbose': False,
    }

    if sample_weight is not None:
        fit_params['sample_weight'] = sample_weight

    if init_model is not None:
        fit_params['init_model'] = init_model

    model.fit(X_train, y_train, **fit_params)
    return model


def train_lightgbm(X_train, y_train, X_val, y_val, config, sample_weight=None):
    """Train LightGBM with optional sample weights."""
    import lightgbm as lgb

    model = lgb.LGBMRegressor(
        objective='regression',
        metric='mae',
        verbosity=-1,
        random_state=42,
        **config
    )

    fit_params = {
        'eval_set': [(X_val, y_val)],
        'callbacks': [lgb.early_stopping(100, verbose=False)],
    }

    if sample_weight is not None:
        fit_params['sample_weight'] = sample_weight

    model.fit(X_train, y_train, **fit_params)
    return model


def train_xgboost(X_train, y_train, X_val, y_val, config, sample_weight=None):
    """Train XGBoost with optional sample weights."""
    import xgboost as xgb

    model = xgb.XGBRegressor(
        random_state=42,
        verbosity=0,
        early_stopping_rounds=100,
        **config
    )

    fit_params = {
        'eval_set': [(X_val, y_val)],
        'verbose': False,
    }

    if sample_weight is not None:
        fit_params['sample_weight'] = sample_weight

    model.fit(X_train, y_train, **fit_params)
    return model


# =============================================================================
# STACKING ENSEMBLE (Reused from winning V1)
# =============================================================================

def run_stacking_ensemble(X_train, y_train, X_test, y_test,
                          sample_weight=None, strategy_name="default"):
    """
    Stacking ensemble with OOF predictions as meta-features.
    This is the winning architecture from optimized_search.py (15.96% baseline).
    """
    from sklearn.model_selection import KFold

    logger.info(f"\n  Training stacking ensemble for {strategy_name}...")
    logger.info(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    # Base model configs (same as V1)
    base_configs = {
        'catboost': {'iterations': 1000, 'depth': 8, 'learning_rate': 0.03, 'l2_leaf_reg': 3},
        'lightgbm': {'n_estimators': 1000, 'max_depth': 10, 'learning_rate': 0.03, 'num_leaves': 63},
        'xgboost': {'n_estimators': 1000, 'max_depth': 10, 'learning_rate': 0.03},
    }

    n_folds = 5
    kf = KFold(n_splits=n_folds, shuffle=False)  # No shuffle for time series

    oof_preds = {name: np.zeros(len(X_train)) for name in base_configs}
    test_preds = {name: np.zeros(len(X_test)) for name in base_configs}

    # Generate sample weights for each fold if provided
    if sample_weight is not None:
        weight_arr = np.array(sample_weight)
    else:
        weight_arr = None

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr = X_train.iloc[train_idx]
        X_vl = X_train.iloc[val_idx]
        y_tr = y_train.iloc[train_idx]
        y_vl = y_train.iloc[val_idx]

        fold_weights = weight_arr[train_idx] if weight_arr is not None else None

        for name, config in base_configs.items():
            if name == 'catboost':
                model = train_catboost(X_tr, y_tr, X_vl, y_vl, config, sample_weight=fold_weights)
            elif name == 'lightgbm':
                model = train_lightgbm(X_tr, y_tr, X_vl, y_vl, config, sample_weight=fold_weights)
            elif name == 'xgboost':
                model = train_xgboost(X_tr, y_tr, X_vl, y_vl, config, sample_weight=fold_weights)

            oof_preds[name][val_idx] = model.predict(X_vl)
            test_preds[name] += model.predict(X_test) / n_folds

        logger.info(f"    Fold {fold+1}/{n_folds} complete")

    # Create meta-features
    meta_train = pd.DataFrame(oof_preds)
    meta_test = pd.DataFrame(test_preds)

    # Add original features
    meta_train = pd.concat([meta_train.reset_index(drop=True),
                           X_train.reset_index(drop=True)], axis=1)
    meta_test = pd.concat([meta_test.reset_index(drop=True),
                          X_test.reset_index(drop=True)], axis=1)

    # Train meta-model
    split_idx = int(len(meta_train) * 0.8)
    meta_X_train = meta_train.iloc[:split_idx]
    meta_X_val = meta_train.iloc[split_idx:]
    meta_y_train = y_train.iloc[:split_idx]
    meta_y_val = y_train.iloc[split_idx:]

    # Meta weights from last 20% of training weights
    meta_weights = weight_arr[:split_idx] if weight_arr is not None else None

    meta_model = train_catboost(
        meta_X_train, meta_y_train,
        meta_X_val, meta_y_val,
        {'iterations': 500, 'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 5},
        sample_weight=meta_weights
    )

    # Predictions
    val_pred = meta_model.predict(meta_X_val)
    test_pred = meta_model.predict(meta_test)

    val_metrics = evaluate(meta_y_val.values, val_pred)
    test_metrics = evaluate(y_test.values, test_pred)
    gap = test_metrics['smape'] / val_metrics['smape'] if val_metrics['smape'] > 0 else float('inf')

    return {
        'val_smape': val_metrics['smape'],
        'test_smape': test_metrics['smape'],
        'test_mae': test_metrics['mae'],
        'test_bias': test_metrics['bias'],
        'gap': gap,
        'test_pred': test_pred,
    }


# =============================================================================
# STRATEGY A: RECENT HISTORY (HARD CUT)
# =============================================================================

def run_strategy_a_recent_history(base_data: Dict, cutoff_date: str = '2023-01-01') -> Dict:
    """
    Strategy A: Train only on recent data (2023+).

    Hypothesis: Removes all crisis noise, but might overfit due to smaller sample size.
    """
    logger.info("\n" + "="*70)
    logger.info("STRATEGY A: RECENT HISTORY (Hard Cut)")
    logger.info(f"Training cutoff: {cutoff_date}")
    logger.info("="*70)

    X_pretrain = base_data['X_pretrain']
    y_pretrain = base_data['y_pretrain']

    # Ensure timezone-aware comparison
    if X_pretrain.index.tz is not None:
        cutoff_dt = pd.Timestamp(cutoff_date, tz=X_pretrain.index.tz)
    else:
        cutoff_dt = pd.Timestamp(cutoff_date)

    # Filter to recent data only
    recent_mask = X_pretrain.index >= cutoff_dt
    X_train = X_pretrain[recent_mask]
    y_train = y_pretrain[recent_mask]

    logger.info(f"  Filtered training: {len(X_train)} rows (from {X_train.index.min()} to {X_train.index.max()})")
    logger.info(f"  Removed: {len(X_pretrain) - len(X_train)} rows of crisis-era data")

    result = run_stacking_ensemble(
        X_train, y_train,
        base_data['X_test'], base_data['y_test'],
        sample_weight=None,
        strategy_name="Recent History (2023+)"
    )

    logger.info(f"\n  STRATEGY A RESULT:")
    logger.info(f"    Val sMAPE:  {result['val_smape']:.2f}%")
    logger.info(f"    Test sMAPE: {result['test_smape']:.2f}%")
    logger.info(f"    Test Bias:  {result['test_bias']:.2f} TL/MWh")
    logger.info(f"    Gap: {result['gap']:.2f}x")

    return {'strategy': 'A_recent_history', **result}


# =============================================================================
# STRATEGY B: TIME DECAY (SOFT CUT)
# =============================================================================

def run_strategy_b_time_decay(base_data: Dict, power: float = 2.0) -> Dict:
    """
    Strategy B: Use full training set with time-based sample weights.

    Formula: Weight_t = ((t - t_start) / (t_end - t_start))^power

    Effect: Recent data (2025) gets weight 1.0. Old data (2020) gets weight near 0.
    Forces model to learn current price levels while peeking at past seasonal patterns.
    """
    logger.info("\n" + "="*70)
    logger.info("STRATEGY B: TIME DECAY (Soft Cut)")
    logger.info(f"Power: {power}")
    logger.info("="*70)

    X_train = base_data['X_pretrain']
    y_train = base_data['y_pretrain']

    # Compute time-based weights
    timestamps = X_train.index.astype(np.int64).values  # Convert to nanoseconds array
    t_start = timestamps.min()
    t_end = timestamps.max()

    # Normalized time position [0, 1]
    t_normalized = (timestamps - t_start) / (t_end - t_start + 1)

    # Apply power function: recent data gets higher weight
    weights = np.power(t_normalized, power)

    # Ensure minimum weight to prevent zero-weight samples
    weights = np.clip(weights, 0.01, 1.0)

    logger.info(f"  Weight range: {weights.min():.4f} to {weights.max():.4f}")
    logger.info(f"  Mean weight: {weights.mean():.4f}")

    # Show weight distribution by year
    weight_df = pd.DataFrame({'weight': weights}, index=X_train.index)
    yearly_weights = weight_df.groupby(weight_df.index.year)['weight'].mean()
    logger.info(f"  Yearly mean weights:")
    for year, w in yearly_weights.items():
        logger.info(f"    {year}: {w:.3f}")

    result = run_stacking_ensemble(
        X_train, y_train,
        base_data['X_test'], base_data['y_test'],
        sample_weight=weights,
        strategy_name=f"Time Decay (power={power})"
    )

    logger.info(f"\n  STRATEGY B RESULT:")
    logger.info(f"    Val sMAPE:  {result['val_smape']:.2f}%")
    logger.info(f"    Test sMAPE: {result['test_smape']:.2f}%")
    logger.info(f"    Test Bias:  {result['test_bias']:.2f} TL/MWh")
    logger.info(f"    Gap: {result['gap']:.2f}x")

    return {'strategy': 'B_time_decay', 'power': power, **result}


# =============================================================================
# STRATEGY C: TRANSFER LEARNING (FINE-TUNING)
# =============================================================================

def run_strategy_c_transfer_learning(base_data: Dict,
                                     finetune_start: str = '2023-01-01',
                                     lr_decay: float = 0.1) -> Dict:
    """
    Strategy C: Pre-train on full data, fine-tune on recent data.

    1. Train CatBoost on full dataset (2020-2025) - learns physics
    2. Fine-tune on 2023+ with reduced learning rate - adapts to current price levels

    Hypothesis: Best of both worlds - learns physics from 5 years, adapts price levels from 2 years.

    Note: Stacking is complex for transfer learning. We use a simplified approach:
    - Base models: trained with fine-tuning
    - Meta-model: trained on OOF from fine-tuned models
    """
    logger.info("\n" + "="*70)
    logger.info("STRATEGY C: TRANSFER LEARNING (Fine-Tuning)")
    logger.info(f"Fine-tune from: {finetune_start}")
    logger.info(f"LR decay: {lr_decay}")
    logger.info("="*70)

    from sklearn.model_selection import KFold
    from catboost import CatBoostRegressor

    X_pretrain = base_data['X_pretrain']
    y_pretrain = base_data['y_pretrain']
    X_test = base_data['X_test']
    y_test = base_data['y_test']

    # Split pretrain into pre and fine-tune periods
    if X_pretrain.index.tz is not None:
        finetune_dt = pd.Timestamp(finetune_start, tz=X_pretrain.index.tz)
    else:
        finetune_dt = pd.Timestamp(finetune_start)

    pretrain_mask = X_pretrain.index < finetune_dt
    finetune_mask = X_pretrain.index >= finetune_dt

    X_full = X_pretrain
    y_full = y_pretrain
    X_finetune = X_pretrain[finetune_mask]
    y_finetune = y_pretrain[finetune_mask]

    logger.info(f"  Pre-train data: {pretrain_mask.sum()} rows")
    logger.info(f"  Fine-tune data: {finetune_mask.sum()} rows")

    # Base configs
    base_lr = 0.03
    base_config = {
        'iterations': 1000,
        'depth': 8,
        'learning_rate': base_lr,
        'l2_leaf_reg': 3,
    }

    finetune_config = {
        'iterations': 500,  # Fewer iterations for fine-tuning
        'depth': 8,
        'learning_rate': base_lr * lr_decay,  # Reduced learning rate
        'l2_leaf_reg': 3,
    }

    # For transfer learning, we'll use CatBoost (supports init_model)
    # Train base model on FULL data
    n_folds = 5
    kf = KFold(n_splits=n_folds, shuffle=False)

    # Use fine-tune period for OOF predictions (since that's what we care about)
    oof_preds = np.zeros(len(X_finetune))
    test_preds = np.zeros(len(X_test))

    logger.info("  Pre-training and fine-tuning CatBoost models...")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_finetune)):
        # Get fine-tune fold indices
        X_ft_tr = X_finetune.iloc[train_idx]
        X_ft_vl = X_finetune.iloc[val_idx]
        y_ft_tr = y_finetune.iloc[train_idx]
        y_ft_vl = y_finetune.iloc[val_idx]

        # Step 1: Pre-train on FULL data (before fine-tune cutoff + train fold)
        X_pretrain_fold = pd.concat([X_pretrain[pretrain_mask], X_ft_tr])
        y_pretrain_fold = pd.concat([y_pretrain[pretrain_mask], y_ft_tr])

        pretrain_model = CatBoostRegressor(
            loss_function='MAE',
            random_state=42,
            verbose=False,
            early_stopping_rounds=100,
            **base_config
        )
        pretrain_model.fit(
            X_pretrain_fold, y_pretrain_fold,
            eval_set=(X_ft_vl, y_ft_vl),
            verbose=False
        )

        # Step 2: Fine-tune on recent data only
        finetune_model = CatBoostRegressor(
            loss_function='MAE',
            random_state=42,
            verbose=False,
            early_stopping_rounds=50,
            **finetune_config
        )
        finetune_model.fit(
            X_ft_tr, y_ft_tr,
            eval_set=(X_ft_vl, y_ft_vl),
            init_model=pretrain_model,
            verbose=False
        )

        oof_preds[val_idx] = finetune_model.predict(X_ft_vl)
        test_preds += finetune_model.predict(X_test) / n_folds

        logger.info(f"    Fold {fold+1}/{n_folds} complete")

    # Evaluate
    oof_metrics = evaluate(y_finetune.values, oof_preds)
    test_metrics = evaluate(y_test.values, test_preds)
    gap = test_metrics['smape'] / oof_metrics['smape'] if oof_metrics['smape'] > 0 else float('inf')

    logger.info(f"\n  STRATEGY C RESULT:")
    logger.info(f"    OOF sMAPE (fine-tune period): {oof_metrics['smape']:.2f}%")
    logger.info(f"    Test sMAPE: {test_metrics['smape']:.2f}%")
    logger.info(f"    Test Bias:  {test_metrics['bias']:.2f} TL/MWh")
    logger.info(f"    Gap: {gap:.2f}x")

    return {
        'strategy': 'C_transfer_learning',
        'val_smape': oof_metrics['smape'],
        'test_smape': test_metrics['smape'],
        'test_mae': test_metrics['mae'],
        'test_bias': test_metrics['bias'],
        'gap': gap,
        'test_pred': test_preds,
    }


# =============================================================================
# BASELINE: FULL DATA (No drift handling)
# =============================================================================

def run_baseline_full_data(base_data: Dict) -> Dict:
    """
    Baseline: Train on full data without any drift handling.
    This reproduces the 15.96% baseline for comparison.
    """
    logger.info("\n" + "="*70)
    logger.info("BASELINE: FULL DATA (No drift handling)")
    logger.info("="*70)

    result = run_stacking_ensemble(
        base_data['X_pretrain'], base_data['y_pretrain'],
        base_data['X_test'], base_data['y_test'],
        sample_weight=None,
        strategy_name="Full Data Baseline"
    )

    logger.info(f"\n  BASELINE RESULT:")
    logger.info(f"    Val sMAPE:  {result['val_smape']:.2f}%")
    logger.info(f"    Test sMAPE: {result['test_smape']:.2f}%")
    logger.info(f"    Test Bias:  {result['test_bias']:.2f} TL/MWh")
    logger.info(f"    Gap: {result['gap']:.2f}x")

    return {'strategy': 'baseline_full_data', **result}


# =============================================================================
# MAIN: DATA STRATEGY COMPARISON
# =============================================================================

def run_data_strategy_comparison():
    """Run all training window strategies and compare results."""
    output_dir = PROJECT_ROOT / 'reports' / 'optimized_search_v4'
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*70)
    logger.info("OPTIMIZED SEARCH V4 - Training Window Strategy for Concept Drift")
    logger.info(f"Baseline to beat: {BASELINE_SMAPE}% sMAPE")
    logger.info("="*70)

    # Load and prepare data
    df = load_data()
    logger.info(f"Loaded data: {df.shape}")
    logger.info(f"Date range: {df.index.min()} to {df.index.max()}")

    base_data = prepare_base_data(df, BEST_FEATURES, test_start='2024-06-01')
    logger.info(f"Features: {len(base_data['features'])}")

    results = []

    # Run baseline first
    try:
        results.append(run_baseline_full_data(base_data))
    except Exception as e:
        logger.error(f"Baseline failed: {e}")
        import traceback
        traceback.print_exc()

    # Strategy A: Recent History
    try:
        results.append(run_strategy_a_recent_history(base_data, cutoff_date='2023-01-01'))
    except Exception as e:
        logger.error(f"Strategy A failed: {e}")
        import traceback
        traceback.print_exc()

    # Strategy B: Time Decay (try multiple power values)
    for power in [2.0, 3.0]:
        try:
            results.append(run_strategy_b_time_decay(base_data, power=power))
        except Exception as e:
            logger.error(f"Strategy B (power={power}) failed: {e}")
            import traceback
            traceback.print_exc()

    # Strategy C: Transfer Learning
    try:
        results.append(run_strategy_c_transfer_learning(base_data, finetune_start='2023-01-01'))
    except Exception as e:
        logger.error(f"Strategy C failed: {e}")
        import traceback
        traceback.print_exc()

    # Summary table
    logger.info("\n" + "="*70)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("="*70)

    logger.info(f"\n{'Strategy':<30} {'Test sMAPE':>12} {'Bias (TL)':>12} {'Gap':>8} {'Beat?':>8}")
    logger.info("-"*75)

    for r in sorted(results, key=lambda x: x['test_smape']):
        beat = "YES" if r['test_smape'] < BASELINE_SMAPE else "no"
        strategy_name = r['strategy']
        if 'power' in r:
            strategy_name += f" (p={r['power']})"
        logger.info(f"{strategy_name:<30} {r['test_smape']:>12.2f}% {r['test_bias']:>12.2f} {r['gap']:>7.2f}x {beat:>8}")

    # Best result
    best = min(results, key=lambda x: x['test_smape'])

    logger.info(f"\n{'='*70}")
    if best['test_smape'] < BASELINE_SMAPE:
        improvement = BASELINE_SMAPE - best['test_smape']
        logger.info(f"SUCCESS! Beat {BASELINE_SMAPE}% by {improvement:.2f}%!")
    else:
        logger.info(f"Best: {best['test_smape']:.2f}% (baseline: {BASELINE_SMAPE}%)")

    logger.info(f"  Best Strategy: {best['strategy']}")
    logger.info(f"  Test sMAPE: {best['test_smape']:.2f}%")
    logger.info(f"  Test Bias: {best['test_bias']:.2f} TL/MWh")
    logger.info(f"  Gap: {best['gap']:.2f}x")

    # Bias analysis
    logger.info(f"\n{'='*70}")
    logger.info("BIAS ANALYSIS")
    logger.info("="*70)
    logger.info("  Positive bias = over-predicting (expecting crisis prices in stable market)")
    logger.info("  Negative bias = under-predicting")

    baseline_result = next((r for r in results if r['strategy'] == 'baseline_full_data'), None)
    if baseline_result:
        baseline_bias = baseline_result['test_bias']
        logger.info(f"\n  Baseline bias: {baseline_bias:.2f} TL/MWh")
        for r in results:
            if r['strategy'] != 'baseline_full_data':
                bias_change = r['test_bias'] - baseline_bias
                logger.info(f"  {r['strategy']}: {r['test_bias']:.2f} TL/MWh (change: {bias_change:+.2f})")

    # Save results
    summary = {
        'timestamp': datetime.now().isoformat(),
        'baseline_smape': BASELINE_SMAPE,
        'test_start': '2024-06-01',
        'features_used': base_data['features'],
        'n_features': len(base_data['features']),
        'best_strategy': best['strategy'],
        'best_test_smape': float(best['test_smape']),
        'best_bias': float(best['test_bias']),
        'all_results': [{k: v for k, v in r.items() if k != 'test_pred'} for r in results],
    }

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Save best predictions
    if 'test_pred' in best:
        pred_df = pd.DataFrame({
            'datetime': base_data['X_test'].index,
            'y_true': base_data['y_test'].values,
            'y_pred': best['test_pred'],
        })
        pred_df.to_parquet(output_dir / 'best_predictions.parquet', index=False)

    logger.info(f"\nResults saved to: {output_dir}")

    return results


if __name__ == "__main__":
    run_data_strategy_comparison()
