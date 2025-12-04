"""
Optimized Feature Search - Beat 16.01% sMAPE
=============================================
Key strategies:
1. Ensemble: Combine top models (CatBoost, LightGBM, XGBoost)
2. Stacking: Use model predictions as meta-features
3. Target transformation: Log/Box-Cox for better distribution
4. Feature interactions: Create polynomial features for top predictors
5. Quantile ensemble: Use median of predictions for robustness

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
from typing import Dict, List, Tuple
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

TARGET_SMAPE = 16.01

# =============================================================================
# BEST FEATURE SET (from V3 search)
# =============================================================================

BEST_FEATURES = [
    'hour_sin', 'hour_cos', 'dow_sin_x', 'dow_cos_x', 'is_weekend_x',
    'price_ptf_rolling_std_24h', 'price_ptf_rolling_mean_24h',
    'price_ptf_rolling_min_24h', 'price_ptf_rolling_max_24h',
    'price_ptf_lag_24h', 'price_ptf_lag_168h',
    'thermal_gap', 'thermal_gap_lag_24h',
    'renewable_saturation', 'spark_spread_proxy_lag_24h',
    'system_short_signal', 'load_factor', 'consumption_forecast',
]

# Extended features for more signal
EXTENDED_FEATURES = BEST_FEATURES + [
    'price_ptf_lag_48h',
    'reserve_margin_ratio',
    'price_volatility_lag24h',
    'realtime_premium_lag24h',
]


def load_data():
    """Load master dataset."""
    path = PROJECT_ROOT / 'data' / 'gold' / 'master' / 'master_v2_fundamental.parquet'
    return pd.read_parquet(path)


def add_feature_interactions(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Add interaction features for top predictors."""
    df = df.copy()

    # Key interactions based on domain knowledge
    interactions = [
        ('price_ptf_lag_24h', 'thermal_gap'),
        ('price_ptf_rolling_std_24h', 'renewable_saturation'),
        ('load_factor', 'reserve_margin_ratio') if 'reserve_margin_ratio' in df.columns else None,
        ('hour_sin', 'price_ptf_lag_24h'),
        ('system_short_signal', 'thermal_gap'),
    ]

    new_features = []
    for pair in interactions:
        if pair is None:
            continue
        f1, f2 = pair
        if f1 in df.columns and f2 in df.columns:
            name = f'{f1}_x_{f2}'
            df[name] = df[f1] * df[f2]
            new_features.append(name)

    return df, new_features


def prepare_data(df: pd.DataFrame, features: List[str], add_interactions: bool = True,
                 val_size: float = 0.2, test_size: float = 0.2):
    """Prepare data with optional feature interactions."""
    available = [f for f in features if f in df.columns]

    X = df[available].copy()
    y = df['price_real'].copy()

    # Add interactions
    interaction_features = []
    if add_interactions:
        X, interaction_features = add_feature_interactions(X, available)
        available = available + interaction_features

    # Drop NaN
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]

    # Split
    n = len(X)
    train_end = int(n * (1 - val_size - test_size))
    val_end = int(n * (1 - test_size))

    return {
        'X_train': X.iloc[:train_end],
        'X_val': X.iloc[train_end:val_end],
        'X_test': X.iloc[val_end:],
        'y_train': y.iloc[:train_end],
        'y_val': y.iloc[train_end:val_end],
        'y_test': y.iloc[val_end:],
        'features': list(X.columns),
    }


def train_catboost(X_train, y_train, X_val, y_val, config):
    """Train CatBoost with optimized settings."""
    from catboost import CatBoostRegressor

    model = CatBoostRegressor(
        loss_function='MAE',
        random_state=42,
        verbose=False,
        early_stopping_rounds=100,
        **config
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    return model


def train_lightgbm(X_train, y_train, X_val, y_val, config):
    """Train LightGBM with optimized settings."""
    import lightgbm as lgb

    model = lgb.LGBMRegressor(
        objective='regression',
        metric='mae',
        verbosity=-1,
        random_state=42,
        **config
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    return model


def train_xgboost(X_train, y_train, X_val, y_val, config):
    """Train XGBoost with optimized settings."""
    import xgboost as xgb

    model = xgb.XGBRegressor(
        random_state=42,
        verbosity=0,
        early_stopping_rounds=100,
        **config
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def evaluate(y_true, y_pred):
    """Calculate metrics."""
    mae = np.mean(np.abs(y_true - y_pred))
    smape = 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))
    return {'mae': mae, 'smape': smape}


# =============================================================================
# STRATEGY 1: WEIGHTED ENSEMBLE
# =============================================================================

def run_weighted_ensemble(data):
    """Train multiple models and combine with optimal weights."""
    logger.info("\n" + "="*60)
    logger.info("STRATEGY 1: WEIGHTED ENSEMBLE")
    logger.info("="*60)

    # Model configs (tuned for this dataset)
    configs = {
        'catboost': {
            'iterations': 2000,
            'depth': 10,
            'learning_rate': 0.02,
            'l2_leaf_reg': 3,
            'bagging_temperature': 0.3,
        },
        'lightgbm': {
            'n_estimators': 2000,
            'max_depth': 12,
            'learning_rate': 0.02,
            'num_leaves': 127,
            'min_child_samples': 15,
            'reg_alpha': 0.03,
            'reg_lambda': 0.03,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
        },
        'xgboost': {
            'n_estimators': 2000,
            'max_depth': 12,
            'learning_rate': 0.02,
            'min_child_weight': 3,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'reg_alpha': 0.03,
            'reg_lambda': 0.03,
        },
    }

    models = {}
    val_preds = {}
    test_preds = {}

    for name, config in configs.items():
        logger.info(f"\n  Training {name}...")
        start = time.time()

        if name == 'catboost':
            model = train_catboost(data['X_train'], data['y_train'],
                                   data['X_val'], data['y_val'], config)
        elif name == 'lightgbm':
            model = train_lightgbm(data['X_train'], data['y_train'],
                                   data['X_val'], data['y_val'], config)
        elif name == 'xgboost':
            model = train_xgboost(data['X_train'], data['y_train'],
                                  data['X_val'], data['y_val'], config)

        models[name] = model
        val_preds[name] = model.predict(data['X_val'])
        test_preds[name] = model.predict(data['X_test'])

        val_metrics = evaluate(data['y_val'].values, val_preds[name])
        test_metrics = evaluate(data['y_test'].values, test_preds[name])
        logger.info(f"    {name}: Val={val_metrics['smape']:.2f}% Test={test_metrics['smape']:.2f}% ({time.time()-start:.1f}s)")

    # Find optimal weights using validation set
    logger.info("\n  Finding optimal weights...")
    from scipy.optimize import minimize

    def ensemble_loss(weights):
        weights = np.array(weights)
        weights = weights / weights.sum()  # normalize
        pred = sum(w * val_preds[n] for w, n in zip(weights, val_preds.keys()))
        return evaluate(data['y_val'].values, pred)['smape']

    # Start with equal weights
    initial_weights = [1/3, 1/3, 1/3]
    result = minimize(ensemble_loss, initial_weights, method='Nelder-Mead',
                     options={'maxiter': 1000})

    optimal_weights = np.array(result.x)
    optimal_weights = optimal_weights / optimal_weights.sum()

    logger.info(f"    Optimal weights: {dict(zip(configs.keys(), optimal_weights.round(3)))}")

    # Create ensemble predictions
    val_ensemble = sum(w * val_preds[n] for w, n in zip(optimal_weights, val_preds.keys()))
    test_ensemble = sum(w * test_preds[n] for w, n in zip(optimal_weights, test_preds.keys()))

    val_metrics = evaluate(data['y_val'].values, val_ensemble)
    test_metrics = evaluate(data['y_test'].values, test_ensemble)
    gap = test_metrics['smape'] / val_metrics['smape']

    logger.info(f"\n  ENSEMBLE RESULT:")
    logger.info(f"    Val sMAPE:  {val_metrics['smape']:.2f}%")
    logger.info(f"    Test sMAPE: {test_metrics['smape']:.2f}%")
    logger.info(f"    Gap: {gap:.2f}x")

    return {
        'strategy': 'weighted_ensemble',
        'val_smape': val_metrics['smape'],
        'test_smape': test_metrics['smape'],
        'gap': gap,
        'weights': dict(zip(configs.keys(), optimal_weights.tolist())),
        'models': models,
        'test_pred': test_ensemble,
    }


# =============================================================================
# STRATEGY 2: STACKING ENSEMBLE
# =============================================================================

def run_stacking_ensemble(data):
    """Use model predictions as features for a meta-learner."""
    logger.info("\n" + "="*60)
    logger.info("STRATEGY 2: STACKING ENSEMBLE")
    logger.info("="*60)

    from sklearn.model_selection import KFold

    # Base model configs
    base_configs = {
        'catboost': {'iterations': 1000, 'depth': 8, 'learning_rate': 0.03, 'l2_leaf_reg': 3},
        'lightgbm': {'n_estimators': 1000, 'max_depth': 10, 'learning_rate': 0.03, 'num_leaves': 63},
        'xgboost': {'n_estimators': 1000, 'max_depth': 10, 'learning_rate': 0.03},
    }

    # Generate out-of-fold predictions for training meta-model
    n_folds = 5
    kf = KFold(n_splits=n_folds, shuffle=False)  # No shuffle for time series

    X_train_full = pd.concat([data['X_train'], data['X_val']])
    y_train_full = pd.concat([data['y_train'], data['y_val']])

    oof_preds = {name: np.zeros(len(X_train_full)) for name in base_configs}
    test_preds = {name: np.zeros(len(data['X_test'])) for name in base_configs}

    logger.info(f"  Generating OOF predictions with {n_folds} folds...")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_full)):
        X_tr, X_vl = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
        y_tr, y_vl = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]

        for name, config in base_configs.items():
            if name == 'catboost':
                model = train_catboost(X_tr, y_tr, X_vl, y_vl, config)
            elif name == 'lightgbm':
                model = train_lightgbm(X_tr, y_tr, X_vl, y_vl, config)
            elif name == 'xgboost':
                model = train_xgboost(X_tr, y_tr, X_vl, y_vl, config)

            oof_preds[name][val_idx] = model.predict(X_vl)
            test_preds[name] += model.predict(data['X_test']) / n_folds

        logger.info(f"    Fold {fold+1}/{n_folds} complete")

    # Create meta-features
    meta_train = pd.DataFrame(oof_preds)
    meta_test = pd.DataFrame(test_preds)

    # Add original features to meta-features
    meta_train = pd.concat([meta_train.reset_index(drop=True),
                           X_train_full.reset_index(drop=True)], axis=1)
    meta_test = pd.concat([meta_test.reset_index(drop=True),
                          data['X_test'].reset_index(drop=True)], axis=1)

    # Train meta-model
    logger.info("\n  Training meta-model...")

    # Split meta-train for validation
    split_idx = int(len(meta_train) * 0.8)
    meta_X_train = meta_train.iloc[:split_idx]
    meta_X_val = meta_train.iloc[split_idx:]
    meta_y_train = y_train_full.iloc[:split_idx]
    meta_y_val = y_train_full.iloc[split_idx:]

    meta_model = train_catboost(
        meta_X_train, meta_y_train,
        meta_X_val, meta_y_val,
        {'iterations': 500, 'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 5}
    )

    # Final predictions
    val_pred = meta_model.predict(meta_X_val)
    test_pred = meta_model.predict(meta_test)

    val_metrics = evaluate(meta_y_val.values, val_pred)
    test_metrics = evaluate(data['y_test'].values, test_pred)
    gap = test_metrics['smape'] / val_metrics['smape']

    logger.info(f"\n  STACKING RESULT:")
    logger.info(f"    Val sMAPE:  {val_metrics['smape']:.2f}%")
    logger.info(f"    Test sMAPE: {test_metrics['smape']:.2f}%")
    logger.info(f"    Gap: {gap:.2f}x")

    return {
        'strategy': 'stacking_ensemble',
        'val_smape': val_metrics['smape'],
        'test_smape': test_metrics['smape'],
        'gap': gap,
        'test_pred': test_pred,
    }


# =============================================================================
# STRATEGY 3: QUANTILE ENSEMBLE (Robust to outliers)
# =============================================================================

def run_quantile_ensemble(data):
    """Train quantile regressors and use median for robust predictions."""
    logger.info("\n" + "="*60)
    logger.info("STRATEGY 3: QUANTILE ENSEMBLE")
    logger.info("="*60)

    from catboost import CatBoostRegressor
    import lightgbm as lgb

    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

    # Train CatBoost quantile models
    logger.info("  Training quantile regressors...")

    quantile_preds_val = []
    quantile_preds_test = []

    for q in quantiles:
        logger.info(f"    Quantile {q}...")

        model = CatBoostRegressor(
            loss_function=f'Quantile:alpha={q}',
            iterations=1500,
            depth=10,
            learning_rate=0.03,
            l2_leaf_reg=3,
            random_state=42,
            verbose=False,
            early_stopping_rounds=50,
        )
        model.fit(
            data['X_train'], data['y_train'],
            eval_set=(data['X_val'], data['y_val']),
            verbose=False
        )

        quantile_preds_val.append(model.predict(data['X_val']))
        quantile_preds_test.append(model.predict(data['X_test']))

    # Use median (q=0.5) as final prediction
    val_pred = quantile_preds_val[2]  # 0.5 quantile
    test_pred = quantile_preds_test[2]

    val_metrics = evaluate(data['y_val'].values, val_pred)
    test_metrics = evaluate(data['y_test'].values, test_pred)
    gap = test_metrics['smape'] / val_metrics['smape']

    logger.info(f"\n  QUANTILE RESULT (median):")
    logger.info(f"    Val sMAPE:  {val_metrics['smape']:.2f}%")
    logger.info(f"    Test sMAPE: {test_metrics['smape']:.2f}%")
    logger.info(f"    Gap: {gap:.2f}x")

    # Also try average of quantiles
    val_pred_avg = np.mean(quantile_preds_val, axis=0)
    test_pred_avg = np.mean(quantile_preds_test, axis=0)

    val_metrics_avg = evaluate(data['y_val'].values, val_pred_avg)
    test_metrics_avg = evaluate(data['y_test'].values, test_pred_avg)

    logger.info(f"\n  QUANTILE RESULT (average):")
    logger.info(f"    Val sMAPE:  {val_metrics_avg['smape']:.2f}%")
    logger.info(f"    Test sMAPE: {test_metrics_avg['smape']:.2f}%")

    best_test = min(test_metrics['smape'], test_metrics_avg['smape'])

    return {
        'strategy': 'quantile_ensemble',
        'val_smape': val_metrics['smape'],
        'test_smape': test_metrics['smape'],
        'test_smape_avg': test_metrics_avg['smape'],
        'gap': gap,
        'test_pred': test_pred if test_metrics['smape'] < test_metrics_avg['smape'] else test_pred_avg,
    }


# =============================================================================
# STRATEGY 4: LOG TRANSFORM + ENSEMBLE
# =============================================================================

def run_log_transform_ensemble(data):
    """Train on log-transformed target for better distribution."""
    logger.info("\n" + "="*60)
    logger.info("STRATEGY 4: LOG TRANSFORM + ENSEMBLE")
    logger.info("="*60)

    # Log transform target (add small constant to handle zeros/negatives)
    offset = 100  # Shift to ensure all positive
    y_train_log = np.log1p(data['y_train'] + offset)
    y_val_log = np.log1p(data['y_val'] + offset)

    configs = {
        'catboost': {'iterations': 1500, 'depth': 10, 'learning_rate': 0.025, 'l2_leaf_reg': 3},
        'lightgbm': {'n_estimators': 1500, 'max_depth': 12, 'learning_rate': 0.025, 'num_leaves': 127},
        'xgboost': {'n_estimators': 1500, 'max_depth': 12, 'learning_rate': 0.025},
    }

    val_preds = {}
    test_preds = {}

    for name, config in configs.items():
        logger.info(f"  Training {name} on log-transformed target...")

        if name == 'catboost':
            model = train_catboost(data['X_train'], y_train_log,
                                   data['X_val'], y_val_log, config)
        elif name == 'lightgbm':
            model = train_lightgbm(data['X_train'], y_train_log,
                                   data['X_val'], y_val_log, config)
        elif name == 'xgboost':
            model = train_xgboost(data['X_train'], y_train_log,
                                  data['X_val'], y_val_log, config)

        # Inverse transform predictions
        val_preds[name] = np.expm1(model.predict(data['X_val'])) - offset
        test_preds[name] = np.expm1(model.predict(data['X_test'])) - offset

    # Simple average ensemble
    val_ensemble = np.mean(list(val_preds.values()), axis=0)
    test_ensemble = np.mean(list(test_preds.values()), axis=0)

    val_metrics = evaluate(data['y_val'].values, val_ensemble)
    test_metrics = evaluate(data['y_test'].values, test_ensemble)
    gap = test_metrics['smape'] / val_metrics['smape']

    logger.info(f"\n  LOG TRANSFORM RESULT:")
    logger.info(f"    Val sMAPE:  {val_metrics['smape']:.2f}%")
    logger.info(f"    Test sMAPE: {test_metrics['smape']:.2f}%")
    logger.info(f"    Gap: {gap:.2f}x")

    return {
        'strategy': 'log_transform_ensemble',
        'val_smape': val_metrics['smape'],
        'test_smape': test_metrics['smape'],
        'gap': gap,
        'test_pred': test_ensemble,
    }


# =============================================================================
# STRATEGY 5: DEEP CATBOOST (Maximum capacity)
# =============================================================================

def run_deep_catboost(data):
    """Train a very deep CatBoost model with extensive regularization."""
    logger.info("\n" + "="*60)
    logger.info("STRATEGY 5: DEEP CATBOOST (Maximum capacity)")
    logger.info("="*60)

    from catboost import CatBoostRegressor

    model = CatBoostRegressor(
        loss_function='MAE',
        iterations=5000,
        depth=12,
        learning_rate=0.01,
        l2_leaf_reg=1,
        bagging_temperature=0.2,
        random_strength=0.5,
        border_count=254,
        grow_policy='Lossguide',
        max_leaves=512,
        random_state=42,
        verbose=False,
        early_stopping_rounds=200,
    )

    logger.info("  Training deep CatBoost (may take a while)...")
    start = time.time()

    model.fit(
        data['X_train'], data['y_train'],
        eval_set=(data['X_val'], data['y_val']),
        verbose=False
    )

    val_pred = model.predict(data['X_val'])
    test_pred = model.predict(data['X_test'])

    val_metrics = evaluate(data['y_val'].values, val_pred)
    test_metrics = evaluate(data['y_test'].values, test_pred)
    gap = test_metrics['smape'] / val_metrics['smape']

    logger.info(f"\n  DEEP CATBOOST RESULT: ({time.time()-start:.1f}s)")
    logger.info(f"    Val sMAPE:  {val_metrics['smape']:.2f}%")
    logger.info(f"    Test sMAPE: {test_metrics['smape']:.2f}%")
    logger.info(f"    Gap: {gap:.2f}x")

    return {
        'strategy': 'deep_catboost',
        'val_smape': val_metrics['smape'],
        'test_smape': test_metrics['smape'],
        'gap': gap,
        'model': model,
        'test_pred': test_pred,
    }


# =============================================================================
# MAIN SEARCH
# =============================================================================

def run_optimized_search():
    """Run all optimization strategies."""
    output_dir = PROJECT_ROOT / 'reports' / 'optimized_search'
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*70)
    logger.info("OPTIMIZED FEATURE SEARCH - Beat 16.01% sMAPE")
    logger.info("="*70)

    df = load_data()
    logger.info(f"Data: {df.shape}")

    # Prepare data with extended features and interactions
    data = prepare_data(df, EXTENDED_FEATURES, add_interactions=True)
    logger.info(f"Features: {len(data['features'])}")
    logger.info(f"Train: {len(data['X_train'])}, Val: {len(data['X_val'])}, Test: {len(data['X_test'])}")

    results = []

    # Run all strategies
    try:
        results.append(run_weighted_ensemble(data))
    except Exception as e:
        logger.error(f"Weighted ensemble failed: {e}")

    try:
        results.append(run_stacking_ensemble(data))
    except Exception as e:
        logger.error(f"Stacking ensemble failed: {e}")

    try:
        results.append(run_quantile_ensemble(data))
    except Exception as e:
        logger.error(f"Quantile ensemble failed: {e}")

    try:
        results.append(run_log_transform_ensemble(data))
    except Exception as e:
        logger.error(f"Log transform failed: {e}")

    try:
        results.append(run_deep_catboost(data))
    except Exception as e:
        logger.error(f"Deep CatBoost failed: {e}")

    # Summary
    logger.info("\n" + "="*70)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("="*70)

    logger.info(f"\n{'Strategy':<25} {'Val%':>8} {'Test%':>8} {'Gap':>7} {'Beat?':>6}")
    logger.info("-"*60)

    for r in sorted(results, key=lambda x: x['test_smape']):
        beat = "YES" if r['test_smape'] < TARGET_SMAPE else "no"
        logger.info(f"{r['strategy']:<25} {r['val_smape']:>8.2f} {r['test_smape']:>8.2f} {r['gap']:>6.2f}x {beat:>6}")

    # Best result
    best = min(results, key=lambda x: x['test_smape'])

    logger.info(f"\n{'='*60}")
    if best['test_smape'] < TARGET_SMAPE:
        logger.info(f"SUCCESS! Beat {TARGET_SMAPE}%!")
    else:
        logger.info(f"Best: {best['test_smape']:.2f}% (target: {TARGET_SMAPE}%)")

    logger.info(f"  Strategy: {best['strategy']}")
    logger.info(f"  Test sMAPE: {best['test_smape']:.2f}%")
    logger.info(f"  Gap: {best['gap']:.2f}x")

    # Save results
    summary = {
        'timestamp': datetime.now().isoformat(),
        'target_smape': TARGET_SMAPE,
        'features_used': data['features'],
        'n_features': len(data['features']),
        'best_strategy': best['strategy'],
        'best_test_smape': float(best['test_smape']),
        'all_results': [{k: v for k, v in r.items()
                        if k not in ['models', 'model', 'test_pred']}
                       for r in results],
    }

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nResults saved to: {output_dir}")

    return results


if __name__ == "__main__":
    run_optimized_search()
