"""
Normalized Target Grid Search
==============================
Comprehensive grid search with normalized target and lag-focused features.

Tests:
- 3 model types (LightGBM, XGBoost, CatBoost)
- 2 model sizes (small, medium)
- 10 feature sets (max 20 features each)
- Normalized vs Absolute target

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
from typing import Dict, List
import logging
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================

MODEL_CONFIGS = {
    # LightGBM
    'lgb_small': {
        'type': 'lightgbm',
        'n_estimators': 300,
        'max_depth': 5,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_child_samples': 30,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
    },
    'lgb_medium': {
        'type': 'lightgbm',
        'n_estimators': 500,
        'max_depth': 7,
        'learning_rate': 0.03,
        'num_leaves': 63,
        'min_child_samples': 20,
        'reg_alpha': 0.05,
        'reg_lambda': 0.05,
    },

    # XGBoost
    'xgb_small': {
        'type': 'xgboost',
        'n_estimators': 300,
        'max_depth': 5,
        'learning_rate': 0.05,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
    },
    'xgb_medium': {
        'type': 'xgboost',
        'n_estimators': 500,
        'max_depth': 7,
        'learning_rate': 0.03,
        'min_child_weight': 3,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
    },

    # CatBoost
    'cat_small': {
        'type': 'catboost',
        'iterations': 300,
        'depth': 5,
        'learning_rate': 0.05,
        'l2_leaf_reg': 3,
    },
    'cat_medium': {
        'type': 'catboost',
        'iterations': 500,
        'depth': 7,
        'learning_rate': 0.03,
        'l2_leaf_reg': 1,
    },
}

# =============================================================================
# FEATURE SETS (max 20 each, lag-focused)
# =============================================================================

FEATURE_SETS = {
    # === LAG-FOCUSED (best generalization) ===

    'lag_core': [
        'hour_sin', 'hour_cos', 'dow_sin_x', 'dow_cos_x', 'is_weekend_x',
        'price_ptf_lag_24h', 'price_ptf_lag_168h',
        'reserve_margin_ratio_lag_24h', 'renewable_saturation_lag_24h',
        'load_factor_lag_24h', 'thermal_gap_lag_24h',
        'spark_spread_proxy_lag_24h',
    ],

    'lag_extended': [
        'hour_sin', 'hour_cos', 'dow_sin_x', 'dow_cos_x', 'is_weekend_x',
        'price_ptf_lag_24h', 'price_ptf_lag_168h',
        'reserve_margin_ratio_lag_24h', 'reserve_margin_ratio_lag_168h',
        'renewable_saturation_lag_24h', 'renewable_saturation_lag_168h',
        'load_factor_lag_24h', 'thermal_gap_lag_24h',
        'spark_spread_proxy_lag_24h', 'spark_spread_proxy_lag_168h',
        'consumption_lag_24h', 'temp_lag_24h',
    ],

    'lag_full': [
        'hour_sin', 'hour_cos', 'dow_sin_x', 'dow_cos_x', 'is_weekend_x',
        'price_ptf_lag_24h', 'price_ptf_lag_168h',
        'consumption_lag_24h', 'consumption_lag_168h',
        'temp_lag_24h', 'temp_lag_168h',
        'reserve_margin_ratio_lag_24h', 'renewable_saturation_lag_24h',
        'load_factor_lag_24h', 'thermal_gap_lag_24h',
        'spark_spread_proxy_lag_24h', 'spark_spread_proxy_lag_168h',
        'reserve_margin_ratio', 'consumption_forecast',
    ],

    # === RATIO-FOCUSED (good accuracy) ===

    'ratios_core': [
        'hour_sin', 'hour_cos', 'dow_sin_x', 'dow_cos_x', 'is_weekend_x',
        'reserve_margin_ratio', 'renewable_saturation', 'load_factor',
        'spark_spread_proxy', 'system_short_signal',
        'price_ptf_lag_24h', 'price_ptf_lag_168h',
    ],

    'ratios_extended': [
        'hour_sin', 'hour_cos', 'dow_sin_x', 'dow_cos_x', 'is_weekend_x',
        'reserve_margin_ratio', 'renewable_saturation', 'load_factor',
        'spark_spread_proxy', 'realtime_premium_lag24h',
        'system_short_signal', 'price_volatility_lag24h',
        'price_ptf_lag_24h', 'price_ptf_lag_168h',
        'thermal_gap', 'consumption_forecast',
    ],

    # === HYBRID (lag + ratio) ===

    'hybrid_lag_ratio': [
        'hour_sin', 'hour_cos', 'dow_sin_x', 'dow_cos_x', 'is_weekend_x',
        'price_ptf_lag_24h', 'price_ptf_lag_168h',
        'reserve_margin_ratio', 'load_factor',
        'reserve_margin_ratio_lag_24h', 'load_factor_lag_24h',
        'spark_spread_proxy_lag_24h', 'renewable_saturation',
        'consumption_forecast',
    ],

    'hybrid_balanced': [
        'hour_sin', 'hour_cos', 'dow_sin_x', 'dow_cos_x', 'is_weekend_x',
        'price_ptf_lag_24h', 'price_ptf_lag_168h',
        'price_ptf_rolling_mean_24h', 'price_ptf_rolling_std_24h',
        'reserve_margin_ratio', 'renewable_saturation',
        'spark_spread_proxy_lag_24h', 'load_factor_lag_24h',
        'consumption_forecast', 'thermal_gap',
    ],

    # === IMPORTANCE-BASED ===

    'importance_top12': [
        'price_ptf_rolling_std_24h', 'thermal_gap', 'price_ptf_lag_168h',
        'hour_cos', 'price_ptf_rolling_mean_24h', 'renewable_saturation',
        'price_ptf_rolling_min_24h', 'price_ptf_lag_24h',
        'spark_spread_proxy_lag_24h', 'spark_spread_proxy_lag_48h',
        'price_ptf_rolling_max_24h', 'hour_sin',
    ],

    'importance_mixed': [
        'price_ptf_rolling_std_24h', 'thermal_gap', 'price_ptf_lag_168h',
        'hour_cos', 'hour_sin', 'dow_sin_x', 'dow_cos_x',
        'renewable_saturation', 'price_ptf_lag_24h',
        'spark_spread_proxy_lag_24h', 'system_short_signal',
        'reserve_margin_ratio', 'load_factor',
    ],

    # === MINIMAL ===

    'minimal_10': [
        'hour_sin', 'hour_cos', 'dow_sin_x', 'dow_cos_x',
        'price_ptf_lag_24h', 'price_ptf_lag_168h',
        'reserve_margin_ratio', 'load_factor',
        'spark_spread_proxy_lag_24h', 'is_weekend_x',
    ],
}


def load_data():
    """Load dataset."""
    path = PROJECT_ROOT / 'data' / 'gold' / 'master' / 'master_v2_fundamental.parquet'
    return pd.read_parquet(path)


def prepare_data(df, features, use_normalized=True):
    """Prepare train/val/test data."""
    available = [f for f in features if f in df.columns]

    X = df[available].copy()
    y_abs = df['price_real'].copy()

    if use_normalized:
        rolling = df['price_ptf_rolling_mean_168h']
        y = y_abs / (rolling + 1e-8)
    else:
        y = y_abs
        rolling = None

    # Drop NaN
    if use_normalized:
        mask = ~(X.isna().any(axis=1) | y.isna() | rolling.isna())
        rolling = rolling[mask]
    else:
        mask = ~(X.isna().any(axis=1) | y.isna())

    X, y, y_abs = X[mask], y[mask], y_abs[mask]

    # Split 60/20/20
    n = len(X)
    t1, t2 = int(n * 0.6), int(n * 0.8)

    return {
        'X_train': X.iloc[:t1], 'X_val': X.iloc[t1:t2], 'X_test': X.iloc[t2:],
        'y_train': y.iloc[:t1], 'y_val': y.iloc[t1:t2], 'y_test': y.iloc[t2:],
        'y_abs_train': y_abs.iloc[:t1], 'y_abs_val': y_abs.iloc[t1:t2], 'y_abs_test': y_abs.iloc[t2:],
        'rolling_val': rolling.iloc[t1:t2] if rolling is not None else None,
        'rolling_test': rolling.iloc[t2:] if rolling is not None else None,
        'features': available,
        'use_normalized': use_normalized,
    }


def train_model(data, config):
    """Train model based on type."""
    model_type = config.pop('type')

    if model_type == 'lightgbm':
        import lightgbm as lgb
        model = lgb.LGBMRegressor(objective='regression', verbosity=-1, random_state=42, **config)
        model.fit(data['X_train'], data['y_train'],
                  eval_set=[(data['X_val'], data['y_val'])],
                  callbacks=[lgb.early_stopping(30, verbose=False)])

    elif model_type == 'xgboost':
        import xgboost as xgb
        model = xgb.XGBRegressor(verbosity=0, random_state=42, early_stopping_rounds=30, **config)
        model.fit(data['X_train'], data['y_train'],
                  eval_set=[(data['X_val'], data['y_val'])], verbose=False)

    elif model_type == 'catboost':
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(verbose=False, random_seed=42, early_stopping_rounds=30, **config)
        model.fit(data['X_train'], data['y_train'],
                  eval_set=(data['X_val'], data['y_val']), verbose=False)

    config['type'] = model_type  # restore
    return model


def evaluate(y_true, y_pred):
    """Calculate metrics."""
    mae = np.mean(np.abs(y_true - y_pred))
    smape = 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))
    return mae, smape


def run_grid_search():
    """Run full grid search."""
    output_dir = PROJECT_ROOT / 'reports' / 'normalized_grid_search'
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*70)
    logger.info("NORMALIZED TARGET GRID SEARCH")
    logger.info("="*70)

    df = load_data()
    logger.info(f"Data: {df.shape}")

    results = []
    total = len(MODEL_CONFIGS) * len(FEATURE_SETS) * 2  # x2 for normalized/absolute
    count = 0

    for feat_name, features in FEATURE_SETS.items():
        for model_name, config in MODEL_CONFIGS.items():
            for use_norm in [True, False]:
                count += 1
                norm_str = 'norm' if use_norm else 'abs'

                logger.info(f"\n[{count}/{total}] {feat_name} + {model_name} ({norm_str})")

                try:
                    start = time.time()
                    data = prepare_data(df, features, use_normalized=use_norm)
                    model = train_model(data, config.copy())

                    # Predict
                    val_pred = model.predict(data['X_val'])
                    test_pred = model.predict(data['X_test'])

                    # Convert to absolute if normalized
                    if use_norm:
                        val_pred_abs = val_pred * data['rolling_val'].values
                        test_pred_abs = test_pred * data['rolling_test'].values
                    else:
                        val_pred_abs = val_pred
                        test_pred_abs = test_pred

                    # Metrics on absolute scale
                    val_mae, val_smape = evaluate(data['y_abs_val'].values, val_pred_abs)
                    test_mae, test_smape = evaluate(data['y_abs_test'].values, test_pred_abs)

                    gap = test_smape / val_smape if val_smape > 0 else 0

                    result = {
                        'feature_set': feat_name,
                        'model': model_name,
                        'normalized': use_norm,
                        'n_features': len(data['features']),
                        'val_mae': round(val_mae, 2),
                        'val_smape': round(val_smape, 2),
                        'test_mae': round(test_mae, 2),
                        'test_smape': round(test_smape, 2),
                        'gap_ratio': round(gap, 3),
                        'time_sec': round(time.time() - start, 1),
                        'status': 'success',
                    }

                    logger.info(f"  Val: {val_smape:.2f}% | Test: {test_smape:.2f}% | Gap: {gap:.2f}x")

                except Exception as e:
                    result = {
                        'feature_set': feat_name,
                        'model': model_name,
                        'normalized': use_norm,
                        'status': 'failed',
                        'error': str(e),
                    }
                    logger.error(f"  Failed: {e}")

                results.append(result)
                gc.collect()

    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_dir / 'all_results.csv', index=False)

    # Analysis
    logger.info("\n" + "="*70)
    logger.info("RESULTS ANALYSIS")
    logger.info("="*70)

    successful = df_results[df_results['status'] == 'success'].copy()

    # Best by normalized
    logger.info("\n📊 NORMALIZED vs ABSOLUTE:")
    for norm in [True, False]:
        subset = successful[successful['normalized'] == norm]
        if len(subset) > 0:
            best = subset.loc[subset['test_smape'].idxmin()]
            avg_gap = subset['gap_ratio'].mean()
            label = 'Normalized' if norm else 'Absolute'
            logger.info(f"  {label}: Best={best['test_smape']:.2f}% ({best['feature_set']}+{best['model']}), Avg Gap={avg_gap:.2f}x")

    # Top 10 overall
    logger.info("\n🏆 TOP 10 BY TEST sMAPE:")
    top10 = successful.nsmallest(10, 'test_smape')
    logger.info(f"{'Rank':<4} {'Features':<20} {'Model':<12} {'Norm':<5} {'Val%':>7} {'Test%':>7} {'Gap':>6}")
    logger.info("-"*70)
    for i, (_, row) in enumerate(top10.iterrows(), 1):
        norm = 'Y' if row['normalized'] else 'N'
        logger.info(f"{i:<4} {row['feature_set']:<20} {row['model']:<12} {norm:<5} {row['val_smape']:>7.2f} {row['test_smape']:>7.2f} {row['gap_ratio']:>6.2f}x")

    # Best generalization
    logger.info("\n🎯 TOP 5 BY GENERALIZATION (lowest gap):")
    top_gap = successful.nsmallest(5, 'gap_ratio')
    for i, (_, row) in enumerate(top_gap.iterrows(), 1):
        norm = 'Y' if row['normalized'] else 'N'
        logger.info(f"{i}. {row['feature_set']} + {row['model']} ({norm}): Gap={row['gap_ratio']:.2f}x, Test={row['test_smape']:.2f}%")

    # Best per model type
    logger.info("\n📈 BEST PER MODEL TYPE:")
    for model in successful['model'].unique():
        subset = successful[successful['model'] == model]
        best = subset.loc[subset['test_smape'].idxmin()]
        norm = 'norm' if best['normalized'] else 'abs'
        logger.info(f"  {model}: {best['test_smape']:.2f}% ({best['feature_set']}, {norm})")

    # Best per feature set
    logger.info("\n📋 BEST PER FEATURE SET:")
    for feat in successful['feature_set'].unique():
        subset = successful[successful['feature_set'] == feat]
        best = subset.loc[subset['test_smape'].idxmin()]
        norm = 'norm' if best['normalized'] else 'abs'
        logger.info(f"  {feat}: {best['test_smape']:.2f}% ({best['model']}, {norm}, gap={best['gap_ratio']:.2f}x)")

    # Save summary
    best_overall = successful.loc[successful['test_smape'].idxmin()]
    best_gap = successful.loc[successful['gap_ratio'].idxmin()]

    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_experiments': len(results),
        'successful': len(successful),
        'best_accuracy': {
            'feature_set': best_overall['feature_set'],
            'model': best_overall['model'],
            'normalized': bool(best_overall['normalized']),
            'test_smape': float(best_overall['test_smape']),
            'val_smape': float(best_overall['val_smape']),
            'gap_ratio': float(best_overall['gap_ratio']),
            'n_features': int(best_overall['n_features']),
        },
        'best_generalization': {
            'feature_set': best_gap['feature_set'],
            'model': best_gap['model'],
            'normalized': bool(best_gap['normalized']),
            'gap_ratio': float(best_gap['gap_ratio']),
            'test_smape': float(best_gap['test_smape']),
        },
        'feature_sets': {k: v for k, v in FEATURE_SETS.items()},
        'model_configs': MODEL_CONFIGS,
    }

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Save top results
    top10.to_csv(output_dir / 'top10_results.csv', index=False)

    logger.info(f"\n✅ Results saved to: {output_dir}")
    logger.info(f"   - all_results.csv ({len(results)} experiments)")
    logger.info(f"   - top10_results.csv")
    logger.info(f"   - summary.json")

    return df_results


if __name__ == "__main__":
    run_grid_search()
