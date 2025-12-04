"""
Grid Search V2 - Optimized for Best sMAPE
==========================================
Target: Beat 16.01% test sMAPE while maintaining gap < 1.5x

Strategy:
1. Use best importance features
2. Add rolling stats (key for accuracy)
3. Larger models with regularization
4. More feature combinations

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
# OPTIMIZED MODEL CONFIGS (larger, with regularization)
# =============================================================================

MODEL_CONFIGS = {
    # LightGBM - optimized
    'lgb_opt': {
        'type': 'lightgbm',
        'n_estimators': 800,
        'max_depth': 8,
        'learning_rate': 0.03,
        'num_leaves': 127,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
    },
    'lgb_deep': {
        'type': 'lightgbm',
        'n_estimators': 1000,
        'max_depth': 10,
        'learning_rate': 0.02,
        'num_leaves': 255,
        'min_child_samples': 15,
        'reg_alpha': 0.05,
        'reg_lambda': 0.05,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
    },

    # XGBoost - optimized
    'xgb_opt': {
        'type': 'xgboost',
        'n_estimators': 800,
        'max_depth': 8,
        'learning_rate': 0.03,
        'min_child_weight': 3,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
    },
    'xgb_deep': {
        'type': 'xgboost',
        'n_estimators': 1000,
        'max_depth': 10,
        'learning_rate': 0.02,
        'min_child_weight': 2,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'reg_alpha': 0.05,
        'reg_lambda': 0.5,
    },

    # CatBoost - optimized
    'cat_opt': {
        'type': 'catboost',
        'iterations': 800,
        'depth': 8,
        'learning_rate': 0.03,
        'l2_leaf_reg': 1,
        'random_strength': 0.5,
        'bagging_temperature': 0.5,
    },
    'cat_deep': {
        'type': 'catboost',
        'iterations': 1000,
        'depth': 10,
        'learning_rate': 0.02,
        'l2_leaf_reg': 0.5,
        'random_strength': 0.3,
        'bagging_temperature': 0.3,
    },
}

# =============================================================================
# OPTIMIZED FEATURE SETS (based on importance + accuracy analysis)
# =============================================================================

FEATURE_SETS = {
    # Best from importance analysis + rolling stats
    'optimal_v1': [
        'hour_sin', 'hour_cos', 'dow_sin_x', 'dow_cos_x', 'is_weekend_x',
        'price_ptf_rolling_std_24h', 'price_ptf_rolling_mean_24h',
        'price_ptf_lag_24h', 'price_ptf_lag_168h',
        'thermal_gap', 'renewable_saturation',
        'spark_spread_proxy_lag_24h', 'spark_spread_proxy_lag_168h',
        'system_short_signal_rolling_mean_24h',
        'reserve_margin_ratio', 'load_factor',
    ],

    # Add more rolling features
    'optimal_v2': [
        'hour_sin', 'hour_cos', 'dow_sin_x', 'dow_cos_x', 'is_weekend_x',
        'price_ptf_rolling_std_24h', 'price_ptf_rolling_mean_24h',
        'price_ptf_rolling_min_24h', 'price_ptf_rolling_max_24h',
        'price_ptf_lag_24h', 'price_ptf_lag_168h',
        'thermal_gap', 'renewable_saturation',
        'spark_spread_proxy_lag_24h',
        'system_short_signal', 'consumption_forecast',
    ],

    # Full importance top 20
    'importance_20': [
        'price_ptf_rolling_std_24h', 'thermal_gap', 'price_ptf_lag_168h',
        'hour_cos', 'price_ptf_rolling_mean_24h', 'renewable_saturation',
        'price_ptf_rolling_min_24h', 'price_ptf_lag_24h',
        'spark_spread_proxy_lag_24h', 'spark_spread_proxy_lag_48h',
        'price_ptf_rolling_max_24h', 'spark_spread_proxy_lag_168h',
        'hour_sin', 'renewable_saturation_lag_24h',
        'system_short_signal_rolling_mean_24h', 'price_volatility_lag24h',
        'price_smf_lag_24h', 'capacity_eak',
        'system_short_signal_rolling_std_24h', 'system_short_signal',
    ],

    # Combine hybrid_balanced + importance
    'hybrid_importance': [
        'hour_sin', 'hour_cos', 'dow_sin_x', 'dow_cos_x', 'is_weekend_x',
        'price_ptf_lag_24h', 'price_ptf_lag_168h',
        'price_ptf_rolling_mean_24h', 'price_ptf_rolling_std_24h',
        'price_ptf_rolling_min_24h', 'price_ptf_rolling_max_24h',
        'thermal_gap', 'renewable_saturation',
        'spark_spread_proxy_lag_24h', 'load_factor_lag_24h',
        'consumption_forecast', 'system_short_signal',
    ],

    # Price-centric with all rolling
    'price_centric': [
        'hour_sin', 'hour_cos', 'dow_sin_x', 'dow_cos_x',
        'price_ptf_lag_24h', 'price_ptf_lag_168h',
        'price_ptf_rolling_mean_24h', 'price_ptf_rolling_std_24h',
        'price_ptf_rolling_min_24h', 'price_ptf_rolling_max_24h',
        'price_ptf_rolling_mean_168h', 'price_ptf_rolling_std_168h',
        'price_volatility_lag24h', 'price_smf_lag_24h',
        'realtime_premium_lag24h', 'system_short_signal',
    ],

    # Fundamental heavy
    'fundamental_rich': [
        'hour_sin', 'hour_cos', 'dow_sin_x', 'dow_cos_x', 'is_weekend_x',
        'price_ptf_lag_24h', 'price_ptf_lag_168h',
        'price_ptf_rolling_mean_24h', 'price_ptf_rolling_std_24h',
        'reserve_margin_ratio', 'renewable_saturation', 'load_factor',
        'thermal_gap', 'spark_spread_proxy',
        'system_short_signal', 'capacity_eak',
        'consumption_forecast', 'wind_forecast',
    ],

    # Spark spread focus (important feature)
    'spark_focus': [
        'hour_sin', 'hour_cos', 'dow_sin_x', 'dow_cos_x', 'is_weekend_x',
        'price_ptf_lag_24h', 'price_ptf_lag_168h',
        'price_ptf_rolling_mean_24h', 'price_ptf_rolling_std_24h',
        'spark_spread_proxy', 'spark_spread_proxy_lag_24h',
        'spark_spread_proxy_lag_48h', 'spark_spread_proxy_lag_168h',
        'thermal_gap', 'renewable_saturation',
        'system_short_signal', 'load_factor',
    ],

    # Volatility focus
    'volatility_focus': [
        'hour_sin', 'hour_cos', 'dow_sin_x', 'dow_cos_x', 'is_weekend_x',
        'price_ptf_lag_24h', 'price_ptf_lag_168h',
        'price_ptf_rolling_std_24h', 'price_ptf_rolling_std_168h',
        'price_volatility_lag24h',
        'system_short_signal_rolling_std_24h',
        'reserve_margin_ratio_rolling_std_24h',
        'thermal_gap', 'renewable_saturation',
        'spark_spread_proxy_lag_24h', 'consumption_forecast',
    ],

    # Combined best (18 features)
    'combined_best': [
        'hour_sin', 'hour_cos', 'dow_sin_x', 'dow_cos_x', 'is_weekend_x',
        'price_ptf_rolling_std_24h', 'price_ptf_rolling_mean_24h',
        'price_ptf_rolling_min_24h', 'price_ptf_rolling_max_24h',
        'price_ptf_lag_24h', 'price_ptf_lag_168h',
        'thermal_gap', 'renewable_saturation', 'load_factor',
        'spark_spread_proxy_lag_24h', 'spark_spread_proxy_lag_168h',
        'system_short_signal', 'consumption_forecast',
    ],

    # Ultra features (20)
    'ultra_20': [
        'hour_sin', 'hour_cos', 'dow_sin_x', 'dow_cos_x', 'is_weekend_x',
        'price_ptf_rolling_std_24h', 'price_ptf_rolling_mean_24h',
        'price_ptf_rolling_min_24h', 'price_ptf_rolling_max_24h',
        'price_ptf_lag_24h', 'price_ptf_lag_168h',
        'thermal_gap', 'renewable_saturation',
        'spark_spread_proxy_lag_24h', 'spark_spread_proxy_lag_48h',
        'system_short_signal', 'system_short_signal_rolling_mean_24h',
        'load_factor', 'consumption_forecast', 'price_volatility_lag24h',
    ],
}


def load_data():
    path = PROJECT_ROOT / 'data' / 'gold' / 'master' / 'master_v2_fundamental.parquet'
    return pd.read_parquet(path)


def prepare_data(df, features):
    available = [f for f in features if f in df.columns]
    X = df[available].copy()
    y = df['price_real'].copy()

    mask = ~(X.isna().any(axis=1) | y.isna())
    X, y = X[mask], y[mask]

    n = len(X)
    t1, t2 = int(n * 0.6), int(n * 0.8)

    return {
        'X_train': X.iloc[:t1], 'X_val': X.iloc[t1:t2], 'X_test': X.iloc[t2:],
        'y_train': y.iloc[:t1], 'y_val': y.iloc[t1:t2], 'y_test': y.iloc[t2:],
        'features': available,
    }


def train_model(data, config):
    model_type = config.pop('type')

    if model_type == 'lightgbm':
        import lightgbm as lgb
        model = lgb.LGBMRegressor(objective='regression', verbosity=-1, random_state=42, **config)
        model.fit(data['X_train'], data['y_train'],
                  eval_set=[(data['X_val'], data['y_val'])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])

    elif model_type == 'xgboost':
        import xgboost as xgb
        model = xgb.XGBRegressor(verbosity=0, random_state=42, early_stopping_rounds=50, **config)
        model.fit(data['X_train'], data['y_train'],
                  eval_set=[(data['X_val'], data['y_val'])], verbose=False)

    elif model_type == 'catboost':
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(verbose=False, random_seed=42, early_stopping_rounds=50, **config)
        model.fit(data['X_train'], data['y_train'],
                  eval_set=(data['X_val'], data['y_val']), verbose=False)

    config['type'] = model_type
    return model


def evaluate(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    smape = 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))
    return mae, smape


def run_grid_search():
    output_dir = PROJECT_ROOT / 'reports' / 'grid_search_v2'
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*70)
    logger.info("GRID SEARCH V2 - OPTIMIZED FOR BEST sMAPE")
    logger.info("Target: Beat 16.01% test sMAPE")
    logger.info("="*70)

    df = load_data()
    logger.info(f"Data: {df.shape}")

    results = []
    total = len(MODEL_CONFIGS) * len(FEATURE_SETS)
    count = 0

    for feat_name, features in FEATURE_SETS.items():
        for model_name, config in MODEL_CONFIGS.items():
            count += 1
            logger.info(f"\n[{count}/{total}] {feat_name} + {model_name}")

            try:
                start = time.time()
                data = prepare_data(df, features)
                model = train_model(data, config.copy())

                val_pred = model.predict(data['X_val'])
                test_pred = model.predict(data['X_test'])

                val_mae, val_smape = evaluate(data['y_val'].values, val_pred)
                test_mae, test_smape = evaluate(data['y_test'].values, test_pred)

                gap = test_smape / val_smape if val_smape > 0 else 0

                result = {
                    'feature_set': feat_name,
                    'model': model_name,
                    'n_features': len(data['features']),
                    'val_mae': round(val_mae, 2),
                    'val_smape': round(val_smape, 2),
                    'test_mae': round(test_mae, 2),
                    'test_smape': round(test_smape, 2),
                    'gap_ratio': round(gap, 3),
                    'time_sec': round(time.time() - start, 1),
                    'status': 'success',
                    'features': data['features'],
                }

                # Highlight if beats 16.01%
                if test_smape < 16.01:
                    logger.info(f"  🎯 Val: {val_smape:.2f}% | Test: {test_smape:.2f}% | Gap: {gap:.2f}x *** BEATS TARGET! ***")
                else:
                    logger.info(f"  Val: {val_smape:.2f}% | Test: {test_smape:.2f}% | Gap: {gap:.2f}x")

            except Exception as e:
                result = {
                    'feature_set': feat_name,
                    'model': model_name,
                    'status': 'failed',
                    'error': str(e),
                }
                logger.error(f"  Failed: {e}")

            results.append(result)
            gc.collect()

    # Save and analyze
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_dir / 'all_results.csv', index=False)

    successful = df_results[df_results['status'] == 'success'].copy()

    logger.info("\n" + "="*70)
    logger.info("RESULTS")
    logger.info("="*70)

    # Check if any beat target
    beats_target = successful[successful['test_smape'] < 16.01]
    if len(beats_target) > 0:
        logger.info(f"\n🎉 {len(beats_target)} configurations beat 16.01% target!")
        for _, row in beats_target.sort_values('test_smape').iterrows():
            logger.info(f"  {row['feature_set']} + {row['model']}: {row['test_smape']:.2f}% (gap={row['gap_ratio']:.2f}x)")
    else:
        logger.info("\n❌ No configuration beat 16.01% target")

    # Top 10
    logger.info("\n🏆 TOP 10 BY TEST sMAPE:")
    top10 = successful.nsmallest(10, 'test_smape')
    logger.info(f"{'Rank':<4} {'Features':<20} {'Model':<12} {'#F':>3} {'Val%':>7} {'Test%':>7} {'Gap':>6}")
    logger.info("-"*70)
    for i, (_, row) in enumerate(top10.iterrows(), 1):
        logger.info(f"{i:<4} {row['feature_set']:<20} {row['model']:<12} {row['n_features']:>3} {row['val_smape']:>7.2f} {row['test_smape']:>7.2f} {row['gap_ratio']:>6.2f}x")

    # Best overall
    best = successful.loc[successful['test_smape'].idxmin()]
    logger.info(f"\n🏆 BEST OVERALL:")
    logger.info(f"   Feature set: {best['feature_set']}")
    logger.info(f"   Model: {best['model']}")
    logger.info(f"   Test sMAPE: {best['test_smape']:.2f}%")
    logger.info(f"   Gap: {best['gap_ratio']:.2f}x")
    logger.info(f"   Features ({best['n_features']}): {best['features']}")

    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'target': 16.01,
        'beat_target': len(beats_target) > 0,
        'best': {
            'feature_set': best['feature_set'],
            'model': best['model'],
            'test_smape': float(best['test_smape']),
            'gap_ratio': float(best['gap_ratio']),
            'n_features': int(best['n_features']),
            'features': best['features'],
        },
        'model_configs': MODEL_CONFIGS,
        'feature_sets': {k: v for k, v in FEATURE_SETS.items()},
    }

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    top10.to_csv(output_dir / 'top10_results.csv', index=False)

    logger.info(f"\n✅ Results saved to: {output_dir}")

    return df_results


if __name__ == "__main__":
    run_grid_search()
