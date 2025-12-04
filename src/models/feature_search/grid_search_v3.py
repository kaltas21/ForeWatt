"""
Grid Search V3 - Absolute Target with Winning Features
======================================================
Uses absolute price target (like original 16.01% model) with our best features.

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
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

TARGET_SMAPE = 16.01

# =============================================================================
# MODEL CONFIGS - Deeper CatBoost + N-HiTS (original winner)
# =============================================================================

MODEL_CONFIGS = {
    'cat_deep': {
        'type': 'catboost',
        'iterations': 1000,
        'depth': 10,
        'learning_rate': 0.03,
        'l2_leaf_reg': 3,
        'bagging_temperature': 0.5,
    },
    'cat_deeper': {
        'type': 'catboost',
        'iterations': 1500,
        'depth': 12,
        'learning_rate': 0.02,
        'l2_leaf_reg': 5,
        'bagging_temperature': 0.3,
    },
    'lgb_deep': {
        'type': 'lightgbm',
        'n_estimators': 1500,
        'max_depth': 12,
        'learning_rate': 0.02,
        'num_leaves': 127,
        'min_child_samples': 20,
        'reg_alpha': 0.05,
        'reg_lambda': 0.05,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
    },
    'xgb_deep': {
        'type': 'xgboost',
        'n_estimators': 1500,
        'max_depth': 12,
        'learning_rate': 0.02,
        'min_child_weight': 3,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'reg_alpha': 0.05,
        'reg_lambda': 0.05,
    },
    'nhits_small': {
        'type': 'nhits',
        'input_size': 48,
        'horizon': 24,
        'n_blocks': [1, 1],
        'hidden_size': 64,
        'n_mlp_layers': 2,
        'batch_size': 256,
        'learning_rate': 0.003,
        'max_steps': 300,
        'early_stop_patience_steps': 50,
    },
    'nhits_medium': {
        'type': 'nhits',
        'input_size': 72,
        'horizon': 24,
        'n_blocks': [1, 1, 1],
        'hidden_size': 128,
        'n_mlp_layers': 2,
        'batch_size': 256,
        'learning_rate': 0.002,
        'max_steps': 500,
        'early_stop_patience_steps': 100,
    },
}

# =============================================================================
# FEATURE SETS - Winners from V2 + variations
# =============================================================================

FEATURE_SETS = {
    # Best from V2 (16.85%)
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

    # Second best from V2 (16.98%)
    'optimal_v1': [
        'hour_sin', 'hour_cos', 'dow_sin_x', 'dow_cos_x', 'is_weekend_x',
        'price_ptf_rolling_std_24h', 'price_ptf_rolling_mean_24h',
        'price_ptf_lag_24h', 'price_ptf_lag_168h',
        'thermal_gap', 'renewable_saturation', 'load_factor',
        'spark_spread_proxy_lag_24h', 'system_short_signal',
        'consumption_forecast', 'reserve_margin_ratio',
    ],

    # Spark focus (16.98%)
    'spark_focus': [
        'hour_sin', 'hour_cos', 'dow_sin_x', 'dow_cos_x', 'is_weekend_x',
        'price_ptf_lag_24h', 'price_ptf_lag_168h',
        'spark_spread_proxy', 'spark_spread_proxy_lag_24h',
        'spark_spread_proxy_lag_48h', 'spark_spread_proxy_lag_168h',
        'thermal_gap', 'renewable_saturation', 'load_factor',
        'system_short_signal', 'consumption_forecast', 'reserve_margin_ratio',
    ],

    # Price rolling heavy (focus on price dynamics)
    'price_rolling_heavy': [
        'hour_sin', 'hour_cos', 'dow_sin_x', 'dow_cos_x',
        'price_ptf_rolling_std_24h', 'price_ptf_rolling_mean_24h',
        'price_ptf_rolling_min_24h', 'price_ptf_rolling_max_24h',
        'price_ptf_rolling_std_168h', 'price_ptf_rolling_mean_168h',
        'price_ptf_lag_24h', 'price_ptf_lag_48h', 'price_ptf_lag_168h',
        'thermal_gap', 'renewable_saturation', 'load_factor',
        'spark_spread_proxy_lag_24h', 'system_short_signal',
    ],

    # Extended with more lags
    'extended_lags': [
        'hour_sin', 'hour_cos', 'dow_sin_x', 'dow_cos_x', 'is_weekend_x',
        'price_ptf_lag_24h', 'price_ptf_lag_48h', 'price_ptf_lag_168h',
        'price_ptf_rolling_std_24h', 'price_ptf_rolling_mean_24h',
        'thermal_gap_lag_24h', 'renewable_saturation_lag_24h',
        'spark_spread_proxy_lag_24h', 'spark_spread_proxy_lag_168h',
        'load_factor_lag_24h', 'reserve_margin_ratio_lag_24h',
        'system_short_signal', 'consumption_forecast',
    ],

    # Hybrid: rolling + fundamentals
    'hybrid_full': [
        'hour_sin', 'hour_cos', 'dow_sin_x', 'dow_cos_x', 'is_weekend_x',
        'price_ptf_rolling_std_24h', 'price_ptf_rolling_mean_24h',
        'price_ptf_rolling_min_24h', 'price_ptf_rolling_max_24h',
        'price_ptf_lag_24h', 'price_ptf_lag_168h',
        'thermal_gap', 'thermal_gap_lag_24h',
        'renewable_saturation', 'spark_spread_proxy_lag_24h',
        'system_short_signal', 'load_factor', 'consumption_forecast',
    ],
}


def load_data():
    """Load master dataset."""
    path = PROJECT_ROOT / 'data' / 'gold' / 'master' / 'master_v2_fundamental.parquet'
    return pd.read_parquet(path)


def prepare_data(df: pd.DataFrame, features: List[str], val_size: float = 0.2, test_size: float = 0.2):
    """Prepare data with ABSOLUTE target (original approach)."""
    available = [f for f in features if f in df.columns]

    X = df[available].copy()
    y = df['price_real'].copy()  # ABSOLUTE target

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
        'features': available,
    }


def train_catboost(data, config):
    """Train CatBoost."""
    from catboost import CatBoostRegressor

    params = {k: v for k, v in config.items() if k != 'type'}
    model = CatBoostRegressor(
        loss_function='MAE',
        random_state=42,
        verbose=False,
        early_stopping_rounds=50,
        **params
    )
    model.fit(
        data['X_train'], data['y_train'],
        eval_set=(data['X_val'], data['y_val']),
        verbose=False
    )
    return model


def train_lightgbm(data, config):
    """Train LightGBM."""
    import lightgbm as lgb

    params = {k: v for k, v in config.items() if k != 'type'}
    model = lgb.LGBMRegressor(
        objective='regression',
        metric='mae',
        verbosity=-1,
        random_state=42,
        **params
    )
    model.fit(
        data['X_train'], data['y_train'],
        eval_set=[(data['X_val'], data['y_val'])],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    return model


def train_xgboost(data, config):
    """Train XGBoost."""
    import xgboost as xgb

    params = {k: v for k, v in config.items() if k != 'type'}
    model = xgb.XGBRegressor(
        random_state=42,
        verbosity=0,
        early_stopping_rounds=50,
        **params
    )
    model.fit(
        data['X_train'], data['y_train'],
        eval_set=[(data['X_val'], data['y_val'])],
        verbose=False
    )
    return model


def train_nhits(data, config):
    """Train N-HiTS (original winning model type)."""
    from src.models.new_experiment.deeplearning.models.nhits_trainer import NHiTSTrainer

    trainer = NHiTSTrainer(
        target='price_real',
        horizon=config['horizon'],
        input_size=config['input_size'],
        random_seed=42,
        device=None
    )

    hyperparams = {k: v for k, v in config.items()
                   if k not in ['type', 'input_size', 'horizon']}

    model, _ = trainer.train(
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val'],
        hyperparams
    )
    return model, trainer


def evaluate(y_true, y_pred):
    """Calculate metrics."""
    mae = np.mean(np.abs(y_true - y_pred))
    smape = 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))
    return {'mae': mae, 'smape': smape}


def run_search():
    """Run grid search V3 with absolute target."""
    output_dir = PROJECT_ROOT / 'reports' / 'grid_search_v3'
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*70)
    logger.info("GRID SEARCH V3 - ABSOLUTE TARGET WITH WINNING FEATURES")
    logger.info(f"Target: Beat {TARGET_SMAPE}% test sMAPE")
    logger.info("="*70)

    df = load_data()
    logger.info(f"Data: {df.shape}")

    results = []
    total = len(FEATURE_SETS) * len(MODEL_CONFIGS)

    for i, (feat_name, features) in enumerate(FEATURE_SETS.items()):
        for j, (model_name, config) in enumerate(MODEL_CONFIGS.items()):
            exp_num = i * len(MODEL_CONFIGS) + j + 1
            logger.info(f"\n[{exp_num}/{total}] {feat_name} + {model_name}")

            start = time.time()

            try:
                data = prepare_data(df, features)

                if config['type'] == 'catboost':
                    model = train_catboost(data, config)
                    val_pred = model.predict(data['X_val'])
                    test_pred = model.predict(data['X_test'])
                elif config['type'] == 'lightgbm':
                    model = train_lightgbm(data, config)
                    val_pred = model.predict(data['X_val'])
                    test_pred = model.predict(data['X_test'])
                elif config['type'] == 'xgboost':
                    model = train_xgboost(data, config)
                    val_pred = model.predict(data['X_val'])
                    test_pred = model.predict(data['X_test'])
                elif config['type'] == 'nhits':
                    model, trainer = train_nhits(data, config)
                    val_pred = trainer.predict(model, data['X_val'])
                    test_pred = trainer.predict(model, data['X_test'])

                val_metrics = evaluate(data['y_val'].values, val_pred)
                test_metrics = evaluate(data['y_test'].values, test_pred)

                gap = test_metrics['smape'] / val_metrics['smape'] if val_metrics['smape'] > 0 else 0

                result = {
                    'feature_set': feat_name,
                    'model': model_name,
                    'n_features': len(data['features']),
                    'val_smape': val_metrics['smape'],
                    'test_smape': test_metrics['smape'],
                    'val_mae': val_metrics['mae'],
                    'test_mae': test_metrics['mae'],
                    'gap_ratio': gap,
                    'time_sec': time.time() - start,
                    'beat_target': test_metrics['smape'] < TARGET_SMAPE,
                    'features': data['features'],
                }

                logger.info(f"  Val: {val_metrics['smape']:.2f}% | Test: {test_metrics['smape']:.2f}% | Gap: {gap:.2f}x")
                if result['beat_target']:
                    logger.info(f"  🎯 BEATS TARGET!")

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

    # Summary
    logger.info("\n" + "="*70)
    logger.info("RESULTS")
    logger.info("="*70)

    successful = [r for r in results if 'test_smape' in r]
    df_results = pd.DataFrame(successful).sort_values('test_smape')

    # Check if any beat target
    beat_target = df_results[df_results['beat_target'] == True]
    if len(beat_target) > 0:
        logger.info(f"\n✅ {len(beat_target)} configurations beat {TARGET_SMAPE}%!")
        for _, row in beat_target.iterrows():
            logger.info(f"  {row['feature_set']} + {row['model']}: {row['test_smape']:.2f}%")
    else:
        logger.info(f"\n❌ No configuration beat {TARGET_SMAPE}% target")

    # Top 10
    logger.info(f"\n🏆 TOP 10 BY TEST sMAPE:")
    logger.info(f"{'Rank':<5}{'Features':<20}{'Model':<15}{'#F':>4}{'Val%':>8}{'Test%':>8}{'Gap':>7}")
    logger.info("-"*70)
    for rank, (_, row) in enumerate(df_results.head(10).iterrows(), 1):
        marker = "🎯" if row.get('beat_target', False) else "  "
        logger.info(f"{rank:<5}{row['feature_set']:<20}{row['model']:<15}{row['n_features']:>4}{row['val_smape']:>8.2f}{row['test_smape']:>8.2f}{row['gap_ratio']:>6.2f}x {marker}")

    # Best result
    best = df_results.iloc[0]
    logger.info(f"\n🏆 BEST OVERALL:")
    logger.info(f"   Feature set: {best['feature_set']}")
    logger.info(f"   Model: {best['model']}")
    logger.info(f"   Test sMAPE: {best['test_smape']:.2f}%")
    logger.info(f"   Gap: {best['gap_ratio']:.2f}x")
    logger.info(f"   Features ({best['n_features']}): {best['features']}")

    # Save results
    df_results.to_csv(output_dir / 'results.csv', index=False)

    summary = {
        'timestamp': datetime.now().isoformat(),
        'target_smape': TARGET_SMAPE,
        'target_beat': len(beat_target) > 0,
        'num_beat': len(beat_target),
        'best': {
            'feature_set': best['feature_set'],
            'model': best['model'],
            'test_smape': float(best['test_smape']),
            'gap_ratio': float(best['gap_ratio']),
            'features': best['features'],
        },
        'all_results': df_results.to_dict('records'),
    }

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"\n✅ Results saved to: {output_dir}")

    return df_results


if __name__ == "__main__":
    run_search()
