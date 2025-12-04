"""
Normalized Target Feature Search
=================================
Fixes distribution shift by predicting relative price (price / rolling_mean).
Tests compact feature sets with small winning model configs.

Key fixes:
1. Target: price_real / price_ptf_rolling_mean_168h (stable distribution)
2. Features: Max 20, focused on lag-based (best generalization)
3. Models: Small configs from winners

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

warnings.filterwarnings('ignore')
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# SMALL MODEL CONFIGS (from winning models, kept minimal)
# =============================================================================

MODEL_CONFIGS = {
    'lightgbm_small': {
        'n_estimators': 300,
        'max_depth': 6,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_child_samples': 30,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
    },
    'xgboost_small': {
        'n_estimators': 300,
        'max_depth': 6,
        'learning_rate': 0.05,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
    },
    'nhits_small': {
        'input_size': 48,  # smaller lookback
        'horizon': 24,
        'n_blocks': [1, 1],
        'hidden_size': 64,  # smaller
        'n_mlp_layers': 2,
        'n_pool_kernel_size': [2, 1],
        'n_freq_downsample': [2, 1],
        'batch_size': 256,
        'learning_rate': 0.003,
        'max_steps': 200,  # faster
        'early_stop_patience_steps': 50,
    },
}

# =============================================================================
# FEATURE SETS (max 20 each, focus on generalization)
# =============================================================================

FEATURE_SETS = {
    # Best generalization from previous test (1.06x gap)
    'lag_generalization': [
        'hour_sin', 'hour_cos', 'dow_sin_x', 'dow_cos_x', 'is_weekend_x',
        'price_ptf_lag_24h', 'price_ptf_lag_168h',
        'consumption_lag_24h', 'consumption_lag_168h',
        'temp_lag_24h', 'temp_lag_168h',
        'reserve_margin_ratio_lag_24h', 'renewable_saturation_lag_24h',
        'load_factor_lag_24h', 'thermal_gap_lag_24h',
        'reserve_margin_ratio', 'temp_national', 'consumption_forecast',
    ],

    # Top importance features
    'importance_top15': [
        'price_ptf_rolling_std_24h', 'thermal_gap', 'price_ptf_lag_168h',
        'hour_cos', 'price_ptf_rolling_mean_24h', 'renewable_saturation',
        'price_ptf_rolling_min_24h', 'price_ptf_lag_24h',
        'spark_spread_proxy_lag_24h', 'spark_spread_proxy_lag_48h',
        'price_ptf_rolling_max_24h', 'spark_spread_proxy_lag_168h',
        'hour_sin', 'renewable_saturation_lag_24h',
        'system_short_signal_rolling_mean_24h',
    ],

    # Ratio-focused (good balance)
    'ratios_compact': [
        'hour_sin', 'hour_cos', 'dow_sin_x', 'dow_cos_x', 'is_weekend_x',
        'reserve_margin_ratio', 'renewable_saturation', 'load_factor',
        'spark_spread_proxy', 'realtime_premium_lag24h',
        'system_short_signal', 'price_volatility_lag24h',
        'price_ptf_lag_24h', 'price_ptf_lag_168h',
    ],

    # Pure lag features (most stable across regimes)
    'pure_lags': [
        'hour_sin', 'hour_cos', 'dow_sin_x', 'dow_cos_x',
        'price_ptf_lag_24h', 'price_ptf_lag_168h',
        'reserve_margin_ratio_lag_24h', 'reserve_margin_ratio_lag_168h',
        'renewable_saturation_lag_24h', 'renewable_saturation_lag_168h',
        'spark_spread_proxy_lag_24h', 'spark_spread_proxy_lag_168h',
        'load_factor_lag_24h', 'thermal_gap_lag_24h',
    ],

    # Minimal calendar + key fundamentals
    'minimal_stable': [
        'hour_sin', 'hour_cos', 'dow_sin_x', 'dow_cos_x', 'is_weekend_x',
        'price_ptf_lag_24h', 'price_ptf_lag_168h',
        'reserve_margin_ratio', 'load_factor',
        'renewable_saturation', 'thermal_gap',
        'consumption_forecast',
    ],

    # Hybrid: lags + rolling (balance)
    'hybrid_balanced': [
        'hour_sin', 'hour_cos', 'dow_sin_x', 'dow_cos_x', 'is_weekend_x',
        'price_ptf_lag_24h', 'price_ptf_lag_168h',
        'price_ptf_rolling_mean_24h', 'price_ptf_rolling_std_24h',
        'reserve_margin_ratio', 'renewable_saturation',
        'spark_spread_proxy_lag_24h', 'load_factor_lag_24h',
        'consumption_forecast', 'thermal_gap',
    ],
}


def load_data():
    """Load master dataset."""
    path = PROJECT_ROOT / 'data' / 'gold' / 'master' / 'master_v2_fundamental.parquet'
    return pd.read_parquet(path)


def prepare_normalized_data(
    df: pd.DataFrame,
    features: List[str],
    val_size: float = 0.2,
    test_size: float = 0.2
):
    """Prepare data with normalized target."""
    available = [f for f in features if f in df.columns]

    X = df[available].copy()

    # Normalized target: price / rolling_mean (fixes distribution shift)
    y_absolute = df['price_real'].copy()
    rolling_mean = df['price_ptf_rolling_mean_168h'].copy()
    y_normalized = y_absolute / (rolling_mean + 1e-8)

    # Drop NaN
    mask = ~(X.isna().any(axis=1) | y_normalized.isna() | rolling_mean.isna())
    X = X[mask]
    y_normalized = y_normalized[mask]
    y_absolute = y_absolute[mask]
    rolling_mean = rolling_mean[mask]

    # Split
    n = len(X)
    train_end = int(n * (1 - val_size - test_size))
    val_end = int(n * (1 - test_size))

    return {
        'X_train': X.iloc[:train_end],
        'X_val': X.iloc[train_end:val_end],
        'X_test': X.iloc[val_end:],
        'y_train': y_normalized.iloc[:train_end],
        'y_val': y_normalized.iloc[train_end:val_end],
        'y_test': y_normalized.iloc[val_end:],
        'y_train_abs': y_absolute.iloc[:train_end],
        'y_val_abs': y_absolute.iloc[train_end:val_end],
        'y_test_abs': y_absolute.iloc[val_end:],
        'rolling_val': rolling_mean.iloc[train_end:val_end],
        'rolling_test': rolling_mean.iloc[val_end:],
        'features': available,
    }


def train_lightgbm(data, config):
    """Train LightGBM."""
    import lightgbm as lgb

    model = lgb.LGBMRegressor(
        objective='regression',
        metric='mae',
        verbosity=-1,
        random_state=42,
        **config
    )
    model.fit(
        data['X_train'], data['y_train'],
        eval_set=[(data['X_val'], data['y_val'])],
        callbacks=[lgb.early_stopping(30, verbose=False)]
    )
    return model


def train_xgboost(data, config):
    """Train XGBoost."""
    import xgboost as xgb

    model = xgb.XGBRegressor(
        random_state=42,
        verbosity=0,
        early_stopping_rounds=30,
        **config
    )
    model.fit(
        data['X_train'], data['y_train'],
        eval_set=[(data['X_val'], data['y_val'])],
        verbose=False
    )
    return model


def train_nhits(data, config):
    """Train N-HiTS."""
    from src.models.new_experiment.deeplearning.models.nhits_trainer import NHiTSTrainer

    trainer = NHiTSTrainer(
        target='price_real',
        horizon=config['horizon'],
        input_size=config['input_size'],
        random_seed=42,
        device=None
    )

    hyperparams = {k: v for k, v in config.items() if k not in ['input_size', 'horizon']}

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
    """Run normalized feature search."""
    output_dir = PROJECT_ROOT / 'reports' / 'normalized_feature_search'
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("\n" + "="*70)
    logger.info("NORMALIZED TARGET FEATURE SEARCH")
    logger.info("Predicting price / rolling_mean to fix distribution shift")
    logger.info("="*70)

    df = load_data()
    logger.info(f"Loaded: {df.shape}")

    # Check distribution shift is fixed
    n = len(df)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    abs_train = df['price_real'].iloc[:train_end].mean()
    abs_test = df['price_real'].iloc[val_end:].mean()
    rel_target = df['price_real'] / (df['price_ptf_rolling_mean_168h'] + 1e-8)
    rel_train = rel_target.iloc[:train_end].mean()
    rel_test = rel_target.iloc[val_end:].mean()

    logger.info(f"\nDistribution shift check:")
    logger.info(f"  Absolute: {abs_train:.0f} → {abs_test:.0f} ({(abs_test-abs_train)/abs_train*100:+.1f}%)")
    logger.info(f"  Relative: {rel_train:.4f} → {rel_test:.4f} ({(rel_test-rel_train)/rel_train*100:+.1f}%)")

    results = []

    # Test each feature set with each model
    for feat_name, features in FEATURE_SETS.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"FEATURE SET: {feat_name} ({len(features)} features)")
        logger.info(f"{'='*60}")

        data = prepare_normalized_data(df, features)
        logger.info(f"Available features: {len(data['features'])}")

        for model_name, config in MODEL_CONFIGS.items():
            if 'nhits' in model_name:
                continue  # Skip DL for speed, test baseline first

            logger.info(f"\n  {model_name}...")
            start = time.time()

            try:
                # Train on NORMALIZED target
                if 'lightgbm' in model_name:
                    model = train_lightgbm(data, config)
                    val_pred_norm = model.predict(data['X_val'])
                    test_pred_norm = model.predict(data['X_test'])
                elif 'xgboost' in model_name:
                    model = train_xgboost(data, config)
                    val_pred_norm = model.predict(data['X_val'])
                    test_pred_norm = model.predict(data['X_test'])

                # Convert back to ABSOLUTE prices
                val_pred_abs = val_pred_norm * data['rolling_val'].values
                test_pred_abs = test_pred_norm * data['rolling_test'].values

                # Evaluate on ABSOLUTE scale
                val_metrics = evaluate(data['y_val_abs'].values, val_pred_abs)
                test_metrics = evaluate(data['y_test_abs'].values, test_pred_abs)

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
                    'status': 'success',
                    'features': data['features'],
                }

                logger.info(f"    Val sMAPE:  {val_metrics['smape']:.2f}%")
                logger.info(f"    Test sMAPE: {test_metrics['smape']:.2f}%")
                logger.info(f"    Gap: {gap:.2f}x")

            except Exception as e:
                result = {
                    'feature_set': feat_name,
                    'model': model_name,
                    'status': 'failed',
                    'error': str(e),
                }
                logger.error(f"    Failed: {e}")

            results.append(result)
            gc.collect()

    # Summary
    logger.info("\n" + "="*70)
    logger.info("RESULTS SUMMARY (sorted by test sMAPE)")
    logger.info("="*70)

    successful = [r for r in results if r['status'] == 'success']
    df_results = pd.DataFrame(successful).sort_values('test_smape')

    logger.info(f"\n{'Feature Set':<25} {'Model':<15} {'#F':>3} {'Val%':>7} {'Test%':>7} {'Gap':>5}")
    logger.info("-"*70)
    for _, row in df_results.iterrows():
        logger.info(f"{row['feature_set']:<25} {row['model']:<15} {row['n_features']:>3} {row['val_smape']:>7.2f} {row['test_smape']:>7.2f} {row['gap_ratio']:>5.2f}x")

    # Best results
    best = df_results.iloc[0]
    best_gap = df_results.loc[df_results['gap_ratio'].idxmin()]

    logger.info(f"\n🏆 BEST TEST sMAPE:")
    logger.info(f"   {best['feature_set']} + {best['model']}")
    logger.info(f"   Test sMAPE: {best['test_smape']:.2f}%")
    logger.info(f"   Gap: {best['gap_ratio']:.2f}x")
    logger.info(f"   Features ({best['n_features']}): {best['features']}")

    logger.info(f"\n🎯 BEST GENERALIZATION:")
    logger.info(f"   {best_gap['feature_set']} + {best_gap['model']}")
    logger.info(f"   Gap: {best_gap['gap_ratio']:.2f}x")
    logger.info(f"   Test sMAPE: {best_gap['test_smape']:.2f}%")

    # Save results
    df_results.to_csv(output_dir / 'results.csv', index=False)

    summary = {
        'timestamp': datetime.now().isoformat(),
        'approach': 'normalized_target (price / rolling_mean_168h)',
        'distribution_shift_fix': f'{(abs_test-abs_train)/abs_train*100:.1f}% → {(rel_test-rel_train)/rel_train*100:.1f}%',
        'best_accuracy': {
            'feature_set': best['feature_set'],
            'model': best['model'],
            'test_smape': float(best['test_smape']),
            'gap_ratio': float(best['gap_ratio']),
            'n_features': int(best['n_features']),
            'features': best['features'],
        },
        'best_generalization': {
            'feature_set': best_gap['feature_set'],
            'model': best_gap['model'],
            'gap_ratio': float(best_gap['gap_ratio']),
            'test_smape': float(best_gap['test_smape']),
            'features': best_gap['features'],
        },
        'model_configs': MODEL_CONFIGS,
    }

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n✅ Results saved to: {output_dir}")

    return df_results


if __name__ == "__main__":
    run_search()
