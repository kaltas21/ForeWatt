"""
Optimized Search V5 - Break 14% sMAPE with Refined Transfer Learning
=====================================================================
Context: V4 achieved 14.29% sMAPE using Transfer Learning (CatBoost pre-train + fine-tune).
Objective: Break the 14% barrier by:
    1. Refining the fine-tuning window (12 months vs 24 months)
    2. Re-integrating Hybrid V3 features (physics + interactions)
    3. Optimizing the fine-tuning learning rate

Configurations to test:
    Config 1: V4 Winner (Control) - Base Features + Fine-Tune on Last 24 Months
    Config 2: Window Optimization - Base Features + Fine-Tune on Last 12 Months
    Config 3: Feature Integration - Hybrid Features + Fine-Tune on Last 12 Months

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
from sklearn.model_selection import KFold

warnings.filterwarnings('ignore')
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# V4 Winner baseline
BASELINE_SMAPE = 14.29


# =============================================================================
# BASE FEATURE SET (from V4 winner)
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
# HYBRID FEATURE ENGINEERING (V3 Physics + Interactions)
# =============================================================================

def add_hybrid_features(df: pd.DataFrame, verbose: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    """
    Add hybrid features combining:
    - V1 winning interaction features
    - V2 physics-based features (stabilized with clipping)

    Returns: (df_with_features, list_of_new_feature_names)
    """
    if verbose:
        logger.info("\n  Adding Hybrid Features (V1 Interactions + V2 Physics)...")

    df = df.copy()
    eps = 1e-6
    new_features = []

    # =========================================================================
    # V1 INTERACTION FEATURES (from winning 15.96% model)
    # =========================================================================

    if 'price_ptf_lag_24h' in df.columns and 'thermal_gap' in df.columns:
        df['price_lag_x_thermal'] = df['price_ptf_lag_24h'] * df['thermal_gap']
        new_features.append('price_lag_x_thermal')

    if 'price_ptf_rolling_std_24h' in df.columns and 'renewable_saturation' in df.columns:
        df['price_std_x_renewable'] = df['price_ptf_rolling_std_24h'] * df['renewable_saturation']
        new_features.append('price_std_x_renewable')

    if 'load_factor' in df.columns and 'reserve_margin_ratio' in df.columns:
        df['load_x_reserve'] = df['load_factor'] * df['reserve_margin_ratio']
        new_features.append('load_x_reserve')

    if 'hour_sin' in df.columns and 'price_ptf_lag_24h' in df.columns:
        df['hour_x_price_lag'] = df['hour_sin'] * df['price_ptf_lag_24h']
        new_features.append('hour_x_price_lag')

    if 'system_short_signal' in df.columns and 'thermal_gap' in df.columns:
        df['short_signal_x_thermal'] = df['system_short_signal'] * df['thermal_gap']
        new_features.append('short_signal_x_thermal')

    # =========================================================================
    # V2 PHYSICS FEATURES (stabilized with clipping)
    # =========================================================================

    # Scarcity index: 1 / reserve_margin - clip to prevent explosion
    if 'reserve_margin_ratio' in df.columns:
        rm = df['reserve_margin_ratio'].clip(lower=0.05)  # Floor at 5%
        df['scarcity_index'] = (1.0 / rm).clip(upper=20)  # Cap at 20
        new_features.append('scarcity_index')

    # Thermal stress polynomials - normalized
    if 'thermal_gap' in df.columns:
        thermal = df['thermal_gap'].fillna(0)
        thermal_std = thermal.std() + eps
        thermal_norm = (thermal / thermal_std).clip(-10, 10)
        df['thermal_stress_sq'] = thermal_norm ** 2
        df['thermal_stress_cb'] = thermal_norm ** 3
        new_features.extend(['thermal_stress_sq', 'thermal_stress_cb'])

    # Net load stress: load_factor / renewable_saturation
    if 'load_factor' in df.columns and 'renewable_saturation' in df.columns:
        rs = df['renewable_saturation'].clip(lower=0.05)  # Floor at 5%
        df['net_load_stress'] = (df['load_factor'] / rs).clip(upper=20)
        new_features.append('net_load_stress')

    # Price coefficient of variation
    if 'price_ptf_rolling_std_24h' in df.columns and 'price_ptf_rolling_mean_24h' in df.columns:
        mean = df['price_ptf_rolling_mean_24h'].clip(lower=10)  # Floor at 10
        df['price_cv'] = (df['price_ptf_rolling_std_24h'] / mean).clip(upper=5)
        new_features.append('price_cv')

    # =========================================================================
    # HANDLE INF AND NAN
    # =========================================================================
    for col in new_features:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())

    if verbose:
        logger.info(f"    Added {len(new_features)} hybrid features")

    return df, new_features


# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

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


def prepare_data(df: pd.DataFrame, features: List[str],
                 add_hybrid: bool = False,
                 test_start: str = '2024-06-01') -> Dict:
    """
    Prepare data with FIXED test set for fair comparison.

    Test set: 2024-06-01 onwards (stable market regime)
    """
    available = [f for f in features if f in df.columns]

    X = df[available].copy()
    y = df['price_real'].copy()

    # Add hybrid features if requested
    hybrid_features = []
    if add_hybrid:
        X, hybrid_features = add_hybrid_features(X, verbose=True)

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

    return {
        'X_train': X[~test_mask],
        'y_train': y[~test_mask],
        'X_test': X[test_mask],
        'y_test': y[test_mask],
        'features': list(X.columns),
        'hybrid_features': hybrid_features,
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
# REFINED TRANSFER LEARNING PIPELINE
# =============================================================================

def run_transfer_learning_experiment(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    finetune_months: int = 24,
    base_lr: float = 0.02,
    finetune_lr: float = 0.005,
    base_depth: int = 10,
    base_iterations: int = 1500,
    finetune_iterations: int = 500,
    config_name: str = "default"
) -> Dict:
    """
    Refined Transfer Learning Pipeline:

    Step A: Base Training
        - Train CatBoost on ENTIRE training set (learns physics & seasonality)
        - Uses robust hyperparameters

    Step B: Fine-Tuning
        - Fine-tune on LAST N months of training data (adapts to current price levels)
        - Uses reduced learning rate to preserve base knowledge

    Parameters:
        finetune_months: How many months of recent data to use for fine-tuning
        base_lr: Learning rate for base training
        finetune_lr: Learning rate for fine-tuning (should be << base_lr)
    """
    from catboost import CatBoostRegressor
    from sklearn.model_selection import KFold

    logger.info(f"\n  Config: {config_name}")
    logger.info(f"  Fine-tune window: Last {finetune_months} months")
    logger.info(f"  Base LR: {base_lr}, Fine-tune LR: {finetune_lr}")

    # =========================================================================
    # IDENTIFY FINE-TUNING SUBSET
    # =========================================================================
    train_end = X_train.index.max()

    # Calculate fine-tune start date
    if X_train.index.tz is not None:
        finetune_start = train_end - pd.DateOffset(months=finetune_months)
    else:
        finetune_start = train_end - pd.DateOffset(months=finetune_months)

    finetune_mask = X_train.index >= finetune_start
    pretrain_mask = ~finetune_mask

    X_pretrain = X_train[pretrain_mask]
    y_pretrain = y_train[pretrain_mask]
    X_finetune = X_train[finetune_mask]
    y_finetune = y_train[finetune_mask]

    # Log the drift we're correcting
    base_mean_price = y_train.mean()
    finetune_mean_price = y_finetune.mean()
    test_mean_price = y_test.mean()

    logger.info(f"\n  PRICE DRIFT ANALYSIS:")
    logger.info(f"    Full Training Set Mean Price: {base_mean_price:.2f} TL/MWh")
    logger.info(f"    Fine-Tune Subset Mean Price:  {finetune_mean_price:.2f} TL/MWh")
    logger.info(f"    Test Set Mean Price:          {test_mean_price:.2f} TL/MWh")
    logger.info(f"    Drift (Train→Test): {((test_mean_price - base_mean_price) / base_mean_price * 100):.1f}%")
    logger.info(f"    Drift (FT→Test):    {((test_mean_price - finetune_mean_price) / finetune_mean_price * 100):.1f}%")

    logger.info(f"\n  Data splits:")
    logger.info(f"    Pre-train: {len(X_pretrain)} rows ({X_pretrain.index.min()} to {X_pretrain.index.max() if len(X_pretrain) > 0 else 'N/A'})")
    logger.info(f"    Fine-tune: {len(X_finetune)} rows ({X_finetune.index.min()} to {X_finetune.index.max()})")
    logger.info(f"    Test:      {len(X_test)} rows")

    # =========================================================================
    # K-FOLD CROSS-VALIDATION FOR ROBUST ESTIMATES
    # =========================================================================
    n_folds = 5
    kf = KFold(n_splits=n_folds, shuffle=False)

    oof_preds = np.zeros(len(X_finetune))
    test_preds = np.zeros(len(X_test))

    logger.info(f"\n  Training with {n_folds}-fold CV...")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_finetune)):
        # Get fine-tune fold indices
        X_ft_tr = X_finetune.iloc[train_idx]
        X_ft_vl = X_finetune.iloc[val_idx]
        y_ft_tr = y_finetune.iloc[train_idx]
        y_ft_vl = y_finetune.iloc[val_idx]

        # Step A: Base Training on FULL data (pre-train + train fold)
        if len(X_pretrain) > 0:
            X_base = pd.concat([X_pretrain, X_ft_tr])
            y_base = pd.concat([y_pretrain, y_ft_tr])
        else:
            X_base = X_ft_tr
            y_base = y_ft_tr

        base_model = CatBoostRegressor(
            loss_function='MAE',
            iterations=base_iterations,
            depth=base_depth,
            learning_rate=base_lr,
            l2_leaf_reg=3,
            random_state=42,
            verbose=False,
            early_stopping_rounds=100,
        )
        base_model.fit(
            X_base, y_base,
            eval_set=(X_ft_vl, y_ft_vl),
            verbose=False
        )

        # Step B: Fine-tune on recent data only
        finetune_model = CatBoostRegressor(
            loss_function='MAE',
            iterations=finetune_iterations,
            depth=base_depth,
            learning_rate=finetune_lr,
            l2_leaf_reg=3,
            random_state=42,
            verbose=False,
            early_stopping_rounds=50,
        )
        finetune_model.fit(
            X_ft_tr, y_ft_tr,
            eval_set=(X_ft_vl, y_ft_vl),
            init_model=base_model,
            verbose=False
        )

        oof_preds[val_idx] = finetune_model.predict(X_ft_vl)
        test_preds += finetune_model.predict(X_test) / n_folds

        # Log fold metrics
        fold_metrics = evaluate(y_ft_vl.values, oof_preds[val_idx])
        logger.info(f"    Fold {fold+1}/{n_folds}: Val sMAPE = {fold_metrics['smape']:.2f}%")

    # =========================================================================
    # FINAL EVALUATION
    # =========================================================================
    oof_metrics = evaluate(y_finetune.values, oof_preds)
    test_metrics = evaluate(y_test.values, test_preds)
    gap = test_metrics['smape'] / oof_metrics['smape'] if oof_metrics['smape'] > 0 else float('inf')

    logger.info(f"\n  RESULT for {config_name}:")
    logger.info(f"    OOF sMAPE:  {oof_metrics['smape']:.2f}%")
    logger.info(f"    Test sMAPE: {test_metrics['smape']:.2f}%")
    logger.info(f"    Test MAE:   {test_metrics['mae']:.2f} TL/MWh")
    logger.info(f"    Test Bias:  {test_metrics['bias']:.2f} TL/MWh")
    logger.info(f"    Gap: {gap:.2f}x")

    return {
        'config': config_name,
        'finetune_months': finetune_months,
        'val_smape': oof_metrics['smape'],
        'test_smape': test_metrics['smape'],
        'test_mae': test_metrics['mae'],
        'test_bias': test_metrics['bias'],
        'gap': gap,
        'test_pred': test_preds,
        'base_mean_price': base_mean_price,
        'finetune_mean_price': finetune_mean_price,
        'test_mean_price': test_mean_price,
    }


# =============================================================================
# MAIN EXPERIMENT LOOP
# =============================================================================

def run_v5_experiments():
    """Run all V5 configurations and compare results."""
    output_dir = PROJECT_ROOT / 'reports' / 'optimized_search_v5'
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*70)
    logger.info("OPTIMIZED SEARCH V5 - Break 14% sMAPE with Refined Transfer Learning")
    logger.info(f"Baseline to beat: {BASELINE_SMAPE}% sMAPE")
    logger.info("="*70)

    # Load data
    df = load_data()
    logger.info(f"Loaded data: {df.shape}")
    logger.info(f"Date range: {df.index.min()} to {df.index.max()}")

    results = []

    # =========================================================================
    # CONFIG 1: V4 Winner (Control) - Base Features + 24 Month Fine-tune
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("CONFIG 1: V4 Winner (Control)")
    logger.info("Base Features + Fine-Tune on Last 24 Months")
    logger.info("="*70)

    data_base = prepare_data(df, BASE_FEATURES, add_hybrid=False)
    logger.info(f"Features: {len(data_base['features'])}")
    logger.info(f"Train: {len(data_base['X_train'])}, Test: {len(data_base['X_test'])}")

    result = run_transfer_learning_experiment(
        data_base['X_train'], data_base['y_train'],
        data_base['X_test'], data_base['y_test'],
        finetune_months=24,
        base_lr=0.02,
        finetune_lr=0.005,
        config_name="Config1_V4Winner_24mo"
    )
    result['features'] = 'base'
    results.append(result)

    # =========================================================================
    # CONFIG 2: Window Optimization - Base Features + 12 Month Fine-tune
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("CONFIG 2: Window Optimization")
    logger.info("Base Features + Fine-Tune on Last 12 Months")
    logger.info("Hypothesis: 2023 data still has elevated prices; 2024 is cleaner")
    logger.info("="*70)

    result = run_transfer_learning_experiment(
        data_base['X_train'], data_base['y_train'],
        data_base['X_test'], data_base['y_test'],
        finetune_months=12,
        base_lr=0.02,
        finetune_lr=0.005,
        config_name="Config2_WindowOpt_12mo"
    )
    result['features'] = 'base'
    results.append(result)

    # =========================================================================
    # CONFIG 3: Feature Integration - Hybrid Features + 12 Month Fine-tune
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("CONFIG 3: Feature Integration")
    logger.info("Hybrid Features (Physics + Interactions) + Fine-Tune on Last 12 Months")
    logger.info("Hypothesis: Physics features help model residuals in stable era")
    logger.info("="*70)

    data_hybrid = prepare_data(df, BASE_FEATURES, add_hybrid=True)
    logger.info(f"Features: {len(data_hybrid['features'])} (including {len(data_hybrid['hybrid_features'])} hybrid)")
    logger.info(f"Train: {len(data_hybrid['X_train'])}, Test: {len(data_hybrid['X_test'])}")

    result = run_transfer_learning_experiment(
        data_hybrid['X_train'], data_hybrid['y_train'],
        data_hybrid['X_test'], data_hybrid['y_test'],
        finetune_months=12,
        base_lr=0.02,
        finetune_lr=0.005,
        config_name="Config3_Hybrid_12mo"
    )
    result['features'] = 'hybrid'
    result['hybrid_feature_list'] = data_hybrid['hybrid_features']
    results.append(result)

    # =========================================================================
    # CONFIG 4: Aggressive Window - Base Features + 6 Month Fine-tune
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("CONFIG 4: Aggressive Window")
    logger.info("Base Features + Fine-Tune on Last 6 Months")
    logger.info("Hypothesis: Even shorter window captures most recent regime better")
    logger.info("="*70)

    result = run_transfer_learning_experiment(
        data_base['X_train'], data_base['y_train'],
        data_base['X_test'], data_base['y_test'],
        finetune_months=6,
        base_lr=0.02,
        finetune_lr=0.005,
        config_name="Config4_Aggressive_6mo"
    )
    result['features'] = 'base'
    results.append(result)

    # =========================================================================
    # CONFIG 5: Feature + LR Tuning - Hybrid + 12mo + Lower LR
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("CONFIG 5: Feature + LR Tuning")
    logger.info("Hybrid Features + 12mo + Lower Fine-tune LR (0.003)")
    logger.info("Hypothesis: More conservative fine-tuning preserves base knowledge better")
    logger.info("="*70)

    result = run_transfer_learning_experiment(
        data_hybrid['X_train'], data_hybrid['y_train'],
        data_hybrid['X_test'], data_hybrid['y_test'],
        finetune_months=12,
        base_lr=0.02,
        finetune_lr=0.003,  # More conservative
        config_name="Config5_Hybrid_12mo_LowLR"
    )
    result['features'] = 'hybrid'
    results.append(result)

    # =========================================================================
    # FINAL LEADERBOARD
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("FINAL LEADERBOARD")
    logger.info("="*70)

    logger.info(f"\n{'Config':<35} {'Test sMAPE':>12} {'Bias':>10} {'Gap':>8} {'Beat?':>8}")
    logger.info("-"*80)

    for r in sorted(results, key=lambda x: x['test_smape']):
        beat = "YES" if r['test_smape'] < BASELINE_SMAPE else "no"
        logger.info(f"{r['config']:<35} {r['test_smape']:>12.2f}% {r['test_bias']:>9.2f} {r['gap']:>7.2f}x {beat:>8}")

    # Best result
    best = min(results, key=lambda x: x['test_smape'])

    logger.info(f"\n{'='*70}")
    if best['test_smape'] < BASELINE_SMAPE:
        improvement = BASELINE_SMAPE - best['test_smape']
        logger.info(f"SUCCESS! Beat {BASELINE_SMAPE}% by {improvement:.2f}%!")
    else:
        logger.info(f"Best: {best['test_smape']:.2f}% (baseline: {BASELINE_SMAPE}%)")

    logger.info(f"  Best Config: {best['config']}")
    logger.info(f"  Test sMAPE: {best['test_smape']:.2f}%")
    logger.info(f"  Test MAE: {best['test_mae']:.2f} TL/MWh")
    logger.info(f"  Test Bias: {best['test_bias']:.2f} TL/MWh")
    logger.info(f"  Gap: {best['gap']:.2f}x")

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    summary = {
        'timestamp': datetime.now().isoformat(),
        'baseline_smape': BASELINE_SMAPE,
        'test_start': '2024-06-01',
        'best_config': best['config'],
        'best_test_smape': float(best['test_smape']),
        'best_bias': float(best['test_bias']),
        'best_gap': float(best['gap']),
        'all_results': [
            {k: v for k, v in r.items() if k not in ['test_pred', 'hybrid_feature_list']}
            for r in results
        ],
    }

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Save best predictions
    if 'test_pred' in best:
        pred_df = pd.DataFrame({
            'datetime': data_base['X_test'].index if best['features'] == 'base' else data_hybrid['X_test'].index,
            'y_true': data_base['y_test'].values if best['features'] == 'base' else data_hybrid['y_test'].values,
            'y_pred': best['test_pred'],
        })
        pred_df.to_parquet(output_dir / 'best_predictions.parquet', index=False)

    logger.info(f"\nResults saved to: {output_dir}")

    return results


if __name__ == "__main__":
    run_v5_experiments()
