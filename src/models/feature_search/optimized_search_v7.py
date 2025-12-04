"""
Optimized Search V7 - Time-Block Transfer Learning (Duck Curve Adaptation)
===========================================================================
Context: We are stuck at 14.08% sMAPE (V5). V5 adapted to new price LEVEL,
         but failed to capture the SHAPE change (Duck Curve from solar adoption).

Insight: The Turkish grid has undergone structural shape change:
         - Noon 2025 is fundamentally different from Noon 2020 (solar depression)
         - Midnight physics are relatively stable
         - Peak hours (17-21) are most critical (solar drop + demand peak)

Objective: Break 13% sMAPE by implementing Time-Block Transfer Learning.

Strategy:
    1. Train ONE global base model on 2020-2023 (learns general physics)
    2. Fine-tune SEPARATE models for each time block on 2024 data
    3. Use block-specific models for prediction

Time Blocks:
    - Night:   00:00 - 06:00 (Stable, thermal driven)
    - Morning: 07:00 - 10:00 (Ramp up)
    - Solar:   11:00 - 16:00 (Renewable depression, high volatility)
    - Peak:    17:00 - 21:00 (Solar drop, demand peak - most critical)
    - Evening: 22:00 - 23:00 (Ramp down)

Author: ForeWatt Team
Date: December 2025
"""

import sys
import os
import copy
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

# V5 Winner baseline
BASELINE_SMAPE = 14.08


# =============================================================================
# TIME BLOCK DEFINITIONS
# =============================================================================

TIME_BLOCKS = {
    'Night':   list(range(0, 7)),      # 00:00 - 06:00 (7 hours)
    'Morning': list(range(7, 11)),     # 07:00 - 10:00 (4 hours)
    'Solar':   list(range(11, 17)),    # 11:00 - 16:00 (6 hours)
    'Peak':    list(range(17, 22)),    # 17:00 - 21:00 (5 hours)
    'Evening': list(range(22, 24)),    # 22:00 - 23:00 (2 hours)
}

BLOCK_ORDER = ['Night', 'Morning', 'Solar', 'Peak', 'Evening']


def get_time_block(hour: int) -> str:
    """Map hour to structural time block."""
    for block_name, hours in TIME_BLOCKS.items():
        if hour in hours:
            return block_name
    return 'Night'  # Fallback


def get_hour_from_index(idx: pd.DatetimeIndex) -> pd.Series:
    """Extract hour from datetime index."""
    return idx.hour


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

    # Ensure hour column exists
    if 'hour' not in df.columns:
        df['hour'] = df.index.hour

    return df


def prepare_stratified_data(df: pd.DataFrame, features: List[str],
                            base_end: str = '2023-12-31',
                            finetune_end: str = '2024-05-31') -> Dict:
    """
    Prepare data with DISJOINT splits:
        - Base Train: Start to base_end (Learn long-term physics)
        - Fine-Tune: base_end to finetune_end (Shape adaptation)
        - Test: finetune_end onwards (Out-of-sample)
    """
    available = [f for f in features if f in df.columns]

    X = df[available].copy()
    y = df['price_real'].copy()

    # Ensure hour is available for stratification
    hours = df.index.hour

    # Drop NaN
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    hours = hours[mask]

    # Timezone-aware splits
    tz = X.index.tz
    base_end_dt = pd.Timestamp(base_end, tz=tz)
    finetune_end_dt = pd.Timestamp(finetune_end, tz=tz)

    # Create masks
    base_mask = X.index <= base_end_dt
    finetune_mask = (X.index > base_end_dt) & (X.index <= finetune_end_dt)
    test_mask = X.index > finetune_end_dt

    logger.info(f"\n  DATA SPLITS (Disjoint):")
    logger.info(f"    Base Train:  {base_mask.sum():,} rows ({X[base_mask].index.min()} to {X[base_mask].index.max()})")
    logger.info(f"    Fine-Tune:   {finetune_mask.sum():,} rows ({X[finetune_mask].index.min()} to {X[finetune_mask].index.max()})")
    logger.info(f"    Test:        {test_mask.sum():,} rows ({X[test_mask].index.min()} to {X[test_mask].index.max()})")

    return {
        'X_base': X[base_mask],
        'y_base': y[base_mask],
        'hours_base': hours[base_mask],
        'X_finetune': X[finetune_mask],
        'y_finetune': y[finetune_mask],
        'hours_finetune': hours[finetune_mask],
        'X_test': X[test_mask],
        'y_test': y[test_mask],
        'hours_test': hours[test_mask],
        'features': list(X.columns),
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


def evaluate_by_block(y_true: np.ndarray, y_pred: np.ndarray, hours: np.ndarray) -> Dict:
    """Calculate metrics per time block."""
    block_metrics = {}

    for block_name in BLOCK_ORDER:
        block_hours = TIME_BLOCKS[block_name]
        mask = np.isin(hours, block_hours)

        if mask.sum() > 0:
            metrics = evaluate(y_true[mask], y_pred[mask])
            metrics['count'] = mask.sum()
            block_metrics[block_name] = metrics

    return block_metrics


# =============================================================================
# MODEL TRAINING FUNCTIONS
# =============================================================================

def train_catboost(X_train, y_train, X_val, y_val, config, init_model=None):
    """Train CatBoost with optional init_model for transfer learning."""
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

    if init_model is not None:
        fit_params['init_model'] = init_model

    model.fit(X_train, y_train, **fit_params)
    return model


# =============================================================================
# STRATIFIED TRANSFER LEARNING PIPELINE
# =============================================================================

def run_stratified_transfer_learning(data: Dict, strategy: str = 'block') -> Dict:
    """
    Run stratified transfer learning pipeline.

    Strategy options:
        - 'global': Global fine-tuning (baseline V5 approach)
        - 'hourly': Fine-tune 24 separate hourly models
        - 'block': Fine-tune 5 time block models
    """
    from catboost import CatBoostRegressor

    logger.info(f"\n  Strategy: {strategy.upper()}")

    # =========================================================================
    # STEP A: GLOBAL BASE TRAINING
    # =========================================================================
    logger.info("\n  Step A: Training Global Base Model...")

    # Split base data for early stopping
    split_idx = int(len(data['X_base']) * 0.85)
    X_base_tr = data['X_base'].iloc[:split_idx]
    X_base_vl = data['X_base'].iloc[split_idx:]
    y_base_tr = data['y_base'].iloc[:split_idx]
    y_base_vl = data['y_base'].iloc[split_idx:]

    base_config = {
        'iterations': 2000,
        'depth': 10,
        'learning_rate': 0.02,
        'l2_leaf_reg': 3,
    }

    base_model = train_catboost(X_base_tr, y_base_tr, X_base_vl, y_base_vl, base_config)
    logger.info(f"    Base model trained ({base_model.tree_count_} trees)")

    # Save base model for reuse
    base_model_path = PROJECT_ROOT / 'reports' / 'optimized_search_v7' / 'base_model.cbm'
    base_model_path.parent.mkdir(parents=True, exist_ok=True)
    base_model.save_model(str(base_model_path))

    # =========================================================================
    # STEP B: STRATIFIED FINE-TUNING
    # =========================================================================
    logger.info(f"\n  Step B: Stratified Fine-Tuning ({strategy})...")

    finetune_config = {
        'iterations': 500,
        'depth': 10,
        'learning_rate': 0.01,
        'l2_leaf_reg': 3,
    }

    specialist_models = {}
    X_ft = data['X_finetune']
    y_ft = data['y_finetune']
    hours_ft = data['hours_finetune']

    if strategy == 'global':
        # Single global fine-tune
        split_idx = int(len(X_ft) * 0.8)
        model = train_catboost(
            X_ft.iloc[:split_idx], y_ft.iloc[:split_idx],
            X_ft.iloc[split_idx:], y_ft.iloc[split_idx:],
            finetune_config,
            init_model=CatBoostRegressor().load_model(str(base_model_path))
        )
        specialist_models['global'] = model
        logger.info(f"    Global model fine-tuned")

    elif strategy == 'hourly':
        # 24 separate hourly models
        for hour in range(24):
            mask = (hours_ft == hour)
            if mask.sum() < 100:
                logger.warning(f"    Hour {hour}: Only {mask.sum()} samples, skipping")
                continue

            X_hour = X_ft[mask]
            y_hour = y_ft[mask]

            split_idx = int(len(X_hour) * 0.8)
            if split_idx < 50:
                specialist_models[hour] = None
                continue

            # Load fresh base model
            fresh_base = CatBoostRegressor()
            fresh_base.load_model(str(base_model_path))

            model = train_catboost(
                X_hour.iloc[:split_idx], y_hour.iloc[:split_idx],
                X_hour.iloc[split_idx:], y_hour.iloc[split_idx:],
                finetune_config,
                init_model=fresh_base
            )
            specialist_models[hour] = model
            logger.info(f"    Hour {hour:02d}: {mask.sum()} samples, fine-tuned")

    elif strategy == 'block':
        # 5 time block models
        for block_name in BLOCK_ORDER:
            block_hours = TIME_BLOCKS[block_name]
            mask = hours_ft.isin(block_hours)

            X_block = X_ft[mask]
            y_block = y_ft[mask]

            logger.info(f"    Block {block_name}: {mask.sum()} samples")

            if mask.sum() < 100:
                logger.warning(f"      Too few samples, using base model")
                specialist_models[block_name] = None
                continue

            split_idx = int(len(X_block) * 0.8)

            # Load fresh base model
            fresh_base = CatBoostRegressor()
            fresh_base.load_model(str(base_model_path))

            model = train_catboost(
                X_block.iloc[:split_idx], y_block.iloc[:split_idx],
                X_block.iloc[split_idx:], y_block.iloc[split_idx:],
                finetune_config,
                init_model=fresh_base
            )
            specialist_models[block_name] = model
            logger.info(f"      Fine-tuned ({model.tree_count_} trees)")

    # =========================================================================
    # STEP C: STRATIFIED PREDICTION
    # =========================================================================
    logger.info(f"\n  Step C: Stratified Prediction on Test Set...")

    X_test = data['X_test']
    y_test = data['y_test']
    hours_test = data['hours_test']

    # Load base model for fallback
    base_model_fallback = CatBoostRegressor()
    base_model_fallback.load_model(str(base_model_path))

    predictions = np.zeros(len(X_test))

    if strategy == 'global':
        predictions = specialist_models['global'].predict(X_test)

    elif strategy == 'hourly':
        for hour in range(24):
            mask = (hours_test == hour)
            if mask.sum() == 0:
                continue

            model = specialist_models.get(hour)
            if model is None:
                predictions[mask] = base_model_fallback.predict(X_test[mask])
            else:
                predictions[mask] = model.predict(X_test[mask])

    elif strategy == 'block':
        for block_name in BLOCK_ORDER:
            block_hours = TIME_BLOCKS[block_name]
            mask = hours_test.isin(block_hours)

            if mask.sum() == 0:
                continue

            model = specialist_models.get(block_name)
            if model is None:
                predictions[mask] = base_model_fallback.predict(X_test[mask])
            else:
                predictions[mask] = model.predict(X_test[mask])

    # =========================================================================
    # EVALUATION
    # =========================================================================
    test_metrics = evaluate(y_test.values, predictions)
    block_metrics = evaluate_by_block(y_test.values, predictions, hours_test.values)

    logger.info(f"\n  RESULT ({strategy.upper()}):")
    logger.info(f"    Test sMAPE: {test_metrics['smape']:.2f}%")
    logger.info(f"    Test MAE:   {test_metrics['mae']:.2f} TL/MWh")
    logger.info(f"    Test Bias:  {test_metrics['bias']:.2f} TL/MWh")

    logger.info(f"\n  Per-Block Performance:")
    for block_name in BLOCK_ORDER:
        if block_name in block_metrics:
            m = block_metrics[block_name]
            logger.info(f"    {block_name:<10}: sMAPE={m['smape']:.2f}%, Bias={m['bias']:+.2f} TL, n={m['count']}")

    return {
        'strategy': strategy,
        'test_smape': test_metrics['smape'],
        'test_mae': test_metrics['mae'],
        'test_bias': test_metrics['bias'],
        'block_metrics': block_metrics,
        'predictions': predictions,
    }


# =============================================================================
# MAIN EXPERIMENT LOOP
# =============================================================================

def run_v7_experiments():
    """Run all V7 experiments: Global vs Hourly vs Block stratified."""
    output_dir = PROJECT_ROOT / 'reports' / 'optimized_search_v7'
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*70)
    logger.info("OPTIMIZED SEARCH V7 - Time-Block Transfer Learning")
    logger.info(f"Baseline to beat: {BASELINE_SMAPE}% sMAPE")
    logger.info("="*70)

    # Load data
    df = load_data()
    logger.info(f"Loaded data: {df.shape}")
    logger.info(f"Date range: {df.index.min()} to {df.index.max()}")

    # Show time block distribution in data
    logger.info("\n  TIME BLOCK DEFINITIONS:")
    for block_name in BLOCK_ORDER:
        hours = TIME_BLOCKS[block_name]
        logger.info(f"    {block_name:<10}: Hours {hours[0]:02d}-{hours[-1]:02d} ({len(hours)} hours)")

    # Prepare stratified data
    data = prepare_stratified_data(df, BASE_FEATURES)
    logger.info(f"Features: {len(data['features'])}")

    results = []

    # =========================================================================
    # MODEL A: GLOBAL (V5 Control)
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("MODEL A: GLOBAL (V5 Control)")
    logger.info("Global Base → Global Fine-Tune")
    logger.info("="*70)

    result_global = run_stratified_transfer_learning(data, strategy='global')
    result_global['model'] = 'A_Global'
    results.append(result_global)

    # =========================================================================
    # MODEL B: HOURLY STRATIFIED
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("MODEL B: HOURLY STRATIFIED")
    logger.info("Global Base → 24 Separate Hourly Models")
    logger.info("="*70)

    result_hourly = run_stratified_transfer_learning(data, strategy='hourly')
    result_hourly['model'] = 'B_Hourly'
    results.append(result_hourly)

    # =========================================================================
    # MODEL C: BLOCK STRATIFIED
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("MODEL C: BLOCK STRATIFIED")
    logger.info("Global Base → 5 Block Models (Night, Morning, Solar, Peak, Evening)")
    logger.info("="*70)

    result_block = run_stratified_transfer_learning(data, strategy='block')
    result_block['model'] = 'C_Block'
    results.append(result_block)

    # =========================================================================
    # FINAL COMPARISON
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("FINAL COMPARISON")
    logger.info("="*70)

    logger.info(f"\n{'Model':<20} {'Strategy':<15} {'sMAPE':>10} {'Bias':>12} {'Beat?':>8}")
    logger.info("-"*70)

    for r in sorted(results, key=lambda x: x['test_smape']):
        beat = "YES" if r['test_smape'] < BASELINE_SMAPE else "no"
        logger.info(f"{r['model']:<20} {r['strategy']:<15} {r['test_smape']:>9.2f}% {r['test_bias']:>+11.2f} {beat:>8}")

    # Best result
    best = min(results, key=lambda x: x['test_smape'])

    logger.info(f"\n{'='*70}")
    if best['test_smape'] < BASELINE_SMAPE:
        improvement = BASELINE_SMAPE - best['test_smape']
        logger.info(f"SUCCESS! Beat {BASELINE_SMAPE}% by {improvement:.2f}%!")
    else:
        logger.info(f"Best: {best['test_smape']:.2f}% (baseline: {BASELINE_SMAPE}%)")

    logger.info(f"  Best Model: {best['model']}")
    logger.info(f"  Strategy: {best['strategy']}")
    logger.info(f"  Test sMAPE: {best['test_smape']:.2f}%")
    logger.info(f"  Test Bias: {best['test_bias']:.2f} TL/MWh")

    # =========================================================================
    # DETAILED BLOCK ANALYSIS FOR BEST MODEL
    # =========================================================================
    logger.info(f"\n{'='*70}")
    logger.info(f"DETAILED BLOCK ANALYSIS ({best['model']})")
    logger.info("="*70)

    logger.info(f"\n{'Block':<12} {'sMAPE':>10} {'MAE':>10} {'Bias':>12} {'Count':>10}")
    logger.info("-"*60)

    block_data = []
    for block_name in BLOCK_ORDER:
        if block_name in best['block_metrics']:
            m = best['block_metrics'][block_name]
            logger.info(f"{block_name:<12} {m['smape']:>9.2f}% {m['mae']:>9.2f} {m['bias']:>+11.2f} {m['count']:>10}")
            block_data.append({
                'block': block_name,
                'smape': m['smape'],
                'mae': m['mae'],
                'bias': m['bias'],
                'count': m['count'],
            })

    # Compare block performance across strategies
    logger.info(f"\n{'='*70}")
    logger.info("BLOCK-BY-BLOCK COMPARISON (All Strategies)")
    logger.info("="*70)

    for block_name in BLOCK_ORDER:
        logger.info(f"\n  {block_name}:")
        for r in results:
            if block_name in r['block_metrics']:
                m = r['block_metrics'][block_name]
                logger.info(f"    {r['model']:<15}: sMAPE={m['smape']:.2f}%, Bias={m['bias']:+.2f}")

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    summary = {
        'timestamp': datetime.now().isoformat(),
        'baseline_smape': BASELINE_SMAPE,
        'best_model': best['model'],
        'best_strategy': best['strategy'],
        'best_test_smape': float(best['test_smape']),
        'best_bias': float(best['test_bias']),
        'all_results': [
            {k: v for k, v in r.items() if k not in ['predictions', 'block_metrics']}
            for r in results
        ],
        'best_block_metrics': best['block_metrics'],
    }

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Save best predictions
    pred_df = pd.DataFrame({
        'datetime': data['X_test'].index,
        'hour': data['hours_test'].values,
        'block': [get_time_block(h) for h in data['hours_test'].values],
        'y_true': data['y_test'].values,
        'y_pred': best['predictions'],
    })
    pred_df.to_csv(output_dir / 'best_predictions.csv', index=False)
    pred_df.to_parquet(output_dir / 'best_predictions.parquet', index=False)

    # Save block performance
    block_df = pd.DataFrame(block_data)
    block_df.to_csv(output_dir / 'block_performance.csv', index=False)

    logger.info(f"\nResults saved to: {output_dir}")

    return results


if __name__ == "__main__":
    run_v7_experiments()
