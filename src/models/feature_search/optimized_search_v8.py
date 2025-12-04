"""
Optimized Search V8 - Morning Doctor (Residual Correction for Sunrise Ramp)
============================================================================
Context: V7 revealed that the Morning Block (07:00-10:00) is the bottleneck:
         - Morning sMAPE: 30.39% with +41 TL bias (terrible!)
         - Peak sMAPE: 6.24% (excellent!)
         - The model overpredicts mornings because hour=8 means HIGH price in
           winter but LOW price in summer (sunrise ramp timing varies).

Objective: Break 13.5% sMAPE by implementing a Residual Correction Model
           specifically for the Morning Block.

Strategy:
    Stage 1: Train V5 Champion (Base Model)
    Stage 2: Train "Morning Doctor" XGBoost on residuals (hours 7-10 only)
    Stage 3: Apply correction: Final_Pred = Base_Pred + Correction (mornings only)

Experiments:
    A. Feature Injection: Add slope features to base model
    B. Bias Subtraction: Simple hourly mean error correction
    C. Residual Boosting: Full ML residual correction

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

# V5 Winner baseline
BASELINE_SMAPE = 14.08

# Morning hours that need fixing
MORNING_HOURS = [7, 8, 9, 10]


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
# SLOPE FEATURE ENGINEERING (Ramp Detection)
# =============================================================================

def add_slope_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Add slope features for detecting ramp dynamics.

    Key features:
    - load_slope: Change in consumption forecast (demand ramp)
    - renewable_slope: Change in renewable saturation (solar ramp)
    - solar_proxy: Hour * month interaction (seasonal sunrise proxy)
    """
    df = df.copy()
    new_features = []

    # Load slope (demand ramp-up)
    if 'consumption_forecast' in df.columns:
        df['load_slope'] = df['consumption_forecast'].diff().fillna(0)
        new_features.append('load_slope')

    # Renewable slope (solar ramp-up)
    if 'renewable_saturation' in df.columns:
        df['renewable_slope'] = df['renewable_saturation'].diff().fillna(0)
        new_features.append('renewable_slope')

    # Thermal gap slope
    if 'thermal_gap' in df.columns:
        df['thermal_slope'] = df['thermal_gap'].diff().fillna(0)
        new_features.append('thermal_slope')

    # Solar proxy: interaction of hour and month for seasonal sunrise timing
    if 'hour_sin' in df.columns:
        # Month from index
        month = df.index.month
        month_cos = np.cos(2 * np.pi * month / 12)
        df['solar_proxy'] = df['hour_sin'] * month_cos
        new_features.append('solar_proxy')

        # Also add month sin/cos directly
        df['month_sin'] = np.sin(2 * np.pi * month / 12)
        df['month_cos'] = month_cos
        new_features.extend(['month_sin', 'month_cos'])

    # Hour as integer (useful for residual model)
    df['hour'] = df.index.hour
    new_features.append('hour')

    # Month as integer
    df['month'] = df.index.month
    new_features.append('month')

    logger.info(f"  Added {len(new_features)} slope/ramp features")

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


def prepare_v5_style_data(df: pd.DataFrame, features: List[str],
                          add_slopes: bool = False,
                          finetune_months: int = 6,
                          test_start: str = '2024-06-01') -> Dict:
    """
    Prepare data with V5-style split:
        - Base Train: Everything before fine-tune window
        - Fine-Tune: Last N months before test
        - Test: test_start onwards
    """
    available = [f for f in features if f in df.columns]

    X = df[available].copy()
    y = df['price_real'].copy()

    # Add slope features if requested
    slope_features = []
    if add_slopes:
        X, slope_features = add_slope_features(X)

    # Drop NaN
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]

    # Get hours for stratified analysis
    hours = X.index.hour

    # Timezone-aware splits
    tz = X.index.tz
    test_start_dt = pd.Timestamp(test_start, tz=tz)

    # Test mask
    test_mask = X.index >= test_start_dt

    # Fine-tune window: last N months before test
    pretrain_data = X[~test_mask]
    finetune_start = pretrain_data.index.max() - pd.DateOffset(months=finetune_months)
    finetune_mask = (~test_mask) & (X.index >= finetune_start)
    base_mask = (~test_mask) & (X.index < finetune_start)

    logger.info(f"\n  DATA SPLITS (V5-style):")
    logger.info(f"    Base Train:  {base_mask.sum():,} rows")
    logger.info(f"    Fine-Tune:   {finetune_mask.sum():,} rows ({X[finetune_mask].index.min()} to {X[finetune_mask].index.max()})")
    logger.info(f"    Test:        {test_mask.sum():,} rows ({X[test_mask].index.min()} to {X[test_mask].index.max()})")

    return {
        'X_base': X[base_mask],
        'y_base': y[base_mask],
        'X_finetune': X[finetune_mask],
        'y_finetune': y[finetune_mask],
        'hours_finetune': hours[finetune_mask],
        'X_test': X[test_mask],
        'y_test': y[test_mask],
        'hours_test': hours[test_mask],
        'features': list(X.columns),
        'slope_features': slope_features,
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


def evaluate_morning_vs_other(y_true: np.ndarray, y_pred: np.ndarray, hours: np.ndarray) -> Dict:
    """Evaluate separately for Morning hours vs Rest."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    hours = np.array(hours)

    morning_mask = np.isin(hours, MORNING_HOURS)
    other_mask = ~morning_mask

    global_metrics = evaluate(y_true, y_pred)

    morning_metrics = evaluate(y_true[morning_mask], y_pred[morning_mask]) if morning_mask.sum() > 0 else {}
    morning_metrics['count'] = morning_mask.sum()

    other_metrics = evaluate(y_true[other_mask], y_pred[other_mask]) if other_mask.sum() > 0 else {}
    other_metrics['count'] = other_mask.sum()

    return {
        'global': global_metrics,
        'morning': morning_metrics,
        'other': other_metrics,
    }


# =============================================================================
# V5-STYLE BASE MODEL TRAINING
# =============================================================================

def train_v5_base_model(data: Dict) -> Tuple[object, np.ndarray, np.ndarray]:
    """
    Train V5-style transfer learning model.

    Returns: (model, finetune_predictions, test_predictions)
    """
    from catboost import CatBoostRegressor

    logger.info("\n  Stage 1: Training V5 Base Model...")

    # Step A: Train on base data
    split_idx = int(len(data['X_base']) * 0.85)
    X_base_tr = data['X_base'].iloc[:split_idx]
    X_base_vl = data['X_base'].iloc[split_idx:]
    y_base_tr = data['y_base'].iloc[:split_idx]
    y_base_vl = data['y_base'].iloc[split_idx:]

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
    base_model.fit(X_base_tr, y_base_tr, eval_set=(X_base_vl, y_base_vl), verbose=False)
    logger.info(f"    Base model: {base_model.tree_count_} trees")

    # Step B: Fine-tune on recent data
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

    # Generate predictions
    finetune_pred = finetune_model.predict(X_ft)
    test_pred = finetune_model.predict(data['X_test'])

    return finetune_model, finetune_pred, test_pred


# =============================================================================
# EXPERIMENT A: FEATURE INJECTION (Slope Features in Base Model)
# =============================================================================

def run_experiment_a_feature_injection(df: pd.DataFrame) -> Dict:
    """
    Experiment A: Add slope features to the base V5 model.
    Hypothesis: The global model can learn the ramp dynamics with better features.
    """
    logger.info("\n" + "="*70)
    logger.info("EXPERIMENT A: FEATURE INJECTION")
    logger.info("Add slope features to base V5 model")
    logger.info("="*70)

    # Prepare data WITH slope features
    data = prepare_v5_style_data(df, BASE_FEATURES, add_slopes=True)

    # Train V5 model with slope features
    model, ft_pred, test_pred = train_v5_base_model(data)

    # Evaluate
    metrics = evaluate_morning_vs_other(
        data['y_test'].values, test_pred, data['hours_test'].values
    )

    logger.info(f"\n  RESULTS (Feature Injection):")
    logger.info(f"    Global sMAPE:  {metrics['global']['smape']:.2f}%")
    logger.info(f"    Morning sMAPE: {metrics['morning']['smape']:.2f}% (n={metrics['morning']['count']})")
    logger.info(f"    Other sMAPE:   {metrics['other']['smape']:.2f}% (n={metrics['other']['count']})")
    logger.info(f"    Morning Bias:  {metrics['morning']['bias']:+.2f} TL")

    return {
        'experiment': 'A_FeatureInjection',
        'test_pred': test_pred,
        'metrics': metrics,
        'data': data,
    }


# =============================================================================
# EXPERIMENT B: BIAS SUBTRACTION (Simple Rule-Based)
# =============================================================================

def run_experiment_b_bias_subtraction(df: pd.DataFrame) -> Dict:
    """
    Experiment B: Simple hourly bias correction.
    Calculate mean error per hour (7,8,9,10) in fine-tune set and subtract from test.
    """
    logger.info("\n" + "="*70)
    logger.info("EXPERIMENT B: BIAS SUBTRACTION")
    logger.info("Simple rule-based hourly correction")
    logger.info("="*70)

    # Prepare data WITHOUT slope features (standard V5)
    data = prepare_v5_style_data(df, BASE_FEATURES, add_slopes=False)

    # Train V5 model
    model, ft_pred, test_pred = train_v5_base_model(data)

    # Calculate hourly bias in fine-tune set
    ft_residuals = ft_pred - data['y_finetune'].values
    ft_hours = data['hours_finetune'].values

    hourly_bias = {}
    for hour in MORNING_HOURS:
        mask = (ft_hours == hour)
        if mask.sum() > 0:
            hourly_bias[hour] = ft_residuals[mask].mean()
            logger.info(f"    Hour {hour:02d} bias: {hourly_bias[hour]:+.2f} TL (n={mask.sum()})")

    # Apply correction to test set
    corrected_pred = test_pred.copy()
    test_hours = data['hours_test'].values

    for hour, bias in hourly_bias.items():
        mask = (test_hours == hour)
        corrected_pred[mask] -= bias
        logger.info(f"    Applied -{bias:.2f} TL to {mask.sum()} test samples at hour {hour}")

    # Evaluate before and after
    metrics_before = evaluate_morning_vs_other(
        data['y_test'].values, test_pred, test_hours
    )
    metrics_after = evaluate_morning_vs_other(
        data['y_test'].values, corrected_pred, test_hours
    )

    logger.info(f"\n  RESULTS (Bias Subtraction):")
    logger.info(f"    BEFORE:")
    logger.info(f"      Global sMAPE:  {metrics_before['global']['smape']:.2f}%")
    logger.info(f"      Morning sMAPE: {metrics_before['morning']['smape']:.2f}%")
    logger.info(f"      Morning Bias:  {metrics_before['morning']['bias']:+.2f} TL")
    logger.info(f"    AFTER:")
    logger.info(f"      Global sMAPE:  {metrics_after['global']['smape']:.2f}%")
    logger.info(f"      Morning sMAPE: {metrics_after['morning']['smape']:.2f}%")
    logger.info(f"      Morning Bias:  {metrics_after['morning']['bias']:+.2f} TL")

    return {
        'experiment': 'B_BiasSubtraction',
        'test_pred': corrected_pred,
        'test_pred_uncorrected': test_pred,
        'metrics_before': metrics_before,
        'metrics_after': metrics_after,
        'hourly_bias': hourly_bias,
        'data': data,
    }


# =============================================================================
# EXPERIMENT C: RESIDUAL BOOSTING (Morning Doctor)
# =============================================================================

def run_experiment_c_residual_boosting(df: pd.DataFrame) -> Dict:
    """
    Experiment C: Full ML residual correction (Morning Doctor).
    Train XGBoost on morning residuals to learn complex correction patterns.
    """
    logger.info("\n" + "="*70)
    logger.info("EXPERIMENT C: RESIDUAL BOOSTING (Morning Doctor)")
    logger.info("Train XGBoost on morning residuals")
    logger.info("="*70)

    import xgboost as xgb

    # Prepare data WITH slope features (needed for residual model)
    data = prepare_v5_style_data(df, BASE_FEATURES, add_slopes=True)

    # Train V5 base model
    model, ft_pred, test_pred = train_v5_base_model(data)

    # Stage 2: Train Morning Doctor on residuals
    logger.info("\n  Stage 2: Training Morning Doctor...")

    # Calculate residuals (what the model got wrong)
    ft_residuals = data['y_finetune'].values - ft_pred  # True - Pred (positive = underpredicted)

    # Filter to morning hours only
    ft_hours = data['hours_finetune'].values
    morning_mask_ft = np.isin(ft_hours, MORNING_HOURS)

    X_morning_ft = data['X_finetune'][morning_mask_ft]
    residuals_morning = ft_residuals[morning_mask_ft]

    logger.info(f"    Morning training samples: {morning_mask_ft.sum()}")
    logger.info(f"    Mean morning residual: {residuals_morning.mean():.2f} TL")
    logger.info(f"    Std morning residual: {residuals_morning.std():.2f} TL")

    # Features for the residual model
    residual_features = [
        'load_slope', 'renewable_slope', 'thermal_slope',
        'thermal_gap', 'hour', 'month', 'solar_proxy',
        'month_sin', 'month_cos', 'hour_sin', 'hour_cos',
        'renewable_saturation', 'load_factor',
    ]
    available_residual_features = [f for f in residual_features if f in X_morning_ft.columns]
    logger.info(f"    Residual model features: {len(available_residual_features)}")

    X_residual = X_morning_ft[available_residual_features]

    # Train XGBoost residual model
    split_idx = int(len(X_residual) * 0.8)

    doctor = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
        early_stopping_rounds=50,
    )

    doctor.fit(
        X_residual.iloc[:split_idx], residuals_morning[:split_idx],
        eval_set=[(X_residual.iloc[split_idx:], residuals_morning[split_idx:])],
        verbose=False
    )
    logger.info(f"    Morning Doctor trained ({doctor.best_iteration} trees)")

    # Stage 3: Apply correction to test set
    logger.info("\n  Stage 3: Applying Morning Surgery...")

    test_hours = data['hours_test'].values
    morning_mask_test = np.isin(test_hours, MORNING_HOURS)

    X_morning_test = data['X_test'][morning_mask_test][available_residual_features]

    # Predict corrections for morning hours
    corrections = np.zeros(len(test_pred))
    morning_corrections = doctor.predict(X_morning_test)
    corrections[morning_mask_test] = morning_corrections

    logger.info(f"    Morning test samples: {morning_mask_test.sum()}")
    logger.info(f"    Mean correction: {morning_corrections.mean():.2f} TL")
    logger.info(f"    Correction range: [{morning_corrections.min():.2f}, {morning_corrections.max():.2f}] TL")

    # Apply correction: Final = Base + Correction
    corrected_pred = test_pred + corrections

    # Evaluate before and after
    metrics_before = evaluate_morning_vs_other(
        data['y_test'].values, test_pred, test_hours
    )
    metrics_after = evaluate_morning_vs_other(
        data['y_test'].values, corrected_pred, test_hours
    )

    logger.info(f"\n  RESULTS (Residual Boosting):")
    logger.info(f"    BEFORE:")
    logger.info(f"      Global sMAPE:  {metrics_before['global']['smape']:.2f}%")
    logger.info(f"      Morning sMAPE: {metrics_before['morning']['smape']:.2f}%")
    logger.info(f"      Morning Bias:  {metrics_before['morning']['bias']:+.2f} TL")
    logger.info(f"    AFTER:")
    logger.info(f"      Global sMAPE:  {metrics_after['global']['smape']:.2f}%")
    logger.info(f"      Morning sMAPE: {metrics_after['morning']['smape']:.2f}%")
    logger.info(f"      Morning Bias:  {metrics_after['morning']['bias']:+.2f} TL")

    # Feature importance
    importance = pd.DataFrame({
        'feature': available_residual_features,
        'importance': doctor.feature_importances_
    }).sort_values('importance', ascending=False)

    logger.info(f"\n  Morning Doctor Feature Importance:")
    for _, row in importance.head(5).iterrows():
        logger.info(f"    {row['feature']:<20}: {row['importance']:.3f}")

    return {
        'experiment': 'C_ResidualBoosting',
        'test_pred': corrected_pred,
        'test_pred_uncorrected': test_pred,
        'metrics_before': metrics_before,
        'metrics_after': metrics_after,
        'corrections': corrections,
        'doctor': doctor,
        'data': data,
    }


# =============================================================================
# MAIN EXPERIMENT LOOP
# =============================================================================

def run_v8_experiments():
    """Run all V8 experiments: Morning Doctor approaches."""
    output_dir = PROJECT_ROOT / 'reports' / 'optimized_search_v8'
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*70)
    logger.info("OPTIMIZED SEARCH V8 - Morning Doctor (Residual Correction)")
    logger.info(f"Baseline to beat: {BASELINE_SMAPE}% sMAPE")
    logger.info(f"Morning hours to fix: {MORNING_HOURS}")
    logger.info("="*70)

    # Load data
    df = load_data()
    logger.info(f"Loaded data: {df.shape}")
    logger.info(f"Date range: {df.index.min()} to {df.index.max()}")

    results = []

    # Run experiments
    result_a = run_experiment_a_feature_injection(df)
    results.append(result_a)

    result_b = run_experiment_b_bias_subtraction(df)
    results.append(result_b)

    result_c = run_experiment_c_residual_boosting(df)
    results.append(result_c)

    # =========================================================================
    # FINAL COMPARISON
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("FINAL COMPARISON")
    logger.info("="*70)

    # Build comparison table
    comparison = []

    # Experiment A
    m = result_a['metrics']
    comparison.append({
        'Experiment': 'A_FeatureInjection',
        'Global_sMAPE': m['global']['smape'],
        'Morning_sMAPE': m['morning']['smape'],
        'Morning_Bias': m['morning']['bias'],
        'Other_sMAPE': m['other']['smape'],
    })

    # Experiment B (after correction)
    m = result_b['metrics_after']
    comparison.append({
        'Experiment': 'B_BiasSubtraction',
        'Global_sMAPE': m['global']['smape'],
        'Morning_sMAPE': m['morning']['smape'],
        'Morning_Bias': m['morning']['bias'],
        'Other_sMAPE': m['other']['smape'],
    })

    # Experiment C (after correction)
    m = result_c['metrics_after']
    comparison.append({
        'Experiment': 'C_ResidualBoosting',
        'Global_sMAPE': m['global']['smape'],
        'Morning_sMAPE': m['morning']['smape'],
        'Morning_Bias': m['morning']['bias'],
        'Other_sMAPE': m['other']['smape'],
    })

    logger.info(f"\n{'Experiment':<25} {'Global':>10} {'Morning':>10} {'Bias':>10} {'Other':>10} {'Beat?':>8}")
    logger.info("-"*80)

    for c in sorted(comparison, key=lambda x: x['Global_sMAPE']):
        beat = "YES" if c['Global_sMAPE'] < BASELINE_SMAPE else "no"
        logger.info(f"{c['Experiment']:<25} {c['Global_sMAPE']:>9.2f}% {c['Morning_sMAPE']:>9.2f}% {c['Morning_Bias']:>+9.2f} {c['Other_sMAPE']:>9.2f}% {beat:>8}")

    # Best result
    best = min(comparison, key=lambda x: x['Global_sMAPE'])

    logger.info(f"\n{'='*70}")
    if best['Global_sMAPE'] < BASELINE_SMAPE:
        improvement = BASELINE_SMAPE - best['Global_sMAPE']
        logger.info(f"SUCCESS! Beat {BASELINE_SMAPE}% by {improvement:.2f}%!")
    else:
        logger.info(f"Best: {best['Global_sMAPE']:.2f}% (baseline: {BASELINE_SMAPE}%)")

    logger.info(f"  Best Experiment: {best['Experiment']}")
    logger.info(f"  Global sMAPE: {best['Global_sMAPE']:.2f}%")
    logger.info(f"  Morning sMAPE: {best['Morning_sMAPE']:.2f}%")
    logger.info(f"  Morning Bias: {best['Morning_Bias']:.2f} TL")

    # =========================================================================
    # BEFORE/AFTER ANALYSIS
    # =========================================================================
    logger.info(f"\n{'='*70}")
    logger.info("BEFORE/AFTER ANALYSIS (Morning Block)")
    logger.info("="*70)

    # Experiment B
    logger.info(f"\n  Experiment B (Bias Subtraction):")
    logger.info(f"    Morning sMAPE: {result_b['metrics_before']['morning']['smape']:.2f}% → {result_b['metrics_after']['morning']['smape']:.2f}%")
    logger.info(f"    Morning Bias:  {result_b['metrics_before']['morning']['bias']:+.2f} → {result_b['metrics_after']['morning']['bias']:+.2f} TL")

    # Experiment C
    logger.info(f"\n  Experiment C (Residual Boosting):")
    logger.info(f"    Morning sMAPE: {result_c['metrics_before']['morning']['smape']:.2f}% → {result_c['metrics_after']['morning']['smape']:.2f}%")
    logger.info(f"    Morning Bias:  {result_c['metrics_before']['morning']['bias']:+.2f} → {result_c['metrics_after']['morning']['bias']:+.2f} TL")

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    # Find the result with best global sMAPE
    best_result = min(results, key=lambda x: (
        x['metrics']['global']['smape'] if 'metrics' in x
        else x['metrics_after']['global']['smape']
    ))

    best_pred = best_result['test_pred']
    best_data = best_result['data']

    summary = {
        'timestamp': datetime.now().isoformat(),
        'baseline_smape': BASELINE_SMAPE,
        'morning_hours': MORNING_HOURS,
        'best_experiment': best['Experiment'],
        'best_global_smape': float(best['Global_sMAPE']),
        'best_morning_smape': float(best['Morning_sMAPE']),
        'best_morning_bias': float(best['Morning_Bias']),
        'comparison': comparison,
    }

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Save best predictions
    pred_df = pd.DataFrame({
        'datetime': best_data['X_test'].index,
        'hour': best_data['hours_test'].values,
        'is_morning': np.isin(best_data['hours_test'].values, MORNING_HOURS),
        'y_true': best_data['y_test'].values,
        'y_pred': best_pred,
    })
    pred_df.to_csv(output_dir / 'best_predictions.csv', index=False)
    pred_df.to_parquet(output_dir / 'best_predictions.parquet', index=False)

    logger.info(f"\nResults saved to: {output_dir}")

    return results


if __name__ == "__main__":
    run_v8_experiments()
