"""
Optimized Search V3 - Log-Target + Stacking + Hybrid Features
==============================================================
Fixes V2 issues:
1. Inverse weighting destabilized training -> Use log-target instead
2. Lost stacking architecture -> Restore meta-learner
3. Combine best of V1 (interactions) + V2 (physics) features

Target: Beat 15.96% sMAPE

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
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings('ignore')
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

TARGET_SMAPE = 15.96  # Baseline to beat


# =============================================================================
# HYBRID FEATURE ENGINEERING (V1 Interactions + V2 Physics)
# =============================================================================

def add_hybrid_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add hybrid features combining:
    - V1 winning interaction features
    - V2 physics-based features (without destabilizing weights)
    """
    logger.info("\n" + "="*60)
    logger.info("HYBRID FEATURE ENGINEERING (V1 + V2)")
    logger.info("="*60)

    df = df.copy()
    eps = 1e-6

    # =========================================================================
    # V1 INTERACTION FEATURES (from winning 15.96% model)
    # =========================================================================
    logger.info("\n  V1 Interaction Features:")

    if 'price_ptf_lag_24h' in df.columns and 'thermal_gap' in df.columns:
        df['price_ptf_lag_24h_x_thermal_gap'] = df['price_ptf_lag_24h'] * df['thermal_gap']
        logger.info("    + price_ptf_lag_24h_x_thermal_gap")

    if 'price_ptf_rolling_std_24h' in df.columns and 'renewable_saturation' in df.columns:
        df['price_ptf_rolling_std_24h_x_renewable_saturation'] = (
            df['price_ptf_rolling_std_24h'] * df['renewable_saturation']
        )
        logger.info("    + price_ptf_rolling_std_24h_x_renewable_saturation")

    if 'load_factor' in df.columns and 'reserve_margin_ratio' in df.columns:
        df['load_factor_x_reserve_margin_ratio'] = df['load_factor'] * df['reserve_margin_ratio']
        logger.info("    + load_factor_x_reserve_margin_ratio")

    if 'hour_sin' in df.columns and 'price_ptf_lag_24h' in df.columns:
        df['hour_sin_x_price_ptf_lag_24h'] = df['hour_sin'] * df['price_ptf_lag_24h']
        logger.info("    + hour_sin_x_price_ptf_lag_24h")

    if 'system_short_signal' in df.columns and 'thermal_gap' in df.columns:
        df['system_short_signal_x_thermal_gap'] = df['system_short_signal'] * df['thermal_gap']
        logger.info("    + system_short_signal_x_thermal_gap")

    # =========================================================================
    # V2 PHYSICS FEATURES (stabilized - no inverse weighting)
    # =========================================================================
    logger.info("\n  V2 Physics Features (stabilized):")

    # Scarcity index - clip to prevent explosion
    if 'reserve_margin_ratio' in df.columns:
        rm = df['reserve_margin_ratio'].clip(lower=0.05)  # Floor at 5%
        df['scarcity_index'] = 1.0 / rm
        df['scarcity_index'] = df['scarcity_index'].clip(upper=20)  # Cap at 20
        logger.info("    + scarcity_index (clipped)")

    # Thermal stress polynomials - normalized
    if 'thermal_gap' in df.columns:
        thermal = df['thermal_gap'].fillna(0)
        thermal_std = thermal.std() + eps
        thermal_norm = thermal / thermal_std
        df['thermal_stress_sq'] = thermal_norm ** 2
        df['thermal_stress_cb'] = thermal_norm ** 3
        logger.info("    + thermal_stress_sq, thermal_stress_cb (normalized)")

    # Net load stress - clipped
    if 'load_factor' in df.columns and 'renewable_saturation' in df.columns:
        rs = df['renewable_saturation'].clip(lower=0.05)  # Floor at 5%
        df['net_load_stress'] = df['load_factor'] / rs
        df['net_load_stress'] = df['net_load_stress'].clip(upper=20)
        logger.info("    + net_load_stress (clipped)")

    # Price coefficient of variation
    if 'price_ptf_rolling_std_24h' in df.columns and 'price_ptf_rolling_mean_24h' in df.columns:
        mean = df['price_ptf_rolling_mean_24h'].clip(lower=10)  # Floor at 10
        df['price_cv'] = df['price_ptf_rolling_std_24h'] / mean
        df['price_cv'] = df['price_cv'].clip(upper=5)
        logger.info("    + price_cv (clipped)")

    # =========================================================================
    # HANDLE INF AND NAN
    # =========================================================================
    for col in df.columns:
        if df[col].dtype in [np.float64, np.float32]:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())

    logger.info(f"\n  Total features: {len(df.columns)}")

    return df


# =============================================================================
# LOG-TARGET TRANSFORMATION
# =============================================================================

def log_transform_target(y: pd.Series) -> pd.Series:
    """
    Apply log1p transformation to target.

    This naturally handles the "hockey stick" non-linearity:
    - Compresses high price variance
    - Makes sMAPE optimization more stable
    - No exploding weights near zero
    """
    return np.log1p(y)


def inverse_log_transform(y_log: np.ndarray) -> np.ndarray:
    """Inverse log1p transformation."""
    return np.expm1(y_log)


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_catboost(X_train, y_train, X_val, y_val, config):
    """Train CatBoost model."""
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
    """Train LightGBM model."""
    import lightgbm as lgb

    model = lgb.LGBMRegressor(
        objective='regression_l1',
        verbosity=-1,
        random_state=42,
        **config
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    return model


def train_xgboost(X_train, y_train, X_val, y_val, config):
    """Train XGBoost model."""
    import xgboost as xgb

    model = xgb.XGBRegressor(
        objective='reg:absoluteerror',
        random_state=42,
        verbosity=0,
        early_stopping_rounds=50,
        **config
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    return model


# =============================================================================
# STACKING ENSEMBLE (Restored from V1)
# =============================================================================

def run_stacking_ensemble(
    X_train_full: pd.DataFrame,
    y_train_full: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    use_log_target: bool = True,
    n_splits: int = 5
) -> Tuple[Dict, np.ndarray]:
    """
    Run stacking ensemble with out-of-fold predictions.

    This is the architecture that achieved 15.96% - restored with hybrid features.

    Args:
        X_train_full: Full training features (train + val combined)
        y_train_full: Full training target
        X_test: Test features
        y_test: Test target
        use_log_target: Whether to train on log-transformed target
        n_splits: Number of CV folds

    Returns:
        Tuple of (results_dict, test_predictions)
    """
    logger.info("\n" + "="*60)
    logger.info("STACKING ENSEMBLE (Restored Architecture)")
    logger.info("="*60)
    logger.info(f"  Log-target: {use_log_target}")
    logger.info(f"  CV folds: {n_splits}")

    # Transform target if requested
    if use_log_target:
        y_train_model = log_transform_target(y_train_full)
        logger.info("  Target: log1p(price_real)")
    else:
        y_train_model = y_train_full

    # Base model configs
    base_configs = {
        'catboost': {
            'iterations': 2000,
            'depth': 10,
            'learning_rate': 0.02,
            'l2_leaf_reg': 3,
            'bagging_temperature': 0.3,
        },
        'lightgbm': {
            'n_estimators': 1500,
            'max_depth': 10,
            'learning_rate': 0.02,
            'num_leaves': 63,
            'min_child_samples': 30,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
        },
        'xgboost': {
            'n_estimators': 1000,
            'max_depth': 8,
            'learning_rate': 0.03,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        },
    }

    # Initialize OOF predictions
    oof_preds = {name: np.zeros(len(X_train_full)) for name in base_configs}
    test_preds = {name: np.zeros(len(X_test)) for name in base_configs}

    # K-Fold CV for OOF predictions
    kf = KFold(n_splits=n_splits, shuffle=False)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_full)):
        logger.info(f"\n  Fold {fold + 1}/{n_splits}")

        X_tr = X_train_full.iloc[train_idx]
        y_tr = y_train_model.iloc[train_idx]
        X_vl = X_train_full.iloc[val_idx]
        y_vl = y_train_model.iloc[val_idx]

        for name, config in base_configs.items():
            if name == 'catboost':
                model = train_catboost(X_tr, y_tr, X_vl, y_vl, config)
            elif name == 'lightgbm':
                model = train_lightgbm(X_tr, y_tr, X_vl, y_vl, config)
            elif name == 'xgboost':
                model = train_xgboost(X_tr, y_tr, X_vl, y_vl, config)

            # OOF predictions
            oof_preds[name][val_idx] = model.predict(X_vl)
            # Test predictions (averaged across folds)
            test_preds[name] += model.predict(X_test) / n_splits

        gc.collect()

    # Transform OOF predictions back if log-target was used
    if use_log_target:
        oof_preds_orig = {name: inverse_log_transform(preds) for name, preds in oof_preds.items()}
        test_preds_orig = {name: inverse_log_transform(preds) for name, preds in test_preds.items()}
    else:
        oof_preds_orig = oof_preds
        test_preds_orig = test_preds

    # =========================================================================
    # BUILD META-FEATURES
    # =========================================================================
    logger.info("\n  Building meta-features...")

    # Meta-train: OOF predictions + original features
    meta_train = pd.DataFrame(oof_preds_orig)
    meta_train = pd.concat([meta_train, X_train_full.reset_index(drop=True)], axis=1)

    # Meta-test: Averaged test predictions + original features
    meta_test = pd.DataFrame(test_preds_orig)
    meta_test = pd.concat([meta_test, X_test.reset_index(drop=True)], axis=1)

    logger.info(f"  Meta-features shape: {meta_train.shape}")

    # =========================================================================
    # TRAIN META-MODEL
    # =========================================================================
    logger.info("\n  Training meta-model...")

    # Split meta-train into train/val for meta-model
    n_meta = len(meta_train)
    meta_train_end = int(n_meta * 0.8)

    meta_X_train = meta_train.iloc[:meta_train_end]
    meta_y_train = y_train_full.iloc[:meta_train_end]
    meta_X_val = meta_train.iloc[meta_train_end:]
    meta_y_val = y_train_full.iloc[meta_train_end:]

    # Train meta-model on original scale (not log)
    meta_config = {
        'iterations': 1500,
        'depth': 8,
        'learning_rate': 0.03,
        'l2_leaf_reg': 5,
        'bagging_temperature': 0.2,
    }
    meta_model = train_catboost(meta_X_train, meta_y_train, meta_X_val, meta_y_val, meta_config)

    # =========================================================================
    # PREDICTIONS AND EVALUATION
    # =========================================================================
    val_pred = meta_model.predict(meta_X_val)
    test_pred = meta_model.predict(meta_test)

    val_metrics = evaluate(meta_y_val.values, val_pred)
    test_metrics = evaluate(y_test.values, test_pred)
    gap = test_metrics['smape'] / val_metrics['smape'] if val_metrics['smape'] > 0 else 0

    result = {
        'model': 'stacking_ensemble',
        'val_smape': val_metrics['smape'],
        'test_smape': test_metrics['smape'],
        'val_mae': val_metrics['mae'],
        'test_mae': test_metrics['mae'],
        'gap': gap,
        'use_log_target': use_log_target,
    }

    beat = " BEAT!" if test_metrics['smape'] < TARGET_SMAPE else ""
    logger.info(f"\n  Stacking Result:")
    logger.info(f"    Val sMAPE:  {val_metrics['smape']:.2f}%")
    logger.info(f"    Test sMAPE: {test_metrics['smape']:.2f}%{beat}")
    logger.info(f"    Gap: {gap:.2f}x")

    return result, test_pred


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate metrics."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae = np.mean(np.abs(y_true - y_pred))
    smape = 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

    return {'mae': mae, 'smape': smape}


# =============================================================================
# FEATURE SETS
# =============================================================================

# Base features from winning V1 model
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

# V1 interaction features
V1_INTERACTION_FEATURES = [
    'price_ptf_lag_24h_x_thermal_gap',
    'price_ptf_rolling_std_24h_x_renewable_saturation',
    'load_factor_x_reserve_margin_ratio',
    'hour_sin_x_price_ptf_lag_24h',
    'system_short_signal_x_thermal_gap',
]

# V2 physics features (stabilized)
V2_PHYSICS_FEATURES = [
    'scarcity_index',
    'thermal_stress_sq',
    'thermal_stress_cb',
    'net_load_stress',
    'price_cv',
]


def load_data() -> pd.DataFrame:
    """Load master dataset."""
    path = PROJECT_ROOT / 'data' / 'gold' / 'master' / 'master_v2_fundamental.parquet'
    return pd.read_parquet(path)


def prepare_data(
    df: pd.DataFrame,
    features: List[str],
    test_size: float = 0.2,
    val_size: float = 0.2
) -> Dict:
    """Prepare train/val/test data."""
    available = [f for f in features if f in df.columns]

    X = df[available].copy()
    y = df['price_real'].copy()

    # Drop NaN
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]

    # Time-based split
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


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_v3_pipeline():
    """
    Run V3 pipeline with:
    1. Hybrid features (V1 interactions + V2 physics)
    2. Log-target transformation
    3. Stacking meta-learner
    """
    output_dir = PROJECT_ROOT / 'reports' / 'optimized_search_v3'
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*70)
    logger.info("OPTIMIZED SEARCH V3")
    logger.info("Log-Target + Stacking + Hybrid Features")
    logger.info("="*70)
    logger.info(f"Baseline to beat: {TARGET_SMAPE}% sMAPE")

    # 1. Load data
    df = load_data()
    logger.info(f"\nLoaded data: {df.shape}")

    # 2. Add hybrid features
    df = add_hybrid_features(df)

    # 3. Combine all feature sets
    all_features = BASE_FEATURES + V1_INTERACTION_FEATURES + V2_PHYSICS_FEATURES
    all_features = [f for f in all_features if f in df.columns]

    # 4. Prepare data
    data = prepare_data(df, all_features)
    logger.info(f"\nFeatures: {len(data['features'])}")
    logger.info(f"Train: {len(data['X_train'])}, Val: {len(data['X_val'])}, Test: {len(data['X_test'])}")

    # Combine train + val for stacking
    X_train_full = pd.concat([data['X_train'], data['X_val']], axis=0)
    y_train_full = pd.concat([data['y_train'], data['y_val']], axis=0)

    results = []

    # =========================================================================
    # EXPERIMENT 1: Stacking with Log-Target (V3 main strategy)
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("EXPERIMENT 1: Stacking + Log-Target")
    logger.info("="*70)

    result_log, pred_log = run_stacking_ensemble(
        X_train_full, y_train_full,
        data['X_test'], data['y_test'],
        use_log_target=True
    )
    result_log['experiment'] = 'stacking_log_target'
    results.append(result_log)

    # =========================================================================
    # EXPERIMENT 2: Stacking without Log-Target (comparison)
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("EXPERIMENT 2: Stacking + Original Target")
    logger.info("="*70)

    result_orig, pred_orig = run_stacking_ensemble(
        X_train_full, y_train_full,
        data['X_test'], data['y_test'],
        use_log_target=False
    )
    result_orig['experiment'] = 'stacking_original_target'
    results.append(result_orig)

    # =========================================================================
    # EXPERIMENT 3: Simple base models (no stacking, for comparison)
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("EXPERIMENT 3: Base Models (no stacking)")
    logger.info("="*70)

    base_configs = {
        'catboost_base': {
            'iterations': 2000,
            'depth': 10,
            'learning_rate': 0.02,
            'l2_leaf_reg': 3,
        },
    }

    for name, config in base_configs.items():
        logger.info(f"\n  Training {name}...")
        model = train_catboost(
            data['X_train'], data['y_train'],
            data['X_val'], data['y_val'],
            config
        )

        val_pred = model.predict(data['X_val'])
        test_pred = model.predict(data['X_test'])

        val_metrics = evaluate(data['y_val'].values, val_pred)
        test_metrics = evaluate(data['y_test'].values, test_pred)
        gap = test_metrics['smape'] / val_metrics['smape'] if val_metrics['smape'] > 0 else 0

        result = {
            'model': name,
            'experiment': 'base_model',
            'val_smape': val_metrics['smape'],
            'test_smape': test_metrics['smape'],
            'gap': gap,
        }
        results.append(result)

        beat = " BEAT!" if test_metrics['smape'] < TARGET_SMAPE else ""
        logger.info(f"    Val: {val_metrics['smape']:.2f}% | Test: {test_metrics['smape']:.2f}%{beat} | Gap: {gap:.2f}x")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("FINAL RESULTS")
    logger.info("="*70)

    results_df = pd.DataFrame(results).sort_values('test_smape')

    logger.info(f"\n{'Experiment':<30} {'Model':<20} {'Val%':>8} {'Test%':>8} {'Gap':>7}")
    logger.info("-"*80)
    for _, row in results_df.iterrows():
        val_str = f"{row['val_smape']:.2f}" if pd.notna(row.get('val_smape')) else "N/A"
        gap_str = f"{row['gap']:.2f}x" if pd.notna(row.get('gap')) else "N/A"
        beat = "*" if row['test_smape'] < TARGET_SMAPE else ""
        logger.info(f"{row['experiment']:<30} {row['model']:<20} {val_str:>8} {row['test_smape']:>8.2f}{beat} {gap_str:>7}")

    # Best result
    best = results_df.iloc[0]
    beaten = results_df[results_df['test_smape'] < TARGET_SMAPE]

    logger.info(f"\n{'='*70}")
    if len(beaten) > 0:
        logger.info(f"SUCCESS! {len(beaten)} experiments beat {TARGET_SMAPE}%!")
        for _, row in beaten.iterrows():
            logger.info(f"  - {row['experiment']}: {row['test_smape']:.2f}%")
    else:
        logger.info(f"Best: {best['test_smape']:.2f}% (target: {TARGET_SMAPE}%)")

    # Save results
    summary = {
        'timestamp': datetime.now().isoformat(),
        'target_smape': TARGET_SMAPE,
        'best_experiment': best['experiment'],
        'best_model': best['model'],
        'best_test_smape': float(best['test_smape']),
        'n_features': len(data['features']),
        'features_used': data['features'],
        'all_results': results_df.to_dict('records'),
    }

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    results_df.to_csv(output_dir / 'results.csv', index=False)

    logger.info(f"\nResults saved to: {output_dir}")

    return results_df


if __name__ == "__main__":
    run_v3_pipeline()
