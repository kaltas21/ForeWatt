"""
Consumption Forecasting Model Training - CatBoost
==================================================
Isolated training script for consumption forecasting using CatBoost.

This is an isolated, self-contained training script that can be run independently.
The trained models are saved to: ForeWatt/models/consumption/

Features:
- CatBoost Gradient Boosting
- Time-series aware feature engineering
- Temporal train/val/test split

Usage:
    python src/models/consumption_train.py

Author: ForeWatt Team
Date: December 2025
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / 'data' / 'gold' / 'master' / 'master_v2_fundamental.parquet'
MODELS_OUTPUT_DIR = PROJECT_ROOT / 'models' / 'consumption'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# FEATURE CONFIGURATION
# =============================================================================

# Consumption forecasting features (23 features)
CONSUMPTION_FEATURES = [
    # Consumption lags
    'consumption_lag_24h',
    'consumption_lag_48h',
    'consumption_lag_168h',
    # Consumption rolling
    'consumption_rolling_mean_24h',
    'consumption_rolling_std_24h',
    # Weather
    'temp_national',
    'humidity_national',
    'HDD',
    'CDD',
    'heat_index',
    'is_hot',
    'is_cold',
    # Temperature lags
    'temp_lag_24h',
    # Calendar
    'hour_sin', 'hour_cos',
    'dow_sin_x', 'dow_cos_x',
    'month_sin', 'month_cos',
    'is_weekend_x',
    'is_holiday_day',
    'is_holiday_hour',
    # Price signal
    'price_ptf_lag_24h',
]

# CatBoost hyperparameters (V2 - reduced overfitting)
# Changes from V1:
#   - depth: 8 → 5 (reduced model capacity)
#   - l2_leaf_reg: 5.0 → 15.0 (increased regularization)
#   - iterations: 2000 → 1000 (fewer trees)
#   - early_stopping_rounds: 100 → 50 (more aggressive early stopping)
#   - Added subsample and rsm for stochasticity
CATBOOST_PARAMS = {
    'iterations': 1000,
    'depth': 5,  # Reduced from 8
    'learning_rate': 0.03,
    'l2_leaf_reg': 15.0,  # Increased from 5.0
    'border_count': 128,  # Reduced from 254
    'random_seed': 42,
    'loss_function': 'RMSE',
    'eval_metric': 'MAE',
    'early_stopping_rounds': 50,  # More aggressive
    'subsample': 0.8,  # Row sampling (stochasticity)
    'rsm': 0.8,  # Feature sampling per tree (colsample_bylevel equivalent)
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data() -> pd.DataFrame:
    """Load master dataset."""
    logger.info(f"Loading data from: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)

    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date' in df.columns:
            df = df.set_index('date')
        elif 'datetime' in df.columns:
            df = df.set_index('datetime')

    logger.info(f"Loaded data: {df.shape}")
    return df


# =============================================================================
# FEATURE PREPARATION
# =============================================================================

def prepare_features(
    df: pd.DataFrame,
    features: List[str],
    target: str = 'consumption',
    val_size: float = 0.2,
    test_size: float = 0.2
) -> Dict:
    """
    Prepare features with train/val/test split.

    Args:
        df: Master dataset
        features: List of feature names
        target: Target variable
        val_size: Validation set fraction
        test_size: Test set fraction

    Returns:
        Dictionary with X_train, X_val, X_test, y_train, y_val, y_test, feature_names
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"PREPARING FEATURES FOR CONSUMPTION")
    logger.info(f"{'='*60}")

    # Filter available features
    available = [f for f in features if f in df.columns]
    missing = set(features) - set(available)

    if missing:
        logger.warning(f"Missing {len(missing)} features: {sorted(list(missing))[:5]}...")

    # Extract features and target
    X = df[available].copy()
    y = df[target].copy()

    logger.info(f"  Features: {len(available)}")
    logger.info(f"  Target: {target}")

    # Drop NaN rows
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    logger.info(f"  Samples after NaN removal: {len(X)}")

    # Temporal split
    n = len(X)
    train_end = int(n * (1 - val_size - test_size))
    val_end = int(n * (1 - test_size))

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    X_val = X.iloc[train_end:val_end]
    y_val = y.iloc[train_end:val_end]
    X_test = X.iloc[val_end:]
    y_test = y.iloc[val_end:]

    logger.info(f"\n  Data splits:")
    logger.info(f"    Train: {len(X_train):,} ({len(X_train)/n*100:.1f}%)")
    logger.info(f"    Val:   {len(X_val):,} ({len(X_val)/n*100:.1f}%)")
    logger.info(f"    Test:  {len(X_test):,} ({len(X_test)/n*100:.1f}%)")

    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'feature_names': available,
    }


# =============================================================================
# METRICS
# =============================================================================

def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate MAE."""
    return np.mean(np.abs(y_true - y_pred))


def symmetric_mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate sMAPE."""
    return 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))


def mean_absolute_scaled_error(y_true: np.ndarray, y_pred: np.ndarray,
                                y_train: np.ndarray, seasonality: int = 24) -> float:
    """Calculate MASE."""
    naive_errors = np.abs(np.diff(y_train[::seasonality]))
    mae_naive = np.mean(naive_errors) if len(naive_errors) > 0 else 1.0
    return np.mean(np.abs(y_true - y_pred)) / max(mae_naive, 1e-8)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray = None) -> Dict:
    """Calculate all metrics."""
    metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'sMAPE': symmetric_mean_absolute_percentage_error(y_true, y_pred),
    }

    if y_train is not None:
        metrics['MASE'] = mean_absolute_scaled_error(y_true, y_pred, y_train)

    return metrics


# =============================================================================
# TRAINING
# =============================================================================

def train_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: Dict = None
) -> Tuple[object, Dict]:
    """
    Train CatBoost model.

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        params: CatBoost hyperparameters

    Returns:
        Tuple of (model, validation_metrics)
    """
    from catboost import CatBoostRegressor

    params = params or CATBOOST_PARAMS

    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING CATBOOST")
    logger.info(f"{'='*60}")
    logger.info(f"  Train: {len(X_train):,} samples")
    logger.info(f"  Val:   {len(X_val):,} samples")
    logger.info(f"  Features: {X_train.shape[1]}")
    logger.info(f"  Params: iterations={params['iterations']}, depth={params['depth']}, lr={params['learning_rate']}")

    model = CatBoostRegressor(**params, verbose=100)

    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True
    )

    logger.info(f"  Best iteration: {model.best_iteration_}")

    # Validation predictions
    val_pred = model.predict(X_val)

    # Calculate metrics
    metrics = evaluate(y_val.values, val_pred, y_train.values)

    logger.info(f"\n  Validation metrics:")
    logger.info(f"    MAE:   {metrics['MAE']:.2f}")
    logger.info(f"    sMAPE: {metrics['sMAPE']:.2f}%")
    logger.info(f"    MASE:  {metrics['MASE']:.4f}")

    return model, metrics


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train_consumption_model():
    """Train consumption forecasting model."""
    MODELS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("="*70)
    logger.info("CONSUMPTION MODEL TRAINING - CatBoost")
    logger.info("="*70)

    # Load data
    df = load_data()

    # Prepare features
    data = prepare_features(df, CONSUMPTION_FEATURES, target='consumption')

    # Train model
    model, val_metrics = train_catboost(
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val'],
        CATBOOST_PARAMS
    )

    # Evaluate on train set (for overfitting analysis)
    logger.info("\n" + "="*60)
    logger.info("TRAIN EVALUATION (Overfitting Check)")
    logger.info("="*60)

    train_pred = model.predict(data['X_train'])
    train_metrics = evaluate(
        data['y_train'].values,
        train_pred,
        data['y_train'].values
    )

    logger.info(f"\n  Train metrics:")
    logger.info(f"    MAE:   {train_metrics['MAE']:.2f}")
    logger.info(f"    sMAPE: {train_metrics['sMAPE']:.2f}%")

    # Evaluate on test set
    logger.info("\n" + "="*60)
    logger.info("TEST EVALUATION")
    logger.info("="*60)

    test_pred = model.predict(data['X_test'])
    test_metrics = evaluate(
        data['y_test'].values,
        test_pred,
        data['y_train'].values
    )

    logger.info(f"\n  Test metrics:")
    logger.info(f"    MAE:   {test_metrics['MAE']:.2f}")
    logger.info(f"    sMAPE: {test_metrics['sMAPE']:.2f}%")
    logger.info(f"    MASE:  {test_metrics['MASE']:.4f}")

    # Overfitting ratio
    overfit_ratio = test_metrics['sMAPE'] / train_metrics['sMAPE'] if train_metrics['sMAPE'] > 0 else 0
    logger.info(f"\n  Overfitting Analysis:")
    logger.info(f"    Train sMAPE: {train_metrics['sMAPE']:.2f}%")
    logger.info(f"    Test sMAPE:  {test_metrics['sMAPE']:.2f}%")
    logger.info(f"    Overfit Ratio: {overfit_ratio:.2f}x")
    if overfit_ratio < 1.2:
        logger.info(f"    Status: GOOD generalization")
    elif overfit_ratio < 1.5:
        logger.info(f"    Status: SLIGHT overfitting")
    else:
        logger.info(f"    Status: SIGNIFICANT overfitting - consider more regularization")

    # Save model
    logger.info("\n" + "="*60)
    logger.info("SAVING MODEL")
    logger.info("="*60)

    model_path = MODELS_OUTPUT_DIR / 'model.cbm'
    model.save_model(str(model_path))
    logger.info(f"  Saved model: {model_path}")

    # Save feature importance
    importance = model.get_feature_importance()
    importance_df = pd.DataFrame({
        'feature': data['feature_names'],
        'importance': importance
    }).sort_values('importance', ascending=False)

    importance_path = MODELS_OUTPUT_DIR / 'feature_importance.csv'
    importance_df.to_csv(importance_path, index=False)
    logger.info(f"  Saved feature importance: {importance_path}")

    # Save features config
    features_path = MODELS_OUTPUT_DIR / 'features.json'
    with open(features_path, 'w') as f:
        json.dump({
            'features': data['feature_names'],
            'n_features': len(data['feature_names']),
            'model_type': 'catboost',
            'model_version': 'V2',
            'target': 'consumption',
            'hyperparameters': {
                'depth': CATBOOST_PARAMS['depth'],
                'iterations': CATBOOST_PARAMS['iterations'],
                'l2_leaf_reg': CATBOOST_PARAMS['l2_leaf_reg'],
                'subsample': CATBOOST_PARAMS.get('subsample', 1.0),
                'rsm': CATBOOST_PARAMS.get('rsm', 1.0),
            },
            'train_mae': train_metrics['MAE'],
            'train_smape': train_metrics['sMAPE'],
            'val_mae': val_metrics['MAE'],
            'val_smape': val_metrics['sMAPE'],
            'test_mae': test_metrics['MAE'],
            'test_smape': test_metrics['sMAPE'],
            'test_mase': test_metrics['MASE'],
            'overfit_ratio': overfit_ratio,
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2)
    logger.info(f"  Saved features config: {features_path}")

    logger.info(f"\n{'='*70}")
    logger.info(f"TRAINING COMPLETE")
    logger.info(f"Models saved to: {MODELS_OUTPUT_DIR}")
    logger.info(f"Test MAE: {test_metrics['MAE']:.2f}")
    logger.info(f"{'='*70}")

    return {
        'model': model,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'overfit_ratio': overfit_ratio,
        'features': data['feature_names'],
        'feature_importance': importance_df,
    }


if __name__ == "__main__":
    train_consumption_model()
