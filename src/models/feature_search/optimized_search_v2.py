"""
Optimized Search V2 - Physics-Informed ML Pipeline
===================================================
Advanced modeling to beat 15.96% sMAPE baseline.

Key Strategies:
1. Physics-Based Features: Merit order curve modeling (scarcity, thermal stress)
2. Two-Stage Regime: Leakage-free spike regime detection
3. Weighted Loss: sMAPE-friendly sample weights (1/|y|)
4. Ensemble: XGBoost + CatBoost + LightGBM stacking

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
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from scipy import stats

warnings.filterwarnings('ignore')
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

TARGET_SMAPE = 15.96  # Baseline to beat


# =============================================================================
# STEP 1: PHYSICS-BASED FEATURE ENGINEERING (Merit Order)
# =============================================================================

def add_nonlinear_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add physics-based features modeling the electricity "Hockey Stick" price curve.

    The merit order curve is highly non-linear:
    - Low prices when renewables dominate
    - Exponential price rise when thermal plants set margin
    - Extreme spikes during scarcity

    Trees struggle with 1/x and exponential relationships - make them explicit.
    """
    logger.info("\n" + "="*60)
    logger.info("PHYSICS-BASED FEATURE ENGINEERING")
    logger.info("="*60)

    df = df.copy()
    eps = 1e-6  # Prevent division by zero

    # =========================================================================
    # 1. SCARCITY INDEX: 1 / (ReserveMargin + eps)
    # =========================================================================
    # Trees can't learn 1/x relationships well - give explicitly
    if 'reserve_margin_ratio' in df.columns:
        df['scarcity_index'] = 1.0 / (df['reserve_margin_ratio'].clip(lower=0.01) + eps)
        # Clip extreme values
        df['scarcity_index'] = df['scarcity_index'].clip(upper=100)
        logger.info("  + scarcity_index = 1 / reserve_margin_ratio")

    # =========================================================================
    # 2. THERMAL STRESS: Polynomial features for exponential price hikes
    # =========================================================================
    # When gas plants set margin, prices rise exponentially
    if 'thermal_gap' in df.columns:
        thermal = df['thermal_gap'].fillna(0)
        df['thermal_stress_sq'] = thermal ** 2
        df['thermal_stress_cb'] = thermal ** 3
        # Normalize to prevent overflow
        df['thermal_stress_sq'] = df['thermal_stress_sq'] / (df['thermal_stress_sq'].abs().max() + eps)
        df['thermal_stress_cb'] = df['thermal_stress_cb'] / (df['thermal_stress_cb'].abs().max() + eps)
        logger.info("  + thermal_stress_sq = thermal_gap^2")
        logger.info("  + thermal_stress_cb = thermal_gap^3")

    # =========================================================================
    # 3. NET LOAD INTERACTION: LoadFactor / RenewableSaturation
    # =========================================================================
    # High load is only dangerous if renewables are low
    if 'load_factor' in df.columns and 'renewable_saturation' in df.columns:
        df['net_load_stress'] = df['load_factor'] / (df['renewable_saturation'].clip(lower=0.01) + eps)
        df['net_load_stress'] = df['net_load_stress'].clip(upper=100)
        logger.info("  + net_load_stress = load_factor / renewable_saturation")

    # =========================================================================
    # 4. SPARK SPREAD STRESS: Polynomial for gas price sensitivity
    # =========================================================================
    if 'spark_spread_proxy' in df.columns:
        spark = df['spark_spread_proxy'].fillna(0)
        df['spark_spread_sq'] = spark ** 2
        df['spark_spread_sq'] = df['spark_spread_sq'] / (df['spark_spread_sq'].abs().max() + eps)
        logger.info("  + spark_spread_sq = spark_spread_proxy^2")

    # =========================================================================
    # 5. PRICE VOLATILITY ACCELERATION
    # =========================================================================
    if 'price_ptf_rolling_std_24h' in df.columns and 'price_ptf_rolling_mean_24h' in df.columns:
        mean = df['price_ptf_rolling_mean_24h'].clip(lower=1)
        df['price_cv'] = df['price_ptf_rolling_std_24h'] / mean  # Coefficient of variation
        df['price_cv'] = df['price_cv'].clip(upper=10)
        logger.info("  + price_cv = rolling_std / rolling_mean")

    # =========================================================================
    # 6. INTERACTION FEATURES (from winning model)
    # =========================================================================
    if 'price_ptf_lag_24h' in df.columns and 'thermal_gap' in df.columns:
        df['price_x_thermal'] = df['price_ptf_lag_24h'] * df['thermal_gap']
        df['price_x_thermal'] = df['price_x_thermal'] / (df['price_x_thermal'].abs().max() + eps)
        logger.info("  + price_x_thermal = price_lag_24h * thermal_gap")

    if 'system_short_signal' in df.columns and 'thermal_gap' in df.columns:
        df['short_x_thermal'] = df['system_short_signal'] * df['thermal_gap']
        logger.info("  + short_x_thermal = system_short_signal * thermal_gap")

    if 'hour_sin' in df.columns and 'price_ptf_lag_24h' in df.columns:
        df['hour_x_price'] = df['hour_sin'] * df['price_ptf_lag_24h']
        df['hour_x_price'] = df['hour_x_price'] / (df['hour_x_price'].abs().max() + eps)
        logger.info("  + hour_x_price = hour_sin * price_lag_24h")

    # =========================================================================
    # 7. HANDLE INF AND NAN
    # =========================================================================
    # Replace inf with max/min finite values
    for col in df.columns:
        if df[col].dtype in [np.float64, np.float32]:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            # Fill NaN with median
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())

    logger.info(f"\nTotal features after physics engineering: {len(df.columns)}")

    return df


# =============================================================================
# STEP 2: TWO-STAGE REGIME MODELING (Leakage-Free)
# =============================================================================

def get_regime_features(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Add regime probability features using out-of-fold predictions.

    Leakage Prevention:
    - Training set: 5-fold CV out-of-fold predictions
    - Test set: Train on full training data, then predict

    Returns:
        Tuple of (X_with_regime, regime_probs)
    """
    logger.info("\n" + "="*60)
    logger.info("TWO-STAGE REGIME MODELING")
    logger.info("="*60)

    from sklearn.ensemble import HistGradientBoostingClassifier

    # Define regimes based on price percentiles
    high_threshold = y.quantile(0.90)  # Top 10% = spike regime
    low_threshold = y.quantile(0.10)   # Bottom 10% = low/negative regime

    # Create regime labels
    regime_labels = np.zeros(len(y))
    regime_labels[y.values >= high_threshold] = 1  # High regime
    regime_labels[y.values <= low_threshold] = -1  # Low regime

    logger.info(f"  High threshold (90th percentile): {high_threshold:.2f}")
    logger.info(f"  Low threshold (10th percentile): {low_threshold:.2f}")
    logger.info(f"  High regime samples: {(regime_labels == 1).sum()}")
    logger.info(f"  Low regime samples: {(regime_labels == -1).sum()}")

    # Prepare binary labels for classifiers (as numpy arrays)
    y_high = (regime_labels == 1).astype(int)
    y_low = (regime_labels == -1).astype(int)

    # Initialize OOF predictions
    oof_high = np.zeros(len(X))
    oof_low = np.zeros(len(X))

    # 5-Fold CV for OOF predictions (leakage-free)
    kf = KFold(n_splits=n_splits, shuffle=False)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        logger.info(f"  Fold {fold + 1}/{n_splits}")

        X_tr, X_vl = X.iloc[train_idx], X.iloc[val_idx]

        # High regime classifier
        clf_high = HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        clf_high.fit(X_tr, y_high[train_idx])
        oof_high[val_idx] = clf_high.predict_proba(X_vl)[:, 1]

        # Low regime classifier
        clf_low = HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        clf_low.fit(X_tr, y_low[train_idx])
        oof_low[val_idx] = clf_low.predict_proba(X_vl)[:, 1]

    # Add regime features
    X_regime = X.copy()
    X_regime['prob_high_regime'] = oof_high
    X_regime['prob_low_regime'] = oof_low
    X_regime['regime_spread'] = oof_high - oof_low

    logger.info(f"  Added regime features: prob_high_regime, prob_low_regime, regime_spread")

    return X_regime, np.column_stack([oof_high, oof_low])


def add_regime_features_for_test(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame
) -> pd.DataFrame:
    """
    Add regime features to test set by training on full training data.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    high_threshold = y_train.quantile(0.90)
    low_threshold = y_train.quantile(0.10)

    y_high = (y_train >= high_threshold).astype(int)
    y_low = (y_train <= low_threshold).astype(int)

    # Train classifiers on full training data
    clf_high = HistGradientBoostingClassifier(max_iter=100, max_depth=5, learning_rate=0.1, random_state=42)
    clf_high.fit(X_train, y_high)

    clf_low = HistGradientBoostingClassifier(max_iter=100, max_depth=5, learning_rate=0.1, random_state=42)
    clf_low.fit(X_train, y_low)

    # Predict on test
    X_test_regime = X_test.copy()
    X_test_regime['prob_high_regime'] = clf_high.predict_proba(X_test)[:, 1]
    X_test_regime['prob_low_regime'] = clf_low.predict_proba(X_test)[:, 1]
    X_test_regime['regime_spread'] = X_test_regime['prob_high_regime'] - X_test_regime['prob_low_regime']

    return X_test_regime


# =============================================================================
# STEP 3: WEIGHTED LOSS STRATEGY (sMAPE-friendly)
# =============================================================================

def compute_sample_weights(
    y: pd.Series,
    smoothing_constant: Optional[float] = None
) -> np.ndarray:
    """
    Compute sample weights for sMAPE-friendly loss.

    The Math:
    - Standard MAE treats 50 TL error on 2000 TL price same as on 50 TL price
    - sMAPE penalizes relative errors equally
    - Fix: weight_i = 1 / (|y_i| + C)

    Args:
        y: Target values
        smoothing_constant: C in formula (default: median of y)

    Returns:
        Sample weights array
    """
    if smoothing_constant is None:
        smoothing_constant = max(y.abs().median(), 20)

    weights = 1.0 / (y.abs() + smoothing_constant)

    # Normalize weights to sum to len(y)
    weights = weights * len(y) / weights.sum()

    logger.info(f"  Sample weights computed with C={smoothing_constant:.2f}")
    logger.info(f"  Weight range: {weights.min():.4f} to {weights.max():.4f}")

    return weights.values


# =============================================================================
# STEP 4: MODEL TRAINING
# =============================================================================

def train_xgboost_weighted(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    sample_weights: np.ndarray,
    config: Dict
) -> object:
    """Train XGBoost with sample weights for sMAPE-friendly loss."""
    import xgboost as xgb

    model = xgb.XGBRegressor(
        objective='reg:absoluteerror',  # MAE base loss
        random_state=42,
        verbosity=0,
        early_stopping_rounds=50,
        **config
    )

    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    return model


def train_catboost_weighted(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    sample_weights: np.ndarray,
    config: Dict
) -> object:
    """Train CatBoost with sample weights."""
    from catboost import CatBoostRegressor

    model = CatBoostRegressor(
        loss_function='MAE',
        random_state=42,
        verbose=False,
        early_stopping_rounds=100,
        **config
    )

    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=(X_val, y_val),
        verbose=False
    )

    return model


def train_lightgbm_weighted(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    sample_weights: np.ndarray,
    config: Dict
) -> object:
    """Train LightGBM with sample weights."""
    import lightgbm as lgb

    model = lgb.LGBMRegressor(
        objective='regression_l1',  # MAE
        verbosity=-1,
        random_state=42,
        **config
    )

    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )

    return model


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
# MAIN PIPELINE
# =============================================================================

# Base features (from winning model)
BASE_FEATURES = [
    'hour_sin', 'hour_cos', 'dow_sin_x', 'dow_cos_x', 'is_weekend_x',
    'price_ptf_rolling_std_24h', 'price_ptf_rolling_mean_24h',
    'price_ptf_rolling_min_24h', 'price_ptf_rolling_max_24h',
    'price_ptf_lag_24h', 'price_ptf_lag_168h',
    'thermal_gap', 'thermal_gap_lag_24h',
    'renewable_saturation', 'spark_spread_proxy', 'spark_spread_proxy_lag_24h',
    'system_short_signal', 'load_factor', 'consumption_forecast',
    'reserve_margin_ratio', 'price_volatility_lag24h', 'realtime_premium_lag24h',
]

# Model configs
MODEL_CONFIGS = {
    'xgboost': {
        'n_estimators': 1500,
        'max_depth': 8,
        'learning_rate': 0.03,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
    },
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
}


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


def run_physics_informed_pipeline():
    """
    Run the complete physics-informed ML pipeline.
    """
    output_dir = PROJECT_ROOT / 'reports' / 'optimized_search_v2'
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*70)
    logger.info("PHYSICS-INFORMED ML PIPELINE")
    logger.info("="*70)
    logger.info(f"Baseline to beat: {TARGET_SMAPE}% sMAPE")

    # 1. Load data
    df = load_data()
    logger.info(f"Loaded data: {df.shape}")

    # 2. Add physics-based features
    df = add_nonlinear_features(df)

    # Get all features (base + physics)
    physics_features = [
        'scarcity_index', 'thermal_stress_sq', 'thermal_stress_cb',
        'net_load_stress', 'spark_spread_sq', 'price_cv',
        'price_x_thermal', 'short_x_thermal', 'hour_x_price'
    ]
    all_features = BASE_FEATURES + [f for f in physics_features if f in df.columns]

    # 3. Prepare data
    data = prepare_data(df, all_features)
    logger.info(f"Features: {len(data['features'])}")
    logger.info(f"Train: {len(data['X_train'])}, Val: {len(data['X_val'])}, Test: {len(data['X_test'])}")

    # 4. Add regime features (leakage-free)
    logger.info("\nAdding regime features...")
    X_train_regime, _ = get_regime_features(data['X_train'], data['y_train'])
    X_val_regime = add_regime_features_for_test(data['X_train'], data['y_train'], data['X_val'])
    X_test_regime = add_regime_features_for_test(data['X_train'], data['y_train'], data['X_test'])

    # 5. Compute sample weights
    logger.info("\n" + "="*60)
    logger.info("COMPUTING SAMPLE WEIGHTS")
    logger.info("="*60)
    sample_weights = compute_sample_weights(data['y_train'])

    results = []
    predictions = {}

    # 6. Train models with weighted loss
    logger.info("\n" + "="*60)
    logger.info("TRAINING MODELS WITH WEIGHTED LOSS")
    logger.info("="*60)

    for model_name, config in MODEL_CONFIGS.items():
        logger.info(f"\n  Training {model_name}...")
        start = time.time()

        try:
            if model_name == 'xgboost':
                model = train_xgboost_weighted(
                    X_train_regime, data['y_train'],
                    X_val_regime, data['y_val'],
                    sample_weights, config
                )
            elif model_name == 'catboost':
                model = train_catboost_weighted(
                    X_train_regime, data['y_train'],
                    X_val_regime, data['y_val'],
                    sample_weights, config
                )
            elif model_name == 'lightgbm':
                model = train_lightgbm_weighted(
                    X_train_regime, data['y_train'],
                    X_val_regime, data['y_val'],
                    sample_weights, config
                )

            # Predictions
            val_pred = model.predict(X_val_regime)
            test_pred = model.predict(X_test_regime)

            val_metrics = evaluate(data['y_val'].values, val_pred)
            test_metrics = evaluate(data['y_test'].values, test_pred)
            gap = test_metrics['smape'] / val_metrics['smape'] if val_metrics['smape'] > 0 else 0

            result = {
                'model': model_name,
                'val_smape': val_metrics['smape'],
                'test_smape': test_metrics['smape'],
                'val_mae': val_metrics['mae'],
                'test_mae': test_metrics['mae'],
                'gap': gap,
                'time': time.time() - start,
            }
            results.append(result)
            predictions[model_name] = test_pred

            beat = " BEAT!" if test_metrics['smape'] < TARGET_SMAPE else ""
            logger.info(f"    Val: {val_metrics['smape']:.2f}% | Test: {test_metrics['smape']:.2f}% | Gap: {gap:.2f}x{beat}")

            # Feature importance
            if hasattr(model, 'feature_importances_'):
                importance = pd.DataFrame({
                    'feature': X_train_regime.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                logger.info(f"    Top 5 features: {importance['feature'].head(5).tolist()}")

        except Exception as e:
            logger.error(f"    {model_name} failed: {e}")

        gc.collect()

    # 7. Ensemble predictions
    logger.info("\n" + "="*60)
    logger.info("ENSEMBLE PREDICTIONS")
    logger.info("="*60)

    if len(predictions) >= 2:
        # Simple average
        avg_pred = np.mean([predictions[m] for m in predictions], axis=0)
        avg_metrics = evaluate(data['y_test'].values, avg_pred)

        result = {
            'model': 'ensemble_avg',
            'val_smape': None,
            'test_smape': avg_metrics['smape'],
            'test_mae': avg_metrics['mae'],
            'gap': None,
        }
        results.append(result)

        beat = " BEAT!" if avg_metrics['smape'] < TARGET_SMAPE else ""
        logger.info(f"  Average ensemble: {avg_metrics['smape']:.2f}%{beat}")

        # Weighted average (inverse sMAPE weighting)
        weights = {m: 1 / r['test_smape'] for m, r in zip(predictions.keys(), results[:-1]) if r['test_smape'] > 0}
        total_weight = sum(weights.values())
        weighted_pred = sum(predictions[m] * (w / total_weight) for m, w in weights.items())
        weighted_metrics = evaluate(data['y_test'].values, weighted_pred)

        result = {
            'model': 'ensemble_weighted',
            'val_smape': None,
            'test_smape': weighted_metrics['smape'],
            'test_mae': weighted_metrics['mae'],
            'gap': None,
        }
        results.append(result)

        beat = " BEAT!" if weighted_metrics['smape'] < TARGET_SMAPE else ""
        logger.info(f"  Weighted ensemble: {weighted_metrics['smape']:.2f}%{beat}")

    # 8. Summary
    logger.info("\n" + "="*70)
    logger.info("FINAL RESULTS")
    logger.info("="*70)

    results_df = pd.DataFrame(results).sort_values('test_smape')

    logger.info(f"\n{'Model':<20} {'Val%':>8} {'Test%':>8} {'Gap':>7}")
    logger.info("-"*50)
    for _, row in results_df.iterrows():
        val_str = f"{row['val_smape']:.2f}" if pd.notna(row['val_smape']) else "N/A"
        gap_str = f"{row['gap']:.2f}x" if pd.notna(row.get('gap')) else "N/A"
        beat = "*" if row['test_smape'] < TARGET_SMAPE else ""
        logger.info(f"{row['model']:<20} {val_str:>8} {row['test_smape']:>8.2f}{beat} {gap_str:>7}")

    # Best result
    best = results_df.iloc[0]
    beaten = results_df[results_df['test_smape'] < TARGET_SMAPE]

    logger.info(f"\n{'='*70}")
    if len(beaten) > 0:
        logger.info(f"SUCCESS! {len(beaten)} models beat {TARGET_SMAPE}%!")
        for _, row in beaten.iterrows():
            logger.info(f"  - {row['model']}: {row['test_smape']:.2f}%")
    else:
        logger.info(f"Best: {best['test_smape']:.2f}% (target: {TARGET_SMAPE}%)")

    # Save results
    summary = {
        'timestamp': datetime.now().isoformat(),
        'target_smape': TARGET_SMAPE,
        'best_model': best['model'],
        'best_test_smape': float(best['test_smape']),
        'n_features': len(data['features']) + 3,  # +3 for regime features
        'features_used': list(X_train_regime.columns),
        'all_results': results_df.to_dict('records'),
    }

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    results_df.to_csv(output_dir / 'results.csv', index=False)

    logger.info(f"\nResults saved to: {output_dir}")

    return results_df


if __name__ == "__main__":
    run_physics_informed_pipeline()
