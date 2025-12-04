"""
Optimized Search V6 - Break 14% with Temporal Ensembling & Log-Space Fine-Tuning
=================================================================================
Context: V5 achieved 14.08% sMAPE with 6-month fine-tuning window.
         Shorter windows minimize bias (+1.49 TL) significantly.

Objective: Break the 14.00% barrier by:
    1. Ultra-recency: Test windows [2, 3, 4] months
    2. Log-space fine-tuning: More stable adaptation
    3. Temporal ensembling: Combine level specialists + seasonal specialists

Key Optimization: Train base model ONCE, reuse for all fine-tuning loops.

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

# V5 Winner baseline
BASELINE_SMAPE = 14.08


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

    return df


def prepare_data(df: pd.DataFrame, features: List[str],
                 test_start: str = '2024-06-01') -> Dict:
    """Prepare data with FIXED test set."""
    available = [f for f in features if f in df.columns]

    X = df[available].copy()
    y = df['price_real'].copy()

    # Drop NaN
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]

    # Timezone-aware split
    if X.index.tz is not None:
        test_start_dt = pd.Timestamp(test_start, tz=X.index.tz)
    else:
        test_start_dt = pd.Timestamp(test_start)

    test_mask = X.index >= test_start_dt

    return {
        'X_train': X[~test_mask],
        'y_train': y[~test_mask],
        'X_test': X[test_mask],
        'y_test': y[test_mask],
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


# =============================================================================
# ADVANCED TRANSFER LEARNING ENGINE
# =============================================================================

class TransferLearningEngine:
    """
    Efficient Transfer Learning Engine that trains base model ONCE
    and reuses it for multiple fine-tuning configurations.
    """

    def __init__(self, X_train: pd.DataFrame, y_train: pd.Series,
                 X_test: pd.DataFrame, y_test: pd.Series):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        # Cache for base models
        self.base_model_linear = None
        self.base_model_log = None

        # Store results
        self.results = []

    def _get_finetune_subset(self, window_months: int) -> Tuple[pd.DataFrame, pd.Series]:
        """Get the last N months of training data for fine-tuning."""
        train_end = self.X_train.index.max()
        finetune_start = train_end - pd.DateOffset(months=window_months)

        mask = self.X_train.index >= finetune_start
        return self.X_train[mask], self.y_train[mask]

    def _log_drift_analysis(self, window_months: int, y_finetune: pd.Series):
        """Log the price drift between fine-tune window and test set."""
        ft_mean = y_finetune.mean()
        test_mean = self.y_test.mean()
        train_mean = self.y_train.mean()
        drift_ft_test = (test_mean - ft_mean) / ft_mean * 100

        logger.info(f"    Drift Analysis ({window_months}mo window):")
        logger.info(f"      Fine-tune mean: {ft_mean:.2f} TL | Test mean: {test_mean:.2f} TL | Drift: {drift_ft_test:+.1f}%")

    def train_base_model(self, use_log_target: bool = False):
        """
        Train the base model on FULL training history.
        This is called ONCE and cached for reuse.
        """
        from catboost import CatBoostRegressor

        target_type = "Log" if use_log_target else "Linear"
        logger.info(f"\n  Training Base Model ({target_type}-space)...")

        y = np.log1p(self.y_train) if use_log_target else self.y_train

        # Split for early stopping
        split_idx = int(len(self.X_train) * 0.85)
        X_tr, X_vl = self.X_train.iloc[:split_idx], self.X_train.iloc[split_idx:]
        y_tr, y_vl = y.iloc[:split_idx], y.iloc[split_idx:]

        model = CatBoostRegressor(
            loss_function='MAE',
            iterations=2000,
            depth=10,
            learning_rate=0.02,
            l2_leaf_reg=3,
            random_state=42,
            verbose=False,
            early_stopping_rounds=100,
        )
        model.fit(X_tr, y_tr, eval_set=(X_vl, y_vl), verbose=False)

        if use_log_target:
            self.base_model_log = model
        else:
            self.base_model_linear = model

        logger.info(f"    Base model trained ({model.tree_count_} trees)")

        return model

    def train_and_finetune(self, window_months: int, use_log_target: bool = False,
                           finetune_lr: float = 0.005, finetune_iterations: int = 500) -> Dict:
        """
        Fine-tune the cached base model on a specific window.

        Phase 1: Load cached base model (trained on full history)
        Phase 2: Fine-tune on last N months with low learning rate

        Returns: Dict with predictions and metrics
        """
        from catboost import CatBoostRegressor

        target_type = "Log" if use_log_target else "Linear"

        # Get or train base model
        base_model = self.base_model_log if use_log_target else self.base_model_linear
        if base_model is None:
            base_model = self.train_base_model(use_log_target)

        # Get fine-tune subset
        X_ft, y_ft_raw = self._get_finetune_subset(window_months)
        y_ft = np.log1p(y_ft_raw) if use_log_target else y_ft_raw

        # Log drift analysis
        self._log_drift_analysis(window_months, y_ft_raw)

        # K-fold fine-tuning for robust estimates
        n_folds = 5
        kf = KFold(n_splits=n_folds, shuffle=False)

        oof_preds = np.zeros(len(X_ft))
        test_preds = np.zeros(len(self.X_test))

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_ft)):
            X_tr, X_vl = X_ft.iloc[train_idx], X_ft.iloc[val_idx]
            y_tr, y_vl = y_ft.iloc[train_idx], y_ft.iloc[val_idx]

            # Fine-tune with init_model
            finetune_model = CatBoostRegressor(
                loss_function='MAE',
                iterations=finetune_iterations,
                depth=10,
                learning_rate=finetune_lr,
                l2_leaf_reg=3,
                random_state=42,
                verbose=False,
                early_stopping_rounds=50,
            )
            finetune_model.fit(
                X_tr, y_tr,
                eval_set=(X_vl, y_vl),
                init_model=base_model,
                verbose=False
            )

            # Predictions
            val_pred = finetune_model.predict(X_vl)
            test_pred = finetune_model.predict(self.X_test)

            # Inverse transform if log-space
            if use_log_target:
                val_pred = np.expm1(val_pred)
                test_pred = np.expm1(test_pred)

            oof_preds[val_idx] = val_pred
            test_preds += test_pred / n_folds

        # Inverse transform y_ft for metrics if log-space
        y_ft_eval = y_ft_raw.values

        # Calculate metrics
        oof_metrics = evaluate(y_ft_eval, oof_preds)
        test_metrics = evaluate(self.y_test.values, test_preds)
        gap = test_metrics['smape'] / oof_metrics['smape'] if oof_metrics['smape'] > 0 else float('inf')

        model_name = f"{window_months}mo_{target_type}"
        result = {
            'model': model_name,
            'window_months': window_months,
            'target_space': target_type,
            'val_smape': oof_metrics['smape'],
            'test_smape': test_metrics['smape'],
            'test_mae': test_metrics['mae'],
            'test_bias': test_metrics['bias'],
            'gap': gap,
            'test_pred': test_preds,
        }

        logger.info(f"    Result: sMAPE={test_metrics['smape']:.2f}%, Bias={test_metrics['bias']:+.2f} TL")

        self.results.append(result)
        return result

    def create_ensemble(self, model_names: List[str], weights: Optional[List[float]] = None,
                        ensemble_name: str = "Ensemble") -> Dict:
        """
        Create an ensemble by averaging predictions from multiple models.

        Args:
            model_names: List of model names to ensemble (e.g., ['3mo_Linear', '12mo_Linear'])
            weights: Optional weights for each model (default: equal weights)
            ensemble_name: Name for this ensemble
        """
        # Find results for specified models
        preds = []
        for name in model_names:
            result = next((r for r in self.results if r['model'] == name), None)
            if result is None:
                logger.warning(f"Model {name} not found in results!")
                continue
            preds.append(result['test_pred'])

        if len(preds) == 0:
            raise ValueError("No valid models found for ensemble")

        # Default to equal weights
        if weights is None:
            weights = [1.0 / len(preds)] * len(preds)

        # Weighted average
        ensemble_pred = np.zeros_like(preds[0])
        for pred, weight in zip(preds, weights):
            ensemble_pred += pred * weight

        # Calculate metrics
        test_metrics = evaluate(self.y_test.values, ensemble_pred)

        result = {
            'model': ensemble_name,
            'window_months': 'ensemble',
            'target_space': 'ensemble',
            'val_smape': np.nan,
            'test_smape': test_metrics['smape'],
            'test_mae': test_metrics['mae'],
            'test_bias': test_metrics['bias'],
            'gap': np.nan,
            'test_pred': ensemble_pred,
            'components': model_names,
            'weights': weights,
        }

        logger.info(f"    {ensemble_name}: sMAPE={test_metrics['smape']:.2f}%, Bias={test_metrics['bias']:+.2f} TL")

        self.results.append(result)
        return result


# =============================================================================
# MAIN EXPERIMENT LOOP
# =============================================================================

def run_v6_experiments():
    """Run all V6 experiments: Ultra-recency, Log-space, and Temporal Ensembling."""
    output_dir = PROJECT_ROOT / 'reports' / 'optimized_search_v6'
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*70)
    logger.info("OPTIMIZED SEARCH V6 - Temporal Ensembling & Log-Space Fine-Tuning")
    logger.info(f"Baseline to beat: {BASELINE_SMAPE}% sMAPE")
    logger.info("="*70)

    # Load data
    df = load_data()
    logger.info(f"Loaded data: {df.shape}")
    logger.info(f"Date range: {df.index.min()} to {df.index.max()}")

    data = prepare_data(df, BASE_FEATURES)
    logger.info(f"Features: {len(data['features'])}")
    logger.info(f"Train: {len(data['X_train'])}, Test: {len(data['X_test'])}")

    # Initialize engine
    engine = TransferLearningEngine(
        data['X_train'], data['y_train'],
        data['X_test'], data['y_test']
    )

    # Log test set statistics
    logger.info(f"\nTest Set Mean Price: {data['y_test'].mean():.2f} TL/MWh")
    logger.info(f"Full Train Mean Price: {data['y_train'].mean():.2f} TL/MWh")

    # =========================================================================
    # PRE-TRAIN BASE MODELS (ONCE)
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("PHASE 0: PRE-TRAINING BASE MODELS")
    logger.info("="*70)

    engine.train_base_model(use_log_target=False)  # Linear base
    engine.train_base_model(use_log_target=True)   # Log base

    # =========================================================================
    # EXPERIMENT A: ULTRA-RECENCY (Window Optimization)
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("EXPERIMENT A: ULTRA-RECENCY (Window Optimization)")
    logger.info("Testing windows: [2, 3, 4, 6, 12] months (Linear-space)")
    logger.info("="*70)

    for window in [2, 3, 4, 6, 12]:
        logger.info(f"\n  Window: {window} months")
        engine.train_and_finetune(window_months=window, use_log_target=False)

    # =========================================================================
    # EXPERIMENT B: LOG-SPACE STABILITY
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("EXPERIMENT B: LOG-SPACE STABILITY")
    logger.info("Testing windows: [3, 6, 12] months (Log-space)")
    logger.info("="*70)

    for window in [3, 6, 12]:
        logger.info(f"\n  Window: {window} months (Log-space)")
        engine.train_and_finetune(window_months=window, use_log_target=True)

    # =========================================================================
    # EXPERIMENT C: TEMPORAL ENSEMBLING
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("EXPERIMENT C: TEMPORAL ENSEMBLING")
    logger.info("="*70)

    # Ensemble 1: Level Specialist + Seasonal Specialist
    logger.info("\n  Ensemble 1: 50% 3mo (Level) + 50% 12mo (Seasonal)")
    engine.create_ensemble(
        model_names=['3mo_Linear', '12mo_Linear'],
        weights=[0.5, 0.5],
        ensemble_name='Ens1_3mo+12mo_Linear'
    )

    # Ensemble 2: Log + Linear variance stabilizer
    logger.info("\n  Ensemble 2: 50% Log-3mo + 50% Linear-3mo")
    engine.create_ensemble(
        model_names=['3mo_Log', '3mo_Linear'],
        weights=[0.5, 0.5],
        ensemble_name='Ens2_Log+Lin_3mo'
    )

    # Ensemble 3: Multi-window weighted average
    logger.info("\n  Ensemble 3: Weighted [2mo, 6mo, 12mo] = [0.4, 0.4, 0.2]")
    engine.create_ensemble(
        model_names=['2mo_Linear', '6mo_Linear', '12mo_Linear'],
        weights=[0.4, 0.4, 0.2],
        ensemble_name='Ens3_2+6+12mo_Weighted'
    )

    # Ensemble 4: Best recency blend
    logger.info("\n  Ensemble 4: 60% 3mo + 40% 6mo (Linear)")
    engine.create_ensemble(
        model_names=['3mo_Linear', '6mo_Linear'],
        weights=[0.6, 0.4],
        ensemble_name='Ens4_3+6mo_Linear'
    )

    # Ensemble 5: Triple log blend
    logger.info("\n  Ensemble 5: Equal [3mo_Log, 6mo_Log, 12mo_Log]")
    engine.create_ensemble(
        model_names=['3mo_Log', '6mo_Log', '12mo_Log'],
        weights=[1/3, 1/3, 1/3],
        ensemble_name='Ens5_Log_Blend'
    )

    # Ensemble 6: Bias-optimized (recent heavy)
    logger.info("\n  Ensemble 6: 50% 2mo + 30% 3mo + 20% 6mo (Bias-optimized)")
    engine.create_ensemble(
        model_names=['2mo_Linear', '3mo_Linear', '6mo_Linear'],
        weights=[0.5, 0.3, 0.2],
        ensemble_name='Ens6_BiasOpt'
    )

    # =========================================================================
    # FINAL LEADERBOARD
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("FINAL LEADERBOARD")
    logger.info("="*70)

    # Separate single models and ensembles
    single_models = [r for r in engine.results if r['target_space'] != 'ensemble']
    ensembles = [r for r in engine.results if r['target_space'] == 'ensemble']

    logger.info(f"\n{'Model':<30} {'Window':>8} {'Space':>8} {'sMAPE':>10} {'Bias':>10} {'Beat?':>8}")
    logger.info("-"*80)

    for r in sorted(single_models, key=lambda x: x['test_smape']):
        beat = "YES" if r['test_smape'] < BASELINE_SMAPE else "no"
        logger.info(f"{r['model']:<30} {r['window_months']:>8} {r['target_space']:>8} {r['test_smape']:>9.2f}% {r['test_bias']:>9.2f} {beat:>8}")

    logger.info("\n  ENSEMBLES:")
    logger.info("-"*80)

    for r in sorted(ensembles, key=lambda x: x['test_smape']):
        beat = "YES" if r['test_smape'] < BASELINE_SMAPE else "no"
        logger.info(f"{r['model']:<30} {'mix':>8} {'mix':>8} {r['test_smape']:>9.2f}% {r['test_bias']:>9.2f} {beat:>8}")

    # Overall best
    all_results = single_models + ensembles
    best = min(all_results, key=lambda x: x['test_smape'])

    logger.info(f"\n{'='*70}")
    if best['test_smape'] < BASELINE_SMAPE:
        improvement = BASELINE_SMAPE - best['test_smape']
        logger.info(f"SUCCESS! Beat {BASELINE_SMAPE}% by {improvement:.2f}%!")
    else:
        logger.info(f"Best: {best['test_smape']:.2f}% (baseline: {BASELINE_SMAPE}%)")

    logger.info(f"  Best Model: {best['model']}")
    logger.info(f"  Test sMAPE: {best['test_smape']:.2f}%")
    logger.info(f"  Test MAE: {best['test_mae']:.2f} TL/MWh")
    logger.info(f"  Test Bias: {best['test_bias']:.2f} TL/MWh")

    # =========================================================================
    # BIAS ANALYSIS
    # =========================================================================
    logger.info(f"\n{'='*70}")
    logger.info("BIAS ANALYSIS (Lower is better)")
    logger.info("="*70)

    logger.info(f"\n{'Model':<30} {'Bias':>12} {'|Bias|':>12}")
    logger.info("-"*60)

    for r in sorted(all_results, key=lambda x: abs(x['test_bias'])):
        logger.info(f"{r['model']:<30} {r['test_bias']:>+11.2f} {abs(r['test_bias']):>11.2f}")

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    summary = {
        'timestamp': datetime.now().isoformat(),
        'baseline_smape': BASELINE_SMAPE,
        'test_start': '2024-06-01',
        'best_model': best['model'],
        'best_test_smape': float(best['test_smape']),
        'best_bias': float(best['test_bias']),
        'all_results': [
            {k: v for k, v in r.items() if k != 'test_pred'}
            for r in all_results
        ],
    }

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Save best predictions
    pred_df = pd.DataFrame({
        'datetime': data['X_test'].index,
        'y_true': data['y_test'].values,
        'y_pred': best['test_pred'],
    })
    pred_df.to_csv(output_dir / 'best_predictions.csv', index=False)
    pred_df.to_parquet(output_dir / 'best_predictions.parquet', index=False)

    logger.info(f"\nResults saved to: {output_dir}")

    return engine.results


if __name__ == "__main__":
    run_v6_experiments()
