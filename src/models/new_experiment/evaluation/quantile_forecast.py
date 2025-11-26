"""
Quantile Regression for Probabilistic Forecasting
==================================================
Provides uncertainty estimates with prediction intervals.

Energy trading needs not just the forecast but the risk:
- Is it 100 EUR/MWh ± 5 or ± 50?
- What's the probability of extreme prices?

Author: ForeWatt Team
Date: November 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QuantilePrediction:
    """Container for quantile predictions."""
    lower: np.ndarray      # Lower bound (e.g., 10th percentile)
    median: np.ndarray     # Median (50th percentile)
    upper: np.ndarray      # Upper bound (e.g., 90th percentile)
    quantiles: List[float] # Quantile levels used


class QuantileCatBoostTrainer:
    """
    CatBoost with Quantile Regression for prediction intervals.

    Trains three models: lower (10%), median (50%), upper (90%) quantiles.
    """

    def __init__(
        self,
        target: str = 'price_real',
        quantiles: List[float] = [0.1, 0.5, 0.9],
        random_seed: int = 42
    ):
        """
        Initialize quantile trainer.

        Args:
            target: Target variable name
            quantiles: List of quantiles to predict
            random_seed: Random seed
        """
        self.target = target
        self.quantiles = sorted(quantiles)
        self.random_seed = random_seed
        self.models: Dict[float, Any] = {}
        self.feature_names = None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        hyperparams: Dict[str, Any]
    ) -> Tuple[Dict[float, Any], Dict[str, float]]:
        """
        Train quantile regression models.

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            hyperparams: Model hyperparameters

        Returns:
            Tuple of (models_dict, metrics_dict)
        """
        try:
            from catboost import CatBoostRegressor
        except ImportError:
            raise ImportError("CatBoost required: pip install catboost")

        print(f"\n{'='*70}")
        print(f"  QUANTILE CATBOOST TRAINING: {self.target}")
        print(f"{'='*70}")
        print(f"  Quantiles: {self.quantiles}")
        print(f"  Data: train={len(X_train)}, val={len(X_val)}, features={X_train.shape[1]}")
        print(f"{'='*70}")

        self.feature_names = list(X_train.columns)
        all_metrics = {}

        for q in self.quantiles:
            print(f"\n  Training Q{int(q*100):02d} model...")

            params = {
                'iterations': hyperparams.get('iterations', 1000),
                'depth': hyperparams.get('depth', 6),
                'learning_rate': hyperparams.get('learning_rate', 0.05),
                'l2_leaf_reg': hyperparams.get('l2_leaf_reg', 3.0),
                'random_seed': self.random_seed,
                'loss_function': f'Quantile:alpha={q}',
                'eval_metric': 'MAE',
                'early_stopping_rounds': 50,
                'verbose': 100,
            }

            model = CatBoostRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                use_best_model=True
            )

            self.models[q] = model

            # Calculate coverage on validation
            val_pred = model.predict(X_val)
            all_metrics[f'Q{int(q*100):02d}_MAE'] = np.mean(np.abs(y_val.values - val_pred))

        # Calculate prediction interval coverage
        if len(self.quantiles) >= 2:
            lower_q = min(self.quantiles)
            upper_q = max(self.quantiles)

            lower_pred = self.models[lower_q].predict(X_val)
            upper_pred = self.models[upper_q].predict(X_val)

            coverage = np.mean((y_val.values >= lower_pred) & (y_val.values <= upper_pred))
            expected_coverage = upper_q - lower_q

            all_metrics['interval_coverage'] = coverage
            all_metrics['expected_coverage'] = expected_coverage
            all_metrics['interval_width_mean'] = np.mean(upper_pred - lower_pred)

            print(f"\n  Prediction Interval ({int(lower_q*100)}-{int(upper_q*100)}%):")
            print(f"    Coverage: {coverage*100:.1f}% (expected: {expected_coverage*100:.0f}%)")
            print(f"    Mean width: {all_metrics['interval_width_mean']:.2f}")

        print(f"  Training complete.")
        return self.models, all_metrics

    def predict(self, X: pd.DataFrame) -> QuantilePrediction:
        """
        Generate quantile predictions.

        Args:
            X: Features dataframe

        Returns:
            QuantilePrediction with lower, median, upper bounds
        """
        if not self.models:
            raise ValueError("Models not trained. Call train() first.")

        predictions = {}
        for q in self.quantiles:
            predictions[q] = self.models[q].predict(X)

        # Find lower, median, upper
        lower_q = min(self.quantiles)
        upper_q = max(self.quantiles)
        median_q = 0.5 if 0.5 in self.quantiles else self.quantiles[len(self.quantiles)//2]

        return QuantilePrediction(
            lower=predictions[lower_q],
            median=predictions[median_q],
            upper=predictions[upper_q],
            quantiles=self.quantiles
        )


class QuantileLightGBMTrainer:
    """
    LightGBM with Quantile Regression for prediction intervals.
    """

    def __init__(
        self,
        target: str = 'price_real',
        quantiles: List[float] = [0.1, 0.5, 0.9],
        random_seed: int = 42
    ):
        self.target = target
        self.quantiles = sorted(quantiles)
        self.random_seed = random_seed
        self.models: Dict[float, Any] = {}
        self.feature_names = None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        hyperparams: Dict[str, Any]
    ) -> Tuple[Dict[float, Any], Dict[str, float]]:
        """Train quantile regression models."""
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("LightGBM required: pip install lightgbm")

        print(f"\n{'='*70}")
        print(f"  QUANTILE LIGHTGBM TRAINING: {self.target}")
        print(f"{'='*70}")
        print(f"  Quantiles: {self.quantiles}")
        print(f"  Data: train={len(X_train)}, val={len(X_val)}, features={X_train.shape[1]}")
        print(f"{'='*70}")

        self.feature_names = list(X_train.columns)
        all_metrics = {}

        for q in self.quantiles:
            print(f"\n  Training Q{int(q*100):02d} model...")

            params = {
                'n_estimators': hyperparams.get('n_estimators', 1000),
                'max_depth': hyperparams.get('max_depth', 6),
                'learning_rate': hyperparams.get('learning_rate', 0.05),
                'num_leaves': hyperparams.get('num_leaves', 63),
                'random_state': self.random_seed,
                'objective': 'quantile',
                'alpha': q,
                'metric': 'mae',
                'verbosity': -1,
                'force_row_wise': True,
            }

            callbacks = [
                lgb.early_stopping(stopping_rounds=50, verbose=True),
                lgb.log_evaluation(period=100)
            ]

            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=callbacks
            )

            self.models[q] = model
            val_pred = model.predict(X_val)
            all_metrics[f'Q{int(q*100):02d}_MAE'] = np.mean(np.abs(y_val.values - val_pred))

        # Calculate coverage
        if len(self.quantiles) >= 2:
            lower_q = min(self.quantiles)
            upper_q = max(self.quantiles)

            lower_pred = self.models[lower_q].predict(X_val)
            upper_pred = self.models[upper_q].predict(X_val)

            coverage = np.mean((y_val.values >= lower_pred) & (y_val.values <= upper_pred))
            expected_coverage = upper_q - lower_q

            all_metrics['interval_coverage'] = coverage
            all_metrics['expected_coverage'] = expected_coverage
            all_metrics['interval_width_mean'] = np.mean(upper_pred - lower_pred)

            print(f"\n  Prediction Interval ({int(lower_q*100)}-{int(upper_q*100)}%):")
            print(f"    Coverage: {coverage*100:.1f}% (expected: {expected_coverage*100:.0f}%)")
            print(f"    Mean width: {all_metrics['interval_width_mean']:.2f}")

        print(f"  Training complete.")
        return self.models, all_metrics

    def predict(self, X: pd.DataFrame) -> QuantilePrediction:
        """Generate quantile predictions."""
        if not self.models:
            raise ValueError("Models not trained. Call train() first.")

        predictions = {}
        for q in self.quantiles:
            predictions[q] = self.models[q].predict(X)

        lower_q = min(self.quantiles)
        upper_q = max(self.quantiles)
        median_q = 0.5 if 0.5 in self.quantiles else self.quantiles[len(self.quantiles)//2]

        return QuantilePrediction(
            lower=predictions[lower_q],
            median=predictions[median_q],
            upper=predictions[upper_q],
            quantiles=self.quantiles
        )


def evaluate_quantile_forecast(
    y_true: np.ndarray,
    predictions: QuantilePrediction
) -> Dict[str, float]:
    """
    Evaluate quantile forecast quality.

    Metrics:
    - Coverage: % of actuals within prediction interval
    - Pinball loss: Quantile-specific loss
    - Interval width: Average prediction interval width
    - Calibration: How close coverage is to expected

    Args:
        y_true: Actual values
        predictions: QuantilePrediction object

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Median accuracy (standard metrics)
    median_mae = np.mean(np.abs(y_true - predictions.median))
    metrics['median_MAE'] = median_mae

    # Pinball loss for each quantile
    for i, q in enumerate(predictions.quantiles):
        if q == min(predictions.quantiles):
            pred = predictions.lower
        elif q == max(predictions.quantiles):
            pred = predictions.upper
        else:
            pred = predictions.median

        # Pinball loss
        errors = y_true - pred
        pinball = np.mean(np.where(errors >= 0, q * errors, (q - 1) * errors))
        metrics[f'Q{int(q*100):02d}_pinball'] = pinball

    # Interval coverage
    coverage = np.mean((y_true >= predictions.lower) & (y_true <= predictions.upper))
    expected_coverage = max(predictions.quantiles) - min(predictions.quantiles)

    metrics['coverage'] = coverage
    metrics['expected_coverage'] = expected_coverage
    metrics['calibration_error'] = abs(coverage - expected_coverage)

    # Interval width
    metrics['interval_width_mean'] = np.mean(predictions.upper - predictions.lower)
    metrics['interval_width_std'] = np.std(predictions.upper - predictions.lower)

    # Winkler score (combines coverage and width)
    alpha = 1 - expected_coverage
    width = predictions.upper - predictions.lower
    penalty_lower = 2/alpha * (predictions.lower - y_true) * (y_true < predictions.lower)
    penalty_upper = 2/alpha * (y_true - predictions.upper) * (y_true > predictions.upper)
    winkler = np.mean(width + penalty_lower + penalty_upper)
    metrics['winkler_score'] = winkler

    return metrics


if __name__ == '__main__':
    # Example usage
    import sys
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(PROJECT_ROOT))

    from src.models.new_experiment.baseline.feature_preparer_baseline import (
        load_master_v2, BaselineFeaturePreparer
    )

    # Load data
    df = load_master_v2()
    preparer = BaselineFeaturePreparer(target='price_real', strategy='baseline_minimal')
    X_train, X_val, X_test, y_train, y_val, y_test, features = preparer.prepare_train_val_test(df)

    # Train quantile model
    trainer = QuantileCatBoostTrainer(
        target='price_real',
        quantiles=[0.1, 0.5, 0.9]
    )

    models, train_metrics = trainer.train(
        X_train, y_train, X_val, y_val,
        hyperparams={'iterations': 500, 'depth': 6, 'learning_rate': 0.05}
    )

    # Predict on test
    predictions = trainer.predict(X_test)

    # Evaluate
    metrics = evaluate_quantile_forecast(y_test.values, predictions)

    print("\n" + "="*70)
    print("  TEST SET QUANTILE METRICS")
    print("="*70)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
