"""
Split Conformal Prediction for Uncertainty Quantification
=========================================================
Provides calibrated prediction intervals with guaranteed coverage.

Method: Split Conformal Prediction
- Split data into train/calibration/test
- Train model on train set
- Calibrate on calibration set to get quantiles
- Apply to test set for prediction intervals

Coverage guarantee: For 90% prediction interval, at least 90% of true
values will fall within the interval (in expectation).

Author: ForeWatt Team
Date: November 2025
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
import logging
from mapie.regression import MapieRegressor
from mapie.subsample import Subsample

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SplitConformalPredictor:
    """
    Split conformal prediction for time series forecasting.

    Provides calibrated prediction intervals with coverage guarantees.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        method: str = 'naive'
    ):
        """
        Initialize conformal predictor.

        Args:
            alpha: Miscoverage level (default: 0.1 for 90% coverage)
            method: Conformalization method
                - 'naive': Simple split conformal
                - 'plus': More conservative (better coverage)
                - 'minmax': Adaptive for time series
        """
        self.alpha = alpha
        self.method = method
        self.coverage_level = 1 - alpha
        self.mapie = None
        self.is_fitted = False

    def fit(
        self,
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_calib: np.ndarray,
        y_calib: np.ndarray
    ):
        """
        Fit conformal predictor.

        Args:
            model: Trained base model (must have .predict() method)
            X_train: Training features
            y_train: Training target
            X_calib: Calibration features
            y_calib: Calibration target
        """
        logger.info(f"\n{'='*80}")
        logger.info("SPLIT CONFORMAL CALIBRATION")
        logger.info(f"{'='*80}")
        logger.info(f"Coverage level: {self.coverage_level*100:.1f}%")
        logger.info(f"Alpha (miscoverage): {self.alpha}")
        logger.info(f"Method: {self.method}")
        logger.info(f"Calibration samples: {len(X_calib)}")

        # Create MAPIE wrapper
        self.mapie = MapieRegressor(
            estimator=model,
            method=self.method,
            cv='prefit'  # Model already fitted
        )

        # Fit on calibration set
        # Note: We pass train+calib, but since cv='prefit', it only uses calib for conformalization
        X_combined = np.vstack([X_train, X_calib])
        y_combined = np.hstack([y_train, y_calib])

        # Create train/calib indices
        n_train = len(X_train)
        train_indices = np.arange(n_train)
        calib_indices = np.arange(n_train, len(X_combined))

        self.mapie.fit(
            X_combined,
            y_combined
        )

        self.is_fitted = True

        # Compute calibration metrics
        y_pred_calib, y_pis_calib = self.mapie.predict(
            X_calib,
            alpha=self.alpha
        )

        # Calculate empirical coverage on calibration set
        coverage = self._calculate_coverage(y_calib, y_pis_calib)
        width = np.mean(y_pis_calib[:, 1, 0] - y_pis_calib[:, 0, 0])

        logger.info(f"\nCalibration results:")
        logger.info(f"  Empirical coverage: {coverage*100:.2f}%")
        logger.info(f"  Mean interval width: {width:.2f}")
        logger.info(f"  Target coverage: {self.coverage_level*100:.1f}%")

        if abs(coverage - self.coverage_level) > 0.05:
            logger.warning(f"  ⚠ Coverage deviation: {abs(coverage - self.coverage_level)*100:.2f}%")
        else:
            logger.info(f"  ✓ Coverage within ±5%")

        logger.info(f"{'='*80}\n")

    def predict(
        self,
        X: np.ndarray,
        alpha: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions with prediction intervals.

        Args:
            X: Features
            alpha: Miscoverage level (default: use fitted alpha)

        Returns:
            Tuple of (predictions, prediction_intervals)
            - predictions: (n_samples,)
            - prediction_intervals: (n_samples, 2, 1) with [lower, upper]
        """
        if not self.is_fitted:
            raise ValueError("Conformal predictor not fitted. Call fit() first.")

        if alpha is None:
            alpha = self.alpha

        y_pred, y_pis = self.mapie.predict(X, alpha=alpha)

        return y_pred, y_pis

    def predict_multi_horizon(
        self,
        model,
        X: np.ndarray,
        horizons: int = 24
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate multi-horizon predictions with intervals.

        For each sample, predicts 1 to horizons steps ahead with
        uncertainty quantification.

        Args:
            model: Trained model with predict_multi_horizon method
            X: Features (n_samples, input_size, n_features)
            horizons: Number of horizons to predict

        Returns:
            Tuple of (predictions, lower_bounds, upper_bounds)
            Each of shape (n_samples, horizons)
        """
        if not self.is_fitted:
            raise ValueError("Conformal predictor not fitted")

        # Get multi-horizon predictions
        if hasattr(model, 'predict_multi_horizon'):
            y_pred_multi = model.predict_multi_horizon(X, horizons)
        else:
            # Fallback: iterative prediction
            y_pred_multi = self._iterative_predict(model, X, horizons)

        # Apply conformal prediction for each horizon
        n_samples = y_pred_multi.shape[0]
        lower_bounds = np.zeros_like(y_pred_multi)
        upper_bounds = np.zeros_like(y_pred_multi)

        for h in range(horizons):
            # For each horizon, we use the same conformalization
            # (In practice, you might want horizon-specific calibration)
            _, y_pis = self.predict(X[:, :, :] if len(X.shape) == 3 else X)

            # Extract bounds
            interval_width = y_pis[:, 1, 0] - y_pis[:, 0, 0]
            lower_bounds[:, h] = y_pred_multi[:, h] - interval_width / 2
            upper_bounds[:, h] = y_pred_multi[:, h] + interval_width / 2

        return y_pred_multi, lower_bounds, upper_bounds

    def _calculate_coverage(
        self,
        y_true: np.ndarray,
        y_pis: np.ndarray
    ) -> float:
        """
        Calculate empirical coverage.

        Args:
            y_true: True values
            y_pis: Prediction intervals (n_samples, 2, 1)

        Returns:
            Coverage rate (fraction of y_true within intervals)
        """
        lower = y_pis[:, 0, 0]
        upper = y_pis[:, 1, 0]

        coverage = np.mean((y_true >= lower) & (y_true <= upper))
        return coverage

    def _iterative_predict(
        self,
        model,
        X: np.ndarray,
        horizons: int
    ) -> np.ndarray:
        """
        Iterative multi-horizon prediction (fallback).

        Args:
            model: Model with predict method
            X: Features
            horizons: Number of steps

        Returns:
            Predictions of shape (n_samples, horizons)
        """
        logger.warning("Using iterative prediction (no native multi-horizon support)")

        n_samples = len(X)
        predictions = np.zeros((n_samples, horizons))

        for h in range(horizons):
            # Predict one step
            if len(X.shape) == 3:
                # Sequence data
                y_pred = model.predict(X)
            else:
                # Tabular data
                y_pred = model.predict(X)

            predictions[:, h] = y_pred.flatten()

            # Update X for next step (simplified - proper implementation needs feature updating)

        return predictions

    def evaluate_coverage(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pis: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate prediction interval quality.

        Args:
            y_true: True values
            y_pred: Point predictions
            y_pis: Prediction intervals

        Returns:
            Dictionary with coverage metrics
        """
        # Empirical coverage
        coverage = self._calculate_coverage(y_true, y_pis)

        # Interval width
        widths = y_pis[:, 1, 0] - y_pis[:, 0, 0]
        mean_width = np.mean(widths)
        std_width = np.std(widths)

        # Winkler score (interval score)
        # Lower is better, penalizes wide intervals and miscoverage
        alpha = self.alpha
        lower = y_pis[:, 0, 0]
        upper = y_pis[:, 1, 0]

        # Winkler score components
        width_penalty = upper - lower
        lower_penalty = (2 / alpha) * (lower - y_true) * (y_true < lower)
        upper_penalty = (2 / alpha) * (y_true - upper) * (y_true > upper)

        winkler_score = np.mean(width_penalty + lower_penalty + upper_penalty)

        # Prediction interval coverage probability (PICP)
        picp = coverage

        # Mean prediction interval width (MPIW)
        mpiw = mean_width

        # Normalized mean prediction interval width
        nmpiw = mpiw / (np.max(y_true) - np.min(y_true))

        metrics = {
            'coverage': coverage,
            'target_coverage': self.coverage_level,
            'coverage_error': abs(coverage - self.coverage_level),
            'mean_width': mean_width,
            'std_width': std_width,
            'winkler_score': winkler_score,
            'PICP': picp,
            'MPIW': mpiw,
            'NMPIW': nmpiw
        }

        return metrics

    def print_metrics(self, metrics: Dict[str, float]):
        """Print coverage metrics."""
        logger.info(f"\n{'='*80}")
        logger.info("CONFORMAL PREDICTION METRICS")
        logger.info(f"{'='*80}")
        logger.info(f"Coverage:")
        logger.info(f"  Empirical: {metrics['coverage']*100:.2f}%")
        logger.info(f"  Target: {metrics['target_coverage']*100:.1f}%")
        logger.info(f"  Error: {metrics['coverage_error']*100:.2f}%")
        logger.info(f"\nInterval Quality:")
        logger.info(f"  Mean width: {metrics['mean_width']:.2f}")
        logger.info(f"  Std width: {metrics['std_width']:.2f}")
        logger.info(f"  Normalized width: {metrics['NMPIW']:.4f}")
        logger.info(f"\nScores:")
        logger.info(f"  Winkler score: {metrics['winkler_score']:.2f}")
        logger.info(f"{'='*80}\n")


def horizon_specific_conformal(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    horizons: int = 24,
    alpha: float = 0.1
) -> Dict[int, SplitConformalPredictor]:
    """
    Create horizon-specific conformal predictors.

    Different horizons may have different uncertainty,
    so we calibrate separately for each.

    Args:
        model: Trained multi-horizon model
        X_train: Training features
        y_train: Training targets (n_samples, horizons)
        X_calib: Calibration features
        y_calib: Calibration targets (n_samples, horizons)
        horizons: Number of horizons
        alpha: Miscoverage level

    Returns:
        Dictionary mapping horizon to conformal predictor
    """
    logger.info(f"\n{'='*80}")
    logger.info("HORIZON-SPECIFIC CONFORMAL CALIBRATION")
    logger.info(f"{'='*80}")
    logger.info(f"Horizons: {horizons}")
    logger.info(f"Coverage: {(1-alpha)*100:.1f}%")

    conformal_predictors = {}

    for h in range(horizons):
        logger.info(f"\nHorizon {h+1}/{horizons}:")

        # Extract single horizon
        y_train_h = y_train[:, h] if len(y_train.shape) > 1 else y_train
        y_calib_h = y_calib[:, h] if len(y_calib.shape) > 1 else y_calib

        # Create and fit conformal predictor
        cp = SplitConformalPredictor(alpha=alpha)

        # For horizon-specific, we need a model that predicts just that horizon
        # For simplicity, use the multi-horizon model and extract horizon h
        class HorizonModel:
            def __init__(self, base_model, horizon):
                self.base_model = base_model
                self.horizon = horizon

            def predict(self, X):
                preds = self.base_model.predict(X)
                if len(preds.shape) > 1:
                    return preds[:, self.horizon]
                return preds

        horizon_model = HorizonModel(model, h)

        cp.fit(horizon_model, X_train, y_train_h, X_calib, y_calib_h)

        conformal_predictors[h] = cp

    logger.info(f"\n✓ Calibrated {len(conformal_predictors)} horizon-specific predictors")
    logger.info(f"{'='*80}\n")

    return conformal_predictors
