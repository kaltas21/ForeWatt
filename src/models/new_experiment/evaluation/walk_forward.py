"""
Walk-Forward (Expanding Window) Cross-Validation
=================================================
Time-series appropriate validation that simulates real-world model retraining.

Handles concept drift by testing model performance across different time periods.

Author: ForeWatt Team
Date: November 2025
"""

import numpy as np
import pandas as pd
from typing import Generator, Tuple, List, Dict, Any, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class WalkForwardFold:
    """Container for a single walk-forward fold."""
    fold_idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_size: int
    test_size: int


def expanding_window_split(
    df: pd.DataFrame,
    min_train_size: int = 8760,  # 1 year in hours
    test_size: int = 720,        # 1 month in hours
    step_size: int = 720,        # Step by 1 month
    max_folds: Optional[int] = None
) -> Generator[Tuple[np.ndarray, np.ndarray, WalkForwardFold], None, None]:
    """
    Generator for walk-forward (expanding window) validation.

    Simulates real-world model retraining cycle:
    - Train on Jan-Mar, Test Apr
    - Train on Jan-Apr, Test May
    - Train on Jan-May, Test Jun
    - etc.

    Args:
        df: DataFrame with datetime index
        min_train_size: Minimum training set size (hours)
        test_size: Test set size for each fold (hours)
        step_size: How much to expand training window each fold (hours)
        max_folds: Maximum number of folds (None for all possible)

    Yields:
        Tuple of (train_indices, test_indices, fold_info)
    """
    n_samples = len(df)
    current_train_end = min_train_size
    fold_idx = 0

    while current_train_end + test_size <= n_samples:
        if max_folds is not None and fold_idx >= max_folds:
            break

        train_idx = np.arange(0, current_train_end)
        test_idx = np.arange(current_train_end, current_train_end + test_size)

        # Create fold info
        fold_info = WalkForwardFold(
            fold_idx=fold_idx,
            train_start=df.index[0],
            train_end=df.index[current_train_end - 1],
            test_start=df.index[current_train_end],
            test_end=df.index[min(current_train_end + test_size - 1, n_samples - 1)],
            train_size=len(train_idx),
            test_size=len(test_idx)
        )

        yield train_idx, test_idx, fold_info

        current_train_end += step_size
        fold_idx += 1


def sliding_window_split(
    df: pd.DataFrame,
    train_size: int = 8760,      # 1 year in hours
    test_size: int = 720,        # 1 month in hours
    step_size: int = 720,        # Step by 1 month
    max_folds: Optional[int] = None
) -> Generator[Tuple[np.ndarray, np.ndarray, WalkForwardFold], None, None]:
    """
    Generator for sliding window validation (fixed training window size).

    Unlike expanding window, training size stays constant:
    - Train on Jan-Dec 2022, Test Jan 2023
    - Train on Feb 2022-Jan 2023, Test Feb 2023
    - etc.

    Args:
        df: DataFrame with datetime index
        train_size: Fixed training set size (hours)
        test_size: Test set size for each fold (hours)
        step_size: How much to slide the window each fold (hours)
        max_folds: Maximum number of folds (None for all possible)

    Yields:
        Tuple of (train_indices, test_indices, fold_info)
    """
    n_samples = len(df)
    current_start = 0
    fold_idx = 0

    while current_start + train_size + test_size <= n_samples:
        if max_folds is not None and fold_idx >= max_folds:
            break

        train_idx = np.arange(current_start, current_start + train_size)
        test_idx = np.arange(current_start + train_size,
                            current_start + train_size + test_size)

        # Create fold info
        fold_info = WalkForwardFold(
            fold_idx=fold_idx,
            train_start=df.index[current_start],
            train_end=df.index[current_start + train_size - 1],
            test_start=df.index[current_start + train_size],
            test_end=df.index[min(current_start + train_size + test_size - 1, n_samples - 1)],
            train_size=len(train_idx),
            test_size=len(test_idx)
        )

        yield train_idx, test_idx, fold_info

        current_start += step_size
        fold_idx += 1


class WalkForwardValidator:
    """
    Walk-forward cross-validation for time series models.

    Handles the full validation loop including metric aggregation.
    """

    def __init__(
        self,
        min_train_size: int = 8760,
        test_size: int = 720,
        step_size: int = 720,
        max_folds: Optional[int] = None,
        mode: str = 'expanding'  # 'expanding' or 'sliding'
    ):
        """
        Initialize walk-forward validator.

        Args:
            min_train_size: Minimum/fixed training size (hours)
            test_size: Test size per fold (hours)
            step_size: Window step size (hours)
            max_folds: Max folds (None for all)
            mode: 'expanding' (growing train) or 'sliding' (fixed train size)
        """
        self.min_train_size = min_train_size
        self.test_size = test_size
        self.step_size = step_size
        self.max_folds = max_folds
        self.mode = mode

        self.fold_results: List[Dict[str, Any]] = []

    def get_splits(self, df: pd.DataFrame):
        """Get train/test splits for the dataframe."""
        if self.mode == 'expanding':
            return expanding_window_split(
                df, self.min_train_size, self.test_size,
                self.step_size, self.max_folds
            )
        else:
            return sliding_window_split(
                df, self.min_train_size, self.test_size,
                self.step_size, self.max_folds
            )

    def validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        trainer,
        hyperparams: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run walk-forward validation.

        Args:
            X: Features dataframe
            y: Target series
            trainer: Model trainer instance with train() and predict() methods
            hyperparams: Hyperparameters for training

        Returns:
            Dictionary with aggregated metrics and fold details
        """
        self.fold_results = []

        print(f"\n{'='*70}")
        print(f"  WALK-FORWARD VALIDATION ({self.mode.upper()})")
        print(f"{'='*70}")
        print(f"  Min train: {self.min_train_size}h, Test: {self.test_size}h, Step: {self.step_size}h")
        print(f"{'='*70}")

        from src.models.evaluate import (
            mean_absolute_error,
            symmetric_mean_absolute_percentage_error,
            mean_absolute_scaled_error
        )

        for train_idx, test_idx, fold_info in self.get_splits(X):
            print(f"\n  Fold {fold_info.fold_idx + 1}: "
                  f"Train {fold_info.train_start.strftime('%Y-%m')} - {fold_info.train_end.strftime('%Y-%m')} "
                  f"({fold_info.train_size}h) | "
                  f"Test {fold_info.test_start.strftime('%Y-%m')} ({fold_info.test_size}h)")

            # Split data
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

            # Use 20% of training for validation during training
            val_split = int(len(X_train) * 0.8)
            X_train_inner = X_train.iloc[:val_split]
            y_train_inner = y_train.iloc[:val_split]
            X_val_inner = X_train.iloc[val_split:]
            y_val_inner = y_train.iloc[val_split:]

            # Train model
            model, _ = trainer.train(
                X_train_inner, y_train_inner,
                X_val_inner, y_val_inner,
                hyperparams
            )

            # Predict on test
            predictions = trainer.predict(X_test)

            # Calculate metrics
            mae = mean_absolute_error(y_test.values, predictions)
            smape = symmetric_mean_absolute_percentage_error(y_test.values, predictions)
            mase = mean_absolute_scaled_error(
                y_test.values, predictions, y_train.values, seasonality=24
            )

            fold_result = {
                'fold_idx': fold_info.fold_idx,
                'train_start': fold_info.train_start,
                'train_end': fold_info.train_end,
                'test_start': fold_info.test_start,
                'test_end': fold_info.test_end,
                'train_size': fold_info.train_size,
                'test_size': fold_info.test_size,
                'MAE': mae,
                'sMAPE': smape,
                'MASE': mase
            }
            self.fold_results.append(fold_result)

            print(f"    -> MAE: {mae:.2f}, sMAPE: {smape:.2f}%, MASE: {mase:.4f}")

        # Aggregate results
        metrics_df = pd.DataFrame(self.fold_results)

        aggregated = {
            'n_folds': len(self.fold_results),
            'MAE_mean': metrics_df['MAE'].mean(),
            'MAE_std': metrics_df['MAE'].std(),
            'sMAPE_mean': metrics_df['sMAPE'].mean(),
            'sMAPE_std': metrics_df['sMAPE'].std(),
            'MASE_mean': metrics_df['MASE'].mean(),
            'MASE_std': metrics_df['MASE'].std(),
            'fold_details': self.fold_results
        }

        print(f"\n{'='*70}")
        print(f"  WALK-FORWARD RESULTS (n={aggregated['n_folds']} folds)")
        print(f"{'='*70}")
        print(f"  MAE:   {aggregated['MAE_mean']:.2f} ± {aggregated['MAE_std']:.2f}")
        print(f"  sMAPE: {aggregated['sMAPE_mean']:.2f}% ± {aggregated['sMAPE_std']:.2f}%")
        print(f"  MASE:  {aggregated['MASE_mean']:.4f} ± {aggregated['MASE_std']:.4f}")
        print(f"{'='*70}\n")

        return aggregated


def analyze_concept_drift(fold_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Analyze concept drift across walk-forward folds.

    Args:
        fold_results: List of fold result dictionaries

    Returns:
        DataFrame with drift analysis
    """
    df = pd.DataFrame(fold_results)

    # Calculate rolling metrics
    df['MAE_rolling_mean'] = df['MAE'].expanding().mean()
    df['sMAPE_rolling_mean'] = df['sMAPE'].expanding().mean()

    # Calculate trend (is performance degrading over time?)
    if len(df) >= 3:
        mae_trend = np.polyfit(range(len(df)), df['MAE'].values, 1)[0]
        smape_trend = np.polyfit(range(len(df)), df['sMAPE'].values, 1)[0]

        df['MAE_trend'] = mae_trend
        df['sMAPE_trend'] = smape_trend

        # Positive trend = degrading performance
        drift_detected = mae_trend > 0 or smape_trend > 0
        df['drift_detected'] = drift_detected

    return df


if __name__ == '__main__':
    # Example usage
    import sys
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(PROJECT_ROOT))

    from src.models.new_experiment.baseline.feature_preparer_baseline import (
        load_master_v2, BaselineFeaturePreparer
    )
    from src.models.new_experiment.baseline.models import CatBoostTrainer

    # Load data
    df = load_master_v2()
    preparer = BaselineFeaturePreparer(target='price_real', strategy='baseline_minimal')
    X, y, features = preparer.prepare_features(df)

    # Drop NaN
    mask = ~(X.isna().any(axis=1) | y.isna())
    X, y = X[mask], y[mask]

    # Create validator
    validator = WalkForwardValidator(
        min_train_size=8760 * 2,  # 2 years minimum training
        test_size=720,             # 1 month test
        step_size=720 * 3,         # Step by 3 months
        max_folds=4,               # Limit folds for demo
        mode='expanding'
    )

    # Create trainer
    trainer = CatBoostTrainer(target='price_real')

    # Run validation
    results = validator.validate(
        X, y, trainer,
        hyperparams={'iterations': 500, 'depth': 6, 'learning_rate': 0.05}
    )

    # Analyze drift
    drift_df = analyze_concept_drift(results['fold_details'])
    print("\nConcept Drift Analysis:")
    print(drift_df[['fold_idx', 'test_start', 'MAE', 'sMAPE', 'MAE_rolling_mean']].to_string())
