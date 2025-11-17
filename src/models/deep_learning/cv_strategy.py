"""
Expanding-Window Temporal Cross-Validation
==========================================
Time series cross-validation that respects temporal ordering.

Strategy:
- Train on increasingly larger windows (expanding)
- Validate on fixed-size future windows
- No data leakage from future to past

Author: ForeWatt Team
Date: November 2025
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Iterator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExpandingWindowCV:
    """
    Expanding-window temporal cross-validation for time series.

    Unlike standard k-fold CV, this:
    1. Preserves temporal order (no shuffling)
    2. Uses expanding training window (grows with each fold)
    3. Validates on future data only

    Example with 3 folds:
        Fold 1: Train[0:100]  Val[100:120]
        Fold 2: Train[0:120]  Val[120:140]
        Fold 3: Train[0:140]  Val[140:160]

    This mimics real-world scenario where model is retrained with
    all available historical data and tested on future.
    """

    def __init__(
        self,
        n_splits: int = 5,
        min_train_size: int = None,
        val_size: int = None,
        gap: int = 0
    ):
        """
        Initialize expanding-window CV.

        Args:
            n_splits: Number of splits/folds
            min_train_size: Minimum training size (default: auto from data)
            val_size: Validation size for each fold (default: auto from data)
            gap: Gap between train and validation (default: 0, no gap)
        """
        self.n_splits = n_splits
        self.min_train_size = min_train_size
        self.val_size = val_size
        self.gap = gap

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices for train/validation splits.

        Args:
            X: Features dataframe (with datetime index)
            y: Target series (not used, for sklearn compatibility)

        Yields:
            Tuple of (train_indices, val_indices)
        """
        n_samples = len(X)

        # Auto-determine sizes if not provided
        if self.val_size is None:
            # Default: validation is ~10% of data
            self.val_size = max(int(n_samples * 0.1), 24)  # At least 24h

        if self.min_train_size is None:
            # Default: start with 50% of data for first fold
            self.min_train_size = max(int(n_samples * 0.5), 168)  # At least 1 week

        # Calculate step size for expanding window
        remaining = n_samples - self.min_train_size - self.gap
        step_size = (remaining - self.val_size) // self.n_splits

        if step_size < self.val_size:
            raise ValueError(
                f"Not enough data for {self.n_splits} splits. "
                f"Need at least {self.min_train_size + self.gap + self.val_size * (self.n_splits + 1)} samples, "
                f"but got {n_samples}"
            )

        logger.info(f"\n{'='*80}")
        logger.info(f"EXPANDING-WINDOW CROSS-VALIDATION")
        logger.info(f"{'='*80}")
        logger.info(f"Splits: {self.n_splits}")
        logger.info(f"Min train size: {self.min_train_size}")
        logger.info(f"Validation size: {self.val_size}")
        logger.info(f"Gap: {self.gap}")
        logger.info(f"Step size: {step_size}")
        logger.info(f"Total samples: {n_samples}")
        logger.info(f"{'='*80}\n")

        for i in range(self.n_splits):
            # Training: from start to expanding end
            train_end = self.min_train_size + (i * step_size)
            train_indices = np.arange(0, train_end)

            # Validation: after gap, for val_size
            val_start = train_end + self.gap
            val_end = val_start + self.val_size

            if val_end > n_samples:
                # Last fold: use remaining data
                val_end = n_samples

            val_indices = np.arange(val_start, val_end)

            # Log split info
            logger.info(f"Fold {i+1}/{self.n_splits}:")
            logger.info(f"  Train: [{train_indices[0]:6d}:{train_indices[-1]:6d}] "
                       f"({len(train_indices):6d} samples) "
                       f"{X.index[train_indices[0]]} to {X.index[train_indices[-1]]}")
            logger.info(f"  Val:   [{val_indices[0]:6d}:{val_indices[-1]:6d}] "
                       f"({len(val_indices):6d} samples) "
                       f"{X.index[val_indices[0]]} to {X.index[val_indices[-1]]}")

            yield train_indices, val_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return number of splits (sklearn compatibility)."""
        return self.n_splits


class ExpandingWindowCVForSequences:
    """
    Expanding-window CV specialized for sequence data.

    Handles the complexity of sequence generation where each sample
    needs input_size past observations and horizon future observations.
    """

    def __init__(
        self,
        n_splits: int = 5,
        min_train_size: int = None,
        val_size: int = None,
        input_size: int = 168,
        horizon: int = 24,
        gap: int = 0
    ):
        """
        Initialize expanding-window CV for sequences.

        Args:
            n_splits: Number of splits/folds
            min_train_size: Minimum training size in raw samples
            val_size: Validation size in raw samples
            input_size: Sequence input window size
            horizon: Forecast horizon
            gap: Gap between train and validation
        """
        self.n_splits = n_splits
        self.min_train_size = min_train_size
        self.val_size = val_size
        self.input_size = input_size
        self.horizon = horizon
        self.gap = gap

    def split_sequences(
        self,
        X: pd.DataFrame,
        y: pd.Series = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices for train/validation splits (accounting for sequences).

        Args:
            X: Features dataframe
            y: Target series

        Yields:
            Tuple of (train_indices, val_indices) for raw data
        """
        n_samples = len(X)

        # Account for sequence requirements
        seq_overhead = self.input_size + self.horizon

        # Auto-determine sizes
        if self.val_size is None:
            self.val_size = max(int(n_samples * 0.1), seq_overhead + 24)

        if self.min_train_size is None:
            self.min_train_size = max(int(n_samples * 0.5), seq_overhead + 168)

        # Calculate splits
        remaining = n_samples - self.min_train_size - self.gap
        step_size = (remaining - self.val_size) // self.n_splits

        if step_size < seq_overhead:
            raise ValueError(
                f"Not enough data for {self.n_splits} splits with "
                f"input_size={self.input_size} and horizon={self.horizon}. "
                f"Need larger dataset or fewer splits."
            )

        logger.info(f"\n{'='*80}")
        logger.info(f"EXPANDING-WINDOW CV FOR SEQUENCES")
        logger.info(f"{'='*80}")
        logger.info(f"Splits: {self.n_splits}")
        logger.info(f"Input size: {self.input_size}h")
        logger.info(f"Horizon: {self.horizon}h")
        logger.info(f"Min train size: {self.min_train_size}")
        logger.info(f"Validation size: {self.val_size}")
        logger.info(f"{'='*80}\n")

        for i in range(self.n_splits):
            # Training data
            train_end = self.min_train_size + (i * step_size)
            train_indices = np.arange(0, train_end)

            # Validation data
            val_start = train_end + self.gap
            val_end = min(val_start + self.val_size, n_samples)
            val_indices = np.arange(val_start, val_end)

            # Ensure enough data for sequences
            if len(train_indices) >= seq_overhead and len(val_indices) >= seq_overhead:
                logger.info(f"Fold {i+1}/{self.n_splits}:")
                logger.info(f"  Train: {len(train_indices):6d} samples "
                           f"({X.index[train_indices[0]]} to {X.index[train_indices[-1]]})")
                logger.info(f"  Val:   {len(val_indices):6d} samples "
                           f"({X.index[val_indices[0]]} to {X.index[val_indices[-1]]})")

                yield train_indices, val_indices
            else:
                logger.warning(f"Fold {i+1}: Skipping due to insufficient data for sequences")


def evaluate_cv_folds(
    cv_results: List[Dict],
    metric: str = 'sMAPE'
) -> Dict:
    """
    Aggregate cross-validation results.

    Args:
        cv_results: List of dicts with metrics for each fold
        metric: Primary metric to aggregate

    Returns:
        Dictionary with mean, std, and all fold results
    """
    if not cv_results:
        return {}

    # Extract metric values
    values = [fold[metric] for fold in cv_results if metric in fold]

    if not values:
        return {}

    summary = {
        f'{metric}_mean': np.mean(values),
        f'{metric}_std': np.std(values),
        f'{metric}_min': np.min(values),
        f'{metric}_max': np.max(values),
        f'{metric}_median': np.median(values),
        'n_folds': len(values),
        'fold_results': cv_results
    }

    return summary


def print_cv_summary(cv_results: List[Dict], metrics: List[str] = None):
    """
    Print cross-validation summary.

    Args:
        cv_results: List of dicts with metrics for each fold
        metrics: Metrics to display (default: sMAPE, MASE, MAE)
    """
    if not cv_results:
        logger.warning("No CV results to display")
        return

    if metrics is None:
        metrics = ['sMAPE', 'MASE', 'MAE', 'RMSE']

    logger.info(f"\n{'='*80}")
    logger.info("CROSS-VALIDATION SUMMARY")
    logger.info(f"{'='*80}")

    # Print per-fold results
    logger.info(f"\nPer-fold results:")
    logger.info(f"{'Fold':<6s} " + " ".join(f"{m:>10s}" for m in metrics))
    logger.info("-" * 80)

    for i, fold in enumerate(cv_results):
        values = [f"{fold.get(m, float('nan')):10.4f}" for m in metrics]
        logger.info(f"{i+1:<6d} " + " ".join(values))

    # Print aggregates
    logger.info("-" * 80)
    logger.info(f"{'Mean':<6s} " + " ".join(
        f"{np.mean([fold.get(m, float('nan')) for fold in cv_results]):10.4f}"
        for m in metrics
    ))
    logger.info(f"{'Std':<6s} " + " ".join(
        f"{np.std([fold.get(m, float('nan')) for fold in cv_results]):10.4f}"
        for m in metrics
    ))

    logger.info(f"{'='*80}\n")
