"""
Unified Evaluation Metrics for Time Series Forecasting
=======================================================
Implements standard metrics for electricity demand forecasting.

Metrics Included:
-----------------
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- MASE (Mean Absolute Scaled Error) - time series specific

Author: ForeWatt Team
Date: November 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        MAE value
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        RMSE value
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Calculate Mean Absolute Percentage Error.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        epsilon: Small value to avoid division by zero

    Returns:
        MAPE value as percentage
    """
    # Avoid division by zero
    y_true = np.where(np.abs(y_true) < epsilon, epsilon, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def mean_absolute_scaled_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonality: int = 24
) -> float:
    """
    Calculate Mean Absolute Scaled Error (MASE).

    MASE is a scale-independent metric that compares forecast error
    to the naive seasonal forecast (persistence model).

    MASE < 1: Better than naive seasonal forecast
    MASE = 1: Same as naive seasonal forecast
    MASE > 1: Worse than naive seasonal forecast

    Args:
        y_true: Actual values (test set)
        y_pred: Predicted values (test set)
        y_train: Training set values (for computing naive forecast error)
        seasonality: Seasonal period (24 for hourly electricity data)

    Returns:
        MASE value
    """
    # MAE of forecast
    mae_forecast = np.mean(np.abs(y_true - y_pred))

    # MAE of naive seasonal forecast on training set
    # Naive forecast: y(t) = y(t - seasonality)
    naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])
    mae_naive = np.mean(naive_errors)

    # Avoid division by zero
    if mae_naive == 0:
        return float('inf') if mae_forecast > 0 else 0.0

    return float(mae_forecast / mae_naive)


def symmetric_mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Symmetric MAPE (sMAPE).

    sMAPE addresses MAPE's asymmetry and is bounded [0, 200].

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        sMAPE value as percentage
    """
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    # Avoid division by zero
    denominator = np.where(denominator == 0, 1e-10, denominator)
    return float(np.mean(numerator / denominator) * 100)


def evaluate_forecast(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: Optional[np.ndarray] = None,
    seasonality: int = 24,
    model_name: str = "Model"
) -> Dict[str, float]:
    """
    Compute all evaluation metrics for a forecast.

    Args:
        y_true: Actual values (test/validation set)
        y_pred: Predicted values
        y_train: Training set (required for MASE)
        seasonality: Seasonal period (24 for hourly data)
        model_name: Name of model for logging

    Returns:
        Dictionary with all metrics
    """
    # Ensure numpy arrays
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # Check shapes match
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

    # Compute metrics
    metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': root_mean_squared_error(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred),
        'sMAPE': symmetric_mean_absolute_percentage_error(y_true, y_pred)
    }

    # Add MASE if training data provided
    if y_train is not None:
        y_train = np.asarray(y_train).flatten()
        metrics['MASE'] = mean_absolute_scaled_error(y_true, y_pred, y_train, seasonality)

    # Log results
    logger.info(f"\n{'='*60}")
    logger.info(f"{model_name} Evaluation Metrics")
    logger.info(f"{'='*60}")
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name:>10}: {value:>12.4f}")
    logger.info(f"{'='*60}\n")

    return metrics


def calculate_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Calculate forecast residuals for diagnostic plots.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        Dictionary with residuals and percentage errors
    """
    residuals = y_true - y_pred
    pct_errors = (residuals / y_true) * 100

    return {
        'residuals': residuals,
        'percentage_errors': pct_errors,
        'abs_residuals': np.abs(residuals),
        'squared_residuals': residuals ** 2
    }


def forecast_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate forecast bias (mean error).

    Positive bias: Systematic over-forecasting
    Negative bias: Systematic under-forecasting

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        Mean bias
    """
    return float(np.mean(y_pred - y_true))


if __name__ == '__main__':
    # Test metrics with sample data
    np.random.seed(42)

    # Generate synthetic data
    y_train = np.random.rand(1000) * 1000 + 20000
    y_true = np.random.rand(100) * 1000 + 20000
    y_pred = y_true + np.random.randn(100) * 500  # Add noise

    # Evaluate
    metrics = evaluate_forecast(
        y_true=y_true,
        y_pred=y_pred,
        y_train=y_train,
        seasonality=24,
        model_name="Test Model"
    )

    print("\nMetrics test passed!")
    print(f"MAE: {metrics['MAE']:.2f} MWh")
    print(f"RMSE: {metrics['RMSE']:.2f} MWh")
    print(f"MAPE: {metrics['MAPE']:.2f}%")
    print(f"MASE: {metrics['MASE']:.4f}")
