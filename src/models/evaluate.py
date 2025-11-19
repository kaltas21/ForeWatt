"""
Evaluation Metrics for Forecasting Models
==========================================
Comprehensive evaluation functions for forecasting models.

Metrics:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- sMAPE (Symmetric Mean Absolute Percentage Error)
- MASE (Mean Absolute Scaled Error)

Author: ForeWatt Team
Date: November 2025
"""

import numpy as np
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        MAE value
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        RMSE value
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        MAPE value in percentage
    """
    # Avoid division by zero
    mask = y_true != 0
    if not np.any(mask):
        return np.nan

    return float(100 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def symmetric_mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        sMAPE value in percentage
    """
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    # Avoid division by zero
    mask = denominator != 0
    if not np.any(mask):
        return np.nan

    return float(100 * np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]))


def mean_absolute_scaled_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonality: int = 1
) -> float:
    """
    Calculate Mean Absolute Scaled Error.

    MASE scales the MAE by the MAE of the naive seasonal forecast on the training set.
    A value < 1 indicates the forecast is better than the naive seasonal forecast.

    Args:
        y_true: True values
        y_pred: Predicted values
        y_train: Training data for scaling
        seasonality: Seasonal period for naive forecast (1 for non-seasonal)

    Returns:
        MASE value
    """
    # Calculate MAE of forecast
    mae_forecast = np.mean(np.abs(y_true - y_pred))

    # Calculate MAE of naive seasonal forecast on training set
    # Naive forecast: y_t = y_{t-seasonality}
    if len(y_train) <= seasonality:
        # If not enough training data, use simple difference
        naive_errors = np.abs(np.diff(y_train))
    else:
        naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])

    mae_naive = np.mean(naive_errors)

    # Avoid division by zero
    if mae_naive == 0:
        return np.nan

    return float(mae_forecast / mae_naive)


def evaluate_forecast(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonality: int = 1,
    model_name: Optional[str] = None
) -> Dict[str, float]:
    """
    Evaluate forecast with comprehensive metrics.

    Args:
        y_true: True values
        y_pred: Predicted values
        y_train: Training data for MASE calculation
        seasonality: Seasonal period for MASE (default: 1)
        model_name: Optional model name for logging

    Returns:
        Dictionary with all metrics
    """
    metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': root_mean_squared_error(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred),
        'sMAPE': symmetric_mean_absolute_percentage_error(y_true, y_pred),
        'MASE': mean_absolute_scaled_error(y_true, y_pred, y_train, seasonality)
    }

    if model_name:
        logger.info(f"\nEvaluation Metrics for {model_name}:")
        logger.info(f"  MAE:   {metrics['MAE']:.4f}")
        logger.info(f"  RMSE:  {metrics['RMSE']:.4f}")
        logger.info(f"  MAPE:  {metrics['MAPE']:.4f}%")
        logger.info(f"  sMAPE: {metrics['sMAPE']:.4f}%")
        logger.info(f"  MASE:  {metrics['MASE']:.4f}")

    return metrics
