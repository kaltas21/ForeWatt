"""
ForeWatt Metrics Calculator
Utilities for calculating forecasting metrics and evaluation statistics.
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error."""
    return mean_absolute_error(y_true, y_pred)


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Calculate Mean Absolute Percentage Error.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        epsilon: Small value to avoid division by zero

    Returns:
        MAPE as percentage
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Avoid division by zero
    mask = np.abs(y_true) > epsilon
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    return mape


def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        epsilon: Small value to avoid division by zero

    Returns:
        sMAPE as percentage
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 + epsilon

    smape = np.mean(numerator / denominator) * 100

    return smape


def calculate_mase(y_true: np.ndarray, y_pred: np.ndarray,
                   y_train: np.ndarray, seasonality: int = 168) -> float:
    """
    Calculate Mean Absolute Scaled Error.

    MASE compares forecast error to naive seasonal forecast.
    Values < 1 indicate better than naive baseline.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        y_train: Training data for calculating naive forecast error
        seasonality: Seasonal period (default: 168 hours = 1 week)

    Returns:
        MASE value
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_train = np.array(y_train)

    # Forecast error
    mae_forecast = np.mean(np.abs(y_true - y_pred))

    # Naive seasonal forecast error on training data
    if len(y_train) > seasonality:
        naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])
        mae_naive = np.mean(naive_errors)
    else:
        # Fallback: use simple differencing
        naive_errors = np.abs(np.diff(y_train))
        mae_naive = np.mean(naive_errors)

    # Avoid division by zero
    if mae_naive < 1e-10:
        return np.inf

    mase = mae_forecast / mae_naive

    return mase


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R² Score."""
    return r2_score(y_true, y_pred)


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                         y_train: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate all standard forecasting metrics.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        y_train: Optional training data for MASE calculation

    Returns:
        Dictionary of metric names to values
    """
    metrics = {
        "MAE": calculate_mae(y_true, y_pred),
        "RMSE": calculate_rmse(y_true, y_pred),
        "MAPE": calculate_mape(y_true, y_pred),
        "sMAPE": calculate_smape(y_true, y_pred),
        "R²": calculate_r2(y_true, y_pred)
    }

    # Add MASE if training data provided
    if y_train is not None:
        metrics["MASE"] = calculate_mase(y_true, y_pred, y_train)

    return metrics


def calculate_horizon_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                              horizon: int = 24) -> pd.DataFrame:
    """
    Calculate metrics for each forecast horizon.

    Args:
        y_true: Actual values (must be multiple of horizon)
        y_pred: Predicted values
        horizon: Forecast horizon

    Returns:
        DataFrame with metrics per horizon
    """
    n_samples = len(y_true) // horizon
    results = []

    for h in range(1, horizon + 1):
        # Get values for this horizon
        indices = [i * horizon + (h - 1) for i in range(n_samples) if i * horizon + (h - 1) < len(y_true)]
        y_true_h = y_true[indices]
        y_pred_h = y_pred[indices]

        if len(y_true_h) > 0:
            metrics = {
                "Horizon": h,
                "MAE": calculate_mae(y_true_h, y_pred_h),
                "RMSE": calculate_rmse(y_true_h, y_pred_h),
                "sMAPE": calculate_smape(y_true_h, y_pred_h)
            }
            results.append(metrics)

    return pd.DataFrame(results)


def calculate_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Calculate forecast residuals."""
    return y_true - y_pred


def calculate_coverage(y_true: np.ndarray, lower: np.ndarray,
                       upper: np.ndarray) -> float:
    """
    Calculate prediction interval coverage.

    Args:
        y_true: Actual values
        lower: Lower bound of prediction intervals
        upper: Upper bound of prediction intervals

    Returns:
        Coverage as percentage (0-100)
    """
    in_interval = (y_true >= lower) & (y_true <= upper)
    coverage = np.mean(in_interval) * 100

    return coverage


def calculate_interval_width(lower: np.ndarray, upper: np.ndarray) -> Dict[str, float]:
    """
    Calculate prediction interval width statistics.

    Args:
        lower: Lower bounds
        upper: Upper bounds

    Returns:
        Dictionary with width statistics
    """
    widths = upper - lower

    return {
        "mean_width": np.mean(widths),
        "median_width": np.median(widths),
        "std_width": np.std(widths),
        "min_width": np.min(widths),
        "max_width": np.max(widths)
    }


def calculate_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate forecast bias (mean error).

    Positive values indicate over-forecasting, negative indicate under-forecasting.
    """
    return np.mean(y_pred - y_true)


def calculate_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate directional accuracy (percentage of correct directional predictions).

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        Directional accuracy as percentage (0-100)
    """
    if len(y_true) < 2:
        return np.nan

    # Calculate direction changes
    true_direction = np.sign(np.diff(y_true))
    pred_direction = np.sign(np.diff(y_pred))

    # Calculate accuracy
    correct = true_direction == pred_direction
    accuracy = np.mean(correct) * 100

    return accuracy


def calculate_error_percentiles(residuals: np.ndarray) -> Dict[str, float]:
    """
    Calculate error distribution percentiles.

    Args:
        residuals: Forecast residuals (actual - predicted)

    Returns:
        Dictionary with percentile values
    """
    abs_residuals = np.abs(residuals)

    return {
        "p5": np.percentile(abs_residuals, 5),
        "p25": np.percentile(abs_residuals, 25),
        "p50": np.percentile(abs_residuals, 50),
        "p75": np.percentile(abs_residuals, 75),
        "p95": np.percentile(abs_residuals, 95),
        "p99": np.percentile(abs_residuals, 99)
    }


def rolling_metric(y_true: np.ndarray, y_pred: np.ndarray,
                   metric_func, window: int = 168) -> np.ndarray:
    """
    Calculate rolling metric over time.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        metric_func: Function to calculate metric (e.g., calculate_mae)
        window: Rolling window size

    Returns:
        Array of rolling metric values
    """
    n = len(y_true)
    rolling_values = np.full(n, np.nan)

    for i in range(window, n + 1):
        rolling_values[i - 1] = metric_func(
            y_true[i - window:i],
            y_pred[i - window:i]
        )

    return rolling_values


def compare_models(models_results: Dict[str, Tuple[np.ndarray, np.ndarray]],
                   y_train: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    Compare multiple models on standard metrics.

    Args:
        models_results: Dictionary of model_name -> (y_true, y_pred) tuples
        y_train: Optional training data for MASE

    Returns:
        DataFrame with models as rows and metrics as columns
    """
    results = []

    for model_name, (y_true, y_pred) in models_results.items():
        metrics = calculate_all_metrics(y_true, y_pred, y_train)
        metrics['Model'] = model_name
        results.append(metrics)

    df = pd.DataFrame(results)
    df = df.set_index('Model')

    # Sort by MAE (lower is better)
    df = df.sort_values('MAE')

    return df


def calculate_skill_score(model_mae: float, baseline_mae: float) -> float:
    """
    Calculate skill score relative to baseline.

    Positive values indicate improvement over baseline.

    Args:
        model_mae: MAE of the model
        baseline_mae: MAE of the baseline

    Returns:
        Skill score as percentage
    """
    if baseline_mae == 0:
        return np.nan

    skill = ((baseline_mae - model_mae) / baseline_mae) * 100

    return skill


def format_metric_value(value: float, metric_name: str) -> str:
    """
    Format metric value for display.

    Args:
        value: Metric value
        metric_name: Name of the metric

    Returns:
        Formatted string
    """
    if np.isnan(value) or np.isinf(value):
        return "N/A"

    if metric_name in ["MAPE", "sMAPE"]:
        return f"{value:.2f}%"
    elif metric_name in ["R²", "MASE"]:
        return f"{value:.3f}"
    else:
        return f"{value:.2f}"
