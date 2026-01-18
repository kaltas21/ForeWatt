"""
Comprehensive Metrics Module
============================
All error metrics for model evaluation.
"""

import numpy as np
from typing import Dict, Optional
from sklearn.metrics import r2_score


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error"""
    return float(np.mean(np.abs(y_true - y_pred)))


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error"""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """Mean Absolute Percentage Error"""
    mask = np.abs(y_true) > epsilon
    if mask.sum() == 0:
        return np.nan
    return float(100 * np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """Symmetric Mean Absolute Percentage Error"""
    denominator = np.abs(y_true) + np.abs(y_pred) + epsilon
    return float(100 * np.mean(2 * np.abs(y_true - y_pred) / denominator))


def calculate_mase(y_true: np.ndarray, y_pred: np.ndarray,
                   y_train: np.ndarray, seasonality: int = 24) -> float:
    """Mean Absolute Scaled Error (vs seasonal naive)"""
    naive_errors = np.abs(np.diff(y_train[::seasonality]))
    mae_naive = np.mean(naive_errors) if len(naive_errors) > 0 else 1.0
    if mae_naive < 1e-8:
        mae_naive = 1.0
    return float(np.mean(np.abs(y_true - y_pred)) / mae_naive)


def calculate_mbe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Bias Error (positive = overestimate, negative = underestimate)"""
    return float(np.mean(y_pred - y_true))


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R-squared (coefficient of determination)"""
    return float(r2_score(y_true, y_pred))


def calculate_max_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Maximum absolute error"""
    return float(np.max(np.abs(y_true - y_pred)))


def calculate_median_ae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Median Absolute Error"""
    return float(np.median(np.abs(y_true - y_pred)))


def calculate_percentile_errors(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate error percentiles (P10, P50, P90, P95, P99)"""
    errors = np.abs(y_true - y_pred)
    return {
        'P10': float(np.percentile(errors, 10)),
        'P50': float(np.percentile(errors, 50)),
        'P90': float(np.percentile(errors, 90)),
        'P95': float(np.percentile(errors, 95)),
        'P99': float(np.percentile(errors, 99)),
    }


def calculate_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: Optional[np.ndarray] = None,
    seasonality: int = 24
) -> Dict[str, float]:
    """
    Calculate all metrics at once.

    Args:
        y_true: True values
        y_pred: Predicted values
        y_train: Training values (for MASE calculation)
        seasonality: Seasonality period for MASE (default 24 for hourly)

    Returns:
        Dictionary with all metrics
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    metrics = {
        'MAE': calculate_mae(y_true, y_pred),
        'RMSE': calculate_rmse(y_true, y_pred),
        'MAPE': calculate_mape(y_true, y_pred),
        'sMAPE': calculate_smape(y_true, y_pred),
        'MBE': calculate_mbe(y_true, y_pred),
        'R2': calculate_r2(y_true, y_pred),
        'MaxError': calculate_max_error(y_true, y_pred),
        'MedianAE': calculate_median_ae(y_true, y_pred),
    }

    if y_train is not None:
        y_train = np.asarray(y_train).flatten()
        metrics['MASE'] = calculate_mase(y_true, y_pred, y_train, seasonality)

    # Add percentiles
    percentiles = calculate_percentile_errors(y_true, y_pred)
    metrics.update({f'Error_{k}': v for k, v in percentiles.items()})

    return metrics


def format_metrics_table(metrics: Dict[str, float], precision: int = 4) -> str:
    """Format metrics as a nice table string"""
    lines = [
        "=" * 40,
        "METRICS SUMMARY",
        "=" * 40,
    ]

    # Primary metrics
    primary = ['MAE', 'RMSE', 'MAPE', 'sMAPE', 'MASE', 'MBE', 'R2']
    for m in primary:
        if m in metrics:
            val = metrics[m]
            if m in ['MAPE', 'sMAPE']:
                lines.append(f"  {m:>10}: {val:>{precision+5}.{precision}f}%")
            else:
                lines.append(f"  {m:>10}: {val:>{precision+5}.{precision}f}")

    lines.append("-" * 40)
    lines.append("Error Percentiles:")

    # Percentiles
    for key in ['Error_P10', 'Error_P50', 'Error_P90', 'Error_P95', 'Error_P99']:
        if key in metrics:
            lines.append(f"  {key.replace('Error_', ''):>10}: {metrics[key]:>{precision+5}.{precision}f}")

    lines.append("=" * 40)
    return "\n".join(lines)


def compare_metrics(metrics_dict: Dict[str, Dict[str, float]], metric_name: str = 'sMAPE') -> str:
    """
    Compare a specific metric across multiple experiments.

    Args:
        metrics_dict: {experiment_name: {metric_name: value, ...}, ...}
        metric_name: Which metric to compare

    Returns:
        Formatted comparison string
    """
    # Sort by the metric value
    sorted_items = sorted(
        metrics_dict.items(),
        key=lambda x: x[1].get(metric_name, float('inf'))
    )

    lines = [
        f"\n{'='*60}",
        f"COMPARISON BY {metric_name}",
        f"{'='*60}",
        f"{'Rank':<6} {'Experiment':<30} {metric_name:>15}",
        "-" * 60,
    ]

    for rank, (name, metrics) in enumerate(sorted_items, 1):
        val = metrics.get(metric_name, float('nan'))
        suffix = '%' if metric_name in ['MAPE', 'sMAPE'] else ''
        lines.append(f"{rank:<6} {name:<30} {val:>14.4f}{suffix}")

    lines.append("=" * 60)
    return "\n".join(lines)
