"""
Baseline Models for Comparison
==============================
Naive and simple baselines to compare against ML models.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from experiments.metrics import calculate_all_metrics


class NaiveBaseline:
    """
    Naive forecaster: predicts the same hour from previous day.
    (Lag-24h persistence)
    """
    def __init__(self):
        self.name = "Naive (Lag-24h)"

    def predict(self, y_train: np.ndarray, n_test: int) -> np.ndarray:
        """Use last day of training repeated"""
        last_day = y_train[-24:]
        repeats = (n_test // 24) + 1
        predictions = np.tile(last_day, repeats)[:n_test]
        return predictions

    def predict_with_lags(self, y_full: np.ndarray, test_start_idx: int) -> np.ndarray:
        """Predict using actual lag-24h values (true naive)"""
        n_test = len(y_full) - test_start_idx
        predictions = np.zeros(n_test)
        for i in range(n_test):
            predictions[i] = y_full[test_start_idx + i - 24]
        return predictions


class SeasonalNaive:
    """
    Seasonal naive forecaster: predicts same hour from previous week.
    (Lag-168h persistence)
    """
    def __init__(self):
        self.name = "Seasonal Naive (Lag-168h)"

    def predict(self, y_train: np.ndarray, n_test: int) -> np.ndarray:
        """Use last week of training repeated"""
        last_week = y_train[-168:]
        repeats = (n_test // 168) + 1
        predictions = np.tile(last_week, repeats)[:n_test]
        return predictions

    def predict_with_lags(self, y_full: np.ndarray, test_start_idx: int) -> np.ndarray:
        """Predict using actual lag-168h values (true seasonal naive)"""
        n_test = len(y_full) - test_start_idx
        predictions = np.zeros(n_test)
        for i in range(n_test):
            predictions[i] = y_full[test_start_idx + i - 168]
        return predictions


class RollingMeanBaseline:
    """
    Rolling mean forecaster: predicts based on rolling average.
    """
    def __init__(self, window: int = 24):
        self.window = window
        self.name = f"Rolling Mean ({window}h)"

    def predict(self, y_train: np.ndarray, n_test: int) -> np.ndarray:
        """Use rolling mean from end of training"""
        rolling_mean = np.mean(y_train[-self.window:])
        return np.full(n_test, rolling_mean)


class HourlyMeanBaseline:
    """
    Hourly mean forecaster: predicts average for each hour from training data.
    """
    def __init__(self):
        self.name = "Hourly Mean"
        self.hourly_means = None

    def fit(self, y_train: np.ndarray, hours_train: np.ndarray) -> None:
        """Calculate mean for each hour"""
        self.hourly_means = {}
        for h in range(24):
            mask = hours_train == h
            if mask.sum() > 0:
                self.hourly_means[h] = np.mean(y_train[mask])
            else:
                self.hourly_means[h] = np.mean(y_train)

    def predict(self, hours_test: np.ndarray) -> np.ndarray:
        """Predict using hourly means"""
        return np.array([self.hourly_means.get(h, 0) for h in hours_test])


class DayOfWeekHourlyMeanBaseline:
    """
    Day-of-week + Hour mean forecaster: predicts average for each (dow, hour) combination.
    """
    def __init__(self):
        self.name = "DOW-Hourly Mean"
        self.dow_hour_means = None

    def fit(self, y_train: np.ndarray, hours_train: np.ndarray, dow_train: np.ndarray) -> None:
        """Calculate mean for each (dow, hour) combination"""
        self.dow_hour_means = {}
        for dow in range(7):
            for hour in range(24):
                mask = (dow_train == dow) & (hours_train == hour)
                if mask.sum() > 0:
                    self.dow_hour_means[(dow, hour)] = np.mean(y_train[mask])
                else:
                    self.dow_hour_means[(dow, hour)] = np.mean(y_train)

    def predict(self, hours_test: np.ndarray, dow_test: np.ndarray) -> np.ndarray:
        """Predict using (dow, hour) means"""
        return np.array([self.dow_hour_means.get((d, h), 0) for d, h in zip(dow_test, hours_test)])


def run_baseline_experiments(
    y_full: np.ndarray,
    hours_full: np.ndarray,
    dow_full: np.ndarray,
    test_start_idx: int,
    y_train: np.ndarray,
    hours_train: np.ndarray = None,
    dow_train: np.ndarray = None
) -> Dict[str, Dict]:
    """
    Run all baseline models and return their predictions and metrics.

    Args:
        y_full: Full target array
        hours_full: Full hours array
        dow_full: Full day-of-week array
        test_start_idx: Index where test set starts
        y_train: Training target values
        hours_train: Training hours (optional, derived from hours_full if not provided)
        dow_train: Training dow (optional, derived from dow_full if not provided)

    Returns:
        Dictionary with baseline results
    """
    y_test = y_full[test_start_idx:]
    hours_test = hours_full[test_start_idx:]
    dow_test = dow_full[test_start_idx:]

    # Use provided or derive from full arrays
    if hours_train is None:
        hours_train = hours_full[:test_start_idx]
    if dow_train is None:
        dow_train = dow_full[:test_start_idx]

    # Ensure train arrays match y_train length
    if len(hours_train) != len(y_train):
        # Adjust to match y_train length (take last n elements)
        hours_train = hours_train[-len(y_train):]
    if len(dow_train) != len(y_train):
        dow_train = dow_train[-len(y_train):]

    results = {}

    # 1. Naive (Lag-24h)
    naive = NaiveBaseline()
    naive_pred = naive.predict_with_lags(y_full, test_start_idx)
    naive_metrics = calculate_all_metrics(y_test, naive_pred, y_train)
    results['naive_24h'] = {
        'name': naive.name,
        'predictions': naive_pred,
        'metrics': naive_metrics
    }

    # 2. Seasonal Naive (Lag-168h)
    seasonal = SeasonalNaive()
    seasonal_pred = seasonal.predict_with_lags(y_full, test_start_idx)
    seasonal_metrics = calculate_all_metrics(y_test, seasonal_pred, y_train)
    results['seasonal_168h'] = {
        'name': seasonal.name,
        'predictions': seasonal_pred,
        'metrics': seasonal_metrics
    }

    # 3. Rolling Mean (24h)
    rolling = RollingMeanBaseline(window=24)
    rolling_pred = rolling.predict(y_train, len(y_test))
    rolling_metrics = calculate_all_metrics(y_test, rolling_pred, y_train)
    results['rolling_mean_24h'] = {
        'name': rolling.name,
        'predictions': rolling_pred,
        'metrics': rolling_metrics
    }

    # 4. Hourly Mean
    hourly = HourlyMeanBaseline()
    hourly.fit(y_train, hours_train)
    hourly_pred = hourly.predict(hours_test)
    hourly_metrics = calculate_all_metrics(y_test, hourly_pred, y_train)
    results['hourly_mean'] = {
        'name': hourly.name,
        'predictions': hourly_pred,
        'metrics': hourly_metrics
    }

    # 5. DOW-Hourly Mean
    dow_hourly = DayOfWeekHourlyMeanBaseline()
    dow_hourly.fit(y_train, hours_train, dow_train)
    dow_hourly_pred = dow_hourly.predict(hours_test, dow_test)
    dow_hourly_metrics = calculate_all_metrics(y_test, dow_hourly_pred, y_train)
    results['dow_hourly_mean'] = {
        'name': dow_hourly.name,
        'predictions': dow_hourly_pred,
        'metrics': dow_hourly_metrics
    }

    return results
