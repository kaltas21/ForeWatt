"""
ForeWatt Model Loader
Utilities for loading models from MLflow and making predictions.
"""
import pandas as pd
import streamlit as st
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import mlflow
import numpy as np
import sys

from .config import MLRUNS_DIR, BASELINE_MODELS, DEEP_LEARNING_MODELS, TARGET_VARIABLE


@st.cache_resource
def get_mlflow_experiments() -> List[Dict]:
    """
    Get all MLflow experiments with their metadata.

    Returns:
        List of experiment dictionaries
    """
    try:
        mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")
        experiments = mlflow.search_experiments()

        exp_list = []
        for exp in experiments:
            exp_dict = {
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "artifact_location": exp.artifact_location,
                "lifecycle_stage": exp.lifecycle_stage
            }
            exp_list.append(exp_dict)

        return exp_list
    except Exception as e:
        st.warning(f"Could not load MLflow experiments: {e}")
        return []


@st.cache_resource
def get_experiment_runs(experiment_name: str) -> pd.DataFrame:
    """
    Get all runs for a specific experiment.

    Args:
        experiment_name: Name of the MLflow experiment

    Returns:
        DataFrame with run metadata and metrics
    """
    try:
        mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")
        experiment = mlflow.get_experiment_by_name(experiment_name)

        if experiment is None:
            return pd.DataFrame()

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"]
        )

        return runs
    except Exception as e:
        st.warning(f"Could not load runs for {experiment_name}: {e}")
        return pd.DataFrame()


@st.cache_resource
def load_best_model(experiment_name: str, metric: str = "metrics.MAE"):
    """
    Load the best model from an experiment based on a metric.

    Args:
        experiment_name: Name of the MLflow experiment
        metric: Metric to optimize (lower is better)

    Returns:
        Loaded model object or None
    """
    try:
        runs = get_experiment_runs(experiment_name)

        if runs.empty:
            return None

        # Try to find the metric column (case-insensitive)
        metric_col = None
        metric_name = metric.replace("metrics.", "").lower()

        for col in runs.columns:
            if col.lower() == f"metrics.{metric_name}":
                metric_col = col
                break

        if metric_col is None:
            st.warning(f"Metric {metric} not found in {experiment_name}")
            return None

        # Filter for successful runs with the metric
        valid_runs = runs[runs[metric_col].notna()].copy()

        if valid_runs.empty:
            return None

        # Get best run (lowest metric value)
        best_run = valid_runs.loc[valid_runs[metric_col].idxmin()]
        run_id = best_run['run_id']

        # Load the model
        mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")
        model_uri = f"runs:/{run_id}/model"
        model = mlflow.pyfunc.load_model(model_uri)

        return model
    except Exception as e:
        st.warning(f"Could not load best model for {experiment_name}: {e}")
        return None


@st.cache_data
def get_model_metrics(experiment_name: str) -> Dict[str, float]:
    """
    Get metrics for the best model in an experiment.

    Args:
        experiment_name: Name of the MLflow experiment

    Returns:
        Dictionary of metrics
    """
    try:
        runs = get_experiment_runs(experiment_name)

        if runs.empty:
            return {}

        # Try to find MAE metric column (case-insensitive)
        metric_col = None
        for col in runs.columns:
            if col.lower() == "metrics.mae":
                metric_col = col
                break

        if metric_col is None:
            st.warning(f"Could not find MAE metric in {experiment_name}")
            return {}

        # Get best run based on MAE
        valid_runs = runs[runs[metric_col].notna()].copy()

        if valid_runs.empty:
            return {}

        best_run = valid_runs.loc[valid_runs[metric_col].idxmin()]

        # Extract metrics
        metrics = {}
        for col in best_run.index:
            if col.lower().startswith("metrics."):
                metric_name = col.replace("metrics.", "").replace("metrics.", "").upper()
                metrics[metric_name] = best_run[col]

        return metrics
    except Exception as e:
        st.warning(f"Could not load metrics for {experiment_name}: {e}")
        return {}


def make_forecast(model, features: pd.DataFrame, horizon: int = 24) -> np.ndarray:
    """
    Generate forecast using a loaded model.

    Args:
        model: Loaded model object
        features: Feature dataframe
        horizon: Forecast horizon (hours)

    Returns:
        Array of predictions
    """
    try:
        # Different models have different predict signatures
        if hasattr(model, 'predict'):
            predictions = model.predict(features)
        else:
            # MLflow wrapped model
            predictions = model._model_impl.python_model.predict(None, features)

        return predictions[:horizon]
    except Exception as e:
        st.error(f"Error making forecast: {e}")
        return np.array([])


@st.cache_data
def get_available_models() -> Dict[str, Dict]:
    """
    Get list of available models with their metadata.

    Returns:
        Dictionary of model names to metadata
    """
    available = {}

    # Check baseline models
    for model_name, meta in BASELINE_MODELS.items():
        exp_name = meta["experiment_name"]
        runs = get_experiment_runs(exp_name)

        if not runs.empty:
            available[model_name] = {
                **meta,
                "available": True,
                "num_runs": len(runs)
            }

    return available


@st.cache_data
def load_all_model_metrics() -> pd.DataFrame:
    """
    Load metrics for all available models.

    Returns:
        DataFrame with model names as index and metrics as columns
    """
    metrics_list = []

    for model_name, meta in BASELINE_MODELS.items():
        exp_name = meta["experiment_name"]
        metrics = get_model_metrics(exp_name)

        if metrics:
            metrics['model'] = model_name
            metrics['type'] = meta['type']
            metrics['description'] = meta['description']
            metrics_list.append(metrics)

    if not metrics_list:
        return pd.DataFrame()

    df = pd.DataFrame(metrics_list)
    df = df.set_index('model')

    return df


def generate_naive_baseline(data: pd.DataFrame, horizon: int = 24) -> np.ndarray:
    """
    Generate naive seasonal baseline forecast (same hour last week).

    Args:
        data: Historical data with target variable
        horizon: Forecast horizon

    Returns:
        Array of naive forecasts
    """
    if TARGET_VARIABLE not in data.columns:
        return np.array([])

    target = data[TARGET_VARIABLE].values

    # Use last week same hours
    if len(target) >= 168 + horizon:
        naive_forecast = target[-168-horizon:-168]
    else:
        # Fallback: use last available values
        naive_forecast = target[-horizon:]

    return naive_forecast


@st.cache_data
def simulate_predictions(model_name: str, test_data: pd.DataFrame,
                         n_samples: int = 100) -> Dict:
    """
    Simulate predictions for visualization (when model loading fails).

    Args:
        model_name: Name of the model
        test_data: Test dataset
        n_samples: Number of samples to generate

    Returns:
        Dictionary with predictions and metadata
    """
    # Get model's expected error from metadata (based on new experiments)
    mae_estimates = {
        # Baseline models (consumption)
        "LIGHTGBM": 836,
        "LightGBM": 836,
        "CATBOOST": 826,
        "CatBoost": 826,
        "XGBOOST": 854,
        "XGBoost": 854,

        # Deep learning models (consumption)
        "TFT": 4451,
        "N-HiTS": 4945,
        "NHITS": 4945,
        "PATCHTST": 5015,
        "PatchTST": 5015,

        # Legacy
        "Prophet": 5039,
        "SARIMAX": 1000
    }

    # Use uppercase for matching
    mae = mae_estimates.get(model_name, mae_estimates.get(model_name.upper(), 500))

    actual = test_data[TARGET_VARIABLE].values[:n_samples]

    # Generate predictions with realistic noise
    np.random.seed(42)  # For reproducibility
    noise = np.random.normal(0, mae * 0.8, size=len(actual))
    predictions = actual + noise

    # Generate prediction intervals (90%)
    lower = predictions - 1.645 * mae
    upper = predictions + 1.645 * mae

    return {
        "predictions": predictions,
        "lower_bound": lower,
        "upper_bound": upper,
        "actual": actual,
        "dates": test_data.index[:n_samples]
    }


def calculate_prediction_intervals(predictions: np.ndarray,
                                   residuals: np.ndarray,
                                   confidence: float = 0.90) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate prediction intervals using residual-based method.

    Args:
        predictions: Point predictions
        residuals: Historical residuals
        confidence: Confidence level (default: 90%)

    Returns:
        Tuple of (lower_bounds, upper_bounds)
    """
    from scipy import stats

    # Calculate standard error from residuals
    std_error = np.std(residuals)

    # Get z-score for confidence level
    z_score = stats.norm.ppf((1 + confidence) / 2)

    # Calculate intervals
    margin = z_score * std_error
    lower = predictions - margin
    upper = predictions + margin

    return lower, upper


@st.cache_resource
def load_deep_learning_model(model_name: str, checkpoint_path: Optional[Path] = None):
    """
    Load a deep learning model from checkpoint.

    Args:
        model_name: Name of the model (N-HiTS, TFT, PatchTST)
        checkpoint_path: Optional path to specific checkpoint

    Returns:
        Loaded model or None
    """
    try:
        from .config import LIGHTNING_LOGS_DIR

        if checkpoint_path is None:
            # Find latest checkpoint for this model
            checkpoints = list(LIGHTNING_LOGS_DIR.glob(f"**/*{model_name}*.ckpt"))
            if not checkpoints:
                return None
            checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)

        # Load using PyTorch Lightning
        import torch
        model = torch.load(checkpoint_path)

        return model
    except Exception as e:
        st.warning(f"Could not load deep learning model {model_name}: {e}")
        return None
