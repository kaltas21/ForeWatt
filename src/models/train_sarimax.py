"""
SARIMAX Baseline Model for Electricity Demand Forecasting
=========================================================
Seasonal ARIMA with exogenous variables (weather, holidays).

Features:
- Seasonal patterns (daily, weekly)
- Exogenous regressors (temperature, holidays)
- Automatic parameter tuning (basic)
- MLflow tracking

Author: ForeWatt Team
Date: November 2025
"""

import os
import sys
import numpy as np
import pandas as pd
import mlflow
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.evaluate import evaluate_forecast

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SARIMAXForecaster:
    """SARIMAX model wrapper for electricity demand forecasting."""

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 24),
        exog_vars: Optional[List[str]] = None
    ):
        """
        Initialize SARIMAX forecaster.

        Args:
            order: (p, d, q) for ARIMA
                p: autoregressive order
                d: differencing order
                q: moving average order
            seasonal_order: (P, D, Q, s) for seasonal ARIMA
                P: seasonal AR order
                D: seasonal differencing order
                Q: seasonal MA order
                s: seasonal period (24 for hourly data)
            exog_vars: List of exogenous variable names
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.exog_vars = exog_vars or [
            'temp_national',
            'HDD',
            'CDD',
            'is_holiday_hour',
            'hour_sin',
            'hour_cos',
            'dow_sin',
            'dow_cos'
        ]
        self.model = None
        self.fitted_model = None

    def prepare_features(
        self,
        df: pd.DataFrame,
        target_col: str = 'consumption'
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Prepare data for SARIMAX.

        Args:
            df: Master dataset
            target_col: Target column name

        Returns:
            Tuple of (y, exog)
        """
        y = df[target_col]

        # Exogenous variables
        exog_cols = [col for col in self.exog_vars if col in df.columns]
        if len(exog_cols) < len(self.exog_vars):
            missing = set(self.exog_vars) - set(exog_cols)
            logger.warning(f"Missing exogenous variables: {missing}")

        exog = df[exog_cols]

        logger.info(f"Target: {target_col}")
        logger.info(f"Exogenous variables: {exog_cols}")

        return y, exog

    def train(
        self,
        y_train: pd.Series,
        exog_train: pd.DataFrame
    ):
        """
        Train SARIMAX model.

        Args:
            y_train: Training target
            exog_train: Training exogenous variables

        Returns:
            Fitted SARIMAX model
        """
        logger.info("Training SARIMAX model...")
        logger.info(f"Order: {self.order}")
        logger.info(f"Seasonal order: {self.seasonal_order}")

        # Initialize model
        self.model = SARIMAX(
            y_train,
            exog=exog_train,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        # Fit model
        self.fitted_model = self.model.fit(disp=False, maxiter=100)

        logger.info("✓ SARIMAX model trained successfully")
        logger.info(f"✓ AIC: {self.fitted_model.aic:.2f}")  # type: ignore
        logger.info(f"✓ BIC: {self.fitted_model.bic:.2f}")  # type: ignore

        return self.fitted_model

    def predict(
        self,
        exog_test: pd.DataFrame,
        start: Optional[int] = None,
        end: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate predictions.

        Args:
            exog_test: Test exogenous variables
            start: Start index for prediction
            end: End index for prediction

        Returns:
            Predictions array
        """
        if self.fitted_model is None:
            raise ValueError("Model must be trained before prediction")

        # Get predictions
        if start is None:
            start = 0
        if end is None:
            end = len(exog_test) - 1

        predictions = self.fitted_model.forecast(steps=len(exog_test), exog=exog_test)  # type: ignore

        return predictions.values


def load_master_data(
    start_date: str = '2020-01-01',
    end_date: str = '2024-12-31'
) -> pd.DataFrame:
    """Load master dataset from Gold layer."""
    gold_master = PROJECT_ROOT / 'data' / 'gold' / 'master'
    master_files = list(gold_master.glob('master_v*.parquet'))

    if not master_files:
        raise FileNotFoundError(f"No master files found in {gold_master}")

    latest_master = sorted(master_files)[-1]
    logger.info(f"Loading master data from {latest_master.name}")

    df = pd.read_parquet(latest_master)

    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')

    df = df.loc[start_date:end_date]

    logger.info(f"Loaded {len(df)} samples from {df.index.min()} to {df.index.max()}")
    logger.info(f"Shape: {df.shape}")

    return df


def train_test_split_temporal(
    df: pd.DataFrame,
    test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data temporally."""
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    logger.info(f"Train: {len(train_df)} samples ({train_df.index.min()} to {train_df.index.max()})")
    logger.info(f"Test:  {len(test_df)} samples ({test_df.index.min()} to {test_df.index.max()})")

    return train_df, test_df


def run_sarimax_baseline(
    experiment_name: str = "ForeWatt-Baseline-SARIMAX",
    run_name: str = "sarimax_v1_baseline",
    test_size: float = 0.2,
    order: Tuple[int, int, int] = (1, 1, 1),
    seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 24),
    mlflow_uri: Optional[str] = None
) -> Dict[str, float]:
    """
    Run SARIMAX baseline experiment.

    Args:
        experiment_name: MLflow experiment name
        run_name: MLflow run name
        test_size: Test set size
        order: ARIMA order (p, d, q)
        seasonal_order: Seasonal ARIMA order (P, D, Q, s)
        mlflow_uri: MLflow tracking URI (default: file-based)

    Returns:
        Dictionary with evaluation metrics
    """
    # Set MLflow tracking URI
    if mlflow_uri is None:
        mlflow_dir = PROJECT_ROOT / 'mlruns'
        mlflow_dir.mkdir(exist_ok=True)
        mlflow_uri = f"file://{mlflow_dir}"
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)

    # Load data
    logger.info("="*80)
    logger.info("SARIMAX BASELINE MODEL TRAINING")
    logger.info("="*80)

    df = load_master_data()

    # Drop rows with missing target
    df = df.dropna(subset=['consumption'])

    # Due to SARIMAX memory constraints, use subset for training
    # Use last 3 months for train, last month for test (faster training)
    logger.info("⚠️  Using subset of data for SARIMAX (computational efficiency)")
    df_subset = df.tail(24 * 30 * 4)  # Last 4 months
    logger.info(f"Subset: {len(df_subset)} samples ({df_subset.index.min()} to {df_subset.index.max()})")

    # Train-test split
    train_df, test_df = train_test_split_temporal(df_subset, test_size=test_size)

    # Initialize forecaster
    forecaster = SARIMAXForecaster(
        order=order,
        seasonal_order=seasonal_order
    )

    # Prepare features
    y_train, exog_train = forecaster.prepare_features(train_df)
    y_test, exog_test = forecaster.prepare_features(test_df)

    # Start MLflow run
    with mlflow.start_run(run_name=run_name) as run:

        # Log parameters
        mlflow.log_param("model_type", "SARIMAX")
        mlflow.log_param("order", str(order))
        mlflow.log_param("seasonal_order", str(seasonal_order))
        mlflow.log_param("exog_vars", ",".join(forecaster.exog_vars))
        mlflow.log_param("train_samples", len(y_train))
        mlflow.log_param("test_samples", len(y_test))

        # Train model
        fitted_model = forecaster.train(y_train, exog_train)

        # Log model diagnostics
        mlflow.log_metric("AIC", fitted_model.aic)  # type: ignore
        mlflow.log_metric("BIC", fitted_model.bic)  # type: ignore

        # Generate predictions
        predictions = forecaster.predict(exog_test)
        y_true = np.asarray(y_test.values)
        y_train_array = np.asarray(y_train.values)

        # Evaluate
        metrics = evaluate_forecast(
            y_true=y_true,
            y_pred=predictions,
            y_train=y_train_array,
            seasonality=24,
            model_name="SARIMAX Baseline"
        )

        # Log metrics
        mlflow.log_metric("MAE", metrics['MAE'])
        mlflow.log_metric("RMSE", metrics['RMSE'])
        mlflow.log_metric("MAPE", metrics['MAPE'])
        mlflow.log_metric("sMAPE", metrics['sMAPE'])
        mlflow.log_metric("MASE", metrics['MASE'])

        # Save model summary
        summary_path = "/tmp/sarimax_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(str(fitted_model.summary()))  # type: ignore
        mlflow.log_artifact(summary_path)

        logger.info(f"✓ MLflow run ID: {run.info.run_id}")

    return metrics


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train SARIMAX baseline model')
    parser.add_argument('--experiment', type=str, default='ForeWatt-Baseline-SARIMAX',
                        help='MLflow experiment name')
    parser.add_argument('--run-name', type=str, default='sarimax_v1_baseline',
                        help='MLflow run name')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set size')

    args = parser.parse_args()

    # Run baseline with basic parameters
    # Note: SARIMAX is computationally expensive, using simple (1,1,1) x (1,1,1,24)
    metrics = run_sarimax_baseline(
        experiment_name=args.experiment,
        run_name=args.run_name,
        test_size=args.test_size,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 24)
    )

    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE")
    logger.info("="*80)
    logger.info(f"MAE:   {metrics['MAE']:.2f} MWh")
    logger.info(f"RMSE:  {metrics['RMSE']:.2f} MWh")
    logger.info(f"MAPE:  {metrics['MAPE']:.2f}%")
    logger.info(f"sMAPE: {metrics['sMAPE']:.2f}%")
    logger.info(f"MASE:  {metrics['MASE']:.4f}")
    logger.info("="*80)
