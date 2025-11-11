"""
Prophet Baseline Model for Electricity Demand Forecasting
=========================================================
Uses Facebook Prophet with:
- Turkish national holidays
- Weather regressors (temperature, humidity, wind speed, etc.)
- Daily/weekly/yearly seasonality
- MLflow tracking for experiments

Author: ForeWatt Team
Date: November 2025
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import mlflow
import mlflow.prophet
from prophet import Prophet
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.evaluate import evaluate_forecast

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProphetForecaster:
    """Prophet model wrapper for electricity demand forecasting."""

    def __init__(
        self,
        seasonality_mode: str = 'multiplicative',
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        holidays_prior_scale: float = 10.0,
        interval_width: float = 0.95,
        weather_regressors: Optional[List[str]] = None
    ):
        """
        Initialize Prophet forecaster.

        Args:
            seasonality_mode: 'additive' or 'multiplicative'
            changepoint_prior_scale: Flexibility of trend (higher = more flexible)
            seasonality_prior_scale: Strength of seasonality
            holidays_prior_scale: Strength of holiday effects
            interval_width: Uncertainty interval width
            weather_regressors: List of weather columns to use as regressors
        """
        self.seasonality_mode = seasonality_mode
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.holidays_prior_scale = holidays_prior_scale
        self.interval_width = interval_width
        self.weather_regressors = weather_regressors or [
            'temp_national',
            'humidity_national',
            'wind_speed_national',
            'HDD',
            'CDD',
            'heat_index',
            'wind_chill'
        ]
        self.model = None

    def load_turkish_holidays(self) -> pd.DataFrame:
        """
        Load Turkish national holidays from static JSON.

        Returns:
            DataFrame with columns: holiday, ds, lower_window, upper_window
        """
        holidays_path = PROJECT_ROOT / 'src' / 'data' / 'static' / 'tr_holidays_2020_2025.json'

        with open(holidays_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        holidays_list = []
        for holiday in data['holidays']:
            # For multi-day holidays, create entries for each day
            start = pd.to_datetime(holiday['start_date'])
            end = pd.to_datetime(holiday['end_date'])

            for single_date in pd.date_range(start, end):
                holidays_list.append({
                    'holiday': holiday['name'],
                    'ds': single_date,
                    'lower_window': 0,
                    'upper_window': 0
                })

        holidays_df = pd.DataFrame(holidays_list)
        logger.info(f"Loaded {len(holidays_df)} Turkish holiday dates ({data['metadata']['years']})")

        return holidays_df

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str = 'consumption'
    ) -> pd.DataFrame:
        """
        Prepare data for Prophet format.

        Prophet requires columns: ds (timestamp), y (target), and regressors

        Args:
            df: Master dataset with all features
            target_col: Name of target column

        Returns:
            DataFrame in Prophet format
        """
        prophet_df = pd.DataFrame()
        prophet_df['ds'] = pd.to_datetime(df.index).tz_localize(None)  # Remove timezone for Prophet
        prophet_df['y'] = df[target_col].values

        # Add weather regressors
        for regressor in self.weather_regressors:
            if regressor in df.columns:
                prophet_df[regressor] = df[regressor].values
            else:
                logger.warning(f"Regressor '{regressor}' not found in data")

        # Remove any rows with missing target or regressors
        prophet_df = prophet_df.dropna()

        logger.info(f"Prepared {len(prophet_df)} samples for Prophet")
        logger.info(f"Regressors: {self.weather_regressors}")

        return prophet_df

    def train(
        self,
        train_df: pd.DataFrame
    ) -> Prophet:
        """
        Train Prophet model.

        Args:
            train_df: Training data in Prophet format

        Returns:
            Trained Prophet model
        """
        logger.info("Training Prophet model...")

        # Initialize model
        self.model = Prophet(
            seasonality_mode=self.seasonality_mode,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            holidays_prior_scale=self.holidays_prior_scale,
            interval_width=self.interval_width,
            yearly_seasonality=True,  # type: ignore
            weekly_seasonality=True,  # type: ignore
            daily_seasonality=True  # type: ignore
        )

        # Add Turkish holidays
        holidays_df = self.load_turkish_holidays()
        self.model.holidays = holidays_df

        # Add weather regressors
        for regressor in self.weather_regressors:
            if regressor in train_df.columns:
                self.model.add_regressor(regressor)

        # Fit model
        self.model.fit(train_df)

        logger.info("✓ Prophet model trained successfully")
        return self.model

    def predict(
        self,
        future_df: pd.DataFrame
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Generate predictions.

        Args:
            future_df: Future dataframe with regressors

        Returns:
            Tuple of (predictions array, full forecast dataframe)
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")

        forecast = self.model.predict(future_df)
        predictions = np.asarray(forecast['yhat'].values)

        return predictions, forecast


def load_master_data(
    start_date: str = '2020-01-01',
    end_date: str = '2024-12-31'
) -> pd.DataFrame:
    """
    Load master dataset from Gold layer.

    Args:
        start_date: Start date for data
        end_date: End date for data

    Returns:
        Master dataset with all features
    """
    # Find latest master file
    gold_master = PROJECT_ROOT / 'data' / 'gold' / 'master'
    master_files = list(gold_master.glob('master_v*.parquet'))

    if not master_files:
        raise FileNotFoundError(f"No master files found in {gold_master}")

    # Get most recent file
    latest_master = sorted(master_files)[-1]
    logger.info(f"Loading master data from {latest_master.name}")

    df = pd.read_parquet(latest_master)

    # Set datetime index
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')

    # Filter date range
    df = df.loc[start_date:end_date]

    logger.info(f"Loaded {len(df)} samples from {df.index.min()} to {df.index.max()}")
    logger.info(f"Shape: {df.shape}")

    return df


def train_test_split_temporal(
    df: pd.DataFrame,
    test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data temporally (no shuffle for time series).

    Args:
        df: Full dataset
        test_size: Fraction of data for test set

    Returns:
        Tuple of (train_df, test_df)
    """
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    logger.info(f"Train: {len(train_df)} samples ({train_df.index.min()} to {train_df.index.max()})")
    logger.info(f"Test:  {len(test_df)} samples ({test_df.index.min()} to {test_df.index.max()})")

    return train_df, test_df


def run_prophet_baseline(
    experiment_name: str = "ForeWatt-Baseline-Prophet",
    run_name: str = "prophet_v1_baseline",
    test_size: float = 0.2,
    mlflow_uri: Optional[str] = None
) -> Dict[str, float]:
    """
    Run Prophet baseline experiment.

    Args:
        experiment_name: MLflow experiment name
        run_name: MLflow run name
        test_size: Test set size
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
    logger.info("PROPHET BASELINE MODEL TRAINING")
    logger.info("="*80)

    df = load_master_data()

    # Train-test split
    train_df, test_df = train_test_split_temporal(df, test_size=test_size)

    # Initialize forecaster
    forecaster = ProphetForecaster(
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        holidays_prior_scale=10.0,
        weather_regressors=[
            'temp_national',
            'humidity_national',
            'wind_speed_national',
            'HDD',
            'CDD',
            'heat_index',
            'wind_chill'
        ]
    )

    # Prepare data
    train_prophet = forecaster.prepare_data(train_df, target_col='consumption')
    test_prophet = forecaster.prepare_data(test_df, target_col='consumption')

    # Start MLflow run
    with mlflow.start_run(run_name=run_name) as run:

        # Log parameters
        mlflow.log_param("model_type", "Prophet")
        mlflow.log_param("seasonality_mode", forecaster.seasonality_mode)
        mlflow.log_param("changepoint_prior_scale", forecaster.changepoint_prior_scale)
        mlflow.log_param("seasonality_prior_scale", forecaster.seasonality_prior_scale)
        mlflow.log_param("holidays_prior_scale", forecaster.holidays_prior_scale)
        mlflow.log_param("weather_regressors", ",".join(forecaster.weather_regressors))
        mlflow.log_param("train_samples", len(train_prophet))
        mlflow.log_param("test_samples", len(test_prophet))
        mlflow.log_param("test_size", test_size)

        # Train model
        model = forecaster.train(train_prophet)

        # Generate predictions
        predictions, forecast = forecaster.predict(test_prophet)
        y_true = np.asarray(test_prophet['y'].values)
        y_train = np.asarray(train_prophet['y'].values)

        # Evaluate
        metrics = evaluate_forecast(
            y_true=y_true,
            y_pred=predictions,
            y_train=y_train,
            seasonality=24,
            model_name="Prophet Baseline"
        )

        # Log metrics to MLflow
        mlflow.log_metric("MAE", metrics['MAE'])
        mlflow.log_metric("RMSE", metrics['RMSE'])
        mlflow.log_metric("MAPE", metrics['MAPE'])
        mlflow.log_metric("sMAPE", metrics['sMAPE'])
        mlflow.log_metric("MASE", metrics['MASE'])

        # Log model
        mlflow.prophet.log_model(model, "prophet_model")  # type: ignore

        logger.info(f"✓ MLflow run ID: {run.info.run_id}")
        logger.info(f"✓ Experiment: {experiment_name}")

    return metrics


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train Prophet baseline model')
    parser.add_argument('--experiment', type=str, default='ForeWatt-Baseline-Prophet',
                        help='MLflow experiment name')
    parser.add_argument('--run-name', type=str, default='prophet_v1_baseline',
                        help='MLflow run name')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set size (default: 0.2)')

    args = parser.parse_args()

    # Run baseline
    metrics = run_prophet_baseline(
        experiment_name=args.experiment,
        run_name=args.run_name,
        test_size=args.test_size
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
