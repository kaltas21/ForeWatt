"""
Consumption Forecast Inference Module
======================================
Loads trained CatBoost model and generates 24-hour consumption forecasts.
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / 'models' / 'consumption'
DATA_PATH = PROJECT_ROOT / 'data' / 'gold' / 'master' / 'master_v2_fundamental.parquet'


class ConsumptionForecaster:
    """Consumption forecasting using CatBoost."""

    def __init__(self, models_dir: str = None):
        """
        Initialize consumption forecaster.

        Args:
            models_dir: Path to models directory
        """
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR

        self.model = None
        self.features = None

        self._load_model()

    def _load_model(self):
        """Load trained model and configuration."""
        from catboost import CatBoostRegressor

        # Load CatBoost model
        model_path = self.models_dir / 'model.cbm'
        self.model = CatBoostRegressor()
        self.model.load_model(str(model_path))
        logger.info(f"Loaded CatBoost from: {model_path}")

        # Load features
        features_path = self.models_dir / 'features.json'
        with open(features_path) as f:
            self.features = json.load(f)
        logger.info(f"Loaded {self.features['n_features']} features, sMAPE={self.features['test_smape']:.2f}%")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate consumption predictions.

        Args:
            df: Input DataFrame with required features

        Returns:
            Predictions array
        """
        # Get available features
        feature_list = self.features['features']
        available = [f for f in feature_list if f in df.columns]

        missing = set(feature_list) - set(available)
        if missing:
            logger.warning(f"Missing features: {missing}")

        X = df[available].fillna(0)

        return self.model.predict(X)

    def forecast_next_24h(
        self,
        current_data: pd.DataFrame,
        forecast_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Generate 24-hour ahead forecast.

        Args:
            current_data: Recent historical data (at least 168 hours for lags)
            forecast_time: Base time for forecast (defaults to now)

        Returns:
            DataFrame with 24-hour forecast
        """
        if forecast_time is None:
            forecast_time = datetime.now()

        # Ensure we have enough history
        if len(current_data) < 168:
            logger.warning(f"Only {len(current_data)} rows of history, need 168 for full lags")

        # Get last row as template
        last_row = current_data.iloc[-1:].copy()

        forecasts = []
        for h in range(1, 25):
            target_time = forecast_time + timedelta(hours=h)

            # Update time features
            row = last_row.copy()

            # Hour features
            hour = target_time.hour
            row['hour_sin'] = np.sin(2 * np.pi * hour / 24)
            row['hour_cos'] = np.cos(2 * np.pi * hour / 24)

            # Day of week features
            dow = target_time.weekday()
            row['dow_sin_x'] = np.sin(2 * np.pi * dow / 7)
            row['dow_cos_x'] = np.cos(2 * np.pi * dow / 7)
            row['is_weekend_x'] = 1 if dow >= 5 else 0

            # Month features
            month = target_time.month
            row['month_sin'] = np.sin(2 * np.pi * month / 12)
            row['month_cos'] = np.cos(2 * np.pi * month / 12)

            # Predict
            pred = self.predict(row)

            forecasts.append({
                'target_time': target_time,
                'forecast_value': float(pred[0]),
                'hour': hour,
            })

        return pd.DataFrame(forecasts)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test inference
    forecaster = ConsumptionForecaster()

    # Load some test data
    df = pd.read_parquet(DATA_PATH)
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date' in df.columns:
            df = df.set_index('date')

    # Use last 200 hours as test
    test_data = df.tail(200)

    # Generate forecast
    forecast = forecaster.forecast_next_24h(test_data)
    print("\n24-Hour Consumption Forecast:")
    print(forecast)
