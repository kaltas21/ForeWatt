"""
Price Forecast Inference Module
================================
Loads trained CHybrid V14 model and generates 24-hour price forecasts.

Uses:
- CatBoost + LightGBM Ensemble
- Hybrid Error Correction (50% Simple AEC + 50% KNN-EC)
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / 'models' / 'price'
DATA_PATH = PROJECT_ROOT / 'data' / 'gold' / 'master' / 'master_v2_fundamental.parquet'


class PriceForecaster:
    """Price forecasting using CHybrid V14 ensemble with error correction."""

    def __init__(self, models_dir: str = None):
        """
        Initialize price forecaster.

        Args:
            models_dir: Path to models directory
        """
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR

        self.catboost_model = None
        self.lightgbm_model = None
        self.config = None
        self.features = None
        self.scaler = None

        # Error correction history
        self.error_history = {}  # hour -> list of (context, error)

        self._load_models()

    def _load_models(self):
        """Load trained models and configuration."""
        from catboost import CatBoostRegressor
        import lightgbm as lgb

        # Load CatBoost
        catboost_path = self.models_dir / 'catboost_v14.cbm'
        self.catboost_model = CatBoostRegressor()
        self.catboost_model.load_model(str(catboost_path))
        logger.info(f"Loaded CatBoost from: {catboost_path}")

        # Load LightGBM
        lightgbm_path = self.models_dir / 'lightgbm_v14.txt'
        self.lightgbm_model = lgb.Booster(model_file=str(lightgbm_path))
        logger.info(f"Loaded LightGBM from: {lightgbm_path}")

        # Load configuration
        config_path = self.models_dir / 'ensemble_config.json'
        with open(config_path) as f:
            self.config = json.load(f)
        logger.info(f"Loaded config: {self.config['best_method']}, sMAPE={self.config['test_smape']:.2f}%")

        # Load features
        features_path = self.models_dir / 'features.json'
        with open(features_path) as f:
            self.features = json.load(f)
        logger.info(f"Loaded {self.features['n_features']} features")

        # Initialize scaler for KNN
        self.scaler = StandardScaler()
        self._is_scaler_fitted = False

    def _create_profile_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create profile evolution features for inference."""
        df = df.copy()

        if 'hour' not in df.columns:
            df['hour'] = df.index.hour

        # Use lagged price for profile calculation
        price_col = 'price_ptf_lag_24h'
        if price_col not in df.columns:
            return df

        # Daily average (using available lagged data)
        df['daily_avg_price'] = df[price_col].rolling(24, min_periods=12).mean()

        # Hourly ratio
        df['hourly_ratio'] = (df[price_col] / df['daily_avg_price'].clip(lower=1)).clip(0.2, 5.0)

        # Profile features (simplified for inference - use available history)
        df['profile_14d'] = df['hourly_ratio'].rolling(14 * 24, min_periods=7 * 24).mean()
        df['profile_28d'] = df['hourly_ratio'].rolling(28 * 24, min_periods=14 * 24).mean()
        df['profile_momentum'] = df['profile_14d'] - df['profile_28d']
        df['daily_avg_momentum'] = df['daily_avg_price'] - df['daily_avg_price'].shift(24)

        # Solar profile features
        if 'renewable_saturation' in df.columns and 'load_factor' in df.columns:
            load = df['load_factor'].clip(lower=0.1)
            df['solar_ratio'] = (df['renewable_saturation'] / load).clip(0, 5)
            df['solar_profile_14d'] = df['solar_ratio'].rolling(14 * 24, min_periods=7 * 24).mean()
            df['solar_profile_28d'] = df['solar_ratio'].rolling(28 * 24, min_periods=14 * 24).mean()
            df['solar_momentum'] = df['solar_profile_14d'] - df['solar_profile_28d']

            if 'profile_14d' in df.columns:
                df['price_solar_interaction'] = df['profile_14d'] * df['solar_momentum']

        # Fill NaN with median
        for col in ['hourly_ratio', 'profile_14d', 'profile_28d', 'profile_momentum',
                    'daily_avg_momentum', 'solar_ratio', 'solar_profile_14d',
                    'solar_profile_28d', 'solar_momentum', 'price_solar_interaction']:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

        return df

    def _apply_simple_aec(self, raw_pred: float, hour: int) -> float:
        """Apply Simple Adaptive Error Correction."""
        aec_params = self.config.get('hourly_aec_params', {})
        params = aec_params.get(str(hour), {'lookback': 7, 'damping': 0.5})

        lookback = params['lookback']
        damping = params['damping']

        # Get error history for this hour
        hour_errors = self.error_history.get(hour, [])

        if len(hour_errors) == 0:
            return raw_pred

        # Use recent errors
        recent_errors = [e for _, e in hour_errors[-lookback:]]
        if len(recent_errors) > 0:
            correction = damping * np.mean(recent_errors)
            return raw_pred - correction

        return raw_pred

    def _apply_knn_correction(self, raw_pred: float, context: np.ndarray, hour: int) -> float:
        """Apply KNN-based error correction."""
        knn_params = self.config.get('knn_params', {})
        k = knn_params.get('k', 5)
        damping = 0.8

        # Get error history for this hour
        hour_history = self.error_history.get(hour, [])

        if len(hour_history) < k:
            return raw_pred

        # Build context matrix and error array
        contexts = np.array([h[0] for h in hour_history])
        errors = np.array([h[1] for h in hour_history])

        # Fit KNN
        k_actual = min(k, len(hour_history))
        knn = NearestNeighbors(n_neighbors=k_actual, metric='euclidean')
        knn.fit(contexts)

        # Find neighbors
        distances, indices = knn.kneighbors(context.reshape(1, -1))
        distances = distances[0]
        indices = indices[0]

        # Weight by inverse distance
        epsilon = 1e-6
        weights = 1.0 / (distances + epsilon)
        weights = weights / weights.sum()

        # Weighted error
        weighted_error = np.sum(weights * errors[indices])
        correction = damping * weighted_error

        return raw_pred - correction

    def update_error_history(
        self,
        hour: int,
        context: np.ndarray,
        actual: float,
        predicted: float,
        max_history: int = 45
    ):
        """
        Update error history with actual vs predicted.
        Call this when actual values become available.

        Args:
            hour: Hour of day (0-23)
            context: Context features [load_factor, renewable_saturation, thermal_gap]
            actual: Actual observed value
            predicted: Model's prediction
            max_history: Maximum history to keep per hour
        """
        error = predicted - actual

        if hour not in self.error_history:
            self.error_history[hour] = []

        self.error_history[hour].append((context, error))

        # Keep only recent history
        if len(self.error_history[hour]) > max_history:
            self.error_history[hour] = self.error_history[hour][-max_history:]

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate raw ensemble predictions (no error correction).

        Args:
            X: Feature DataFrame

        Returns:
            Raw predictions array
        """
        # CatBoost prediction
        cat_pred = self.catboost_model.predict(X)

        # LightGBM prediction
        lgb_pred = self.lightgbm_model.predict(X)

        # Weighted ensemble
        w_cat = self.config['catboost_weight']
        w_lgb = self.config['lightgbm_weight']

        return w_cat * cat_pred + w_lgb * lgb_pred

    def predict(
        self,
        df: pd.DataFrame,
        apply_correction: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions with optional error correction.

        Args:
            df: Input DataFrame with required features
            apply_correction: Whether to apply hybrid error correction

        Returns:
            Tuple of (corrected_predictions, raw_predictions)
        """
        # Prepare features
        df = self._create_profile_features(df)

        # Ensure hour column
        if 'hour' not in df.columns:
            df['hour'] = df.index.hour

        # Get available features
        feature_list = self.features['features']
        available = [f for f in feature_list if f in df.columns]

        X = df[available].fillna(0)

        # Raw predictions
        raw_pred = self.predict_raw(X)

        if not apply_correction:
            return raw_pred, raw_pred

        # Apply hybrid correction (50% Simple AEC + 50% KNN)
        corrected = np.zeros_like(raw_pred)

        # Context features for KNN
        context_features = self.config.get('knn_params', {}).get(
            'context_features', ['load_factor', 'renewable_saturation', 'thermal_gap']
        )
        available_context = [f for f in context_features if f in df.columns]

        # Fit scaler if not fitted
        if not self._is_scaler_fitted and len(df) > 10:
            self.scaler.fit(df[available_context].fillna(0))
            self._is_scaler_fitted = True

        for i in range(len(raw_pred)):
            hour = int(df['hour'].iloc[i])

            # Simple AEC
            simple_corrected = self._apply_simple_aec(raw_pred[i], hour)

            # KNN correction
            if self._is_scaler_fitted and len(available_context) > 0:
                context = self.scaler.transform(
                    df[available_context].iloc[i:i+1].fillna(0)
                )[0]
                knn_corrected = self._apply_knn_correction(raw_pred[i], context, hour)
            else:
                knn_corrected = raw_pred[i]

            # Hybrid: 50% Simple + 50% KNN
            corrected[i] = 0.5 * simple_corrected + 0.5 * knn_corrected

        return corrected, raw_pred

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

        # Get predictions for next 24 hours
        # For now, use last available features (in production, would need weather forecasts)
        last_row = current_data.iloc[-1:].copy()

        forecasts = []
        for h in range(1, 25):
            target_time = forecast_time + timedelta(hours=h)

            # Update hour features
            row = last_row.copy()
            row['hour'] = target_time.hour
            row['hour_sin'] = np.sin(2 * np.pi * target_time.hour / 24)
            row['hour_cos'] = np.cos(2 * np.pi * target_time.hour / 24)

            # Predict
            corrected, raw = self.predict(row, apply_correction=True)

            forecasts.append({
                'target_time': target_time,
                'forecast_value': float(corrected[0]),
                'raw_value': float(raw[0]),
                'hour': target_time.hour,
            })

        return pd.DataFrame(forecasts)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test inference
    forecaster = PriceForecaster()

    # Load some test data
    df = pd.read_parquet(DATA_PATH)
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date' in df.columns:
            df = df.set_index('date')

    # Use last 200 hours as test
    test_data = df.tail(200)

    # Generate forecast
    forecast = forecaster.forecast_next_24h(test_data)
    print("\n24-Hour Price Forecast:")
    print(forecast)
