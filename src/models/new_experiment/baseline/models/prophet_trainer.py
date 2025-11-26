"""
Prophet Trainer
===============
Facebook Prophet for time series forecasting with seasonality.

Author: ForeWatt Team - New Experiment V2
Date: November 2025
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List
import logging
import warnings

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logging.warning("Prophet not installed. Install with: pip install prophet")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProphetTrainer:
    """Prophet time series trainer with exogenous regressors."""

    def __init__(
        self,
        target: str = 'price_real',
        random_seed: int = 42,
        verbose: bool = False
    ):
        self.target = target
        self.random_seed = random_seed
        self.verbose = verbose
        self.model = None
        self.feature_names = None
        self.train_end_date = None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        hyperparams: Dict[str, Any]
    ) -> Tuple[Any, Dict[str, float]]:
        """
        Train Prophet model.

        Returns:
            Tuple of (trained_model, validation_metrics)
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet not installed")

        logger.info(f"\n{'='*60}")
        logger.info(f"TRAINING PROPHET: {self.target}")
        logger.info(f"{'='*60}")
        logger.info(f"Train samples: {len(X_train)}")
        logger.info(f"Val samples: {len(X_val)}")
        logger.info(f"Features: {X_train.shape[1]}")

        # Suppress Prophet's cmdstanpy logs
        if not self.verbose:
            warnings.filterwarnings('ignore')
            logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
            logging.getLogger('prophet').setLevel(logging.WARNING)

        # Extract hyperparameters
        params = {
            'changepoint_prior_scale': hyperparams.get('changepoint_prior_scale', 0.1),
            'seasonality_prior_scale': hyperparams.get('seasonality_prior_scale', 10.0),
            'seasonality_mode': hyperparams.get('seasonality_mode', 'additive'),
            'yearly_seasonality': hyperparams.get('yearly_seasonality', True),
            'weekly_seasonality': hyperparams.get('weekly_seasonality', True),
            'daily_seasonality': hyperparams.get('daily_seasonality', True),
            'n_changepoints': hyperparams.get('n_changepoints', 25),
        }

        logger.info(f"Hyperparameters: {params}")

        # Prepare Prophet dataframe
        # Prophet requires 'ds' (datetime) and 'y' (target) columns
        train_df = pd.DataFrame({
            'ds': X_train.index,
            'y': y_train.values
        })

        # Store feature names (for adding regressors)
        self.feature_names = list(X_train.columns)

        # Add exogenous regressors (top features only to avoid overfitting)
        # Select numeric features that are not calendar-based (Prophet handles those)
        regressor_features = [f for f in self.feature_names if not any(
            x in f.lower() for x in ['hour', 'dow', 'month', 'weekend', 'holiday', 'sin', 'cos']
        )]

        # Limit to top 10 regressors for efficiency
        regressor_features = regressor_features[:10]

        for feature in regressor_features:
            train_df[feature] = X_train[feature].values

        # Initialize Prophet
        self.model = Prophet(
            changepoint_prior_scale=params['changepoint_prior_scale'],
            seasonality_prior_scale=params['seasonality_prior_scale'],
            seasonality_mode=params['seasonality_mode'],
            yearly_seasonality=params['yearly_seasonality'],
            weekly_seasonality=params['weekly_seasonality'],
            daily_seasonality=params['daily_seasonality'],
            n_changepoints=params['n_changepoints'],
        )

        # Add regressors
        for feature in regressor_features:
            self.model.add_regressor(feature)

        # Fit model
        self.model.fit(train_df)

        # Store training end date
        self.train_end_date = train_df['ds'].max()

        # Prepare validation dataframe
        val_df = pd.DataFrame({
            'ds': X_val.index
        })
        for feature in regressor_features:
            val_df[feature] = X_val[feature].values

        # Predict on validation
        forecast = self.model.predict(val_df)
        val_pred = forecast['yhat'].values

        # Calculate metrics
        from src.models.evaluate import (
            mean_absolute_error,
            symmetric_mean_absolute_percentage_error,
            mean_absolute_scaled_error
        )

        metrics = {
            'MAE': mean_absolute_error(y_val.values, val_pred),
            'sMAPE': symmetric_mean_absolute_percentage_error(y_val.values, val_pred),
            'MASE': mean_absolute_scaled_error(y_val.values, val_pred, y_train.values, seasonality=24)
        }

        logger.info(f"\nValidation metrics:")
        logger.info(f"  MAE:   {metrics['MAE']:.2f}")
        logger.info(f"  sMAPE: {metrics['sMAPE']:.2f}%")
        logger.info(f"  MASE:  {metrics['MASE']:.4f}")

        # Store regressor features for prediction
        self._regressor_features = regressor_features

        return self.model, metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        if self.model is None:
            raise ValueError("Model not trained")

        # Prepare prediction dataframe
        pred_df = pd.DataFrame({'ds': X.index})

        for feature in self._regressor_features:
            if feature in X.columns:
                pred_df[feature] = X[feature].values
            else:
                pred_df[feature] = 0  # Default if missing

        forecast = self.model.predict(pred_df)
        return forecast['yhat'].values

    def get_seasonality_components(self) -> pd.DataFrame:
        """Get seasonality components from the model."""
        if self.model is None:
            raise ValueError("Model not trained")

        # Create a future dataframe for component analysis
        future = self.model.make_future_dataframe(periods=168, freq='H')

        # Add regressor values (use zeros for simplicity)
        for feature in self._regressor_features:
            future[feature] = 0

        forecast = self.model.predict(future)

        components = ['ds', 'trend']
        if 'daily' in forecast.columns:
            components.append('daily')
        if 'weekly' in forecast.columns:
            components.append('weekly')
        if 'yearly' in forecast.columns:
            components.append('yearly')

        return forecast[components]
