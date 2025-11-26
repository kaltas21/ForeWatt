"""
XGBoost Trainer
===============
Extreme Gradient Boosting for regression.

Author: ForeWatt Team - New Experiment V2
Date: November 2025
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import logging

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not installed. Install with: pip install xgboost")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XGBoostTrainer:
    """XGBoost regression trainer."""

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

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        hyperparams: Dict[str, Any]
    ) -> Tuple[Any, Dict[str, float]]:
        """
        Train XGBoost model.

        Returns:
            Tuple of (trained_model, validation_metrics)
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed")

        logger.info(f"\n{'='*60}")
        logger.info(f"TRAINING XGBOOST: {self.target}")
        logger.info(f"{'='*60}")
        logger.info(f"Train samples: {len(X_train)}")
        logger.info(f"Val samples: {len(X_val)}")
        logger.info(f"Features: {X_train.shape[1]}")

        # Extract hyperparameters
        params = {
            'n_estimators': hyperparams.get('n_estimators', 1000),
            'max_depth': hyperparams.get('max_depth', 6),
            'learning_rate': hyperparams.get('learning_rate', 0.05),
            'subsample': hyperparams.get('subsample', 0.8),
            'colsample_bytree': hyperparams.get('colsample_bytree', 0.8),
            'reg_alpha': hyperparams.get('reg_alpha', 0.1),
            'reg_lambda': hyperparams.get('reg_lambda', 1.0),
            'random_state': self.random_seed,
            'objective': 'reg:squarederror',
            'early_stopping_rounds': 50,
            'verbosity': 1 if self.verbose else 0,
        }

        logger.info(f"Hyperparameters: {params}")

        self.model = xgb.XGBRegressor(**params)
        self.feature_names = list(X_train.columns)

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=self.verbose
        )

        # Validation predictions
        val_pred = self.model.predict(X_val)

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

        return self.model, metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        if self.model is None:
            raise ValueError("Model not trained")
        return self.model.predict(X)

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance."""
        if self.model is None:
            raise ValueError("Model not trained")
        importance = self.model.feature_importances_
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
