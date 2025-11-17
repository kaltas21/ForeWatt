"""
Model Trainer for Baseline Models
==================================
Unified trainer for all baseline models with intelligent feature selection.

Supports:
- CatBoost
- XGBoost
- LightGBM
- Prophet
- SARIMAX

Targets:
- Demand (consumption)
- Price (price_real - today's Turkish Lira)

Author: ForeWatt Team
Date: November 2025
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from catboost import CatBoostRegressor, Pool
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.baseline.feature_selector import FeatureSelector, get_categorical_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Unified trainer for baseline models with intelligent feature selection.
    """

    def __init__(
        self,
        model_type: str,
        target: str = 'consumption',
        hyperparams: Optional[Dict] = None,
        random_seed: int = 42
    ):
        """
        Initialize model trainer.

        Args:
            model_type: One of ['catboost', 'xgboost', 'lightgbm', 'prophet', 'sarimax']
            target: Target variable ('consumption' or 'price_real')
            hyperparams: Model hyperparameters
            random_seed: Random seed for reproducibility
        """
        self.model_type = model_type.lower()
        self.target = target
        self.hyperparams = hyperparams or self._get_default_hyperparams()
        self.random_seed = random_seed
        self.model = None
        self.feature_selector = FeatureSelector(target=target)
        self.feature_names = None
        self.categorical_features = None

    def _get_default_hyperparams(self) -> Dict[str, Any]:
        """Get default hyperparameters for each model type."""
        defaults = {
            'catboost': {
                'iterations': 1000,
                'learning_rate': 0.1,
                'depth': 6,
                'l2_leaf_reg': 3.0,
                'early_stopping_rounds': 50
            },
            'xgboost': {
                'n_estimators': 1000,
                'learning_rate': 0.1,
                'max_depth': 6,
                'reg_lambda': 1.0,
                'early_stopping_rounds': 50
            },
            'lightgbm': {
                'n_estimators': 1000,
                'learning_rate': 0.1,
                'max_depth': 6,
                'reg_lambda': 1.0,
                'early_stopping_rounds': 50
            },
            'prophet': {
                'growth': 'linear',
                'changepoint_prior_scale': 0.05,
                'seasonality_prior_scale': 10.0,
                'holidays_prior_scale': 10.0,
                'seasonality_mode': 'additive'
            },
            'sarimax': {
                'order': (1, 0, 1),
                'seasonal_order': (1, 0, 1, 24),
                'trend': 'c'
            }
        }
        return defaults.get(self.model_type, {})

    def prepare_features(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Prepare features using intelligent feature selector.

        Args:
            df: Master dataset

        Returns:
            Tuple of (X, y, feature_names)
        """
        X, y, feature_names = self.feature_selector.prepare_features(
            df, self.model_type, self.target
        )

        self.feature_names = feature_names

        # Identify categorical features for boosting models
        if self.model_type in ['catboost', 'xgboost', 'lightgbm']:
            self.categorical_features = get_categorical_features(feature_names)
            logger.info(f"Categorical features: {len(self.categorical_features)}")

        return X, y, feature_names

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Any:
        """
        Train model.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (for early stopping)
            y_val: Validation target

        Returns:
            Trained model
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"TRAINING {self.model_type.upper()} MODEL")
        logger.info(f"{'='*80}")
        logger.info(f"Target: {self.target}")
        logger.info(f"Train samples: {len(X_train)}")
        if X_val is not None:
            logger.info(f"Val samples: {len(X_val)}")
        logger.info(f"Features: {len(self.feature_names)}")

        if self.model_type == 'catboost':
            self.model = self._train_catboost(X_train, y_train, X_val, y_val)
        elif self.model_type == 'xgboost':
            self.model = self._train_xgboost(X_train, y_train, X_val, y_val)
        elif self.model_type == 'lightgbm':
            self.model = self._train_lightgbm(X_train, y_train, X_val, y_val)
        elif self.model_type == 'prophet':
            self.model = self._train_prophet(X_train, y_train)
        elif self.model_type == 'sarimax':
            self.model = self._train_sarimax(X_train, y_train)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        logger.info(f"✓ {self.model_type.upper()} training complete")
        return self.model

    def _train_catboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series]
    ) -> CatBoostRegressor:
        """Train CatBoost model."""
        # Create training pool
        train_pool = Pool(
            data=X_train,
            label=y_train,
            cat_features=self.categorical_features
        )

        # Create validation pool if provided
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = Pool(
                data=X_val,
                label=y_val,
                cat_features=self.categorical_features
            )

        # Initialize and train
        model = CatBoostRegressor(
            iterations=self.hyperparams.get('iterations', 1000),
            learning_rate=self.hyperparams.get('learning_rate', 0.1),
            depth=self.hyperparams.get('depth', 6),
            l2_leaf_reg=self.hyperparams.get('l2_leaf_reg', 3.0),
            random_seed=self.random_seed,
            loss_function='RMSE',
            eval_metric='RMSE',
            early_stopping_rounds=self.hyperparams.get('early_stopping_rounds', 50),
            verbose=100,
            cat_features=self.categorical_features
        )

        model.fit(
            train_pool,
            eval_set=eval_set,
            use_best_model=True if eval_set is not None else False
        )

        logger.info(f"Best iteration: {model.best_iteration_}")
        return model

    def _train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series]
    ) -> XGBRegressor:
        """Train XGBoost model."""
        # Handle categorical features (convert to category dtype)
        X_train_copy = X_train.copy()
        if self.categorical_features:
            for cat_col in self.categorical_features:
                if cat_col in X_train_copy.columns:
                    X_train_copy[cat_col] = X_train_copy[cat_col].astype('category')

        eval_set = None
        if X_val is not None and y_val is not None:
            X_val_copy = X_val.copy()
            if self.categorical_features:
                for cat_col in self.categorical_features:
                    if cat_col in X_val_copy.columns:
                        X_val_copy[cat_col] = X_val_copy[cat_col].astype('category')
            eval_set = [(X_val_copy, y_val)]

        model = XGBRegressor(
            n_estimators=self.hyperparams.get('n_estimators', 1000),
            learning_rate=self.hyperparams.get('learning_rate', 0.1),
            max_depth=self.hyperparams.get('max_depth', 6),
            reg_lambda=self.hyperparams.get('reg_lambda', 1.0),
            random_state=self.random_seed,
            enable_categorical=True,
            tree_method='hist',
            early_stopping_rounds=self.hyperparams.get('early_stopping_rounds', 50) if eval_set else None,
            verbosity=1
        )

        model.fit(
            X_train_copy, y_train,
            eval_set=eval_set,
            verbose=100
        )

        return model

    def _train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series]
    ) -> LGBMRegressor:
        """Train LightGBM model."""
        # Handle categorical features
        X_train_copy = X_train.copy()
        if self.categorical_features:
            for cat_col in self.categorical_features:
                if cat_col in X_train_copy.columns:
                    X_train_copy[cat_col] = X_train_copy[cat_col].astype('category')

        eval_set = None
        if X_val is not None and y_val is not None:
            X_val_copy = X_val.copy()
            if self.categorical_features:
                for cat_col in self.categorical_features:
                    if cat_col in X_val_copy.columns:
                        X_val_copy[cat_col] = X_val_copy[cat_col].astype('category')
            eval_set = [(X_val_copy, y_val)]

        model = LGBMRegressor(
            n_estimators=self.hyperparams.get('n_estimators', 1000),
            learning_rate=self.hyperparams.get('learning_rate', 0.1),
            max_depth=self.hyperparams.get('max_depth', 6),
            reg_lambda=self.hyperparams.get('reg_lambda', 1.0),
            random_state=self.random_seed,
            verbosity=1
        )

        callbacks = None
        if eval_set:
            from lightgbm import early_stopping, log_evaluation
            callbacks = [
                early_stopping(stopping_rounds=self.hyperparams.get('early_stopping_rounds', 50)),
                log_evaluation(period=100)
            ]

        model.fit(
            X_train_copy, y_train,
            eval_set=eval_set,
            callbacks=callbacks
        )

        return model

    def _train_prophet(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Prophet:
        """Train Prophet model."""
        # Prepare data for Prophet
        df_prophet = pd.DataFrame({
            'ds': y_train.index,
            'y': y_train.values
        })

        # Add regressors from X_train
        for col in X_train.columns:
            if col != 'holiday_name':  # Skip text column
                df_prophet[col] = X_train[col].values

        # Initialize Prophet
        model = Prophet(
            growth=self.hyperparams.get('growth', 'linear'),
            changepoint_prior_scale=self.hyperparams.get('changepoint_prior_scale', 0.05),
            seasonality_prior_scale=self.hyperparams.get('seasonality_prior_scale', 10.0),
            holidays_prior_scale=self.hyperparams.get('holidays_prior_scale', 10.0),
            seasonality_mode=self.hyperparams.get('seasonality_mode', 'additive'),
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True
        )

        # Add regressors
        for col in X_train.columns:
            if col != 'holiday_name':
                model.add_regressor(col)

        # Fit model
        model.fit(df_prophet)

        return model

    def _train_sarimax(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> SARIMAX:
        """Train SARIMAX model."""
        # Limit features for SARIMAX (computational constraints)
        # Use top 10 most correlated features
        correlations = X_train.corrwith(y_train).abs().sort_values(ascending=False)
        top_features = correlations.head(10).index.tolist()

        X_train_subset = X_train[top_features]

        logger.info(f"SARIMAX using top {len(top_features)} features: {top_features}")

        # Fit SARIMAX
        model = SARIMAX(
            endog=y_train,
            exog=X_train_subset,
            order=self.hyperparams.get('order', (1, 0, 1)),
            seasonal_order=self.hyperparams.get('seasonal_order', (1, 0, 1, 24)),
            trend=self.hyperparams.get('trend', 'c'),
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        fitted_model = model.fit(disp=False, maxiter=100)

        # Store subset features for prediction
        self.sarimax_features = top_features

        return fitted_model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions.

        Args:
            X: Features

        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")

        if self.model_type in ['catboost', 'xgboost', 'lightgbm']:
            # Handle categorical features
            X_copy = X.copy()
            if self.categorical_features and self.model_type in ['xgboost', 'lightgbm']:
                for cat_col in self.categorical_features:
                    if cat_col in X_copy.columns:
                        X_copy[cat_col] = X_copy[cat_col].astype('category')

            return self.model.predict(X_copy)

        elif self.model_type == 'prophet':
            # Prepare data for Prophet
            df_prophet = pd.DataFrame({'ds': X.index})
            for col in X.columns:
                if col != 'holiday_name':
                    df_prophet[col] = X[col].values

            forecast = self.model.predict(df_prophet)
            return forecast['yhat'].values

        elif self.model_type == 'sarimax':
            # Use subset of features
            X_subset = X[self.sarimax_features]
            predictions = self.model.forecast(steps=len(X), exog=X_subset)
            return predictions.values

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance.

        Returns:
            DataFrame with feature names and importance scores (None for statistical models)
        """
        if self.model is None:
            raise ValueError("Model must be trained first")

        if self.model_type == 'catboost':
            importance = self.model.get_feature_importance()
            return pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)

        elif self.model_type in ['xgboost', 'lightgbm']:
            importance = self.model.feature_importances_
            return pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)

        elif self.model_type in ['prophet', 'sarimax']:
            logger.info("Feature importance not available for statistical models")
            return None

        else:
            return None
