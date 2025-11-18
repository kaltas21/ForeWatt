"""
Model Trainer for Baseline Models
==================================
Unified trainer for all baseline models with intelligent feature selection.

Supports:
- CatBoost
- XGBoost
- LightGBM
- Prophet

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
            model_type: One of ['catboost', 'xgboost', 'lightgbm', 'prophet']
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
        self.prophet_features = None  # Store numeric features for Prophet
        self.numeric_features = None  # Store numeric features for boosting models
        self.used_categorical_features = None  # Store actual categorical features used in training

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
        # Filter to numeric columns only, then add categorical back
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        X_train_clean = X_train[numeric_cols].copy()

        # Add categorical features that should be categorical
        cat_features_to_add = []
        if self.categorical_features:
            for cat_col in self.categorical_features:
                if cat_col in X_train.columns and cat_col not in numeric_cols:
                    X_train_clean[cat_col] = X_train[cat_col].astype(str)
                    cat_features_to_add.append(cat_col)

        # Store for prediction
        self.numeric_features = numeric_cols
        self.used_categorical_features = cat_features_to_add

        logger.info(f"Using {len(numeric_cols)} numeric + {len(cat_features_to_add)} categorical features")

        # Create training pool
        train_pool = Pool(
            data=X_train_clean,
            label=y_train,
            cat_features=cat_features_to_add
        )

        # Create validation pool if provided
        eval_set = None
        if X_val is not None and y_val is not None:
            X_val_clean = X_val[numeric_cols].copy()
            for cat_col in cat_features_to_add:
                if cat_col in X_val.columns:
                    X_val_clean[cat_col] = X_val[cat_col].astype(str)

            eval_set = Pool(
                data=X_val_clean,
                label=y_val,
                cat_features=cat_features_to_add
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
            cat_features=cat_features_to_add
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
        # Filter to numeric columns only, then add categorical back
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        X_train_copy = X_train[numeric_cols].copy()

        # Add categorical features that should be categorical
        cat_features_to_add = []
        if self.categorical_features:
            for cat_col in self.categorical_features:
                if cat_col in X_train.columns and cat_col not in numeric_cols:
                    X_train_copy[cat_col] = X_train[cat_col].astype('category')
                    cat_features_to_add.append(cat_col)

        # Store for prediction
        self.numeric_features = numeric_cols
        self.used_categorical_features = cat_features_to_add

        logger.info(f"Using {len(numeric_cols)} numeric + {len(cat_features_to_add)} categorical features")

        eval_set = None
        if X_val is not None and y_val is not None:
            X_val_copy = X_val[numeric_cols].copy()
            for cat_col in cat_features_to_add:
                if cat_col in X_val.columns:
                    X_val_copy[cat_col] = X_val[cat_col].astype('category')
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
        # LightGBM: Use ONLY numeric features (no categorical to avoid mismatch errors)
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        X_train_copy = X_train[numeric_cols].copy()

        # Store for prediction (no categorical features for LightGBM)
        self.numeric_features = numeric_cols
        self.used_categorical_features = []  # Empty - LightGBM doesn't use categorical

        logger.info(f"Using {len(numeric_cols)} numeric features (no categorical to avoid LightGBM issues)")

        eval_set = None
        if X_val is not None and y_val is not None:
            X_val_copy = X_val[numeric_cols].copy()
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
        # Filter to numeric columns only (Prophet can't handle categorical)
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        X_train_numeric = X_train[numeric_cols]

        logger.info(f"Filtered to {len(numeric_cols)} numeric features (removed categorical)")

        # Prepare data for Prophet
        # Remove timezone from datetime index (Prophet doesn't support it)
        ds_values = y_train.index
        if hasattr(ds_values, 'tz') and ds_values.tz is not None:
            ds_values = ds_values.tz_localize(None)

        df_prophet = pd.DataFrame({
            'ds': ds_values,
            'y': y_train.values
        })

        # Add numeric regressors from X_train
        for col in numeric_cols:
            df_prophet[col] = X_train_numeric[col].values

        # Drop rows with NaN values (from lag/rolling features)
        rows_before = len(df_prophet)
        df_prophet = df_prophet.dropna()
        rows_after = len(df_prophet)
        if rows_before > rows_after:
            logger.info(f"Dropped {rows_before - rows_after} rows with NaN values ({rows_after} remaining)")

        # Update numeric_cols to only include columns without NaN
        numeric_cols = [col for col in numeric_cols if col in df_prophet.columns]

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

        # Add numeric regressors
        for col in numeric_cols:
            model.add_regressor(col)

        # Store numeric columns for prediction
        self.prophet_features = numeric_cols

        # Fit model
        model.fit(df_prophet)

        return model

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
            # Apply same transformation as training: numeric + categorical features
            if self.numeric_features is None:
                raise ValueError("Model must be trained first (numeric_features not set)")

            # Start with numeric features
            X_copy = X[self.numeric_features].copy()

            # Add categorical features with appropriate dtype (not for LightGBM)
            if self.used_categorical_features and self.model_type != 'lightgbm':
                for cat_col in self.used_categorical_features:
                    if cat_col in X.columns:
                        if self.model_type == 'catboost':
                            # CatBoost needs string dtype for categorical
                            X_copy[cat_col] = X[cat_col].astype(str)
                        elif self.model_type == 'xgboost':
                            # XGBoost needs category dtype
                            X_copy[cat_col] = X[cat_col].astype('category')

            return self.model.predict(X_copy)

        elif self.model_type == 'prophet':
            # Prepare data for Prophet (numeric columns only)
            # Remove timezone from datetime index (Prophet doesn't support it)
            ds_values = X.index
            if hasattr(ds_values, 'tz') and ds_values.tz is not None:
                ds_values = ds_values.tz_localize(None)

            df_prophet = pd.DataFrame({'ds': ds_values})
            # Use only the numeric features stored during training
            for col in self.prophet_features:
                if col in X.columns:
                    df_prophet[col] = X[col].values

            # Fill NaN values with forward fill then backward fill
            # (can't drop rows during prediction as we need same-length output)
            df_prophet = df_prophet.fillna(method='ffill').fillna(method='bfill')

            forecast = self.model.predict(df_prophet)
            return forecast['yhat'].values

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
            # Use actual features used in training
            used_features = self.numeric_features + (self.used_categorical_features or [])
            return pd.DataFrame({
                'feature': used_features,
                'importance': importance
            }).sort_values('importance', ascending=False)

        elif self.model_type in ['xgboost', 'lightgbm']:
            importance = self.model.feature_importances_
            # Use actual features used in training
            used_features = self.numeric_features + (self.used_categorical_features or [])
            return pd.DataFrame({
                'feature': used_features,
                'importance': importance
            }).sort_values('importance', ascending=False)

        elif self.model_type == 'prophet':
            logger.info("Feature importance not available for Prophet")
            return None

        else:
            return None
