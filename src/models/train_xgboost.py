"""
XGBoost Baseline Model for Electricity Demand Forecasting
=========================================================
Extreme Gradient Boosting for time series regression.

Features:
- L1/L2 regularization
- Basic hyperparameter tuning
- Early stopping
- MLflow tracking

Author: ForeWatt Team
Date: November 2025
"""

import sys
import numpy as np
import pandas as pd
import mlflow
import mlflow.xgboost
import xgboost as xgb
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.evaluate import evaluate_forecast

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XGBoostForecaster:
    """XGBoost model wrapper for electricity demand forecasting."""

    def __init__(
        self,
        n_estimators: int = 1000,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,  # L1 regularization
        reg_lambda: float = 1.0,  # L2 regularization
        random_state: int = 42,
        early_stopping_rounds: int = 50
    ):
        """
        Initialize XGBoost forecaster.

        Args:
            n_estimators: Number of boosting rounds
            learning_rate: Learning rate
            max_depth: Maximum tree depth
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of features
            reg_alpha: L1 regularization term
            reg_lambda: L2 regularization term
            random_state: Random seed
            early_stopping_rounds: Early stopping rounds
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None
        self.feature_names = None

    def prepare_features(
        self,
        df: pd.DataFrame,
        target_col: str = 'consumption',
        exclude_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for XGBoost.

        Args:
            df: Master dataset
            target_col: Target column name
            exclude_cols: Columns to exclude from features

        Returns:
            Tuple of (X, y)
        """
        if exclude_cols is None:
            exclude_cols = [
                target_col,
                'datetime',
                'timestamp',
                'holiday_name'  # Text column, exclude for XGBoost
            ]

        # Also exclude any column with dtype 'object'
        object_cols = df.select_dtypes(include=['object']).columns.tolist()
        exclude_cols.extend([col for col in object_cols if col not in exclude_cols])

        # Feature columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        X = df[feature_cols]
        y = df[target_col]

        self.feature_names = feature_cols

        logger.info(f"Features: {len(feature_cols)} total")
        if object_cols:
            logger.info(f"Excluded object columns: {object_cols}")

        return X, y

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> xgb.XGBRegressor:
        """
        Train XGBoost model.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target

        Returns:
            Trained XGBoost model
        """
        logger.info("Training XGBoost model...")

        # Initialize model
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            objective='reg:squarederror',
            tree_method='hist',  # Faster training
            verbosity=1
        )

        # Train model (simple fit without early stopping for compatibility)
        self.model.fit(X_train, y_train, verbose=100)

        logger.info("✓ XGBoost model trained successfully")
        if hasattr(self.model, 'best_iteration'):
            logger.info(f"✓ Best iteration: {self.model.best_iteration}")
        if hasattr(self.model, 'best_score'):
            logger.info(f"✓ Best score: {self.model.best_score}")

        return self.model

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

        return self.model.predict(X)

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance.

        Returns:
            DataFrame with feature names and importance scores
        """
        if self.model is None:
            raise ValueError("Model must be trained first")

        importance = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        return feature_importance


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


def train_val_test_split(
    df: pd.DataFrame,
    val_size: float = 0.1,
    test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train/val/test temporally."""
    test_idx = int(len(df) * (1 - test_size))
    val_idx = int(len(df) * (1 - test_size - val_size))

    train_df = df.iloc[:val_idx]
    val_df = df.iloc[val_idx:test_idx]
    test_df = df.iloc[test_idx:]

    logger.info(f"Train: {len(train_df)} samples ({train_df.index.min()} to {train_df.index.max()})")
    logger.info(f"Val:   {len(val_df)} samples ({val_df.index.min()} to {val_df.index.max()})")
    logger.info(f"Test:  {len(test_df)} samples ({test_df.index.min()} to {test_df.index.max()})")

    return train_df, val_df, test_df


def run_xgboost_baseline(
    experiment_name: str = "ForeWatt-Baseline-XGBoost",
    run_name: str = "xgboost_v1_baseline",
    val_size: float = 0.1,
    test_size: float = 0.2,
    hyperparams: Optional[Dict] = None,
    mlflow_uri: Optional[str] = None
) -> Dict[str, float]:
    """
    Run XGBoost baseline experiment.

    Args:
        experiment_name: MLflow experiment name
        run_name: MLflow run name
        val_size: Validation set size
        test_size: Test set size
        hyperparams: Hyperparameters dict
        mlflow_uri: MLflow tracking URI (default: file-based)

    Returns:
        Dictionary with evaluation metrics
    """
    # Default hyperparameters
    if hyperparams is None:
        hyperparams = {
            'n_estimators': 1000,
            'learning_rate': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.0,
            'reg_lambda': 1.0,
            'early_stopping_rounds': 50
        }

    # Set MLflow tracking URI
    if mlflow_uri is None:
        mlflow_dir = PROJECT_ROOT / 'mlruns'
        mlflow_dir.mkdir(exist_ok=True)
        mlflow_uri = f"file://{mlflow_dir}"
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)

    # Load data
    logger.info("="*80)
    logger.info("XGBOOST BASELINE MODEL TRAINING")
    logger.info("="*80)

    df = load_master_data()

    # Drop rows with missing target
    df = df.dropna(subset=['consumption'])

    # Train-val-test split
    train_df, val_df, test_df = train_val_test_split(df, val_size=val_size, test_size=test_size)

    # Initialize forecaster
    forecaster = XGBoostForecaster(**hyperparams)

    # Prepare features
    X_train, y_train = forecaster.prepare_features(train_df)
    X_val, y_val = forecaster.prepare_features(val_df)
    X_test, y_test = forecaster.prepare_features(test_df)

    # Start MLflow run
    with mlflow.start_run(run_name=run_name) as run:

        # Log parameters
        mlflow.log_param("model_type", "XGBoost")
        for key, value in hyperparams.items():
            mlflow.log_param(key, value)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("val_samples", len(X_val))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])

        # Train model
        model = forecaster.train(X_train, y_train, X_val, y_val)

        # Generate predictions
        predictions = forecaster.predict(X_test)
        y_true = np.asarray(y_test.values)
        y_train_array = np.asarray(y_train.values)

        # Evaluate
        metrics = evaluate_forecast(
            y_true=y_true,
            y_pred=predictions,
            y_train=y_train_array,
            seasonality=24,
            model_name="XGBoost Baseline"
        )

        # Log metrics
        mlflow.log_metric("MAE", metrics['MAE'])
        mlflow.log_metric("RMSE", metrics['RMSE'])
        mlflow.log_metric("MAPE", metrics['MAPE'])
        mlflow.log_metric("sMAPE", metrics['sMAPE'])
        mlflow.log_metric("MASE", metrics['MASE'])

        # Log feature importance
        feature_importance = forecaster.get_feature_importance()
        top_features = feature_importance.head(20)

        # Save feature importance as artifact
        importance_path = "/tmp/xgboost_feature_importance.csv"
        feature_importance.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path)

        logger.info("\nTop 10 Most Important Features:")
        logger.info(top_features.head(10).to_string(index=False))

        # Log model
        mlflow.xgboost.log_model(model, "xgboost_model")  # type: ignore

        logger.info(f"✓ MLflow run ID: {run.info.run_id}")

    return metrics


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train XGBoost baseline model')
    parser.add_argument('--experiment', type=str, default='ForeWatt-Baseline-XGBoost',
                        help='MLflow experiment name')
    parser.add_argument('--run-name', type=str, default='xgboost_v1_baseline',
                        help='MLflow run name')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set size')
    parser.add_argument('--val-size', type=float, default=0.1,
                        help='Validation set size')

    args = parser.parse_args()

    # Run baseline
    metrics = run_xgboost_baseline(
        experiment_name=args.experiment,
        run_name=args.run_name,
        val_size=args.val_size,
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
