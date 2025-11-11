"""
CatBoost Baseline Model for Electricity Demand Forecasting
==========================================================
Gradient boosting with categorical features support.

Features:
- Automatic handling of categorical features
- Basic hyperparameter tuning
- Early stopping to prevent overfitting
- MLflow tracking

Author: ForeWatt Team
Date: November 2025
"""

import sys
import numpy as np
import pandas as pd
import mlflow
import mlflow.catboost
from catboost import CatBoostRegressor, Pool
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.evaluate import evaluate_forecast

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CatBoostForecaster:
    """CatBoost model wrapper for electricity demand forecasting."""

    def __init__(
        self,
        iterations: int = 1000,
        learning_rate: float = 0.1,
        depth: int = 6,
        l2_leaf_reg: float = 3.0,
        random_seed: int = 42,
        early_stopping_rounds: int = 50
    ):
        """
        Initialize CatBoost forecaster.

        Args:
            iterations: Number of boosting iterations
            learning_rate: Learning rate
            depth: Tree depth
            l2_leaf_reg: L2 regularization coefficient
            random_seed: Random seed for reproducibility
            early_stopping_rounds: Early stopping rounds
        """
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.random_seed = random_seed
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None
        self.feature_names = None
        self.categorical_features = None

    def prepare_features(
        self,
        df: pd.DataFrame,
        target_col: str = 'consumption',
        exclude_cols: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Prepare features for CatBoost.

        Args:
            df: Master dataset
            target_col: Target column name
            exclude_cols: Columns to exclude from features

        Returns:
            Tuple of (X, y, categorical_features)
        """
        # Default columns to exclude
        if exclude_cols is None:
            exclude_cols = [
                target_col,
                'datetime',
                'timestamp',
                'holiday_name',  # Text column
                'date_only'  # Date column if present
            ]

        # Also exclude any column with dtype 'object' that's not holiday_name
        object_cols = df.select_dtypes(include=['object']).columns.tolist()
        object_cols = [col for col in object_cols if col not in exclude_cols and col != 'holiday_name']
        exclude_cols.extend(object_cols)

        # Identify categorical features
        categorical_features = []
        for col in df.columns:
            if col not in exclude_cols:
                # Holiday name is text categorical
                if col == 'holiday_name':
                    categorical_features.append(col)
                # Binary flags (0/1) can be treated as categorical
                elif df[col].dtype == 'int64' and df[col].nunique() <= 2:
                    categorical_features.append(col)

        # Feature columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        # Handle holiday_name NaNs (mark as "no_holiday")
        df = df.copy()
        if 'holiday_name' in feature_cols:
            df['holiday_name'] = df['holiday_name'].fillna('no_holiday')

        X = df[feature_cols]
        y = df[target_col]

        self.feature_names = feature_cols
        self.categorical_features = categorical_features

        logger.info(f"Features: {len(feature_cols)} total, {len(categorical_features)} categorical")
        logger.info(f"Categorical features: {categorical_features}")

        return X, y, categorical_features

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> CatBoostRegressor:
        """
        Train CatBoost model.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (for early stopping)
            y_val: Validation target

        Returns:
            Trained CatBoost model
        """
        logger.info("Training CatBoost model...")

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

        # Initialize model
        self.model = CatBoostRegressor(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            random_seed=self.random_seed,
            loss_function='RMSE',
            eval_metric='RMSE',
            early_stopping_rounds=self.early_stopping_rounds,
            verbose=100,
            cat_features=self.categorical_features
        )

        # Train
        self.model.fit(
            train_pool,
            eval_set=eval_set,
            use_best_model=True if eval_set is not None else False
        )

        logger.info("✓ CatBoost model trained successfully")
        logger.info(f"✓ Best iteration: {self.model.best_iteration_}")
        logger.info(f"✓ Best score: {self.model.best_score_}")

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

        importance = self.model.get_feature_importance()
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
    """
    Split data into train/val/test temporally.

    Args:
        df: Full dataset
        val_size: Validation set fraction
        test_size: Test set fraction

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    test_idx = int(len(df) * (1 - test_size))
    val_idx = int(len(df) * (1 - test_size - val_size))

    train_df = df.iloc[:val_idx]
    val_df = df.iloc[val_idx:test_idx]
    test_df = df.iloc[test_idx:]

    logger.info(f"Train: {len(train_df)} samples ({train_df.index.min()} to {train_df.index.max()})")
    logger.info(f"Val:   {len(val_df)} samples ({val_df.index.min()} to {val_df.index.max()})")
    logger.info(f"Test:  {len(test_df)} samples ({test_df.index.min()} to {test_df.index.max()})")

    return train_df, val_df, test_df


def run_catboost_baseline(
    experiment_name: str = "ForeWatt-Baseline-CatBoost",
    run_name: str = "catboost_v1_baseline",
    val_size: float = 0.1,
    test_size: float = 0.2,
    hyperparams: Optional[Dict] = None,
    mlflow_uri: Optional[str] = None
) -> Dict[str, float]:
    """
    Run CatBoost baseline experiment.

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
            'iterations': 1000,
            'learning_rate': 0.1,
            'depth': 6,
            'l2_leaf_reg': 3.0,
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
    logger.info("CATBOOST BASELINE MODEL TRAINING")
    logger.info("="*80)

    df = load_master_data()

    # Drop rows with missing target
    df = df.dropna(subset=['consumption'])

    # Train-val-test split
    train_df, val_df, test_df = train_val_test_split(df, val_size=val_size, test_size=test_size)

    # Initialize forecaster
    forecaster = CatBoostForecaster(**hyperparams)

    # Prepare features
    X_train, y_train, cat_features = forecaster.prepare_features(train_df)
    X_val, y_val, _ = forecaster.prepare_features(val_df)
    X_test, y_test, _ = forecaster.prepare_features(test_df)

    # Start MLflow run
    with mlflow.start_run(run_name=run_name) as run:

        # Log parameters
        mlflow.log_param("model_type", "CatBoost")
        for key, value in hyperparams.items():
            mlflow.log_param(key, value)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("val_samples", len(X_val))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_categorical_features", len(cat_features))

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
            model_name="CatBoost Baseline"
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
        importance_path = "/tmp/catboost_feature_importance.csv"
        feature_importance.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path)

        logger.info("\nTop 10 Most Important Features:")
        logger.info(top_features.head(10).to_string(index=False))

        # Log model
        mlflow.catboost.log_model(model, "catboost_model")  # type: ignore

        logger.info(f"✓ MLflow run ID: {run.info.run_id}")

    return metrics


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train CatBoost baseline model')
    parser.add_argument('--experiment', type=str, default='ForeWatt-Baseline-CatBoost',
                        help='MLflow experiment name')
    parser.add_argument('--run-name', type=str, default='catboost_v1_baseline',
                        help='MLflow run name')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set size')
    parser.add_argument('--val-size', type=float, default=0.1,
                        help='Validation set size')

    args = parser.parse_args()

    # Run baseline
    metrics = run_catboost_baseline(
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
