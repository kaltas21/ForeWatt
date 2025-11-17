"""
PatchTST Model Trainer with Optuna Optimization
===============================================
Patched Time Series Transformer for efficient long-term forecasting.

Paper: "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers" (ICLR 2023)

Features:
- Patch-based input representation
- Channel-independent processing
- Efficient long-sequence modeling
- Transformer encoder architecture
- Bayesian hyperparameter optimization with Optuna

Author: ForeWatt Team
Date: November 2025
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import PatchTST
    from neuralforecast.losses.pytorch import MAE, SMAPE
    import torch
except ImportError:
    logging.warning("NeuralForecast not installed. Install with: pip install neuralforecast")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PatchTSTTrainer:
    """
    PatchTST trainer with Optuna hyperparameter optimization.
    """

    def __init__(
        self,
        target: str = 'consumption',
        horizon: int = 24,
        input_size: int = 168,
        random_seed: int = 42
    ):
        """
        Initialize PatchTST trainer.

        Args:
            target: Target variable ('consumption' or 'price_real')
            horizon: Forecast horizon
            input_size: Lookback window size
            random_seed: Random seed
        """
        self.target = target
        self.horizon = horizon
        self.input_size = input_size
        self.random_seed = random_seed
        self.model = None
        self.best_params = None

    def get_search_space(self) -> Dict[str, Any]:
        """
        Define constrained search space for PatchTST.

        Returns:
            Dictionary with parameter ranges
        """
        return {
            # Patch length (how many time steps per patch)
            'patch_len': [8, 12, 16, 24],

            # Stride (overlap between patches)
            'stride': [4, 8, 12, 16],

            # Number of encoder layers
            'n_layers': [2, 3, 4],

            # Hidden dimension
            'hidden_size': [64, 128, 256, 512],

            # Number of attention heads
            'n_heads': [4, 8, 16],

            # Dropout rate
            'dropout': [0.1, 0.2, 0.3],

            # Learning rate
            'learning_rate': [1e-4, 1e-3, 1e-2],

            # Batch size
            'batch_size': [32, 64, 128, 256],

            # Max training steps
            'max_steps': [500, 1000, 2000, 3000],

            # Early stopping
            'early_stop_patience_steps': [50, 100, 150]
        }

    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters for a trial.

        Args:
            trial: Optuna trial

        Returns:
            Dictionary of suggested hyperparameters
        """
        search_space = self.get_search_space()

        params = {
            'patch_len': trial.suggest_categorical('patch_len', search_space['patch_len']),
            'stride': trial.suggest_categorical('stride', search_space['stride']),
            'n_layers': trial.suggest_categorical('n_layers', search_space['n_layers']),
            'hidden_size': trial.suggest_categorical('hidden_size', search_space['hidden_size']),
            'n_heads': trial.suggest_categorical('n_heads', search_space['n_heads']),
            'dropout': trial.suggest_uniform('dropout', 0.1, 0.3),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-2),
            'batch_size': trial.suggest_categorical('batch_size', search_space['batch_size']),
            'max_steps': trial.suggest_categorical('max_steps', search_space['max_steps']),
            'early_stop_patience_steps': trial.suggest_categorical(
                'early_stop_patience_steps',
                search_space['early_stop_patience_steps']
            )
        }

        # Ensure stride <= patch_len
        if params['stride'] > params['patch_len']:
            params['stride'] = params['patch_len']

        return params

    def create_model(self, hyperparams: Dict[str, Any]) -> PatchTST:
        """
        Create PatchTST model with given hyperparameters.

        Args:
            hyperparams: Model hyperparameters

        Returns:
            Configured PatchTST model
        """
        model = PatchTST(
            h=self.horizon,
            input_size=self.input_size,
            patch_len=hyperparams['patch_len'],
            stride=hyperparams['stride'],
            n_layers=hyperparams['n_layers'],
            hidden_size=hyperparams['hidden_size'],
            n_heads=hyperparams['n_heads'],
            dropout=hyperparams['dropout'],
            learning_rate=hyperparams['learning_rate'],
            batch_size=hyperparams['batch_size'],
            max_steps=hyperparams['max_steps'],
            early_stop_patience_steps=hyperparams['early_stop_patience_steps'],
            val_check_steps=50,
            num_lr_decays=3,
            random_seed=self.random_seed,
            loss=MAE(),
            valid_loss=SMAPE(),
            scaler_type='robust',
            revin=True,  # Reversible instance normalization
            affine=True,
            subtract_last=False
        )

        return model

    def prepare_neuralforecast_data(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> pd.DataFrame:
        """
        Prepare data in NeuralForecast format.

        Args:
            X: Features dataframe with datetime index
            y: Target series with datetime index

        Returns:
            DataFrame in NeuralForecast format
        """
        df = pd.DataFrame({
            'unique_id': 'series_1',
            'ds': X.index,
            'y': y.values
        })

        # Add exogenous features
        for col in X.columns:
            df[col] = X[col].values

        return df

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        hyperparams: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, float]]:
        """
        Train PatchTST model.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            hyperparams: Model hyperparameters

        Returns:
            Tuple of (trained_model, validation_metrics)
        """
        if hyperparams is None:
            hyperparams = self.get_default_hyperparameters()

        logger.info(f"\n{'='*80}")
        logger.info(f"TRAINING PATCHTST: {self.target}")
        logger.info(f"{'='*80}")
        logger.info(f"Horizon: {self.horizon}h")
        logger.info(f"Input size: {self.input_size}h")
        logger.info(f"Patch length: {hyperparams['patch_len']}")
        logger.info(f"Stride: {hyperparams['stride']}")
        logger.info(f"Train samples: {len(X_train)}")
        logger.info(f"Val samples: {len(X_val)}")
        logger.info(f"Hyperparameters: {hyperparams}")

        # Prepare data
        train_df = self.prepare_neuralforecast_data(X_train, y_train)
        val_df = self.prepare_neuralforecast_data(X_val, y_val)
        full_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)

        # Create model
        model = self.create_model(hyperparams)

        # Create NeuralForecast wrapper
        nf = NeuralForecast(models=[model], freq='H')

        # Train
        try:
            nf.fit(
                df=full_df,
                val_size=len(val_df),
                use_init_models=False
            )

            # Predict on validation
            val_pred = nf.predict(df=full_df)
            val_predictions = val_pred['PatchTST'].values[-len(y_val):]

            # Calculate validation metrics
            from src.models.evaluate import (
                mean_absolute_error,
                symmetric_mean_absolute_percentage_error,
                mean_absolute_scaled_error
            )

            metrics = {
                'MAE': mean_absolute_error(y_val.values, val_predictions),
                'sMAPE': symmetric_mean_absolute_percentage_error(y_val.values, val_predictions),
                'MASE': mean_absolute_scaled_error(
                    y_val.values,
                    val_predictions,
                    y_train.values,
                    seasonality=24
                )
            }

            logger.info(f"\nValidation metrics:")
            logger.info(f"  MAE: {metrics['MAE']:.2f}")
            logger.info(f"  sMAPE: {metrics['sMAPE']:.2f}%")
            logger.info(f"  MASE: {metrics['MASE']:.4f}")

            self.model = nf
            return nf, metrics

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    def predict(
        self,
        X: pd.DataFrame,
        horizon: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate predictions.

        Args:
            X: Features
            horizon: Forecast horizon

        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        if horizon is None:
            horizon = self.horizon

        # Prepare data
        dummy_y = pd.Series(np.zeros(len(X)), index=X.index)
        df = self.prepare_neuralforecast_data(X, dummy_y)

        # Predict
        predictions = self.model.predict(df=df, horizon=horizon)

        return predictions['PatchTST'].values

    def get_default_hyperparameters(self) -> Dict[str, Any]:
        """Get default hyperparameters."""
        return {
            'patch_len': 16,
            'stride': 8,
            'n_layers': 3,
            'hidden_size': 128,
            'n_heads': 8,
            'dropout': 0.2,
            'learning_rate': 1e-3,
            'batch_size': 64,
            'max_steps': 1000,
            'early_stop_patience_steps': 100
        }


def optimize_patchtst(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    target: str = 'consumption',
    horizon: int = 24,
    input_size: int = 168,
    n_trials: int = 50,
    timeout: Optional[int] = None,
    cv_folds: Optional[List[Tuple]] = None
) -> Tuple[Dict[str, Any], float]:
    """
    Optimize PatchTST hyperparameters using Optuna.

    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        target: Target variable name
        horizon: Forecast horizon
        input_size: Lookback window
        n_trials: Number of optimization trials
        timeout: Timeout in seconds
        cv_folds: Optional CV folds

    Returns:
        Tuple of (best_hyperparameters, best_score)
    """
    logger.info(f"\n{'█'*80}")
    logger.info(f"PATCHTST BAYESIAN OPTIMIZATION: {target}")
    logger.info(f"{'█'*80}")
    logger.info(f"Trials: {n_trials}")
    logger.info(f"Horizon: {horizon}h")
    logger.info(f"Input size: {input_size}h")

    trainer = PatchTSTTrainer(target=target, horizon=horizon, input_size=input_size)

    def objective(trial: optuna.Trial) -> float:
        """Objective function for Optuna."""
        hyperparams = trainer.suggest_hyperparameters(trial)

        try:
            if cv_folds is not None:
                fold_scores = []
                for fold_idx, (train_idx, val_idx) in enumerate(cv_folds):
                    X_train_fold = X_train.iloc[train_idx]
                    y_train_fold = y_train.iloc[train_idx]
                    X_val_fold = X_train.iloc[val_idx]
                    y_val_fold = y_train.iloc[val_idx]

                    _, metrics = trainer.train(
                        X_train_fold, y_train_fold,
                        X_val_fold, y_val_fold,
                        hyperparams
                    )

                    fold_scores.append(metrics['sMAPE'])
                    trial.report(metrics['sMAPE'], fold_idx)

                    if trial.should_prune():
                        raise optuna.TrialPruned()

                score = np.mean(fold_scores)
            else:
                _, metrics = trainer.train(X_train, y_train, X_val, y_val, hyperparams)
                score = metrics['sMAPE']

            return score

        except Exception as e:
            logger.warning(f"Trial {trial.number} failed: {e}")
            return float('inf')

    # Create study
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    )

    # Optimize
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True
    )

    best_params = study.best_params
    best_score = study.best_value

    logger.info(f"\n{'='*80}")
    logger.info("OPTIMIZATION COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Best sMAPE: {best_score:.4f}%")
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"{'='*80}\n")

    return best_params, best_score
