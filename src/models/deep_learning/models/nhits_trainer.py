"""
N-HiTS Model Trainer with Optuna Optimization
==============================================
Neural Hierarchical Interpolation for Time Series Forecasting.

Paper: "N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting" (AAAI 2023)

Features:
- Multi-rate sampling with hierarchical interpolation
- Handles multiple seasonalities (daily, weekly, yearly)
- Stack-based architecture (identity, trend, seasonality)
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
    from neuralforecast.models import NHITS
    from neuralforecast.losses.pytorch import MAE, MSE, SMAPE
    import torch
except ImportError:
    logging.warning("NeuralForecast not installed. Install with: pip install neuralforecast")

from src.models.deep_learning.hardware_config import get_hardware_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NHiTSTrainer:
    """
    N-HiTS model trainer with Optuna hyperparameter optimization.
    """

    def __init__(
        self,
        target: str = 'consumption',
        horizon: int = 24,
        input_size: int = 168,
        random_seed: int = 42,
        device: Optional[str] = None
    ):
        """
        Initialize N-HiTS trainer.

        Args:
            target: Target variable ('consumption' or 'price_real')
            horizon: Forecast horizon (number of steps ahead)
            input_size: Lookback window size
            random_seed: Random seed for reproducibility
            device: Device to use ('cuda', 'mps', 'cpu', or None for auto-detect)
        """
        self.target = target
        self.horizon = horizon
        self.input_size = input_size
        self.random_seed = random_seed
        self.model = None
        self.best_params = None

        # Hardware configuration
        self.hw_config = get_hardware_config(force_device=device)
        self.device = self.hw_config.device
        self.device_type = self.hw_config.device_type

        logger.info(f"N-HiTS Trainer initialized with device: {self.device_type.upper()}")
        if self.hw_config.memory_gb:
            logger.info(f"Available memory: {self.hw_config.memory_gb:.2f} GB")

    def get_search_space(self) -> Dict[str, Any]:
        """
        Define constrained search space for N-HiTS.

        Returns:
            Dictionary with parameter ranges
        """
        return {
            # Stack configuration (fixed: identity, trend, seasonality)
            'stack_types': [['identity', 'trend', 'seasonality']],

            # Number of blocks per stack
            'n_blocks': [1, 2, 3],

            # Pooling kernel size (for downsampling)
            'n_pool_kernel_size': [[2, 2, 1], [4, 4, 1], [8, 4, 1]],

            # Frequency downsampling
            'n_freq_downsample': [[2, 1, 1], [4, 2, 1], [8, 4, 1]],

            # Hidden layer size
            'hidden_size': [128, 256, 512, 1024],

            # MLP layers
            'n_mlp_layers': [1, 2, 3],

            # Learning rate
            'learning_rate': [1e-4, 1e-3, 1e-2],

            # Batch size
            'batch_size': [32, 64, 128, 256],

            # Max training steps
            'max_steps': [500, 1000, 2000, 3000],

            # Early stopping patience
            'early_stop_patience_steps': [50, 100, 150]
        }

    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters for a trial.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of suggested hyperparameters
        """
        search_space = self.get_search_space()

        params = {
            'stack_types': search_space['stack_types'][0],  # Fixed
            'n_blocks': trial.suggest_categorical('n_blocks', search_space['n_blocks']),
            'n_pool_kernel_size': trial.suggest_categorical(
                'n_pool_kernel_size',
                [str(x) for x in search_space['n_pool_kernel_size']]
            ),
            'n_freq_downsample': trial.suggest_categorical(
                'n_freq_downsample',
                [str(x) for x in search_space['n_freq_downsample']]
            ),
            'hidden_size': trial.suggest_categorical('hidden_size', search_space['hidden_size']),
            'n_mlp_layers': trial.suggest_categorical('n_mlp_layers', search_space['n_mlp_layers']),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-2),
            'batch_size': trial.suggest_categorical('batch_size', search_space['batch_size']),
            'max_steps': trial.suggest_categorical('max_steps', search_space['max_steps']),
            'early_stop_patience_steps': trial.suggest_categorical(
                'early_stop_patience_steps',
                search_space['early_stop_patience_steps']
            )
        }

        # Convert string representations back to lists
        params['n_pool_kernel_size'] = eval(params['n_pool_kernel_size'])
        params['n_freq_downsample'] = eval(params['n_freq_downsample'])

        return params

    def create_model(self, hyperparams: Dict[str, Any]) -> NHITS:
        """
        Create N-HiTS model with given hyperparameters.

        Args:
            hyperparams: Model hyperparameters

        Returns:
            Configured N-HiTS model
        """
        # Determine accelerator for PyTorch Lightning
        if self.device_type == 'cuda':
            accelerator = 'gpu'
            devices = 1  # Use first GPU
        elif self.device_type == 'mps':
            accelerator = 'mps'  # Apple Silicon
            devices = 1
        else:
            accelerator = 'cpu'
            devices = 'auto'

        # Extract only valid N-HiTS parameters
        nhits_params = {
            'h': self.horizon,
            'input_size': self.input_size,
            'stack_types': hyperparams['stack_types'],
            'n_blocks': hyperparams['n_blocks'],  # Already a list matching stack_types
            'n_pool_kernel_size': hyperparams['n_pool_kernel_size'],
            'n_freq_downsample': hyperparams['n_freq_downsample'],
            'mlp_units': [[hyperparams['hidden_size']] * hyperparams['n_mlp_layers']] * len(hyperparams['stack_types']),
            'interpolation_mode': 'linear',
            'pooling_mode': 'MaxPool1d',
            'batch_size': hyperparams['batch_size'],
            'max_steps': hyperparams['max_steps'],
            'learning_rate': hyperparams['learning_rate'],
            'num_lr_decays': 3,
            'early_stop_patience_steps': hyperparams['early_stop_patience_steps'],
            'val_check_steps': 50,
            'random_seed': self.random_seed,
            'loss': MAE(),
            'valid_loss': SMAPE(),
            'scaler_type': 'standard',  # 'robust' uses nanmedian which is not supported on MPS
            'accelerator': accelerator,
            'devices': devices
        }

        # Add optional weight_decay if present
        if 'weight_decay' in hyperparams:
            nhits_params['weight_decay'] = hyperparams['weight_decay']

        # Add optional dropout if present
        if 'dropout' in hyperparams:
            nhits_params['dropout'] = hyperparams['dropout']

        model = NHITS(**nhits_params)

        logger.info(f"N-HiTS model created with accelerator={accelerator}, devices={devices}")

        return model

    def prepare_neuralforecast_data(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> pd.DataFrame:
        """
        Prepare data in NeuralForecast format.

        NeuralForecast expects:
        - 'unique_id': identifier (we use 'series_1' for single time series)
        - 'ds': timestamp
        - 'y': target variable
        - Other columns: exogenous features

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

        # Drop rows with NaN values (from lag features)
        initial_len = len(df)
        df = df.dropna()
        dropped = initial_len - len(df)
        if dropped > 0:
            logger.info(f"Dropped {dropped} rows with NaN values from lag features")

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
        Train N-HiTS model.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            hyperparams: Model hyperparameters (if None, use defaults)

        Returns:
            Tuple of (trained_model, validation_metrics)
        """
        if hyperparams is None:
            hyperparams = self.get_default_hyperparameters()

        logger.info(f"\n{'='*80}")
        logger.info(f"TRAINING N-HiTS: {self.target}")
        logger.info(f"{'='*80}")
        logger.info(f"Horizon: {self.horizon}h")
        logger.info(f"Input size: {self.input_size}h")
        logger.info(f"Train samples: {len(X_train)}")
        logger.info(f"Val samples: {len(X_val)}")
        logger.info(f"Hyperparameters: {hyperparams}")

        # Prepare data
        train_df = self.prepare_neuralforecast_data(X_train, y_train)
        val_df = self.prepare_neuralforecast_data(X_val, y_val)

        # Combine for NeuralForecast (it splits internally)
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
        except RuntimeError as e:
            logger.error(f"Training failed with RuntimeError: {e}")
            raise

        # Predict on validation (returns forecast for next h steps)
        val_pred = nf.predict(df=full_df)
        val_predictions = val_pred['NHITS'].values

        # Use only first h validation samples for metrics (single forecast evaluation)
        y_val_horizon = y_val.values[:self.horizon]

        # Calculate validation metrics
        from src.models.evaluate import (
            mean_absolute_error,
            symmetric_mean_absolute_percentage_error,
            mean_absolute_scaled_error
        )

        metrics = {
            'MAE': mean_absolute_error(y_val_horizon, val_predictions),
            'sMAPE': symmetric_mean_absolute_percentage_error(y_val_horizon, val_predictions),
            'MASE': mean_absolute_scaled_error(
                y_val_horizon,
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

    def predict(
        self,
        X: pd.DataFrame,
        horizon: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate predictions.

        Args:
            X: Features
            horizon: Forecast horizon (default: use self.horizon)

        Returns:
            Predictions array of shape (n_samples, horizon)
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

        return predictions['NHITS'].values

    def get_default_hyperparameters(self) -> Dict[str, Any]:
        """Get default hyperparameters."""
        return {
            'stack_types': ['identity', 'trend', 'seasonality'],
            'n_blocks': 2,
            'n_pool_kernel_size': [4, 4, 1],
            'n_freq_downsample': [4, 2, 1],
            'hidden_size': 512,
            'n_mlp_layers': 2,
            'learning_rate': 1e-3,
            'batch_size': 64,
            'max_steps': 1000,
            'early_stop_patience_steps': 100
        }


def optimize_nhits(
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
    Optimize N-HiTS hyperparameters using Optuna.

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
        cv_folds: Optional CV folds for robust optimization

    Returns:
        Tuple of (best_hyperparameters, best_score)
    """
    logger.info(f"\n{'█'*80}")
    logger.info(f"N-HiTS BAYESIAN OPTIMIZATION: {target}")
    logger.info(f"{'█'*80}")
    logger.info(f"Trials: {n_trials}")
    logger.info(f"Horizon: {horizon}h")
    logger.info(f"Input size: {input_size}h")

    trainer = NHiTSTrainer(target=target, horizon=horizon, input_size=input_size)

    def objective(trial: optuna.Trial) -> float:
        """Objective function for Optuna."""
        # Suggest hyperparameters
        hyperparams = trainer.suggest_hyperparameters(trial)

        try:
            if cv_folds is not None:
                # Cross-validation
                fold_scores = []
                for fold_idx, (train_idx, val_idx) in enumerate(cv_folds):
                    X_train_fold = X_train.iloc[train_idx]
                    y_train_fold = y_train.iloc[train_idx]
                    X_val_fold = X_train.iloc[val_idx]  # Val is within train set for CV
                    y_val_fold = y_train.iloc[val_idx]

                    _, metrics = trainer.train(
                        X_train_fold, y_train_fold,
                        X_val_fold, y_val_fold,
                        hyperparams
                    )

                    fold_scores.append(metrics['sMAPE'])

                    # Report intermediate value for pruning
                    trial.report(metrics['sMAPE'], fold_idx)

                    # Prune if not promising
                    if trial.should_prune():
                        raise optuna.TrialPruned()

                # Return mean score across folds
                score = np.mean(fold_scores)

            else:
                # Single train/val split
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

    # Best results
    best_params = study.best_params
    best_score = study.best_value

    # Convert back to proper format
    best_params['n_pool_kernel_size'] = eval(best_params['n_pool_kernel_size'])
    best_params['n_freq_downsample'] = eval(best_params['n_freq_downsample'])
    best_params['stack_types'] = ['identity', 'trend', 'seasonality']

    logger.info(f"\n{'='*80}")
    logger.info("OPTIMIZATION COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Best sMAPE: {best_score:.4f}%")
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"{'='*80}\n")

    return best_params, best_score
