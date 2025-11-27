"""
High-Performance Grid Search Runner V2
======================================
Runs massive combinatorial grid search on RTX 5090 (32GB VRAM).

Features:
- Utilizes GPU with accelerator='gpu' and precision='16-mixed'
- Saves results after every training run (append mode)
- Skip logic for resuming via config hash
- Memory management with garbage collection
- Comprehensive logging: file logs, JSON metrics, MLflow, CSV results

Author: ForeWatt Team - New Experiment V2
Date: November 2025
"""

import sys
import os
import gc
import time
import json
import hashlib
import argparse
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import logging
import warnings

import numpy as np
import pandas as pd

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# MLflow integration
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Configure base logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import feature preparer
from src.models.new_experiment.deeplearning.feature_preparer_v2 import (
    FundamentalFeaturePreparerV2,
    load_master_v2
)

# Import grid generator
from src.models.new_experiment.deeplearning.grid_config_generator_v2 import (
    GridConfigGeneratorV2,
    get_full_grid,
    TARGETS,
    TARGET_STRATEGIES
)

# Import trainers from local models module
try:
    from src.models.new_experiment.deeplearning.models.patchtst_trainer import PatchTSTTrainer
    from src.models.new_experiment.deeplearning.models.nhits_trainer import NHiTSTrainer
    from src.models.new_experiment.deeplearning.models.tft_trainer import TFTTrainer
except ImportError as e:
    logger.error(f"Failed to import trainers: {e}")
    raise

# Import metrics
try:
    from src.models.evaluate import (
        mean_absolute_error,
        symmetric_mean_absolute_percentage_error,
        mean_absolute_scaled_error
    )
except ImportError:
    logger.warning("Custom metrics not found, will use basic implementations")

    def mean_absolute_error(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    def symmetric_mean_absolute_percentage_error(y_true, y_pred):
        return 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

    def mean_absolute_scaled_error(y_true, y_pred, y_train, seasonality=24):
        naive_errors = np.abs(np.diff(y_train[::seasonality]))
        mae_naive = np.mean(naive_errors) if len(naive_errors) > 0 else 1.0
        return np.mean(np.abs(y_true - y_pred)) / max(mae_naive, 1e-8)


class FundamentalGridSearchRunnerV2:
    """
    High-performance grid search runner for fundamental price forecasting.

    Optimized for RTX 5090 (32GB VRAM):
    - Large batch sizes
    - Mixed precision training
    - Memory management
    - Checkpoint/resume capability
    - Comprehensive logging: file logs, JSON metrics, MLflow, CSV results
    """

    def __init__(
        self,
        output_dir: Path = None,
        device: str = None,
        use_mixed_precision: bool = True,
        val_size: float = 0.2,
        test_size: float = 0.2,
        experiment_name: str = "deeplearning_grid_search_v2"
    ):
        """
        Initialize runner.

        Args:
            output_dir: Directory for results
            device: Device to use ('cuda', 'mps', 'cpu')
            use_mixed_precision: Use fp16 mixed precision
            val_size: Validation set fraction
            test_size: Test set fraction
            experiment_name: MLflow experiment name
        """
        self.output_dir = output_dir or PROJECT_ROOT / 'reports' / 'new_experiment' / 'deeplearning'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for organized logging and model storage
        self.logs_dir = self.output_dir / 'logs'
        self.metrics_dir = self.output_dir / 'metrics'
        self.mlruns_dir = self.output_dir / 'mlruns'
        self.models_dir = self.output_dir / 'models'

        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.mlruns_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.results_file = self.output_dir / 'results.csv'
        self.device = device
        self.use_mixed_precision = use_mixed_precision
        self.val_size = val_size
        self.test_size = test_size
        self.experiment_name = experiment_name

        # Track completed configs
        self.completed_hashes = self._load_completed_hashes()

        # Setup MLflow
        self._setup_mlflow()

        # Setup file logging for this run session
        self.run_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._setup_file_logging()

        # Track saved models
        self.saved_model_hashes = self._scan_saved_models()

        logger.info(f"Runner initialized")
        logger.info(f"  Output dir: {self.output_dir}")
        logger.info(f"  Logs dir: {self.logs_dir}")
        logger.info(f"  Metrics dir: {self.metrics_dir}")
        logger.info(f"  Models dir: {self.models_dir}")
        logger.info(f"  MLflow dir: {self.mlruns_dir}")
        logger.info(f"  Results file: {self.results_file}")
        logger.info(f"  Completed configs: {len(self.completed_hashes)}")
        logger.info(f"  Saved models: {len(self.saved_model_hashes)}")
        logger.info(f"  MLflow available: {MLFLOW_AVAILABLE}")

    def _setup_mlflow(self):
        """Setup MLflow tracking."""
        if not MLFLOW_AVAILABLE:
            logger.warning("MLflow not available. Install with: pip install mlflow")
            return

        # Set tracking URI to local directory with new database
        tracking_uri = f"sqlite:///{self.mlruns_dir / 'mlflow_v2.db'}"
        mlflow.set_tracking_uri(tracking_uri)

        # Create or get experiment
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                self.mlflow_experiment_id = mlflow.create_experiment(
                    self.experiment_name,
                    artifact_location=str(self.mlruns_dir / 'artifacts')
                )
            else:
                self.mlflow_experiment_id = experiment.experiment_id
            mlflow.set_experiment(self.experiment_name)
            logger.info(f"MLflow experiment: {self.experiment_name} (ID: {self.mlflow_experiment_id})")
        except Exception as e:
            logger.warning(f"Failed to setup MLflow experiment: {e}")
            self.mlflow_experiment_id = None

    def _setup_file_logging(self):
        """Setup file handler for logging."""
        log_file = self.logs_dir / f"run_{self.run_session_id}.log"

        # Create file handler
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))

        # Add handler to logger
        logger.addHandler(file_handler)
        logger.info(f"File logging initialized: {log_file}")

    def _load_completed_hashes(self) -> set:
        """Load hashes of completed configurations."""
        if not self.results_file.exists():
            return set()

        try:
            df = pd.read_csv(self.results_file)
            if 'config_hash' in df.columns:
                return set(df['config_hash'].unique())
        except Exception as e:
            logger.warning(f"Failed to load completed hashes: {e}")

        return set()

    def _save_result(self, result: Dict[str, Any]):
        """
        Save single result to CSV, JSON, and MLflow.

        Args:
            result: Result dictionary
        """
        config_hash = result.get('config_hash', 'unknown')

        # 1. Save to CSV (append mode)
        df = pd.DataFrame([result])
        if self.results_file.exists():
            df.to_csv(self.results_file, mode='a', header=False, index=False)
        else:
            df.to_csv(self.results_file, index=False)

        # 2. Save JSON metrics file
        json_file = self.metrics_dir / f"{config_hash}.json"
        metrics_data = {
            'config_hash': config_hash,
            'timestamp': result.get('timestamp'),
            'target': result.get('target'),
            'model_type': result.get('model_type'),
            'feature_tier': result.get('feature_tier'),
            'feature_strategy': result.get('feature_strategy'),
            'status': result.get('status'),
            'validation_metrics': {
                'MAE': result.get('val_mae'),
                'sMAPE': result.get('val_smape'),
                'MASE': result.get('val_mase'),
            },
            'test_metrics': {
                'MAE': result.get('test_mae'),
                'sMAPE': result.get('test_smape'),
                'MASE': result.get('test_mase'),
            },
            'training_time_seconds': result.get('training_time_seconds'),
            'config': json.loads(result.get('config_json', '{}')),
            'error': result.get('error'),
        }
        with open(json_file, 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)

        # 3. Log to MLflow
        self._log_to_mlflow(result)

        logger.info(f"Result saved: CSV, JSON ({json_file.name}), MLflow")

    def _log_to_mlflow(self, result: Dict[str, Any]):
        """Log result to MLflow."""
        if not MLFLOW_AVAILABLE or self.mlflow_experiment_id is None:
            return

        try:
            config = json.loads(result.get('config_json', '{}'))
            run_name = f"{result.get('model_type', 'unknown')}_{result.get('config_hash', 'unknown')[:8]}"

            with mlflow.start_run(run_name=run_name):
                # Log parameters
                mlflow.log_param("target", result.get('target'))
                mlflow.log_param("model_type", result.get('model_type'))
                mlflow.log_param("feature_tier", result.get('feature_tier'))
                mlflow.log_param("feature_strategy", result.get('feature_strategy'))
                mlflow.log_param("config_hash", result.get('config_hash'))
                mlflow.log_param("status", result.get('status'))

                # Log model-specific hyperparameters
                for key, value in config.items():
                    if key not in ['config_hash', 'model_type', 'feature_tier', 'feature_strategy', 'target']:
                        try:
                            mlflow.log_param(key, value)
                        except Exception:
                            pass  # Skip non-loggable params

                # Log metrics
                if result.get('status') == 'success':
                    if result.get('val_mae') is not None:
                        mlflow.log_metric("val_mae", result.get('val_mae'))
                    if result.get('val_smape') is not None:
                        mlflow.log_metric("val_smape", result.get('val_smape'))
                    if result.get('val_mase') is not None:
                        mlflow.log_metric("val_mase", result.get('val_mase'))
                    if result.get('test_mae') is not None:
                        mlflow.log_metric("test_mae", result.get('test_mae'))
                    if result.get('test_smape') is not None:
                        mlflow.log_metric("test_smape", result.get('test_smape'))
                    if result.get('test_mase') is not None:
                        mlflow.log_metric("test_mase", result.get('test_mase'))

                if result.get('training_time_seconds') is not None:
                    mlflow.log_metric("training_time_seconds", result.get('training_time_seconds'))

                # Add tags
                mlflow.set_tag("run_session", self.run_session_id)

        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")

    def _save_model(
        self,
        model: Any,
        trainer: Any,
        config_hash: str,
        model_type: str,
        target: str
    ) -> Optional[Path]:
        """
        Save trained model to file.

        Args:
            model: Trained model (NeuralForecast model)
            trainer: Trainer instance
            config_hash: Configuration hash
            model_type: Model type (patchtst, nhits, tft)
            target: Target variable

        Returns:
            Path to saved model or None if failed
        """
        try:
            # Create model filename
            model_filename = f"{model_type}_{target}_{config_hash}"
            model_dir = self.models_dir / model_filename
            model_dir.mkdir(parents=True, exist_ok=True)

            # Save the NeuralForecast model
            # NeuralForecast models have a save method
            if hasattr(model, 'save'):
                model_path = model_dir / 'model'
                model.save(str(model_path))
                logger.info(f"Model saved to: {model_path}")

                # Also save model metadata
                metadata = {
                    'config_hash': config_hash,
                    'model_type': model_type,
                    'target': target,
                    'timestamp': datetime.now().isoformat(),
                    'model_class': str(type(model)),
                }
                metadata_path = model_dir / 'metadata.json'
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)

                return model_dir
            else:
                # Fallback: try to save using pickle
                import pickle
                model_path = model_dir / 'model.pkl'
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                logger.info(f"Model saved (pickle) to: {model_path}")
                return model_dir

        except Exception as e:
            logger.warning(f"Failed to save model: {e}")
            return None

    def _scan_saved_models(self) -> Dict[str, Path]:
        """
        Scan models directory for saved models.

        Returns:
            Dictionary mapping config_hash to model directory path
        """
        saved_models = {}
        if not self.models_dir.exists():
            return saved_models

        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir():
                metadata_path = model_dir / 'metadata.json'
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        config_hash = metadata.get('config_hash')
                        if config_hash:
                            saved_models[config_hash] = model_dir
                    except Exception:
                        pass

        return saved_models

    def _load_model(
        self,
        config_hash: str,
        model_type: str,
        config: Dict[str, Any],
        target: str
    ) -> Tuple[Optional[Any], Optional[Any]]:
        """
        Load a previously trained deep learning model.

        Args:
            config_hash: Configuration hash
            model_type: Model type (patchtst, nhits, tft)
            config: Configuration dictionary
            target: Target variable

        Returns:
            Tuple of (model, trainer) or (None, None) if not found
        """
        if config_hash not in self.saved_model_hashes:
            return None, None

        model_dir = self.saved_model_hashes[config_hash]
        logger.info(f"Loading saved model from: {model_dir}")

        try:
            from neuralforecast import NeuralForecast

            # Create trainer
            trainer = self._create_trainer(model_type, config, target)

            # Try to load NeuralForecast model
            model_path = model_dir / 'model'
            if model_path.exists():
                nf = NeuralForecast.load(str(model_path))
                trainer.model = nf
                logger.info(f"Model loaded successfully: {model_type}")
                return nf, trainer
            else:
                # Try pickle fallback
                import pickle
                pkl_path = model_dir / 'model.pkl'
                if pkl_path.exists():
                    with open(pkl_path, 'rb') as f:
                        nf = pickle.load(f)
                    trainer.model = nf
                    logger.info(f"Model loaded (pickle) successfully: {model_type}")
                    return nf, trainer

            return None, None

        except Exception as e:
            logger.warning(f"Failed to load model: {e}")
            return None, None

    def _prepare_data(
        self,
        target: str = 'price_real',
        strategy: str = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Load and prepare data for training.

        Args:
            target: Target variable ('price_real' or 'consumption')
            strategy: Feature strategy (auto-detected from TARGET_STRATEGIES if None)

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info(f"\nLoading master_v2_fundamental data for target: {target}")

        # Auto-select strategy if not provided
        if strategy is None:
            strategy = TARGET_STRATEGIES.get(target, 'fundamental_v2')
        logger.info(f"Using feature strategy: {strategy}")

        # Load master data
        df = load_master_v2()
        logger.info(f"Loaded data shape: {df.shape}")

        # Initialize feature preparer
        preparer = FundamentalFeaturePreparerV2(
            target=target,
            strategy=strategy
        )

        # Prepare features
        X, y, feature_names = preparer.prepare_features(df)

        # Drop NaN rows
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        logger.info(f"After dropping NaN: {len(X)} samples")

        # Temporal split
        n = len(X)
        train_end = int(n * (1 - self.val_size - self.test_size))
        val_end = int(n * (1 - self.test_size))

        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        X_val = X.iloc[train_end:val_end]
        y_val = y.iloc[train_end:val_end]
        X_test = X.iloc[val_end:]
        y_test = y.iloc[val_end:]

        logger.info(f"\nData splits:")
        logger.info(f"  Train: {len(X_train)} ({len(X_train)/n*100:.1f}%)")
        logger.info(f"  Val:   {len(X_val)} ({len(X_val)/n*100:.1f}%)")
        logger.info(f"  Test:  {len(X_test)} ({len(X_test)/n*100:.1f}%)")
        logger.info(f"  Features: {len(feature_names)}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _create_trainer(
        self,
        model_type: str,
        config: Dict[str, Any],
        target: str = 'price_real'
    ):
        """
        Create trainer for model type.

        Args:
            model_type: 'patchtst', 'nhits', 'tft'
            config: Configuration dictionary
            target: Target variable

        Returns:
            Initialized trainer
        """
        input_size = config.get('input_size', 168)
        horizon = config.get('horizon', 24)

        if model_type == 'patchtst':
            return PatchTSTTrainer(
                target=target,
                horizon=horizon,
                input_size=input_size,
                random_seed=config.get('random_seed', 42),
                device=self.device
            )
        elif model_type == 'nhits':
            return NHiTSTrainer(
                target=target,
                horizon=horizon,
                input_size=input_size,
                random_seed=config.get('random_seed', 42),
                device=self.device
            )
        elif model_type == 'tft':
            return TFTTrainer(
                target=target,
                horizon=horizon,
                input_size=input_size,
                random_seed=config.get('random_seed', 42),
                device=self.device
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _config_to_hyperparams(
        self,
        model_type: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Convert grid config to trainer hyperparameters.

        Args:
            model_type: Model type
            config: Grid configuration

        Returns:
            Hyperparameters dictionary for trainer
        """
        if model_type == 'patchtst':
            return {
                'patch_len': config['patch_len'],
                'stride': config['stride'],
                'encoder_layers': config['n_layers'],  # Map n_layers -> encoder_layers
                'hidden_size': config['d_model'],      # Map d_model -> hidden_size
                'n_heads': config['n_heads'],
                'dropout': config['dropout'],
                'learning_rate': config['learning_rate'],
                'batch_size': config['batch_size'],
                'max_steps': config['max_steps'],
                'early_stop_patience_steps': config.get('early_stop_patience_steps', 100),
            }
        elif model_type == 'nhits':
            return {
                'n_blocks': config['n_blocks'],
                'hidden_size': config['hidden_size'],
                'n_mlp_layers': config['n_mlp_layers'],
                'n_pool_kernel_size': config['n_pool_kernel_size'],
                'n_freq_downsample': config['n_freq_downsample'],
                'learning_rate': config['learning_rate'],
                'batch_size': config['batch_size'],
                'max_steps': config['max_steps'],
                'early_stop_patience_steps': config.get('early_stop_patience_steps', 100),
            }
        elif model_type == 'tft':
            return {
                'hidden_size': config['hidden_size'],
                'n_head': config['n_head'],
                'dropout': config['dropout'],
                'lstm_n_layers': config['lstm_n_layers'],
                'learning_rate': config['learning_rate'],
                'batch_size': config['batch_size'],
                'max_steps': config['max_steps'],
                'early_stop_patience_steps': config.get('early_stop_patience_steps', 100),
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _run_single_config(
        self,
        config: Dict[str, Any],
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        y_test: pd.Series,
        target: str = 'price_real',
        load_if_trained: bool = False
    ) -> Dict[str, Any]:
        """
        Run training for a single configuration.

        Args:
            config: Configuration dictionary
            X_train, X_val, X_test: Feature dataframes
            y_train, y_val, y_test: Target series
            target: Target variable
            load_if_trained: If True, load saved model instead of retraining

        Returns:
            Result dictionary
        """
        model_type = config['model_type']
        config_hash = config['config_hash']

        # Check if model exists and should be loaded
        model_loaded = False
        if load_if_trained and config_hash in self.saved_model_hashes:
            logger.info(f"\n{'='*80}")
            logger.info(f"LOADING SAVED MODEL: {model_type.upper()} | Target: {target} | Hash: {config_hash}")
            logger.info(f"{'='*80}")
            model_loaded = True
        else:
            logger.info(f"\n{'='*80}")
            logger.info(f"TRAINING: {model_type.upper()} | Target: {target} | Hash: {config_hash}")
            logger.info(f"{'='*80}")
            

        result = {
            'timestamp': datetime.now().isoformat(),
            'target': target,
            'model_type': model_type,
            'feature_tier': config.get('feature_tier', 'core'),
            'feature_strategy': config.get('feature_strategy', 'fundamental_v2'),
            'config_hash': config_hash,
            'config_json': json.dumps(config, default=str),
        }

        start_time = time.time()

        try:
            # Either load saved model or train new one
            if model_loaded:
                model, trainer = self._load_model(config_hash, model_type, config, target)
                if model is None:
                    logger.warning(f"Failed to load model, falling back to training")
                    model_loaded = False

            if not model_loaded:
                # Create trainer
                trainer = self._create_trainer(model_type, config, target)

                # Convert config to hyperparameters
                hyperparams = self._config_to_hyperparams(model_type, config)

                logger.info(f"Hyperparameters: {hyperparams}")

                # Train model
                model, val_metrics = trainer.train(
                    X_train, y_train,
                    X_val, y_val,
                    hyperparams=hyperparams
                )

                # Record validation metrics
                result['val_mae'] = val_metrics.get('MAE', np.nan)
                result['val_smape'] = val_metrics.get('sMAPE', np.nan)
                result['val_mase'] = val_metrics.get('MASE', np.nan)
            else:
                # For loaded models, we don't have validation metrics from training
                result['val_mae'] = np.nan
                result['val_smape'] = np.nan
                result['val_mase'] = np.nan
                result['loaded_from_cache'] = True

            # Evaluate on test set
            test_predictions = trainer.predict(X_test, y_test)

            # Handle prediction array shape
            if len(test_predictions.shape) > 1:
                test_predictions = test_predictions.flatten()

            # Align lengths
            min_len = min(len(test_predictions), len(y_test))
            test_predictions = test_predictions[:min_len]
            y_test_aligned = y_test.values[:min_len]

            # Calculate test metrics
            result['test_mae'] = mean_absolute_error(y_test_aligned, test_predictions)
            result['test_smape'] = symmetric_mean_absolute_percentage_error(
                y_test_aligned, test_predictions
            )
            result['test_mase'] = mean_absolute_scaled_error(
                y_test_aligned, test_predictions,
                y_train.values, seasonality=24
            )

            result['status'] = 'success'

            logger.info(f"\nTest metrics:")
            logger.info(f"  MAE:   {result['test_mae']:.2f}")
            logger.info(f"  sMAPE: {result['test_smape']:.2f}%")
            logger.info(f"  MASE:  {result['test_mase']:.4f}")

            # Save model only if we trained it (not if loaded from cache)
            if not model_loaded:
                model_path = self._save_model(model, trainer, config_hash, model_type, target)
                result['model_path'] = str(model_path) if model_path else None
                # Update saved model cache
                if model_path:
                    self.saved_model_hashes[config_hash] = model_path
            else:
                result['model_path'] = str(self.saved_model_hashes.get(config_hash, ''))

        except Exception as e:
            result['status'] = 'failed'
            result['error'] = str(e)
            result['traceback'] = traceback.format_exc()
            logger.error(f"Training failed: {e}")
            logger.error(traceback.format_exc())

        # Record timing
        result['training_time_seconds'] = time.time() - start_time

        # Clean up memory
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        return result

    def run_grid_search(
        self,
        model_types: List[str] = None,
        targets: List[str] = None,
        max_configs: int = None,
        skip_completed: bool = True,
        load_if_trained: bool = False
    ):
        """
        Run full grid search for multiple targets.

        Args:
            model_types: List of model types to run (default: all)
            targets: List of targets to run (default: all from TARGETS)
            max_configs: Maximum configs to run per target (default: all)
            skip_completed: Skip configs that have already been run
            load_if_trained: Load saved models instead of retraining
        """
        logger.info("\n" + "="*80)
        logger.info("FUNDAMENTAL V2 GRID SEARCH - RTX 5090 OPTIMIZED")
        logger.info("D-1 Safe Features for Day-Ahead Forecasting")
        logger.info("="*80)

        # Generate grid
        generator = GridConfigGeneratorV2()
        summary = generator.get_grid_summary()

        logger.info(f"\nGrid summary:")
        logger.info(f"  PatchTST: {summary['patchtst']['count']} configurations per target")
        logger.info(f"  N-HiTS:   {summary['nhits']['count']} configurations per target")
        logger.info(f"  TFT:      {summary['tft']['count']} configurations per target")
        logger.info(f"  Total per target: {summary['total_per_target']} configurations")
        logger.info(f"  Total all targets: {summary['total_all_targets']} configurations")

        # Filter by model types and targets
        if model_types is None:
            model_types = ['patchtst', 'nhits', 'tft']
        if targets is None:
            targets = TARGETS

        logger.info(f"\nRunning for targets: {targets}")
        logger.info(f"Running for models: {model_types}")

        # Run grid search for each target
        total_start = time.time()
        total_successful = 0
        total_failed = 0

        for target in targets:
            logger.info(f"\n{'#'*80}")
            logger.info(f"# TARGET: {target.upper()}")
            logger.info(f"# Feature strategy: {TARGET_STRATEGIES.get(target, 'N/A')}")
            logger.info(f"{'#'*80}")

            # Get configs for this target
            all_configs = []
            for model_type in model_types:
                configs = get_full_grid(model_type, target)
                all_configs.extend(configs)
                logger.info(f"  {model_type}: {len(configs)} configs")

            # Apply max_configs limit
            if max_configs is not None:
                all_configs = all_configs[:max_configs]
                logger.info(f"\nLimited to first {max_configs} configs for {target}")

            # Skip completed if requested
            if skip_completed:
                pending_configs = [
                    c for c in all_configs
                    if c['config_hash'] not in self.completed_hashes
                ]
                skipped = len(all_configs) - len(pending_configs)
                if skipped > 0:
                    logger.info(f"\nSkipping {skipped} already-completed configurations")
                all_configs = pending_configs

            logger.info(f"\nConfigurations to run for {target}: {len(all_configs)}")

            if len(all_configs) == 0:
                logger.info(f"No configurations to run for {target}. Skipping.")
                continue

            # Run grid search
            successful = 0
            failed = 0

            # Cache data by feature_strategy to avoid reloading
            data_cache = {}

            for i, config in enumerate(all_configs):
                # Load data for this config's feature strategy (cached)
                strategy = config.get('feature_strategy', TARGET_STRATEGIES.get(target))
                if strategy not in data_cache:
                    logger.info(f"Loading data for strategy: {strategy}")
                    data_cache[strategy] = self._prepare_data(target=target, strategy=strategy)
                X_train, X_val, X_test, y_train, y_val, y_test = data_cache[strategy]
                logger.info(f"\n{'='*80}")
                logger.info(f"[{target}] Configuration {i+1}/{len(all_configs)}")
                logger.info(f"Feature tier: {config.get('feature_tier', 'core')} ({X_train.shape[1]} features)")
                logger.info(f"Progress: {i/len(all_configs)*100:.1f}%")
                logger.info(f"{'='*80}")

                # Run single config
                result = self._run_single_config(
                    config,
                    X_train, X_val, X_test,
                    y_train, y_val, y_test,
                    target=target,
                    load_if_trained=load_if_trained
                )

                # Save result immediately
                self._save_result(result)

                # Update tracking
                self.completed_hashes.add(config['config_hash'])

                if result['status'] == 'success':
                    successful += 1
                else:
                    failed += 1

                # Progress summary
                elapsed = time.time() - total_start
                avg_time = elapsed / (total_successful + total_failed + successful + failed)
                remaining_this_target = avg_time * (len(all_configs) - i - 1)

                logger.info(f"\n[{target}] Progress: {successful} successful, {failed} failed")
                logger.info(f"Estimated time remaining for {target}: {remaining_this_target/60:.1f} min")

            total_successful += successful
            total_failed += failed

            logger.info(f"\n{'='*80}")
            logger.info(f"[{target}] COMPLETE: {successful} successful, {failed} failed")
            logger.info(f"{'='*80}")

        # Final summary
        total_time = time.time() - total_start
        logger.info("\n" + "="*80)
        logger.info("GRID SEARCH COMPLETE - ALL TARGETS")
        logger.info("="*80)
        logger.info(f"Targets: {targets}")
        logger.info(f"Total successful: {total_successful}")
        logger.info(f"Total failed: {total_failed}")
        logger.info(f"Total time: {total_time/3600:.2f} hours")
        logger.info(f"Results saved to: {self.results_file}")

        # Generate summary report
        self._generate_summary_report(total_successful, total_failed, total_time, targets)

    def _generate_summary_report(
        self,
        total_successful: int,
        total_failed: int,
        total_time: float,
        targets: List[str]
    ):
        """Generate comprehensive summary report."""
        summary_file = self.output_dir / f"summary_{self.run_session_id}.json"

        summary = {
            'run_session_id': self.run_session_id,
            'timestamp': datetime.now().isoformat(),
            'total_successful': total_successful,
            'total_failed': total_failed,
            'total_time_seconds': total_time,
            'total_time_hours': total_time / 3600,
            'targets': targets,
            'results_file': str(self.results_file),
            'logs_dir': str(self.logs_dir),
            'metrics_dir': str(self.metrics_dir),
            'mlruns_dir': str(self.mlruns_dir),
        }

        # Add best results per target/model if results exist
        if self.results_file.exists():
            try:
                df = pd.read_csv(self.results_file)
                df_success = df[df['status'] == 'success']

                if len(df_success) > 0:
                    summary['best_results'] = {}

                    for target in df_success['target'].unique():
                        df_target = df_success[df_success['target'] == target]
                        target_best = {}

                        # Overall best for this target
                        overall_best_idx = df_target['test_smape'].idxmin()
                        overall_best = df_target.loc[overall_best_idx]
                        target_best['overall'] = {
                            'model_type': overall_best['model_type'],
                            'config_hash': overall_best['config_hash'],
                            'test_smape': float(overall_best['test_smape']),
                            'test_mae': float(overall_best['test_mae']),
                            'test_mase': float(overall_best['test_mase']),
                        }

                        # Best per model type
                        for model_type in df_target['model_type'].unique():
                            df_model = df_target[df_target['model_type'] == model_type]
                            best_idx = df_model['test_smape'].idxmin()
                            best = df_model.loc[best_idx]
                            target_best[model_type] = {
                                'config_hash': best['config_hash'],
                                'test_smape': float(best['test_smape']),
                                'test_mae': float(best['test_mae']),
                                'test_mase': float(best['test_mase']),
                            }

                        summary['best_results'][target] = target_best
            except Exception as e:
                logger.warning(f"Failed to add best results to summary: {e}")

        # Save summary
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"\nSummary report saved to: {summary_file}")

    def analyze_results(self, target: str = None) -> pd.DataFrame:
        """
        Analyze grid search results.

        Args:
            target: Specific target to analyze (None for all)

        Returns:
            DataFrame with results and analysis
        """
        if not self.results_file.exists():
            logger.error(f"No results file found: {self.results_file}")
            return None

        df = pd.read_csv(self.results_file)

        # Filter successful runs
        df_success = df[df['status'] == 'success'].copy()

        if len(df_success) == 0:
            logger.warning("No successful runs found")
            return df

        # Filter by target if specified
        if target:
            df_success = df_success[df_success['target'] == target]
            if len(df_success) == 0:
                logger.warning(f"No successful runs found for target: {target}")
                return df

        logger.info("\n" + "="*80)
        logger.info("GRID SEARCH RESULTS ANALYSIS")
        logger.info("="*80)

        # Analyze by target
        targets_in_results = df_success['target'].unique() if 'target' in df_success.columns else ['price_real']

        for tgt in targets_in_results:
            df_target = df_success[df_success['target'] == tgt] if 'target' in df_success.columns else df_success

            logger.info(f"\n{'#'*80}")
            logger.info(f"# TARGET: {tgt.upper()}")
            logger.info(f"{'#'*80}")

            # Best by model type for this target
            for model_type in df_target['model_type'].unique():
                df_model = df_target[df_target['model_type'] == model_type]
                best_idx = df_model['test_smape'].idxmin()
                best = df_model.loc[best_idx]

                logger.info(f"\n{model_type.upper()} - Best configuration:")
                logger.info(f"  Config hash: {best['config_hash']}")
                logger.info(f"  Test sMAPE: {best['test_smape']:.4f}%")
                logger.info(f"  Test MAE: {best['test_mae']:.2f}")
                logger.info(f"  Test MASE: {best['test_mase']:.4f}")
                logger.info(f"  Val sMAPE: {best['val_smape']:.4f}%")

            # Overall best for this target
            overall_best_idx = df_target['test_smape'].idxmin()
            overall_best = df_target.loc[overall_best_idx]

            logger.info(f"\n{'='*80}")
            logger.info(f"OVERALL BEST for {tgt.upper()}")
            logger.info(f"{'='*80}")
            logger.info(f"Model: {overall_best['model_type']}")
            logger.info(f"Config hash: {overall_best['config_hash']}")
            logger.info(f"Test sMAPE: {overall_best['test_smape']:.4f}%")
            logger.info(f"Test MAE: {overall_best['test_mae']:.2f}")
            logger.info(f"Test MASE: {overall_best['test_mase']:.4f}")

            # Target-specific benchmarks
            if tgt == 'price_real':
                benchmark = 10.87
                if overall_best['test_smape'] < benchmark:
                    logger.info(f"\n*** SUCCESS: Beat {benchmark}% sMAPE floor! ***")
                    improvement = benchmark - overall_best['test_smape']
                    logger.info(f"Improvement: {improvement:.4f}%")
                else:
                    gap = overall_best['test_smape'] - benchmark
                    logger.info(f"\nGap to {benchmark}% floor: {gap:.4f}%")

        return df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run Fundamental V2 Grid Search (D-1 Safe Features)'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        choices=['patchtst', 'nhits', 'tft'],
        default=None,
        help='Model types to run (default: all)'
    )
    parser.add_argument(
        '--targets',
        nargs='+',
        choices=['price_real', 'consumption'],
        default=None,
        help='Targets to run (default: both price_real and consumption)'
    )
    parser.add_argument(
        '--max-configs',
        type=int,
        default=None,
        help='Maximum configurations to run per target'
    )
    parser.add_argument(
        '--no-skip',
        action='store_true',
        help='Do not skip completed configurations'
    )
    parser.add_argument(
        '--load-trained',
        action='store_true',
        help='Load saved models instead of retraining (skip training if model exists)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'mps', 'cpu'],
        help='Device to use'
    )
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Analyze existing results instead of running'
    )
    parser.add_argument(
        '--analyze-target',
        type=str,
        choices=['price_real', 'consumption'],
        default=None,
        help='Analyze results for specific target only'
    )

    args = parser.parse_args()

    # Initialize runner
    runner = FundamentalGridSearchRunnerV2(device=args.device)

    if args.analyze:
        # Analyze existing results
        runner.analyze_results(target=args.analyze_target)
    else:
        # Run grid search
        runner.run_grid_search(
            model_types=args.models,
            targets=args.targets,
            max_configs=args.max_configs,
            skip_completed=not args.no_skip,
            load_if_trained=args.load_trained
        )


if __name__ == "__main__":
    main()
