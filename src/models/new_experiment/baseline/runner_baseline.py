"""
Baseline Grid Search Runner
===========================
Runs grid search for CatBoost, XGBoost, LightGBM, and Prophet.

Features:
- Saves results after every training run (append mode)
- Skip logic for resuming via config hash
- Feature importance logging
- Comprehensive logging: file logs, JSON metrics, MLflow, CSV results

Author: ForeWatt Team - New Experiment V2
Date: November 2025
"""

import sys
import os
import gc
import time
import json
import argparse
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import logging
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# MLflow integration
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import feature preparer
from src.models.new_experiment.baseline.feature_preparer_baseline import (
    BaselineFeaturePreparer,
    BASELINE_FEATURE_STRATEGIES,
    get_feature_strategy_for_tier,
    load_master_v2
)

# Import grid generator
from src.models.new_experiment.baseline.grid_config_baseline import (
    get_baseline_grid,
    BaselineGridConfigGenerator,
    TARGETS,
    TARGET_STRATEGIES
)

# Import trainers
from src.models.new_experiment.baseline.models import (
    CatBoostTrainer,
    XGBoostTrainer,
    LightGBMTrainer,
    ProphetTrainer
)

# Import metrics
try:
    from src.models.evaluate import (
        mean_absolute_error,
        symmetric_mean_absolute_percentage_error,
        mean_absolute_scaled_error
    )
except ImportError:
    logger.warning("Custom metrics not found, using basic implementations")

    def mean_absolute_error(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    def symmetric_mean_absolute_percentage_error(y_true, y_pred):
        return 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

    def mean_absolute_scaled_error(y_true, y_pred, y_train, seasonality=24):
        naive_errors = np.abs(np.diff(y_train[::seasonality]))
        mae_naive = np.mean(naive_errors) if len(naive_errors) > 0 else 1.0
        return np.mean(np.abs(y_true - y_pred)) / max(mae_naive, 1e-8)


class BaselineGridSearchRunner:
    """
    Grid search runner for baseline models.

    Features:
    - CatBoost, XGBoost, LightGBM, Prophet support
    - Multiple feature strategies
    - Resume capability via config hash
    - Results saved after each run
    - Comprehensive logging: file logs, JSON metrics, MLflow, CSV results
    """

    def __init__(
        self,
        output_dir: Path = None,
        val_size: float = 0.2,
        test_size: float = 0.2,
        experiment_name: str = "baseline_grid_search"
    ):
        """
        Initialize runner.

        Args:
            output_dir: Directory for results
            val_size: Validation set fraction
            test_size: Test set fraction
            experiment_name: MLflow experiment name
        """
        self.output_dir = output_dir or PROJECT_ROOT / 'reports' / 'new_experiment' / 'baseline'
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

        logger.info(f"Baseline Runner initialized")
        logger.info(f"  Output dir: {self.output_dir}")
        logger.info(f"  Logs dir: {self.logs_dir}")
        logger.info(f"  Metrics dir: {self.metrics_dir}")
        logger.info(f"  Models dir: {self.models_dir}")
        logger.info(f"  MLflow dir: {self.mlruns_dir}")
        logger.info(f"  Results file: {self.results_file}")
        logger.info(f"  Completed configs: {len(self.completed_hashes)}")
        logger.info(f"  MLflow available: {MLFLOW_AVAILABLE}")

    def _setup_mlflow(self):
        """Setup MLflow tracking."""
        if not MLFLOW_AVAILABLE:
            logger.warning("MLflow not available. Install with: pip install mlflow")
            return

        # Set tracking URI to local directory
        tracking_uri = f"sqlite:///{self.mlruns_dir / 'mlflow.db'}"
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
        """Save single result to CSV, JSON, and MLflow."""
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
            'config_name': result.get('config_name'),
            'feature_tier': result.get('feature_tier'),
            'feature_strategy': result.get('feature_strategy'),
            'n_features': result.get('n_features'),
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
            'top_features': result.get('top_features'),
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
                mlflow.log_param("config_name", result.get('config_name'))
                mlflow.log_param("feature_tier", result.get('feature_tier'))
                mlflow.log_param("feature_strategy", result.get('feature_strategy'))
                mlflow.log_param("config_hash", result.get('config_hash'))
                mlflow.log_param("status", result.get('status'))
                mlflow.log_param("n_features", result.get('n_features'))

                # Log model-specific hyperparameters
                for key, value in config.items():
                    if key not in ['config_hash', 'model_type', 'feature_tier', 'feature_strategy', 'target', 'config_name']:
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
            model: Trained model
            trainer: Trainer instance
            config_hash: Configuration hash
            model_type: Model type (catboost, xgboost, lightgbm, prophet)
            target: Target variable

        Returns:
            Path to saved model or None if failed
        """
        try:
            import joblib

            # Create model filename
            model_filename = f"{model_type}_{target}_{config_hash}"
            model_dir = self.models_dir / model_filename
            model_dir.mkdir(parents=True, exist_ok=True)

            # Save model based on type
            if model_type == 'catboost':
                model_path = model_dir / 'model.cbm'
                model.save_model(str(model_path))
            elif model_type == 'xgboost':
                model_path = model_dir / 'model.json'
                model.save_model(str(model_path))
            elif model_type == 'lightgbm':
                model_path = model_dir / 'model.txt'
                model.booster_.save_model(str(model_path))
            elif model_type == 'prophet':
                # Prophet models are serialized with joblib
                model_path = model_dir / 'model.pkl'
                joblib.dump(model, model_path)
            else:
                # Fallback: use joblib
                model_path = model_dir / 'model.pkl'
                joblib.dump(model, model_path)

            logger.info(f"Model saved to: {model_path}")

            # Save model metadata
            metadata = {
                'config_hash': config_hash,
                'model_type': model_type,
                'target': target,
                'timestamp': datetime.now().isoformat(),
                'model_class': str(type(model)),
                'model_file': model_path.name,
            }
            metadata_path = model_dir / 'metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            # Save feature importance if available
            if hasattr(trainer, 'get_feature_importance'):
                try:
                    importance = trainer.get_feature_importance()
                    importance_path = model_dir / 'feature_importance.csv'
                    importance.to_csv(importance_path, index=False)
                except Exception:
                    pass

            return model_dir

        except Exception as e:
            logger.warning(f"Failed to save model: {e}")
            return None

    def _prepare_data(
        self,
        target: str = 'price_real',
        strategy: str = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Load and prepare data for training.

        Args:
            target: Target variable
            strategy: Feature strategy

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info(f"\nLoading data for target: {target}")

        if strategy is None:
            strategy = TARGET_STRATEGIES.get(target, 'baseline_core')
        logger.info(f"Using feature strategy: {strategy}")

        # Load master data
        df = load_master_v2()
        logger.info(f"Loaded data shape: {df.shape}")

        # Initialize feature preparer
        preparer = BaselineFeaturePreparer(
            target=target,
            strategy=strategy
        )

        # Prepare features with split
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = preparer.prepare_train_val_test(
            df,
            val_size=self.val_size,
            test_size=self.test_size
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _create_trainer(self, model_type: str, target: str = 'price_real'):
        """Create trainer for model type."""
        trainers = {
            'catboost': CatBoostTrainer,
            'xgboost': XGBoostTrainer,
            'lightgbm': LightGBMTrainer,
            'prophet': ProphetTrainer,
        }

        if model_type not in trainers:
            raise ValueError(f"Unknown model type: {model_type}")

        return trainers[model_type](target=target)

    def _run_single_config(
        self,
        config: Dict[str, Any],
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        y_test: pd.Series,
        target: str = 'price_real'
    ) -> Dict[str, Any]:
        """Run training for a single configuration."""
        model_type = config['model_type']
        config_hash = config['config_hash']

        logger.info(f"\n{'='*80}")
        logger.info(f"TRAINING: {model_type.upper()} | Target: {target} | Hash: {config_hash}")
        logger.info(f"{'='*80}")

        result = {
            'timestamp': datetime.now().isoformat(),
            'target': target,
            'model_type': model_type,
            'feature_tier': config.get('feature_tier', 'core'),
            'feature_strategy': config.get('feature_strategy', 'baseline_core'),
            'config_hash': config_hash,
            'config_name': config.get('config_name', ''),
            'config_json': json.dumps(config, default=str),
            'n_features': X_train.shape[1],
        }

        start_time = time.time()

        try:
            # Create trainer
            trainer = self._create_trainer(model_type, target)

            # Train model
            model, val_metrics = trainer.train(
                X_train, y_train,
                X_val, y_val,
                hyperparams=config
            )

            # Record validation metrics
            result['val_mae'] = val_metrics.get('MAE', np.nan)
            result['val_smape'] = val_metrics.get('sMAPE', np.nan)
            result['val_mase'] = val_metrics.get('MASE', np.nan)

            # Evaluate on test set
            test_predictions = trainer.predict(X_test)

            # Calculate test metrics
            result['test_mae'] = mean_absolute_error(y_test.values, test_predictions)
            result['test_smape'] = symmetric_mean_absolute_percentage_error(
                y_test.values, test_predictions
            )
            result['test_mase'] = mean_absolute_scaled_error(
                y_test.values, test_predictions,
                y_train.values, seasonality=24
            )

            result['status'] = 'success'

            logger.info(f"\nTest metrics:")
            logger.info(f"  MAE:   {result['test_mae']:.2f}")
            logger.info(f"  sMAPE: {result['test_smape']:.2f}%")
            logger.info(f"  MASE:  {result['test_mase']:.4f}")

            # Get feature importance if available
            if hasattr(trainer, 'get_feature_importance'):
                try:
                    importance = trainer.get_feature_importance()
                    top_features = importance.head(5)['feature'].tolist()
                    result['top_features'] = ','.join(top_features)
                except Exception:
                    result['top_features'] = ''

            # Save model
            model_path = self._save_model(model, trainer, config_hash, model_type, target)
            result['model_path'] = str(model_path) if model_path else None

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

        return result

    def run_grid_search(
        self,
        model_types: List[str] = None,
        targets: List[str] = None,
        max_configs: int = None,
        skip_completed: bool = True
    ):
        """
        Run full grid search.

        Args:
            model_types: List of model types (default: all)
            targets: List of targets (default: all)
            max_configs: Maximum configs to run per target
            skip_completed: Skip already-completed configs
        """
        logger.info("\n" + "="*80)
        logger.info("BASELINE GRID SEARCH")
        logger.info("CatBoost | XGBoost | LightGBM | Prophet")
        logger.info("="*80)

        # Generate grid
        generator = BaselineGridConfigGenerator()
        summary = generator.get_grid_summary()

        logger.info(f"\nGrid summary:")
        logger.info(f"  CatBoost:  {summary['catboost']['count']} configurations")
        logger.info(f"  XGBoost:   {summary['xgboost']['count']} configurations")
        logger.info(f"  LightGBM:  {summary['lightgbm']['count']} configurations")
        logger.info(f"  Prophet:   {summary['prophet']['count']} configurations")
        logger.info(f"  Total per target: {summary['total_per_target']} configurations")

        # Filter by model types and targets
        if model_types is None:
            model_types = ['catboost', 'xgboost', 'lightgbm', 'prophet']
        if targets is None:
            targets = TARGETS

        logger.info(f"\nRunning for targets: {targets}")
        logger.info(f"Running for models: {model_types}")

        # Run grid search
        total_start = time.time()
        total_successful = 0
        total_failed = 0

        for target in targets:
            logger.info(f"\n{'#'*80}")
            logger.info(f"# TARGET: {target.upper()}")
            logger.info(f"{'#'*80}")

            # Get configs for this target
            all_configs = []
            for model_type in model_types:
                configs = get_baseline_grid(model_type, target)
                all_configs.extend(configs)
                logger.info(f"  {model_type}: {len(configs)} configs")

            # Apply max_configs limit
            if max_configs is not None:
                all_configs = all_configs[:max_configs]
                logger.info(f"\nLimited to first {max_configs} configs")

            # Skip completed
            if skip_completed:
                pending_configs = [
                    c for c in all_configs
                    if c['config_hash'] not in self.completed_hashes
                ]
                skipped = len(all_configs) - len(pending_configs)
                if skipped > 0:
                    logger.info(f"\nSkipping {skipped} already-completed configs")
                all_configs = pending_configs

            logger.info(f"\nConfigs to run: {len(all_configs)}")

            if len(all_configs) == 0:
                logger.info(f"No configs to run for {target}. Skipping.")
                continue

            # Cache data by feature_strategy
            data_cache = {}
            successful = 0
            failed = 0

            for i, config in enumerate(all_configs):
                # Load data for this config's feature strategy
                strategy = config.get('feature_strategy', TARGET_STRATEGIES.get(target))
                if strategy not in data_cache:
                    logger.info(f"Loading data for strategy: {strategy}")
                    data_cache[strategy] = self._prepare_data(target=target, strategy=strategy)
                X_train, X_val, X_test, y_train, y_val, y_test = data_cache[strategy]

                logger.info(f"\n[{target}] Config {i+1}/{len(all_configs)}")
                logger.info(f"Feature tier: {config.get('feature_tier', 'core')} ({X_train.shape[1]} features)")

                # Run single config
                result = self._run_single_config(
                    config,
                    X_train, X_val, X_test,
                    y_train, y_val, y_test,
                    target=target
                )

                # Save result
                self._save_result(result)

                # Update tracking
                self.completed_hashes.add(config['config_hash'])

                if result['status'] == 'success':
                    successful += 1
                else:
                    failed += 1

                # Progress
                elapsed = time.time() - total_start
                avg_time = elapsed / (total_successful + total_failed + successful + failed)
                remaining = avg_time * (len(all_configs) - i - 1)

                logger.info(f"Progress: {successful} successful, {failed} failed")
                logger.info(f"Est. remaining: {remaining/60:.1f} min")

            total_successful += successful
            total_failed += failed

            logger.info(f"\n[{target}] COMPLETE: {successful} successful, {failed} failed")

        # Final summary
        total_time = time.time() - total_start
        logger.info("\n" + "="*80)
        logger.info("GRID SEARCH COMPLETE")
        logger.info("="*80)
        logger.info(f"Total successful: {total_successful}")
        logger.info(f"Total failed: {total_failed}")
        logger.info(f"Total time: {total_time/60:.2f} minutes")
        logger.info(f"Results: {self.results_file}")

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
            'total_time_minutes': total_time / 60,
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
                            'config_name': overall_best.get('config_name', ''),
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
                                'config_name': best.get('config_name', ''),
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
        """Analyze grid search results."""
        if not self.results_file.exists():
            logger.error(f"No results file found: {self.results_file}")
            return None

        df = pd.read_csv(self.results_file)
        df_success = df[df['status'] == 'success'].copy()

        if len(df_success) == 0:
            logger.warning("No successful runs found")
            return df

        if target:
            df_success = df_success[df_success['target'] == target]

        logger.info("\n" + "="*80)
        logger.info("BASELINE RESULTS ANALYSIS")
        logger.info("="*80)

        targets_in_results = df_success['target'].unique()

        for tgt in targets_in_results:
            df_target = df_success[df_success['target'] == tgt]

            logger.info(f"\n{'#'*60}")
            logger.info(f"# TARGET: {tgt.upper()}")
            logger.info(f"{'#'*60}")

            # Best by model type
            for model_type in df_target['model_type'].unique():
                df_model = df_target[df_target['model_type'] == model_type]
                best_idx = df_model['test_smape'].idxmin()
                best = df_model.loc[best_idx]

                logger.info(f"\n{model_type.upper()} - Best:")
                logger.info(f"  Config: {best['config_name']}")
                logger.info(f"  Test sMAPE: {best['test_smape']:.4f}%")
                logger.info(f"  Test MAE: {best['test_mae']:.2f}")

            # Overall best
            overall_best_idx = df_target['test_smape'].idxmin()
            overall_best = df_target.loc[overall_best_idx]

            logger.info(f"\n{'='*60}")
            logger.info(f"OVERALL BEST for {tgt.upper()}")
            logger.info(f"  Model: {overall_best['model_type']}")
            logger.info(f"  Config: {overall_best['config_name']}")
            logger.info(f"  Test sMAPE: {overall_best['test_smape']:.4f}%")

            if tgt == 'price_real':
                benchmark = 10.87
                if overall_best['test_smape'] < benchmark:
                    logger.info(f"\n*** Beat {benchmark}% benchmark! ***")

        return df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run Baseline Grid Search')
    parser.add_argument(
        '--models',
        nargs='+',
        choices=['catboost', 'xgboost', 'lightgbm', 'prophet'],
        default=None,
        help='Model types to run'
    )
    parser.add_argument(
        '--targets',
        nargs='+',
        choices=['price_real', 'consumption'],
        default=None,
        help='Targets to run'
    )
    parser.add_argument(
        '--max-configs',
        type=int,
        default=None,
        help='Maximum configs per target'
    )
    parser.add_argument(
        '--no-skip',
        action='store_true',
        help='Do not skip completed configs'
    )
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Analyze existing results'
    )

    args = parser.parse_args()

    runner = BaselineGridSearchRunner()

    if args.analyze:
        runner.analyze_results()
    else:
        runner.run_grid_search(
            model_types=args.models,
            targets=args.targets,
            max_configs=args.max_configs,
            skip_completed=not args.no_skip
        )


if __name__ == "__main__":
    main()
