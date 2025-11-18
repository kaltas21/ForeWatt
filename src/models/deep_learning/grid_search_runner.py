"""
Grid Search Runner for Deep Learning Models
===========================================
Runs all hyperparameter configurations for deep learning models (N-HiTS, TFT, PatchTST)
and compares results across different parameter combinations.

Targets:
- consumption: Electricity demand (MWh)
- price_real: Day-ahead market price (PTF, inflation-adjusted TL/MWh)

Usage:
    # Run all configs for both targets
    python src/models/deep_learning/grid_search_runner.py

    # Run specific model configs
    python src/models/deep_learning/grid_search_runner.py --models nhits tft

    # Run specific target
    python src/models/deep_learning/grid_search_runner.py --targets price_real

    # Enable Optuna optimization (slower but better results)
    python src/models/deep_learning/grid_search_runner.py --use-optuna --n-trials 20

Author: ForeWatt Team
Date: November 2025
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import json
import time
from datetime import datetime
import mlflow

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.deep_learning.hyperparameter_configs import (
    DeepLearningHyperparameterConfigs,
    DL_FEATURE_SELECTION_STRATEGIES,
    OPTUNA_SEARCH_SPACES
)
from src.models.deep_learning.feature_preparer import DeepLearningFeaturePreparer
from src.models.deep_learning.cv_strategy import ExpandingWindowCV
from src.models.deep_learning.evaluator import HorizonWiseEvaluator
from src.models.deep_learning.hardware_config import get_hardware_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeepLearningGridSearchRunner:
    """
    Grid search runner for testing all deep learning hyperparameter configurations.
    """

    def __init__(
        self,
        targets: List[str] = None,
        models: List[str] = None,
        val_size: float = 0.2,
        test_size: float = 0.2,
        cv_folds: int = 3,
        use_optuna: bool = False,
        n_trials: int = 20,
        mlflow_uri: Optional[str] = None
    ):
        """
        Initialize grid search runner.

        Args:
            targets: List of targets to test (default: ['consumption', 'price_real'])
            models: List of models to test (default: ['nhits', 'tft', 'patchtst'])
            val_size: Validation set size for train/val split
            test_size: Test set size
            cv_folds: Number of cross-validation folds
            use_optuna: Whether to use Optuna for optimization (slower but better)
            n_trials: Number of Optuna trials (if use_optuna=True)
            mlflow_uri: MLflow tracking URI
        """
        self.targets = targets or ['consumption', 'price_real']
        self.models = models or ['nhits', 'tft', 'patchtst']
        self.val_size = val_size
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.use_optuna = use_optuna
        self.n_trials = n_trials
        self.mlflow_uri = mlflow_uri

        self.all_results = {}
        self.configs = DeepLearningHyperparameterConfigs.get_all_deep_learning_configs()

        # Hardware detection
        self.hw_config = get_hardware_config()
        logger.info(f"Hardware: {self.hw_config.device_type.upper()}")

        # Setup MLflow
        if mlflow_uri is None:
            import os
            mlflow_uri = os.getenv('MLFLOW_TRACKING_URI')
            if mlflow_uri is None:
                # Default to Docker MLflow server
                mlflow_uri = 'http://localhost:5050'
                logger.info("Using Docker MLflow server at http://localhost:5050")
        self.mlflow_uri = mlflow_uri

        mlflow.set_tracking_uri(self.mlflow_uri)

        # Set local artifact directory when using remote MLflow server
        if self.mlflow_uri.startswith('http'):
            import os
            artifact_dir = PROJECT_ROOT / 'mlflow_artifacts'
            artifact_dir.mkdir(exist_ok=True)
            os.environ['MLFLOW_ARTIFACT_ROOT'] = str(artifact_dir)
            logger.info(f"Using local artifact storage: {artifact_dir}")

    def load_and_prepare_data(
        self,
        target: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load and prepare data for deep learning.

        Args:
            target: Target variable

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        from src.models.baseline.data_loader import load_master_data

        logger.info(f"\n{'='*100}")
        logger.info(f"LOADING DATA FOR {target.upper()}")
        logger.info(f"{'='*100}")

        # Load master dataset
        df = load_master_data()

        # Temporal split
        n_samples = len(df)
        train_end = int(n_samples * (1 - self.val_size - self.test_size))
        val_end = int(n_samples * (1 - self.test_size))

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()

        logger.info(f"Train: {len(train_df)} samples ({train_df.index[0]} to {train_df.index[-1]})")
        logger.info(f"Val:   {len(val_df)} samples ({val_df.index[0]} to {val_df.index[-1]})")
        logger.info(f"Test:  {len(test_df)} samples ({test_df.index[0]} to {test_df.index[-1]})")
        logger.info(f"{'='*100}\n")

        return train_df, val_df, test_df

    def train_single_config(
        self,
        model_type: str,
        config_name: str,
        config_params: Dict,
        target: str,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> Dict:
        """
        Train a single configuration.

        Args:
            model_type: Model type ('nhits', 'tft', 'patchtst')
            config_name: Configuration name
            config_params: Configuration parameters
            target: Target variable
            train_df: Training data
            val_df: Validation data
            test_df: Test data

        Returns:
            Dictionary with results
        """
        # Create log directory for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = PROJECT_ROOT / 'reports' / 'deep_learning' / 'logs' / target / model_type
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f'{config_name}_{timestamp}.log'

        # Add file handler to logger
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info(f"\n{'─'*100}")
        logger.info(f"Training: {model_type.upper()} - {config_name}")
        logger.info(f"Log file: {log_file}")
        logger.info(f"{'─'*100}")

        try:
            # Extract metadata
            description = config_params.pop('description', '')
            feature_selection = config_params.pop('feature_selection', 'standard_dl')
            horizon = config_params.pop('horizon', 24)
            input_size = config_params.pop('input_size', 168)

            logger.info(f"Description: {description}")
            logger.info(f"Feature selection: {feature_selection}")
            logger.info(f"Horizon: {horizon}h, Input size: {input_size}h")
            # Prepare features
            preparer = DeepLearningFeaturePreparer(
                target=target,
                max_lag=input_size
            )

            X_train, y_train, feature_names = preparer.prepare_features(train_df)
            X_val, y_val, _ = preparer.prepare_features(val_df)
            X_test, y_test, _ = preparer.prepare_features(test_df)

            logger.info(f"Features prepared: {len(feature_names)} features")

            # Start MLflow run
            experiment_name = f"ForeWatt-DeepLearning-{target.upper()}-GridSearch"
            mlflow.set_experiment(experiment_name)

            with mlflow.start_run(run_name=f"{model_type}_{config_name}"):
                # Log configuration
                mlflow.log_param("model_type", model_type)
                mlflow.log_param("config_name", config_name)
                mlflow.log_param("target", target)
                mlflow.log_param("feature_selection", feature_selection)
                mlflow.log_param("n_features", len(feature_names))
                mlflow.log_param("horizon", horizon)
                mlflow.log_param("input_size", input_size)

                for key, value in config_params.items():
                    mlflow.log_param(key, value)

                # Train model (placeholder - actual implementation depends on model type)
                start_time = time.time()

                # Import model-specific trainer
                if model_type == 'nhits':
                    from src.models.deep_learning.models.nhits_trainer import NHiTSTrainer
                    trainer = NHiTSTrainer(
                        target=target,
                        horizon=horizon,
                        input_size=input_size
                    )
                elif model_type == 'tft':
                    from src.models.deep_learning.models.tft_trainer import TFTTrainer
                    trainer = TFTTrainer(
                        target=target,
                        horizon=horizon,
                        input_size=input_size
                    )
                elif model_type == 'patchtst':
                    from src.models.deep_learning.models.patchtst_trainer import PatchTSTTrainer
                    trainer = PatchTSTTrainer(
                        target=target,
                        horizon=horizon,
                        input_size=input_size
                    )
                else:
                    raise ValueError(f"Unknown model type: {model_type}")

                # Train
                model, val_metrics = trainer.train(
                    X_train, y_train,
                    X_val, y_val,
                    hyperparams=config_params
                )

                # Predict on test
                test_predictions = trainer.predict(X_test)

                # Evaluate
                evaluator = HorizonWiseEvaluator(horizon=horizon)
                test_metrics = evaluator.evaluate(
                    y_true=y_test.values,
                    y_pred=test_predictions,
                    y_train=y_train.values
                )

                training_time = time.time() - start_time

                # Log metrics
                mlflow.log_metric("val_sMAPE", val_metrics['sMAPE'])
                mlflow.log_metric("val_MASE", val_metrics['MASE'])
                mlflow.log_metric("test_sMAPE", test_metrics['sMAPE_mean'])
                mlflow.log_metric("test_MASE", test_metrics['MASE_mean'])
                mlflow.log_metric("training_time_seconds", training_time)

                # Log per-horizon metrics
                for h in range(1, horizon + 1):
                    if f'sMAPE_h{h}' in test_metrics:
                        mlflow.log_metric(f"sMAPE_h{h}", test_metrics[f'sMAPE_h{h}'])
                        mlflow.log_metric(f"MASE_h{h}", test_metrics[f'MASE_h{h}'])

                logger.info(f"\n✓ {config_name} completed in {training_time:.2f}s")
                logger.info(f"  Val sMAPE: {val_metrics['sMAPE']:.2f}%")
                logger.info(f"  Test sMAPE: {test_metrics['sMAPE_mean']:.2f}%")
                logger.info(f"  Test MASE: {test_metrics['MASE_mean']:.4f}")

                result = {
                    'metrics': {
                        'val_sMAPE': val_metrics['sMAPE'],
                        'val_MASE': val_metrics['MASE'],
                        'test_sMAPE': test_metrics['sMAPE_mean'],
                        'test_MASE': test_metrics['MASE_mean'],
                        'training_time_seconds': training_time,
                        **test_metrics  # Include all per-horizon metrics
                    },
                    'config': config_params,
                    'description': description,
                    'feature_selection': feature_selection
                }

                return result

        except Exception as e:
            logger.error(f"✗ {config_name} failed: {e}", exc_info=True)
            return {'error': str(e)}

        finally:
            # Remove file handler
            logger.removeHandler(file_handler)
            file_handler.close()

    def run_grid_search(self) -> Dict:
        """
        Run grid search across all configurations.

        Returns:
            Dictionary with all results
        """
        # Create overall run log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = PROJECT_ROOT / 'reports' / 'deep_learning' / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        overall_log_file = log_dir / f'grid_search_run_{timestamp}.log'

        # Add file handler for overall run
        overall_file_handler = logging.FileHandler(overall_log_file)
        overall_file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        overall_file_handler.setFormatter(formatter)
        logger.addHandler(overall_file_handler)

        logger.info(f"\n{'█'*100}")
        logger.info("DEEP LEARNING MODELS GRID SEARCH")
        logger.info(f"Overall log file: {overall_log_file}")
        logger.info(f"{'█'*100}")
        logger.info(f"Targets: {', '.join(self.targets)}")
        logger.info(f"Models: {', '.join(self.models)}")
        logger.info(f"Hardware: {self.hw_config.device_type.upper()}")

        total_configs = sum(
            len(self.configs[model])
            for model in self.models
            if model in self.configs
        )
        total_runs = total_configs * len(self.targets)

        logger.info(f"Total configurations: {total_configs}")
        logger.info(f"Total runs: {total_runs}")
        logger.info(f"Use Optuna: {self.use_optuna}")
        if self.use_optuna:
            logger.info(f"Trials per config: {self.n_trials}")
        logger.info(f"{'█'*100}\n")

        start_time = time.time()

        try:
            for target in self.targets:
                logger.info(f"\n{'#'*100}")
                logger.info(f"# TARGET: {target.upper()}")
                logger.info(f"{'#'*100}\n")

                self.all_results[target] = {}

                # Load data once per target
                train_df, val_df, test_df = self.load_and_prepare_data(target)

                for model_type in self.models:
                    if model_type not in self.configs:
                        logger.warning(f"No configs found for {model_type}")
                        continue

                    model_configs = self.configs[model_type]
                    logger.info(f"\n{'='*100}")
                    logger.info(f"MODEL: {model_type.upper()} ({len(model_configs)} configurations)")
                    logger.info(f"{'='*100}\n")

                    self.all_results[target][model_type] = {}

                    for config_name, config_params in model_configs.items():
                        # Make a copy to avoid modifying original
                        config_copy = config_params.copy()

                        result = self.train_single_config(
                            model_type=model_type,
                            config_name=config_name,
                            config_params=config_copy,
                            target=target,
                            train_df=train_df,
                            val_df=val_df,
                            test_df=test_df
                        )

                        self.all_results[target][model_type][config_name] = result

            elapsed_time = time.time() - start_time
            logger.info(f"\n{'█'*100}")
            logger.info(f"GRID SEARCH COMPLETED in {elapsed_time/60:.2f} minutes")
            logger.info(f"{'█'*100}\n")

            # Generate summary reports
            self.generate_summary_reports()

            return self.all_results

        finally:
            # Remove overall log file handler
            logger.removeHandler(overall_file_handler)
            overall_file_handler.close()

    def generate_summary_reports(self):
        """Generate summary reports comparing all configurations."""
        logger.info(f"\n{'='*100}")
        logger.info("GENERATING SUMMARY REPORTS")
        logger.info(f"{'='*100}\n")

        report_dir = PROJECT_ROOT / 'reports' / 'deep_learning' / 'grid_search'
        report_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for target in self.targets:
            logger.info(f"\nTarget: {target.upper()}")

            # Create comparison dataframe
            comparison_data = []

            for model_type, configs in self.all_results[target].items():
                for config_name, result in configs.items():
                    if 'error' in result:
                        continue

                    metrics = result['metrics']

                    comparison_data.append({
                        'model': model_type,
                        'config': config_name,
                        'description': result.get('description', ''),
                        'feature_selection': result.get('feature_selection', ''),
                        'val_sMAPE': metrics.get('val_sMAPE', np.nan),
                        'val_MASE': metrics.get('val_MASE', np.nan),
                        'test_sMAPE': metrics.get('test_sMAPE', np.nan),
                        'test_MASE': metrics.get('test_MASE', np.nan),
                        'test_sMAPE_h1': metrics.get('sMAPE_h1', np.nan),
                        'test_sMAPE_h6': metrics.get('sMAPE_h6', np.nan),
                        'test_sMAPE_h24': metrics.get('sMAPE_h24', np.nan),
                        'training_time_seconds': metrics.get('training_time_seconds', np.nan)
                    })

            if not comparison_data:
                logger.warning(f"No successful runs for {target}")
                continue

            df = pd.DataFrame(comparison_data)

            # Sort by test MASE (primary metric)
            df = df.sort_values('test_MASE')

            # Save full comparison
            csv_path = report_dir / f'grid_search_results_{target}_{timestamp}.csv'
            df.to_csv(csv_path, index=False)
            logger.info(f"  ✓ Saved: {csv_path}")

            # Print top 10 configurations
            logger.info(f"\n  Top 10 configurations for {target}:")
            top_10 = df.head(10)[['model', 'config', 'test_sMAPE', 'test_MASE', 'test_sMAPE_h1', 'test_sMAPE_h24']]
            logger.info("\n" + top_10.to_string(index=False))

            # Best per model
            logger.info(f"\n  Best configuration per model:")
            best_per_model = df.loc[df.groupby('model')['test_MASE'].idxmin()]
            best_summary = best_per_model[['model', 'config', 'test_sMAPE', 'test_MASE']]
            logger.info("\n" + best_summary.to_string(index=False))

            # Save best per model
            best_path = report_dir / f'best_per_model_{target}_{timestamp}.csv'
            best_per_model.to_csv(best_path, index=False)
            logger.info(f"\n  ✓ Saved: {best_path}")

        # Save complete results as JSON
        json_path = report_dir / f'grid_search_complete_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump(self.all_results, f, indent=2, default=str)
        logger.info(f"\n✓ Complete results saved: {json_path}")

        logger.info(f"\n{'='*100}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Grid search for deep learning model hyperparameters'
    )
    parser.add_argument(
        '--targets',
        nargs='+',
        choices=['consumption', 'price_real'],
        help='Targets to test (default: both)'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        choices=['nhits', 'tft', 'patchtst'],
        help='Models to test (default: all)'
    )
    parser.add_argument(
        '--val-size',
        type=float,
        default=0.2,
        help='Validation set size (default: 0.2)'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set size (default: 0.2)'
    )
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=3,
        help='Number of CV folds (default: 3)'
    )
    parser.add_argument(
        '--use-optuna',
        action='store_true',
        help='Use Optuna for optimization (slower but better)'
    )
    parser.add_argument(
        '--n-trials',
        type=int,
        default=20,
        help='Number of Optuna trials (default: 20)'
    )
    parser.add_argument(
        '--mlflow-uri',
        type=str,
        default=None,
        help='MLflow tracking URI'
    )

    args = parser.parse_args()

    # Run grid search
    runner = DeepLearningGridSearchRunner(
        targets=args.targets,
        models=args.models,
        val_size=args.val_size,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
        use_optuna=args.use_optuna,
        n_trials=args.n_trials,
        mlflow_uri=args.mlflow_uri
    )

    results = runner.run_grid_search()

    logger.info("\n✓ Grid search completed successfully!")

    return 0


if __name__ == '__main__':
    sys.exit(main())
