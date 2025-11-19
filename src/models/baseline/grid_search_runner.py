"""
Grid Search Runner for Baseline Models
======================================
Runs all hyperparameter configurations for baseline models and
compares results across different parameter combinations.

Targets:
- consumption: Electricity demand (MWh)
- price_real: Day-ahead market price (PTF, inflation-adjusted TL/MWh)

Usage:
    # Run all configs for both targets
    python src/models/baseline/grid_search_runner.py

    # Run specific model configs
    python src/models/baseline/grid_search_runner.py --models catboost xgboost

    # Run specific target
    python src/models/baseline/grid_search_runner.py --targets price_real

Author: ForeWatt Team
Date: November 2025
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging
import json
import time
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.baseline.hyperparameter_configs import (
    BaselineHyperparameterConfigs,
    FEATURE_SELECTION_STRATEGIES,
    check_gpu_available
)
from src.models.baseline.pipeline_runner import BaselinePipeline
from src.models.baseline.data_loader import load_master_data, train_val_test_split

# Configure root logger to capture ALL output from ALL modules
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check GPU availability once at module load
GPU_AVAILABLE, GPU_TYPE = check_gpu_available()


class GridSearchRunner:
    """
    Grid search runner for testing all hyperparameter configurations.
    """

    def __init__(
        self,
        targets: List[str] = None,
        models: List[str] = None,
        val_size: float = 0.1,
        test_size: float = 0.2,
        mlflow_uri: Optional[str] = None
    ):
        """
        Initialize grid search runner.

        Args:
            targets: List of targets to test (default: ['consumption', 'price_real'])
            models: List of models to test (default: all)
            val_size: Validation set size
            test_size: Test set size
            mlflow_uri: MLflow tracking URI
        """
        self.targets = targets or ['consumption', 'price_real']
        self.models = models or ['catboost', 'xgboost', 'lightgbm', 'prophet']
        self.val_size = val_size
        self.test_size = test_size
        self.mlflow_uri = mlflow_uri

        self.all_results = {}
        self.configs = BaselineHyperparameterConfigs.get_all_baseline_configs()

    def run_grid_search(self) -> Dict:
        """
        Run grid search across all configurations.

        Returns:
            Dictionary with all results
        """
        # Create overall run log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = PROJECT_ROOT / 'reports' / 'baseline' / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        overall_log_file = log_dir / f'grid_search_run_{timestamp}.log'

        # Add file handler to ROOT logger to capture ALL output from ALL modules
        root_logger = logging.getLogger()
        overall_file_handler = logging.FileHandler(overall_log_file)
        overall_file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        overall_file_handler.setFormatter(formatter)
        root_logger.addHandler(overall_file_handler)

        logger.info(f"\n{'█'*100}")
        logger.info("BASELINE MODELS GRID SEARCH")
        logger.info(f"Overall log file: {overall_log_file}")
        logger.info(f"{'█'*100}")
        logger.info(f"Targets: {', '.join(self.targets)}")
        logger.info(f"Models: {', '.join(self.models)}")

        total_configs = sum(
            len(self.configs[model])
            for model in self.models
            if model in self.configs
        )
        total_runs = total_configs * len(self.targets)

        logger.info(f"Total configurations: {total_configs}")
        logger.info(f"Total runs: {total_runs}")
        logger.info(f"{'█'*100}\n")

        start_time = time.time()

        try:
            for target in self.targets:
                logger.info(f"\n{'#'*100}")
                logger.info(f"# TARGET: {target.upper()}")
                logger.info(f"{'#'*100}\n")

                self.all_results[target] = {}

                # Load and split data once per target
                df = load_master_data()
                train_df, val_df, test_df = train_val_test_split(
                    df, self.val_size, self.test_size
                )

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
                        # Create log directory for this run
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        log_dir = PROJECT_ROOT / 'reports' / 'baseline' / 'logs' / target / model_type
                        log_dir.mkdir(parents=True, exist_ok=True)
                        log_file = log_dir / f'{config_name}_{timestamp}.log'

                        # Add file handler to ROOT logger to capture detailed per-config logs
                        file_handler = logging.FileHandler(log_file)
                        file_handler.setLevel(logging.INFO)
                        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                        file_handler.setFormatter(formatter)
                        root_logger.addHandler(file_handler)

                        logger.info(f"\n{'─'*100}")
                        logger.info(f"Config: {config_name}")
                        logger.info(f"Log file: {log_file}")
                        logger.info(f"{'─'*100}")

                        # Extract description and feature_selection
                        description = config_params.pop('description', '')
                        feature_selection = config_params.pop('feature_selection', 'all')

                        # Remove GPU params if no GPU available
                        if not GPU_AVAILABLE:
                            config_params.pop('task_type', None)  # CatBoost
                            config_params.pop('tree_method', None)  # XGBoost
                            config_params.pop('device', None)  # LightGBM

                        logger.info(f"Description: {description}")
                        logger.info(f"Feature selection: {feature_selection}")
                        logger.info(f"Parameters: {config_params}")
                        if GPU_AVAILABLE:
                            logger.info("✓ Using GPU acceleration")

                        try:
                            # Initialize pipeline
                            pipeline = BaselinePipeline(
                                target=target,
                                mlflow_uri=self.mlflow_uri
                            )

                            # Run model with this config
                            metrics = pipeline.run_model(
                                model_type=model_type,
                                train_df=train_df,
                                val_df=val_df,
                                test_df=test_df,
                                hyperparams=config_params
                            )

                            # Store results
                            self.all_results[target][model_type][config_name] = {
                                'metrics': metrics,
                                'config': config_params,
                                'description': description,
                                'feature_selection': feature_selection
                            }

                            logger.info(f"✓ {config_name} completed successfully")

                        except Exception as e:
                            logger.error(f"✗ {config_name} failed: {e}", exc_info=True)
                            self.all_results[target][model_type][config_name] = {
                                'error': str(e)
                            }

                        finally:
                            # Remove file handler from root logger
                            root_logger.removeHandler(file_handler)
                            file_handler.close()

            elapsed_time = time.time() - start_time
            logger.info(f"\n{'█'*100}")
            logger.info(f"GRID SEARCH COMPLETED in {elapsed_time/60:.2f} minutes")
            logger.info(f"{'█'*100}\n")

            # Generate summary reports
            self.generate_summary_reports()

            return self.all_results

        finally:
            # Remove overall log file handler from root logger
            root_logger.removeHandler(overall_file_handler)
            overall_file_handler.close()

    def generate_summary_reports(self):
        """Generate summary reports comparing all configurations."""
        logger.info(f"\n{'='*100}")
        logger.info("GENERATING SUMMARY REPORTS")
        logger.info(f"{'='*100}\n")

        report_dir = PROJECT_ROOT / 'reports' / 'baseline' / 'grid_search'
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
                    if 'error' in metrics:
                        continue

                    comparison_data.append({
                        'model': model_type,
                        'config': config_name,
                        'description': result.get('description', ''),
                        'feature_selection': result.get('feature_selection', ''),
                        'MAE': metrics.get('MAE', np.nan),
                        'RMSE': metrics.get('RMSE', np.nan),
                        'MAPE': metrics.get('MAPE', np.nan),
                        'sMAPE': metrics.get('sMAPE', np.nan),
                        'MASE': metrics.get('MASE', np.nan),
                        'training_time_seconds': metrics.get('training_time_seconds', np.nan)
                    })

            if not comparison_data:
                logger.warning(f"No successful runs for {target}")
                continue

            df = pd.DataFrame(comparison_data)

            # Sort by MASE (primary metric)
            df = df.sort_values('MASE')

            # Save full comparison
            csv_path = report_dir / f'grid_search_results_{target}_{timestamp}.csv'
            df.to_csv(csv_path, index=False)
            logger.info(f"  ✓ Saved: {csv_path}")

            # Print top 10 configurations
            logger.info(f"\n  Top 10 configurations for {target}:")
            top_10 = df.head(10)[['model', 'config', 'sMAPE', 'MASE', 'training_time_seconds']]
            logger.info("\n" + top_10.to_string(index=False))

            # Best per model
            logger.info(f"\n  Best configuration per model:")
            best_per_model = df.loc[df.groupby('model')['MASE'].idxmin()]
            best_summary = best_per_model[['model', 'config', 'sMAPE', 'MASE']]
            logger.info("\n" + best_summary.to_string(index=False))

            # Save best per model
            best_path = report_dir / f'best_per_model_{target}_{timestamp}.csv'
            best_per_model.to_csv(best_path, index=False)
            logger.info(f"\n  ✓ Saved: {best_path}")

            # Save feature selection analysis
            feature_analysis = df.groupby('feature_selection').agg({
                'MASE': ['mean', 'std', 'min'],
                'sMAPE': ['mean', 'std', 'min'],
                'config': 'count'
            }).round(4)
            feature_analysis.columns = ['_'.join(col).strip() for col in feature_analysis.columns]

            feature_path = report_dir / f'feature_selection_analysis_{target}_{timestamp}.csv'
            feature_analysis.to_csv(feature_path)
            logger.info(f"  ✓ Saved: {feature_path}")

        # Save complete results as JSON
        json_path = report_dir / f'grid_search_complete_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump(self.all_results, f, indent=2, default=str)
        logger.info(f"\n✓ Complete results saved: {json_path}")

        logger.info(f"\n{'='*100}\n")

    def get_best_configs(self) -> Dict[str, Dict[str, str]]:
        """
        Get best configuration for each model and target.

        Returns:
            Dictionary mapping target -> model -> best_config_name
        """
        best_configs = {}

        for target in self.targets:
            best_configs[target] = {}

            for model_type, configs in self.all_results[target].items():
                best_mase = float('inf')
                best_config = None

                for config_name, result in configs.items():
                    if 'error' in result:
                        continue

                    metrics = result['metrics']
                    if 'error' in metrics:
                        continue

                    mase = metrics.get('MASE', float('inf'))
                    if mase < best_mase:
                        best_mase = mase
                        best_config = config_name

                if best_config:
                    best_configs[target][model_type] = best_config

        return best_configs


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Grid search for baseline model hyperparameters'
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
        choices=['catboost', 'xgboost', 'lightgbm', 'prophet', 'sarimax'],
        help='Models to test (default: all)'
    )
    parser.add_argument(
        '--val-size',
        type=float,
        default=0.1,
        help='Validation set size (default: 0.1)'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set size (default: 0.2)'
    )
    parser.add_argument(
        '--mlflow-uri',
        type=str,
        default=None,
        help='MLflow tracking URI'
    )

    args = parser.parse_args()

    # Run grid search
    runner = GridSearchRunner(
        targets=args.targets,
        models=args.models,
        val_size=args.val_size,
        test_size=args.test_size,
        mlflow_uri=args.mlflow_uri
    )

    results = runner.run_grid_search()

    # Print best configurations
    best_configs = runner.get_best_configs()

    logger.info(f"\n{'█'*100}")
    logger.info("BEST CONFIGURATIONS")
    logger.info(f"{'█'*100}")

    for target, model_configs in best_configs.items():
        logger.info(f"\n{target.upper()}:")
        for model_type, config_name in model_configs.items():
            metrics = results[target][model_type][config_name]['metrics']
            logger.info(f"  {model_type:15s}: {config_name:30s} (MASE: {metrics['MASE']:.4f}, sMAPE: {metrics['sMAPE']:.2f}%)")

    logger.info(f"\n{'█'*100}\n")
    logger.info("✓ Grid search completed successfully!")

    return 0


if __name__ == '__main__':
    sys.exit(main())
