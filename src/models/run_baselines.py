"""
Baseline Model Orchestrator for ForeWatt
========================================
Runs all baseline models and generates comparison report.

Models:
- Prophet (with holidays & weather regressors)
- CatBoost (gradient boosting with categorical features)
- XGBoost (extreme gradient boosting)
- SARIMAX (seasonal ARIMA with exogenous variables)

All results logged to MLflow with MAE/RMSE/MAPE/sMAPE/MASE metrics.

Usage:
    python src/models/run_baselines.py --all
    python src/models/run_baselines.py --models prophet catboost

Author: ForeWatt Team
Date: November 2025
"""

import sys
import pandas as pd
import mlflow
from pathlib import Path
from typing import List, Dict, Optional
import logging
import time

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.train_prophet import run_prophet_baseline
from src.models.train_catboost import run_catboost_baseline
from src.models.train_xgboost import run_xgboost_baseline
from src.models.train_sarimax import run_sarimax_baseline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaselineOrchestrator:
    """Orchestrates training of all baseline models."""

    def __init__(self, mlflow_uri: Optional[str] = None):
        """
        Initialize orchestrator.

        Args:
            mlflow_uri: MLflow tracking server URI (default: file-based)
        """
        if mlflow_uri is None:
            mlflow_dir = PROJECT_ROOT / 'mlruns'
            mlflow_dir.mkdir(exist_ok=True)
            mlflow_uri = f"file://{mlflow_dir}"
        self.mlflow_uri = mlflow_uri
        self.results = {}

    def run_prophet(
        self,
        experiment_name: str = "ForeWatt-Baseline-Prophet",
        run_name: str = "prophet_v1_baseline"
    ) -> Optional[Dict[str, float]]:
        """Run Prophet baseline."""
        logger.info("\n" + "═"*80)
        logger.info("TRAINING PROPHET BASELINE")
        logger.info("═"*80)

        start_time = time.time()
        try:
            metrics = run_prophet_baseline(
                experiment_name=experiment_name,
                run_name=run_name,
                test_size=0.2,
                mlflow_uri=self.mlflow_uri
            )
            elapsed_time = time.time() - start_time
            metrics['training_time_seconds'] = elapsed_time
            self.results['Prophet'] = metrics
            logger.info(f"✓ Prophet completed in {elapsed_time:.2f}s")
            return metrics
        except Exception as e:
            logger.error(f"✗ Prophet failed: {e}")
            self.results['Prophet'] = {'error': str(e)}
            return None

    def run_catboost(
        self,
        experiment_name: str = "ForeWatt-Baseline-CatBoost",
        run_name: str = "catboost_v1_baseline"
    ) -> Optional[Dict[str, float]]:
        """Run CatBoost baseline."""
        logger.info("\n" + "═"*80)
        logger.info("TRAINING CATBOOST BASELINE")
        logger.info("═"*80)

        start_time = time.time()
        try:
            metrics = run_catboost_baseline(
                experiment_name=experiment_name,
                run_name=run_name,
                val_size=0.1,
                test_size=0.2,
                mlflow_uri=self.mlflow_uri
            )
            elapsed_time = time.time() - start_time
            metrics['training_time_seconds'] = elapsed_time
            self.results['CatBoost'] = metrics
            logger.info(f"✓ CatBoost completed in {elapsed_time:.2f}s")
            return metrics
        except Exception as e:
            logger.error(f"✗ CatBoost failed: {e}")
            self.results['CatBoost'] = {'error': str(e)}
            return None

    def run_xgboost(
        self,
        experiment_name: str = "ForeWatt-Baseline-XGBoost",
        run_name: str = "xgboost_v1_baseline"
    ) -> Optional[Dict[str, float]]:
        """Run XGBoost baseline."""
        logger.info("\n" + "═"*80)
        logger.info("TRAINING XGBOOST BASELINE")
        logger.info("═"*80)

        start_time = time.time()
        try:
            metrics = run_xgboost_baseline(
                experiment_name=experiment_name,
                run_name=run_name,
                val_size=0.1,
                test_size=0.2,
                mlflow_uri=self.mlflow_uri
            )
            elapsed_time = time.time() - start_time
            metrics['training_time_seconds'] = elapsed_time
            self.results['XGBoost'] = metrics
            logger.info(f"✓ XGBoost completed in {elapsed_time:.2f}s")
            return metrics
        except Exception as e:
            logger.error(f"✗ XGBoost failed: {e}")
            self.results['XGBoost'] = {'error': str(e)}
            return None

    def run_sarimax(
        self,
        experiment_name: str = "ForeWatt-Baseline-SARIMAX",
        run_name: str = "sarimax_v1_baseline"
    ) -> Optional[Dict[str, float]]:
        """Run SARIMAX baseline."""
        logger.info("\n" + "═"*80)
        logger.info("TRAINING SARIMAX BASELINE")
        logger.info("═"*80)

        start_time = time.time()
        try:
            metrics = run_sarimax_baseline(
                experiment_name=experiment_name,
                run_name=run_name,
                test_size=0.2,
                mlflow_uri=self.mlflow_uri
            )
            elapsed_time = time.time() - start_time
            metrics['training_time_seconds'] = elapsed_time
            self.results['SARIMAX'] = metrics
            logger.info(f"✓ SARIMAX completed in {elapsed_time:.2f}s")
            return metrics
        except Exception as e:
            logger.error(f"✗ SARIMAX failed: {e}")
            self.results['SARIMAX'] = {'error': str(e)}
            return None

    def run_all(self, models: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """
        Run all baseline models.

        Args:
            models: List of models to run (default: all)
                    Options: 'prophet', 'catboost', 'xgboost', 'sarimax'

        Returns:
            Dictionary with results for each model
        """
        if models is None:
            models = ['prophet', 'catboost', 'xgboost', 'sarimax']

        models = [m.lower() for m in models]

        logger.info("\n" + "█"*80)
        logger.info("FOREWATT BASELINE MODEL TRAINING")
        logger.info("█"*80)
        logger.info(f"Models to train: {', '.join(models).upper()}")
        logger.info(f"MLflow URI: {self.mlflow_uri}")
        logger.info("█"*80)

        # Run models
        if 'prophet' in models:
            self.run_prophet()

        if 'catboost' in models:
            self.run_catboost()

        if 'xgboost' in models:
            self.run_xgboost()

        if 'sarimax' in models:
            self.run_sarimax()

        # Generate comparison report
        self.print_comparison_report()

        return self.results

    def print_comparison_report(self):
        """Print comparison report of all models."""
        logger.info("\n\n" + "█"*80)
        logger.info("BASELINE MODEL COMPARISON REPORT")
        logger.info("█"*80)

        # Create comparison dataframe
        comparison_data = []
        for model_name, metrics in self.results.items():
            if 'error' in metrics:
                logger.warning(f"{model_name}: FAILED - {metrics['error']}")
                continue

            comparison_data.append({
                'Model': model_name,
                'MAE': metrics.get('MAE', float('nan')),
                'RMSE': metrics.get('RMSE', float('nan')),
                'MAPE': metrics.get('MAPE', float('nan')),
                'sMAPE': metrics.get('sMAPE', float('nan')),
                'MASE': metrics.get('MASE', float('nan')),
                'Training Time (s)': metrics.get('training_time_seconds', float('nan'))
            })

        if comparison_data:
            df = pd.DataFrame(comparison_data)

            # Sort by MASE (lower is better)
            df = df.sort_values('MASE')

            logger.info("\n" + df.to_string(index=False))

            # Highlight best model
            logger.info("\n" + "─"*80)
            best_model = df.iloc[0]['Model']
            best_mase = df.iloc[0]['MASE']
            logger.info(f"🏆 BEST MODEL: {best_model} (MASE: {best_mase:.4f})")
            logger.info("─"*80)

            # Save report
            report_path = PROJECT_ROOT / 'reports' / 'baseline_comparison.csv'
            report_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(report_path, index=False)
            logger.info(f"\n✓ Comparison report saved to: {report_path}")

        logger.info("█"*80 + "\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run baseline model training')
    parser.add_argument('--all', action='store_true',
                        help='Run all baseline models')
    parser.add_argument('--models', nargs='+',
                        choices=['prophet', 'catboost', 'xgboost', 'sarimax'],
                        help='Specific models to run')
    parser.add_argument('--mlflow-uri', type=str, default='http://localhost:5050',
                        help='MLflow tracking URI')

    args = parser.parse_args()

    # Determine which models to run
    if args.all:
        models = ['prophet', 'catboost', 'xgboost', 'sarimax']
    elif args.models:
        models = args.models
    else:
        # Default: run all
        models = ['prophet', 'catboost', 'xgboost', 'sarimax']

    # Run orchestrator
    orchestrator = BaselineOrchestrator(mlflow_uri=args.mlflow_uri)
    results = orchestrator.run_all(models=models)

    # Exit with error if all models failed
    successful_models = [m for m, r in results.items() if 'error' not in r]
    if not successful_models:
        logger.error("All models failed!")
        sys.exit(1)
    else:
        logger.info(f"\n✓ Successfully trained {len(successful_models)} models")
        sys.exit(0)
