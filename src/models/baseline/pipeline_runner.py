"""
Baseline Pipeline Runner
========================
Orchestrates training of all baseline models with intelligent feature selection
for both demand and price prediction.

Author: ForeWatt Team
Date: November 2025
"""

import sys
import numpy as np
import pandas as pd
import mlflow
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
import time
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.baseline.data_loader import (
    load_master_data,
    train_val_test_split,
    prepare_target_data
)
from src.models.baseline.model_trainer import ModelTrainer
from src.models.evaluate import evaluate_forecast

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaselinePipeline:
    """
    Orchestrates training of all baseline models.
    """

    def __init__(
        self,
        target: str = 'consumption',
        mlflow_uri: Optional[str] = None,
        experiment_name: Optional[str] = None
    ):
        """
        Initialize baseline pipeline.

        Args:
            target: Target variable ('consumption' or 'price_real')
            mlflow_uri: MLflow tracking server URI
            experiment_name: MLflow experiment name
        """
        self.target = target
        self.results = {}
        self.feature_importance_results = {}

        # Setup MLflow
        if mlflow_uri is None:
            # Try to use environment variable first, then Docker MLflow, then local
            import os
            mlflow_uri = os.getenv('MLFLOW_TRACKING_URI')
            if mlflow_uri is None:
                # Default to Docker MLflow server if available
                mlflow_uri = 'http://localhost:5050'
                logger.info("Using Docker MLflow server at http://localhost:5050")
        self.mlflow_uri = mlflow_uri

        if experiment_name is None:
            target_name = target.replace('_', '-').title()
            experiment_name = f"ForeWatt-Baseline-{target_name}"
        self.experiment_name = experiment_name

        mlflow.set_tracking_uri(self.mlflow_uri)

        # Set local artifact directory when using remote MLflow server
        if self.mlflow_uri.startswith('http'):
            # Using remote server - artifacts need local storage
            artifact_dir = PROJECT_ROOT / 'mlflow_artifacts'
            artifact_dir.mkdir(exist_ok=True)
            import os
            os.environ['MLFLOW_ARTIFACT_ROOT'] = str(artifact_dir)
            logger.info(f"Using local artifact storage: {artifact_dir}")

        mlflow.set_experiment(self.experiment_name)

    def _estimate_training_time(
        self,
        model_type: str,
        n_samples: int,
        hyperparams: Optional[Dict] = None
    ) -> Optional[str]:
        """
        Estimate training time for a model.

        Args:
            model_type: Model type
            n_samples: Number of training samples
            hyperparams: Model hyperparameters

        Returns:
            Estimated time as string (e.g., "~2-5 minutes") or None
        """
        # Base estimates for 30k samples
        estimates = {
            'catboost': (0.5, 2),    # 30s - 2 min (fast with GPU)
            'xgboost': (0.5, 2),     # 30s - 2 min (fast with GPU)
            'lightgbm': (0.3, 1.5),  # 20s - 90s (fastest)
            'prophet': (2, 5)        # 2-5 min (medium)
        }

        if model_type not in estimates:
            return None

        base_min, base_max = estimates[model_type]

        # Adjust for sample size (rough scaling)
        scale_factor = (n_samples / 30000) ** 0.8  # Sublinear scaling

        # Adjust for iterations/steps
        if hyperparams:
            if model_type in ['catboost', 'xgboost', 'lightgbm']:
                iterations = hyperparams.get('iterations') or hyperparams.get('n_estimators', 1000)
                scale_factor *= (iterations / 1000)

        est_min = base_min * scale_factor
        est_max = base_max * scale_factor

        # Format output
        if est_max < 1:
            return f"~{int(est_min * 60)}-{int(est_max * 60)} seconds"
        elif est_max < 60:
            return f"~{est_min:.1f}-{est_max:.1f} minutes"
        else:
            return f"~{est_min/60:.1f}-{est_max/60:.1f} hours"

    def run_model(
        self,
        model_type: str,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        hyperparams: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Run a single model.

        Args:
            model_type: Model type
            train_df: Training data
            val_df: Validation data
            test_df: Test data
            hyperparams: Model hyperparameters

        Returns:
            Dictionary with metrics and feature importance
        """
        logger.info(f"\n{'█'*80}")
        logger.info(f"TRAINING {model_type.upper()} → {self.target}")
        logger.info(f"{'█'*80}")

        # Estimate training time
        n_samples = len(train_df)
        estimated_time = self._estimate_training_time(model_type, n_samples, hyperparams)
        if estimated_time:
            logger.info(f"Estimated training time: {estimated_time}")

        start_time = time.time()

        try:
            import tempfile  # Import here for artifact saving

            # Initialize trainer
            trainer = ModelTrainer(
                model_type=model_type,
                target=self.target,
                hyperparams=hyperparams
            )

            # Prepare features
            X_train, y_train, feature_names = trainer.prepare_features(train_df)
            X_val, y_val, _ = trainer.prepare_features(val_df)
            X_test, y_test, _ = trainer.prepare_features(test_df)

            # Start MLflow run
            run_name = f"{model_type}_{self.target}"
            with mlflow.start_run(run_name=run_name) as run:

                # Log parameters
                mlflow.log_param("model_type", model_type)
                mlflow.log_param("target", self.target)
                mlflow.log_param("train_samples", len(X_train))
                mlflow.log_param("val_samples", len(X_val))
                mlflow.log_param("test_samples", len(X_test))
                mlflow.log_param("n_features", len(feature_names))

                if hyperparams:
                    for key, value in hyperparams.items():
                        mlflow.log_param(key, value)

                # Train model
                model = trainer.train(X_train, y_train, X_val, y_val)

                # Generate predictions
                predictions = trainer.predict(X_test)
                y_true = np.asarray(y_test.values)
                y_train_array = np.asarray(y_train.values)

                # Evaluate
                metrics = evaluate_forecast(
                    y_true=y_true,
                    y_pred=predictions,
                    y_train=y_train_array,
                    seasonality=24,
                    model_name=f"{model_type} - {self.target}"
                )

                # Add training time
                elapsed_time = time.time() - start_time
                metrics['training_time_seconds'] = elapsed_time

                # Log metrics to MLflow
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)):
                        mlflow.log_metric(metric_name, metric_value)

                # Get feature importance
                feature_importance = trainer.get_feature_importance()
                if feature_importance is not None:
                    # Save feature importance
                    temp_dir = tempfile.gettempdir()
                    importance_path = f"{temp_dir}/{model_type}_{self.target}_feature_importance.csv"
                    feature_importance.to_csv(importance_path, index=False)

                    try:
                        mlflow.log_artifact(importance_path)
                        logger.info(f"Feature importance logged to MLflow")
                    except (OSError, Exception) as e:
                        logger.info(f"Artifact saved locally (MLflow remote): {importance_path}")

                    # Log top features
                    top_10 = feature_importance.head(10)
                    logger.info("\nTop 10 Most Important Features:")
                    logger.info(top_10.to_string(index=False))

                    self.feature_importance_results[model_type] = feature_importance

                # Save predictions
                pred_df = pd.DataFrame({
                    'datetime': test_df.index,
                    'y_true': y_true,
                    'y_pred': predictions
                })
                temp_dir = tempfile.gettempdir()
                pred_path = f"{temp_dir}/{model_type}_{self.target}_predictions.csv"
                pred_df.to_csv(pred_path, index=False)

                try:
                    mlflow.log_artifact(pred_path)
                    logger.info(f"Predictions logged to MLflow")
                except (OSError, Exception) as e:
                    logger.info(f"Predictions saved locally (MLflow remote): {pred_path}")

                logger.info(f"\n✓ {model_type.upper()} completed in {elapsed_time:.2f}s")
                logger.info(f"  MLflow run ID: {run.info.run_id}")
                logger.info(f"  MAE: {metrics['MAE']:.2f}")
                logger.info(f"  RMSE: {metrics['RMSE']:.2f}")
                logger.info(f"  sMAPE: {metrics['sMAPE']:.2f}%")
                logger.info(f"  MASE: {metrics['MASE']:.4f}")

                self.results[model_type] = metrics
                return metrics

        except Exception as e:
            logger.error(f"✗ {model_type.upper()} failed: {e}", exc_info=True)
            self.results[model_type] = {'error': str(e)}
            return {'error': str(e)}

    def run_all_models(
        self,
        models: Optional[List[str]] = None,
        val_size: float = 0.1,
        test_size: float = 0.2
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run all baseline models.

        Args:
            models: List of models to run (default: all)
            val_size: Validation set size
            test_size: Test set size

        Returns:
            Dictionary with results for each model
        """
        if models is None:
            models = ['catboost', 'xgboost', 'lightgbm', 'prophet', 'sarimax']

        models = [m.lower() for m in models]

        logger.info(f"\n{'█'*80}")
        logger.info(f"FOREWATT BASELINE PIPELINE")
        logger.info(f"{'█'*80}")
        logger.info(f"Target: {self.target}")
        logger.info(f"Models: {', '.join(models).upper()}")
        logger.info(f"MLflow URI: {self.mlflow_uri}")
        logger.info(f"Experiment: {self.experiment_name}")
        logger.info(f"{'█'*80}")

        # Load data
        df = load_master_data()

        # Train-val-test split
        train_df, val_df, test_df = train_val_test_split(df, val_size, test_size)

        # Prepare target-specific data
        train_df, val_df, test_df = prepare_target_data(train_df, val_df, test_df, self.target)

        # Run each model
        for model_type in models:
            self.run_model(model_type, train_df, val_df, test_df)

        # Generate comparison report
        self.print_comparison_report()
        self.save_feature_importance_summary()

        return self.results

    def print_comparison_report(self):
        """Print comparison report of all models."""
        logger.info(f"\n\n{'█'*80}")
        logger.info(f"MODEL COMPARISON REPORT: {self.target.upper()}")
        logger.info(f"{'█'*80}")

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
            logger.info(f"\n{'─'*80}")
            best_model = df.iloc[0]['Model']
            best_mase = df.iloc[0]['MASE']
            best_smape = df.iloc[0]['sMAPE']
            logger.info(f"🏆 BEST MODEL: {best_model.upper()}")
            logger.info(f"   MASE: {best_mase:.4f} | sMAPE: {best_smape:.2f}%")
            logger.info(f"{'─'*80}")

            # Save report
            report_dir = PROJECT_ROOT / 'reports' / 'baseline'
            report_dir.mkdir(parents=True, exist_ok=True)

            report_path = report_dir / f'baseline_comparison_{self.target}.csv'
            df.to_csv(report_path, index=False)
            logger.info(f"\n✓ Comparison report saved to: {report_path}")

            # Save as JSON
            json_path = report_dir / f'baseline_results_{self.target}.json'
            with open(json_path, 'w') as f:
                json.dump(self.results, f, indent=2)
            logger.info(f"✓ Results JSON saved to: {json_path}")

        logger.info(f"{'█'*80}\n")

    def save_feature_importance_summary(self):
        """Save feature importance summary across all models."""
        if not self.feature_importance_results:
            logger.info("No feature importance results to save")
            return

        logger.info(f"\n{'='*80}")
        logger.info("FEATURE IMPORTANCE SUMMARY")
        logger.info(f"{'='*80}")

        # Aggregate feature importance across models
        all_features = {}
        for model_name, importance_df in self.feature_importance_results.items():
            for _, row in importance_df.iterrows():
                feature = row['feature']
                importance = row['importance']

                if feature not in all_features:
                    all_features[feature] = []
                all_features[feature].append((model_name, importance))

        # Create summary dataframe
        summary_data = []
        for feature, importances in all_features.items():
            avg_importance = np.mean([imp for _, imp in importances])
            model_count = len(importances)

            summary_data.append({
                'feature': feature,
                'avg_importance': avg_importance,
                'model_count': model_count,
                'models': ', '.join([model for model, _ in importances])
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('avg_importance', ascending=False)

        # Save summary
        report_dir = PROJECT_ROOT / 'reports' / 'baseline'
        report_dir.mkdir(parents=True, exist_ok=True)

        summary_path = report_dir / f'feature_importance_summary_{self.target}.csv'
        summary_df.to_csv(summary_path, index=False)

        logger.info(f"\nTop 20 Most Important Features (Averaged across models):")
        logger.info(summary_df.head(20)[['feature', 'avg_importance', 'model_count']].to_string(index=False))
        logger.info(f"\n✓ Feature importance summary saved to: {summary_path}")
        logger.info(f"{'='*80}\n")


def run_baseline_pipeline(
    targets: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    val_size: float = 0.1,
    test_size: float = 0.2,
    mlflow_uri: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Run complete baseline pipeline for specified targets.

    Args:
        targets: List of targets ['consumption', 'price_real'] (default: both)
        models: List of models to run (default: all)
        val_size: Validation set size
        test_size: Test set size
        mlflow_uri: MLflow tracking URI

    Returns:
        Dictionary with results for each target
    """
    if targets is None:
        targets = ['consumption', 'price_real']

    all_results = {}

    for target in targets:
        logger.info(f"\n\n{'#'*80}")
        logger.info(f"# STARTING PIPELINE FOR TARGET: {target.upper()}")
        logger.info(f"{'#'*80}\n")

        pipeline = BaselinePipeline(
            target=target,
            mlflow_uri=mlflow_uri
        )

        results = pipeline.run_all_models(
            models=models,
            val_size=val_size,
            test_size=test_size
        )

        all_results[target] = results

    # Print final summary
    logger.info(f"\n\n{'█'*80}")
    logger.info("FINAL SUMMARY: ALL TARGETS")
    logger.info(f"{'█'*80}")

    for target, results in all_results.items():
        successful_models = [m for m, r in results.items() if 'error' not in r]
        failed_models = [m for m, r in results.items() if 'error' in r]

        logger.info(f"\nTarget: {target.upper()}")
        logger.info(f"  ✓ Successful: {len(successful_models)} models - {', '.join(successful_models)}")
        if failed_models:
            logger.info(f"  ✗ Failed: {len(failed_models)} models - {', '.join(failed_models)}")

        # Best model
        if successful_models:
            best_model = min(
                successful_models,
                key=lambda m: results[m].get('MASE', float('inf'))
            )
            best_mase = results[best_model]['MASE']
            logger.info(f"  🏆 Best: {best_model.upper()} (MASE: {best_mase:.4f})")

    logger.info(f"{'█'*80}\n")

    return all_results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run baseline model pipeline')
    parser.add_argument('--targets', nargs='+',
                        choices=['consumption', 'price_real'],
                        help='Targets to predict (default: both)')
    parser.add_argument('--models', nargs='+',
                        choices=['catboost', 'xgboost', 'lightgbm', 'prophet', 'sarimax'],
                        help='Models to run (default: all)')
    parser.add_argument('--val-size', type=float, default=0.1,
                        help='Validation set size')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set size')
    parser.add_argument('--mlflow-uri', type=str, default=None,
                        help='MLflow tracking URI')

    args = parser.parse_args()

    # Run pipeline
    results = run_baseline_pipeline(
        targets=args.targets,
        models=args.models,
        val_size=args.val_size,
        test_size=args.test_size,
        mlflow_uri=args.mlflow_uri
    )

    # Exit with error if all models failed
    all_failed = all(
        all('error' in r for r in target_results.values())
        for target_results in results.values()
    )

    if all_failed:
        logger.error("All models failed for all targets!")
        sys.exit(1)
    else:
        logger.info("\n✓ Pipeline completed successfully")
        sys.exit(0)
