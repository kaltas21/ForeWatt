"""
Walk-Forward Grid Search Runner for Deep Learning
==================================================
Addresses distribution shift and concept drift by using proper time-series
cross-validation instead of simple train/val/test split.

Key differences from runner_v2.py:
1. Walk-forward (expanding window) validation
2. Multiple test folds to detect regime changes
3. Robust metrics across time periods
4. Optional recent-data-only mode

Author: ForeWatt Team
Date: December 2025
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

warnings.filterwarnings('ignore')
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from src.models.new_experiment.deeplearning.feature_preparer_v2 import (
    FundamentalFeaturePreparerV2,
    load_master_v2
)

from src.models.new_experiment.deeplearning.grid_config_generator_v2 import (
    GridConfigGeneratorV2,
    get_full_grid,
    TARGETS,
    TARGET_STRATEGIES
)

try:
    from src.models.new_experiment.deeplearning.models.patchtst_trainer import PatchTSTTrainer
    from src.models.new_experiment.deeplearning.models.nhits_trainer import NHiTSTrainer
    from src.models.new_experiment.deeplearning.models.tft_trainer import TFTTrainer
except ImportError as e:
    logger.error(f"Failed to import trainers: {e}")
    raise

try:
    from src.models.evaluate import (
        mean_absolute_error,
        symmetric_mean_absolute_percentage_error,
        mean_absolute_scaled_error
    )
except ImportError:
    def mean_absolute_error(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    def symmetric_mean_absolute_percentage_error(y_true, y_pred):
        return 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

    def mean_absolute_scaled_error(y_true, y_pred, y_train, seasonality=24):
        naive_errors = np.abs(np.diff(y_train[::seasonality]))
        mae_naive = np.mean(naive_errors) if len(naive_errors) > 0 else 1.0
        return np.mean(np.abs(y_true - y_pred)) / max(mae_naive, 1e-8)


class WalkForwardRunner:
    """
    Walk-Forward validation runner for deep learning models.

    Addresses distribution shift by:
    1. Testing across multiple time periods
    2. Detecting performance degradation over time
    3. Providing robust cross-temporal metrics
    """

    def __init__(
        self,
        output_dir: Path = None,
        device: str = None,
        min_train_months: int = 24,      # Minimum 2 years training
        test_months: int = 3,            # 3 months per test fold
        step_months: int = 3,            # Step by 3 months
        max_folds: int = 4,              # Limit folds
        use_recent_only: bool = False,   # Only use data from 2023+
        experiment_name: str = "walkforward_experiment"
    ):
        """
        Initialize walk-forward runner.

        Args:
            output_dir: Output directory
            device: Device ('cuda', 'mps', 'cpu')
            min_train_months: Minimum training data in months
            test_months: Test period per fold in months
            step_months: Step size between folds in months
            max_folds: Maximum number of folds
            use_recent_only: If True, only use data from 2023 onwards
            experiment_name: Name for this experiment
        """
        self.output_dir = output_dir or PROJECT_ROOT / 'reports' / 'new_experiment' / 'walkforward'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results_file = self.output_dir / 'results.csv'
        self.device = device
        self.min_train_hours = min_train_months * 30 * 24  # Convert to hours
        self.test_hours = test_months * 30 * 24
        self.step_hours = step_months * 30 * 24
        self.max_folds = max_folds
        self.use_recent_only = use_recent_only
        self.experiment_name = experiment_name

        self.run_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Setup logging
        log_file = self.output_dir / f"run_{self.run_session_id}.log"
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

        logger.info(f"WalkForwardRunner initialized")
        logger.info(f"  Min train: {min_train_months} months")
        logger.info(f"  Test size: {test_months} months")
        logger.info(f"  Step size: {step_months} months")
        logger.info(f"  Max folds: {max_folds}")
        logger.info(f"  Recent only: {use_recent_only}")

    def _prepare_data(
        self,
        target: str = 'price_real',
        strategy: str = None
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Load and prepare full dataset."""
        logger.info(f"\nLoading data for target: {target}")

        if strategy is None:
            strategy = TARGET_STRATEGIES.get(target, 'fundamental_v2')

        df = load_master_v2()

        # Filter to recent data if requested
        if self.use_recent_only:
            cutoff = pd.Timestamp('2023-01-01', tz='UTC')
            df = df[df.index >= cutoff]
            logger.info(f"Filtered to recent data: {len(df)} samples from {df.index[0]}")

        preparer = FundamentalFeaturePreparerV2(
            target=target,
            strategy=strategy
        )

        X, y, feature_names = preparer.prepare_features(df)

        # Drop NaN
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]

        logger.info(f"Data prepared: {len(X)} samples, {len(feature_names)} features")

        return X, y, feature_names

    def _get_walk_forward_splits(
        self,
        n_samples: int
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
        """
        Generate walk-forward splits.

        Returns:
            List of (train_idx, val_idx, test_idx, fold_idx) tuples
        """
        splits = []
        current_train_end = self.min_train_hours
        fold_idx = 0

        # Reserve 10% of training for validation
        val_ratio = 0.1

        while current_train_end + self.test_hours <= n_samples:
            if fold_idx >= self.max_folds:
                break

            # Split training into train/val
            val_size = int(current_train_end * val_ratio)
            train_size = current_train_end - val_size

            train_idx = np.arange(0, train_size)
            val_idx = np.arange(train_size, current_train_end)
            test_idx = np.arange(current_train_end, min(current_train_end + self.test_hours, n_samples))

            splits.append((train_idx, val_idx, test_idx, fold_idx))

            current_train_end += self.step_hours
            fold_idx += 1

        return splits

    def _create_trainer(
        self,
        model_type: str,
        config: Dict[str, Any],
        target: str
    ):
        """Create trainer for model type."""
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
        """Convert config to trainer hyperparameters."""
        if model_type == 'patchtst':
            return {
                'patch_len': config['patch_len'],
                'stride': config['stride'],
                'encoder_layers': config['n_layers'],
                'hidden_size': config['d_model'],
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

    def run_single_config_walkforward(
        self,
        config: Dict[str, Any],
        X: pd.DataFrame,
        y: pd.Series,
        target: str = 'price_real'
    ) -> Dict[str, Any]:
        """
        Run walk-forward validation for a single configuration.

        Returns results with metrics per fold and aggregated metrics.
        """
        model_type = config['model_type']
        config_hash = config['config_hash']

        logger.info(f"\n{'='*80}")
        logger.info(f"WALK-FORWARD: {model_type.upper()} | Hash: {config_hash}")
        logger.info(f"{'='*80}")

        splits = self._get_walk_forward_splits(len(X))
        logger.info(f"Generated {len(splits)} walk-forward folds")

        fold_results = []
        start_time = time.time()

        for train_idx, val_idx, test_idx, fold_idx in splits:
            logger.info(f"\n--- Fold {fold_idx + 1}/{len(splits)} ---")
            logger.info(f"Train: {len(train_idx)} samples ({X.index[train_idx[0]]} to {X.index[train_idx[-1]]})")
            logger.info(f"Val:   {len(val_idx)} samples")
            logger.info(f"Test:  {len(test_idx)} samples ({X.index[test_idx[0]]} to {X.index[test_idx[-1]]})")

            # Get data for this fold
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

            try:
                # Create trainer
                trainer = self._create_trainer(model_type, config, target)
                hyperparams = self._config_to_hyperparams(model_type, config)

                # Train
                model, val_metrics = trainer.train(
                    X_train, y_train,
                    X_val, y_val,
                    hyperparams=hyperparams
                )

                # Predict on test
                test_predictions = trainer.predict(X_test, y_test)

                if len(test_predictions.shape) > 1:
                    test_predictions = test_predictions.flatten()

                min_len = min(len(test_predictions), len(y_test))
                test_predictions = test_predictions[:min_len]
                y_test_aligned = y_test.values[:min_len]

                # Calculate metrics
                test_mae = mean_absolute_error(y_test_aligned, test_predictions)
                test_smape = symmetric_mean_absolute_percentage_error(y_test_aligned, test_predictions)
                test_mase = mean_absolute_scaled_error(y_test_aligned, test_predictions, y_train.values)

                fold_result = {
                    'fold_idx': fold_idx,
                    'test_start': str(X.index[test_idx[0]]),
                    'test_end': str(X.index[test_idx[-1]]),
                    'train_size': len(train_idx),
                    'test_size': len(test_idx),
                    'val_mae': val_metrics.get('MAE', np.nan),
                    'val_smape': val_metrics.get('sMAPE', np.nan),
                    'test_mae': test_mae,
                    'test_smape': test_smape,
                    'test_mase': test_mase,
                    'status': 'success'
                }

                logger.info(f"Fold {fold_idx + 1} - Test sMAPE: {test_smape:.2f}%, MAE: {test_mae:.2f}")

            except Exception as e:
                logger.error(f"Fold {fold_idx + 1} failed: {e}")
                fold_result = {
                    'fold_idx': fold_idx,
                    'status': 'failed',
                    'error': str(e)
                }

            fold_results.append(fold_result)

            # Cleanup
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

        # Aggregate results
        successful_folds = [f for f in fold_results if f.get('status') == 'success']

        if successful_folds:
            test_maes = [f['test_mae'] for f in successful_folds]
            test_smapes = [f['test_smape'] for f in successful_folds]
            test_mases = [f['test_mase'] for f in successful_folds]

            result = {
                'timestamp': datetime.now().isoformat(),
                'config_hash': config_hash,
                'model_type': model_type,
                'target': target,
                'feature_strategy': config.get('feature_strategy'),
                'feature_tier': config.get('feature_tier'),
                'n_folds': len(successful_folds),
                'test_mae_mean': np.mean(test_maes),
                'test_mae_std': np.std(test_maes),
                'test_smape_mean': np.mean(test_smapes),
                'test_smape_std': np.std(test_smapes),
                'test_mase_mean': np.mean(test_mases),
                'test_mase_std': np.std(test_mases),
                'test_smape_min': np.min(test_smapes),
                'test_smape_max': np.max(test_smapes),
                'fold_details': json.dumps(fold_results),
                'training_time_seconds': time.time() - start_time,
                'status': 'success',
                'config_json': json.dumps(config, default=str)
            }

            # Detect concept drift (is performance degrading over time?)
            if len(test_smapes) >= 3:
                trend = np.polyfit(range(len(test_smapes)), test_smapes, 1)[0]
                result['smape_trend'] = trend  # Positive = degrading
                result['drift_detected'] = trend > 1.0  # >1% increase per fold

            logger.info(f"\n{'='*80}")
            logger.info(f"WALK-FORWARD COMPLETE: {model_type.upper()}")
            logger.info(f"{'='*80}")
            logger.info(f"Test sMAPE: {result['test_smape_mean']:.2f}% ± {result['test_smape_std']:.2f}%")
            logger.info(f"Test MAE:   {result['test_mae_mean']:.2f} ± {result['test_mae_std']:.2f}")
            logger.info(f"Range:      {result['test_smape_min']:.2f}% - {result['test_smape_max']:.2f}%")
            if 'drift_detected' in result:
                logger.info(f"Drift:      {'YES' if result['drift_detected'] else 'NO'} (trend={result.get('smape_trend', 0):.2f})")

        else:
            result = {
                'timestamp': datetime.now().isoformat(),
                'config_hash': config_hash,
                'model_type': model_type,
                'target': target,
                'status': 'failed',
                'error': 'All folds failed'
            }

        return result

    def _save_result(self, result: Dict[str, Any]):
        """Save result to CSV."""
        # Remove fold_details from CSV (too long), save separately
        result_for_csv = {k: v for k, v in result.items() if k != 'fold_details'}

        df = pd.DataFrame([result_for_csv])
        if self.results_file.exists():
            df.to_csv(self.results_file, mode='a', header=False, index=False)
        else:
            df.to_csv(self.results_file, index=False)

        # Save full result with fold details as JSON
        config_hash = result.get('config_hash', 'unknown')
        json_file = self.output_dir / f"metrics_{config_hash}.json"
        with open(json_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)

    def run_experiment(
        self,
        model_types: List[str] = None,
        targets: List[str] = None,
        max_configs_per_model: int = 3,  # Limit configs since WF is expensive
        config_filter: str = 'minimal'   # Only run minimal tier for speed
    ):
        """
        Run walk-forward experiment.

        Args:
            model_types: Model types to run
            targets: Targets to run
            max_configs_per_model: Max configs per model type (WF is expensive)
            config_filter: Feature tier filter ('minimal', 'core', 'extended', 'full', or None)
        """
        logger.info("\n" + "="*80)
        logger.info("WALK-FORWARD EXPERIMENT")
        logger.info("Robust validation across time periods")
        logger.info("="*80)

        if model_types is None:
            model_types = ['nhits']  # Start with best model
        if targets is None:
            targets = ['price_real']

        total_start = time.time()
        all_results = []

        for target in targets:
            logger.info(f"\n{'#'*80}")
            logger.info(f"TARGET: {target}")
            logger.info(f"{'#'*80}")

            # Prepare data once per target
            strategy = TARGET_STRATEGIES.get(target, 'fundamental_v2')
            X, y, features = self._prepare_data(target=target, strategy=strategy)

            for model_type in model_types:
                logger.info(f"\n--- Model: {model_type} ---")

                # Get configs
                all_configs = get_full_grid(model_type, target)

                # Filter by feature tier if specified
                if config_filter:
                    all_configs = [c for c in all_configs if c.get('feature_tier') == config_filter]

                # Limit configs
                configs_to_run = all_configs[:max_configs_per_model]

                logger.info(f"Running {len(configs_to_run)} configs (filtered by tier='{config_filter}')")

                for i, config in enumerate(configs_to_run):
                    logger.info(f"\nConfig {i+1}/{len(configs_to_run)}: {config['config_hash']}")

                    # Need to reload data with correct strategy for this config
                    config_strategy = config.get('feature_strategy', strategy)
                    if config_strategy != strategy:
                        X, y, features = self._prepare_data(target=target, strategy=config_strategy)
                        strategy = config_strategy

                    result = self.run_single_config_walkforward(config, X, y, target)
                    self._save_result(result)
                    all_results.append(result)

        # Final summary
        total_time = time.time() - total_start

        logger.info("\n" + "="*80)
        logger.info("EXPERIMENT COMPLETE")
        logger.info("="*80)
        logger.info(f"Total time: {total_time/60:.1f} minutes")
        logger.info(f"Results: {self.results_file}")

        # Print best results
        successful = [r for r in all_results if r.get('status') == 'success']
        if successful:
            best = min(successful, key=lambda x: x.get('test_smape_mean', float('inf')))
            logger.info(f"\nBest configuration:")
            logger.info(f"  Model: {best['model_type']}")
            logger.info(f"  Hash: {best['config_hash']}")
            logger.info(f"  sMAPE: {best['test_smape_mean']:.2f}% ± {best['test_smape_std']:.2f}%")

            # Check drift
            drift_models = [r for r in successful if r.get('drift_detected', False)]
            if drift_models:
                logger.info(f"\n⚠️  {len(drift_models)} configs show concept drift!")


def main():
    parser = argparse.ArgumentParser(description='Walk-Forward Deep Learning Experiment')
    parser.add_argument('--models', nargs='+', choices=['patchtst', 'nhits', 'tft'], default=['nhits'])
    parser.add_argument('--targets', nargs='+', choices=['price_real', 'consumption'], default=['price_real'])
    parser.add_argument('--max-configs', type=int, default=3, help='Max configs per model')
    parser.add_argument('--tier', type=str, default='minimal', choices=['minimal', 'core', 'extended', 'full', 'all'])
    parser.add_argument('--folds', type=int, default=4, help='Max walk-forward folds')
    parser.add_argument('--recent-only', action='store_true', help='Only use data from 2023+')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'mps', 'cpu'])

    args = parser.parse_args()

    runner = WalkForwardRunner(
        device=args.device,
        max_folds=args.folds,
        use_recent_only=args.recent_only
    )

    tier_filter = None if args.tier == 'all' else args.tier

    runner.run_experiment(
        model_types=args.models,
        targets=args.targets,
        max_configs_per_model=args.max_configs,
        config_filter=tier_filter
    )


if __name__ == "__main__":
    main()
