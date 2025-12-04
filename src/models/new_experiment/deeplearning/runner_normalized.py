"""
Normalized Target Runner for Deep Learning
===========================================
Uses relative price targets to eliminate distribution shift.

Key insight: Predicting price/rolling_mean instead of absolute price
reduces train-test distribution shift from 37% to <1%.

Usage:
    python runner_normalized.py --models nhits --max-configs 3
    python runner_normalized.py --compare  # Compare with original

Author: ForeWatt Team
Date: December 2025
"""

import sys
import os
import gc
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import logging
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from src.models.new_experiment.deeplearning.feature_preparer_normalized import (
    NormalizedFeaturePreparerV2,
    load_master_v2
)

from src.models.new_experiment.deeplearning.grid_config_generator_v2 import (
    get_full_grid,
    TARGET_STRATEGIES
)

try:
    from src.models.new_experiment.deeplearning.models.nhits_trainer import NHiTSTrainer
    from src.models.new_experiment.deeplearning.models.patchtst_trainer import PatchTSTTrainer
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


class NormalizedRunner:
    """
    Runner that uses normalized (relative) price targets.

    The model predicts price/rolling_mean, then converts back to absolute.
    """

    def __init__(
        self,
        output_dir: Path = None,
        device: str = None,
        val_size: float = 0.2,
        test_size: float = 0.2,
        normalization_window: int = 168
    ):
        self.output_dir = output_dir or PROJECT_ROOT / 'reports' / 'new_experiment' / 'normalized'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results_file = self.output_dir / 'results.csv'
        self.device = device
        self.val_size = val_size
        self.test_size = test_size
        self.normalization_window = normalization_window

        self.run_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        logger.info(f"NormalizedRunner initialized")
        logger.info(f"  Output: {self.output_dir}")
        logger.info(f"  Normalization window: {normalization_window}h")

    def _prepare_data(
        self,
        target: str = 'price_real',
        strategy: str = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
               pd.Series, pd.Series, pd.Series,
               NormalizedFeaturePreparerV2, pd.DataFrame]:
        """Prepare data with normalized target."""
        if strategy is None:
            strategy = TARGET_STRATEGIES.get(target, 'fundamental_v2')

        df = load_master_v2()

        # Use normalized preparer
        preparer = NormalizedFeaturePreparerV2(
            target=target,
            strategy=strategy,
            normalization_window=self.normalization_window,
            use_relative_target=True
        )

        X, y, features = preparer.prepare_features(df)

        # Drop NaN
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        df = df[mask]

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

        logger.info(f"Data prepared (normalized target):")
        logger.info(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        logger.info(f"  Features: {len(features)}")

        return X_train, X_val, X_test, y_train, y_val, y_test, preparer, df

    def _create_trainer(self, model_type: str, config: Dict, target: str):
        """Create model trainer."""
        input_size = config.get('input_size', 168)
        horizon = config.get('horizon', 24)

        if model_type == 'nhits':
            return NHiTSTrainer(
                target=target,
                horizon=horizon,
                input_size=input_size,
                random_seed=config.get('random_seed', 42),
                device=self.device
            )
        elif model_type == 'patchtst':
            return PatchTSTTrainer(
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

    def _config_to_hyperparams(self, model_type: str, config: Dict) -> Dict:
        """Convert config to hyperparameters."""
        if model_type == 'nhits':
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
        elif model_type == 'patchtst':
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
        return {}

    def run_single_config(
        self,
        config: Dict,
        X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame,
        y_train: pd.Series, y_val: pd.Series, y_test: pd.Series,
        preparer: NormalizedFeaturePreparerV2,
        df: pd.DataFrame,
        target: str = 'price_real'
    ) -> Dict[str, Any]:
        """Run training and evaluation for a single config."""
        model_type = config['model_type']
        config_hash = config['config_hash']

        logger.info(f"\n{'='*80}")
        logger.info(f"NORMALIZED TRAINING: {model_type.upper()} | Hash: {config_hash}")
        logger.info(f"{'='*80}")

        result = {
            'timestamp': datetime.now().isoformat(),
            'config_hash': config_hash,
            'model_type': model_type,
            'target': target,
            'normalization': 'relative',
            'feature_strategy': config.get('feature_strategy'),
            'feature_tier': config.get('feature_tier'),
        }

        start_time = time.time()

        try:
            # Create trainer
            trainer = self._create_trainer(model_type, config, target)
            hyperparams = self._config_to_hyperparams(model_type, config)

            # Train on NORMALIZED target
            model, val_metrics = trainer.train(
                X_train, y_train,
                X_val, y_val,
                hyperparams=hyperparams
            )

            # Get indices for test set
            test_indices = X_test.index

            # Predict (normalized scale)
            test_predictions_normalized = trainer.predict(X_test, y_test)

            if len(test_predictions_normalized.shape) > 1:
                test_predictions_normalized = test_predictions_normalized.flatten()

            # Convert back to absolute prices
            test_predictions_absolute = preparer.inverse_transform_predictions(
                test_predictions_normalized,
                indices=test_indices
            )

            # Get actual absolute prices for evaluation
            y_test_absolute = df.loc[test_indices, target].values

            # Align lengths
            min_len = min(len(test_predictions_absolute), len(y_test_absolute))
            test_predictions_absolute = test_predictions_absolute[:min_len]
            y_test_absolute = y_test_absolute[:min_len]

            # Calculate metrics on ABSOLUTE scale (for fair comparison)
            test_mae = mean_absolute_error(y_test_absolute, test_predictions_absolute)
            test_smape = symmetric_mean_absolute_percentage_error(y_test_absolute, test_predictions_absolute)

            # Get train absolute for MASE
            train_end = int(len(df) * (1 - self.val_size - self.test_size))
            y_train_absolute = df[target].iloc[:train_end].values
            test_mase = mean_absolute_scaled_error(y_test_absolute, test_predictions_absolute, y_train_absolute)

            # Also get validation metrics on absolute scale
            val_indices = X_val.index
            val_predictions_normalized = trainer.predict(X_val, y_val)
            if len(val_predictions_normalized.shape) > 1:
                val_predictions_normalized = val_predictions_normalized.flatten()
            val_predictions_absolute = preparer.inverse_transform_predictions(
                val_predictions_normalized,
                indices=val_indices
            )
            y_val_absolute = df.loc[val_indices, target].values[:len(val_predictions_absolute)]

            val_mae_absolute = mean_absolute_error(y_val_absolute, val_predictions_absolute)
            val_smape_absolute = symmetric_mean_absolute_percentage_error(y_val_absolute, val_predictions_absolute)

            result['val_mae'] = val_mae_absolute
            result['val_smape'] = val_smape_absolute
            result['test_mae'] = test_mae
            result['test_smape'] = test_smape
            result['test_mase'] = test_mase
            result['status'] = 'success'

            # Calculate gap
            gap_ratio = test_smape / val_smape_absolute if val_smape_absolute > 0 else float('inf')
            result['test_val_ratio'] = gap_ratio

            logger.info(f"\n📊 RESULTS (Absolute Scale):")
            logger.info(f"   Val  sMAPE: {val_smape_absolute:.2f}%")
            logger.info(f"   Test sMAPE: {test_smape:.2f}%")
            logger.info(f"   Test MAE:   {test_mae:.2f}")
            logger.info(f"   Gap ratio:  {gap_ratio:.2f}x")

            if gap_ratio < 1.5:
                logger.info(f"   ✅ GOOD GENERALIZATION (gap < 1.5x)")
            elif gap_ratio < 2.0:
                logger.info(f"   🟡 MODERATE GAP (1.5x-2.0x)")
            else:
                logger.info(f"   🔴 LARGE GAP (>2.0x)")

        except Exception as e:
            result['status'] = 'failed'
            result['error'] = str(e)
            logger.error(f"Training failed: {e}")
            import traceback
            traceback.print_exc()

        result['training_time_seconds'] = time.time() - start_time

        gc.collect()

        return result

    def _save_result(self, result: Dict):
        """Save result to CSV."""
        df = pd.DataFrame([result])
        if self.results_file.exists():
            df.to_csv(self.results_file, mode='a', header=False, index=False)
        else:
            df.to_csv(self.results_file, index=False)

    def run_experiment(
        self,
        model_types: List[str] = None,
        targets: List[str] = None,
        max_configs: int = 3,
        tier_filter: str = 'minimal'
    ):
        """Run normalized experiment."""
        logger.info("\n" + "="*80)
        logger.info("NORMALIZED TARGET EXPERIMENT")
        logger.info("Predicting price/rolling_mean to eliminate distribution shift")
        logger.info("="*80)

        if model_types is None:
            model_types = ['nhits']
        if targets is None:
            targets = ['price_real']

        all_results = []

        for target in targets:
            logger.info(f"\n{'#'*80}")
            logger.info(f"TARGET: {target}")
            logger.info(f"{'#'*80}")

            # Prepare data
            X_train, X_val, X_test, y_train, y_val, y_test, preparer, df = self._prepare_data(target=target)

            for model_type in model_types:
                configs = get_full_grid(model_type, target)

                if tier_filter:
                    configs = [c for c in configs if c.get('feature_tier') == tier_filter]

                configs = configs[:max_configs]

                logger.info(f"\nRunning {len(configs)} {model_type} configs")

                for config in configs:
                    result = self.run_single_config(
                        config,
                        X_train, X_val, X_test,
                        y_train, y_val, y_test,
                        preparer, df, target
                    )
                    self._save_result(result)
                    all_results.append(result)

        # Summary
        logger.info("\n" + "="*80)
        logger.info("EXPERIMENT COMPLETE")
        logger.info("="*80)

        successful = [r for r in all_results if r.get('status') == 'success']
        if successful:
            best = min(successful, key=lambda x: x.get('test_smape', float('inf')))
            avg_gap = np.mean([r.get('test_val_ratio', 2) for r in successful])

            logger.info(f"Successful runs: {len(successful)}")
            logger.info(f"Average gap ratio: {avg_gap:.2f}x")
            logger.info(f"\nBest config:")
            logger.info(f"  Model: {best['model_type']}")
            logger.info(f"  Test sMAPE: {best['test_smape']:.2f}%")
            logger.info(f"  Gap ratio: {best['test_val_ratio']:.2f}x")

        logger.info(f"\nResults: {self.results_file}")


def compare_approaches():
    """Compare original vs normalized approach on same config."""
    from src.models.new_experiment.deeplearning.feature_preparer_v2 import FundamentalFeaturePreparerV2

    print("\n" + "="*80)
    print("COMPARISON: Original vs Normalized Target")
    print("="*80)

    df = load_master_v2()
    n = len(df)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    # Original approach stats
    print("\n1. ORIGINAL (price_real):")
    train_mean = df['price_real'].iloc[:train_end].mean()
    val_mean = df['price_real'].iloc[train_end:val_end].mean()
    test_mean = df['price_real'].iloc[val_end:].mean()
    print(f"   Train mean: {train_mean:.2f}")
    print(f"   Val mean:   {val_mean:.2f}")
    print(f"   Test mean:  {test_mean:.2f}")
    print(f"   Shift:      {(test_mean - train_mean) / train_mean * 100:+.1f}%")

    # Normalized approach stats
    print("\n2. NORMALIZED (price / rolling_mean_168h):")
    df['relative'] = df['price_real'] / (df['price_ptf_rolling_mean_168h'] + 1)
    train_mean = df['relative'].iloc[:train_end].mean()
    val_mean = df['relative'].iloc[train_end:val_end].mean()
    test_mean = df['relative'].iloc[val_end:].mean()
    print(f"   Train mean: {train_mean:.4f}")
    print(f"   Val mean:   {val_mean:.4f}")
    print(f"   Test mean:  {test_mean:.4f}")
    print(f"   Shift:      {(test_mean - train_mean) / train_mean * 100:+.1f}%")

    print("\n" + "="*80)
    print("To run normalized experiment:")
    print("  python runner_normalized.py --models nhits --max-configs 3")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Normalized Target Runner')
    parser.add_argument('--models', nargs='+', choices=['nhits', 'patchtst', 'tft'], default=['nhits'])
    parser.add_argument('--targets', nargs='+', choices=['price_real', 'consumption'], default=['price_real'])
    parser.add_argument('--max-configs', type=int, default=3)
    parser.add_argument('--tier', type=str, default='minimal')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--compare', action='store_true', help='Compare original vs normalized')

    args = parser.parse_args()

    if args.compare:
        compare_approaches()
    else:
        runner = NormalizedRunner(device=args.device)
        runner.run_experiment(
            model_types=args.models,
            targets=args.targets,
            max_configs=args.max_configs,
            tier_filter=args.tier
        )


if __name__ == "__main__":
    main()
