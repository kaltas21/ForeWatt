"""
Feature Search Runner
======================
Tests which features/strategies work best with winning model configurations.

Frozen model hyperparameters, only varies:
- Feature strategies (minimal, core, extended, full)
- Normalization (absolute vs relative target)

Author: ForeWatt Team
Date: December 2025
"""

import sys
import os
import gc
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import logging
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# WINNING MODEL CONFIGURATIONS (FROZEN)
# =============================================================================

WINNING_CONFIGS = {
    'price_real': {
        # Best DL: N-HiTS (16.01% sMAPE)
        'nhits': {
            'config_hash': '9626ebea245b',
            'input_size': 72,
            'horizon': 24,
            'n_blocks': [1, 1],
            'hidden_size': 128,
            'n_mlp_layers': 2,
            'n_pool_kernel_size': [2, 1],
            'n_freq_downsample': [2, 1],
            'batch_size': 512,
            'learning_rate': 0.002,
            'max_steps': 300,
            'early_stop_patience_steps': 100,
        },
        # Best baseline: LightGBM (16.83% sMAPE)
        'lightgbm': {
            'config_hash': '775364cc5f8d',
            'n_estimators': 800,
            'max_depth': 8,
            'learning_rate': 0.05,
            'num_leaves': 63,
            'min_child_samples': 30,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
        },
        # 3rd: XGBoost (16.84% sMAPE)
        'xgboost': {
            'config_hash': 'bd09c11ca8e3',
            'n_estimators': 500,
            'max_depth': 8,
            'learning_rate': 0.05,
            'min_child_weight': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
        },
    },
    'consumption': {
        # Best: CatBoost (2.03% sMAPE)
        'catboost': {
            'config_hash': '8327b57030a0',
            'iterations': 1000,
            'depth': 8,
            'learning_rate': 0.05,
            'l2_leaf_reg': 3,
            'random_strength': 1,
        },
        # 2nd: LightGBM (2.04% sMAPE)
        'lightgbm': {
            'config_hash': '1c06e2e4e41f',
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'min_child_samples': 50,
            'reg_alpha': 0.5,
            'reg_lambda': 0.5,
        },
    }
}

# =============================================================================
# FEATURE STRATEGIES TO TEST
# =============================================================================

FEATURE_STRATEGIES_PRICE = [
    'fundamental_v2_minimal',  # ~20 features
    'fundamental_v2',          # ~35 features (core)
    'fundamental_v2_extended', # ~50 features
    'fundamental_v2_full',     # ~65 features
]

FEATURE_STRATEGIES_CONSUMPTION = [
    'consumption_v2_minimal',  # ~15 features
    'consumption_v2',          # ~25 features
    'consumption_v2_extended', # ~40 features
    'consumption_v2_full',     # ~55 features
]

# =============================================================================
# IMPORTS
# =============================================================================

from src.models.new_experiment.deeplearning.feature_preparer_v2 import (
    FundamentalFeaturePreparerV2,
    load_master_v2
)

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


class FeatureSearchRunner:
    """Fast feature search with frozen winning model configs."""

    def __init__(
        self,
        output_dir: Path = None,
        device: str = None,
        val_size: float = 0.2,
        test_size: float = 0.2
    ):
        self.output_dir = output_dir or PROJECT_ROOT / 'reports' / 'feature_search'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results_file = self.output_dir / 'results.csv'
        self.device = device
        self.val_size = val_size
        self.test_size = test_size

        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        logger.info(f"FeatureSearchRunner initialized")
        logger.info(f"  Output: {self.output_dir}")

    def _prepare_data(
        self,
        target: str,
        strategy: str,
        use_normalized: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
               pd.Series, pd.Series, pd.Series, List[str]]:
        """Prepare data with specified feature strategy."""
        df = load_master_v2()

        preparer = FundamentalFeaturePreparerV2(
            target=target,
            strategy=strategy
        )

        X, y, features = preparer.prepare_features(df)

        # Normalized target option
        if use_normalized:
            rolling_mean = df['price_ptf_rolling_mean_168h'] if target == 'price_real' else df['consumption_rolling_mean_168h']
            y = y / (rolling_mean.loc[y.index] + 1e-8)

        # Drop NaN
        mask = ~(X.isna().any(axis=1) | y.isna())
        X, y = X[mask], y[mask]

        # Split
        n = len(X)
        train_end = int(n * (1 - self.val_size - self.test_size))
        val_end = int(n * (1 - self.test_size))

        return (
            X.iloc[:train_end], X.iloc[train_end:val_end], X.iloc[val_end:],
            y.iloc[:train_end], y.iloc[train_end:val_end], y.iloc[val_end:],
            features
        )

    def _train_lightgbm(
        self,
        X_train, y_train, X_val, y_val,
        config: Dict
    ):
        """Train LightGBM model."""
        import lightgbm as lgb

        params = {
            'objective': 'regression',
            'metric': 'mae',
            'verbosity': -1,
            'n_estimators': config.get('n_estimators', 500),
            'max_depth': config.get('max_depth', 6),
            'learning_rate': config.get('learning_rate', 0.05),
            'num_leaves': config.get('num_leaves', 31),
            'min_child_samples': config.get('min_child_samples', 30),
            'reg_alpha': config.get('reg_alpha', 0.1),
            'reg_lambda': config.get('reg_lambda', 0.1),
            'random_state': 42,
        }

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        return model

    def _train_xgboost(
        self,
        X_train, y_train, X_val, y_val,
        config: Dict
    ):
        """Train XGBoost model."""
        import xgboost as xgb

        model = xgb.XGBRegressor(
            n_estimators=config.get('n_estimators', 500),
            max_depth=config.get('max_depth', 6),
            learning_rate=config.get('learning_rate', 0.05),
            min_child_weight=config.get('min_child_weight', 5),
            subsample=config.get('subsample', 0.8),
            colsample_bytree=config.get('colsample_bytree', 0.8),
            reg_alpha=config.get('reg_alpha', 0.1),
            reg_lambda=config.get('reg_lambda', 1.0),
            random_state=42,
            verbosity=0,
            early_stopping_rounds=50,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return model

    def _train_catboost(
        self,
        X_train, y_train, X_val, y_val,
        config: Dict
    ):
        """Train CatBoost model."""
        from catboost import CatBoostRegressor

        model = CatBoostRegressor(
            iterations=config.get('iterations', 500),
            depth=config.get('depth', 6),
            learning_rate=config.get('learning_rate', 0.05),
            l2_leaf_reg=config.get('l2_leaf_reg', 3),
            random_strength=config.get('random_strength', 1),
            random_seed=42,
            verbose=False,
            early_stopping_rounds=50,
        )
        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        return model

    def _train_nhits(
        self,
        X_train, y_train, X_val, y_val,
        config: Dict, target: str
    ):
        """Train N-HiTS model."""
        from src.models.new_experiment.deeplearning.models.nhits_trainer import NHiTSTrainer

        trainer = NHiTSTrainer(
            target=target,
            horizon=config.get('horizon', 24),
            input_size=config.get('input_size', 72),
            random_seed=42,
            device=self.device
        )

        hyperparams = {
            'n_blocks': config['n_blocks'],
            'hidden_size': config['hidden_size'],
            'n_mlp_layers': config['n_mlp_layers'],
            'n_pool_kernel_size': config['n_pool_kernel_size'],
            'n_freq_downsample': config['n_freq_downsample'],
            'learning_rate': config['learning_rate'],
            'batch_size': config['batch_size'],
            'max_steps': config['max_steps'],
            'early_stop_patience_steps': config['early_stop_patience_steps'],
        }

        model, _ = trainer.train(X_train, y_train, X_val, y_val, hyperparams)
        return model, trainer

    def run_single_test(
        self,
        model_type: str,
        target: str,
        strategy: str,
        use_normalized: bool = False
    ) -> Dict[str, Any]:
        """Run single feature strategy test."""
        config = WINNING_CONFIGS[target].get(model_type)
        if config is None:
            return {'status': 'skipped', 'reason': f'No config for {model_type}/{target}'}

        logger.info(f"\n{'='*60}")
        logger.info(f"{model_type.upper()} | {target} | {strategy}")
        logger.info(f"Normalized: {use_normalized}")
        logger.info(f"{'='*60}")

        result = {
            'timestamp': datetime.now().isoformat(),
            'model_type': model_type,
            'target': target,
            'feature_strategy': strategy,
            'use_normalized': use_normalized,
        }

        start = time.time()

        try:
            # Prepare data
            X_train, X_val, X_test, y_train, y_val, y_test, features = \
                self._prepare_data(target, strategy, use_normalized)

            result['n_features'] = len(features)
            result['n_train'] = len(X_train)

            # Train model
            if model_type == 'lightgbm':
                model = self._train_lightgbm(X_train, y_train, X_val, y_val, config)
                val_pred = model.predict(X_val)
                test_pred = model.predict(X_test)
            elif model_type == 'xgboost':
                model = self._train_xgboost(X_train, y_train, X_val, y_val, config)
                val_pred = model.predict(X_val)
                test_pred = model.predict(X_test)
            elif model_type == 'catboost':
                model = self._train_catboost(X_train, y_train, X_val, y_val, config)
                val_pred = model.predict(X_val)
                test_pred = model.predict(X_test)
            elif model_type == 'nhits':
                model, trainer = self._train_nhits(X_train, y_train, X_val, y_val, config, target)
                val_pred = trainer.predict(X_val, y_val).flatten()[:len(y_val)]
                test_pred = trainer.predict(X_test, y_test).flatten()[:len(y_test)]
            else:
                raise ValueError(f"Unknown model: {model_type}")

            # Metrics
            result['val_mae'] = mean_absolute_error(y_val.values, val_pred)
            result['val_smape'] = symmetric_mean_absolute_percentage_error(y_val.values, val_pred)
            result['test_mae'] = mean_absolute_error(y_test.values, test_pred)
            result['test_smape'] = symmetric_mean_absolute_percentage_error(y_test.values, test_pred)
            result['test_mase'] = mean_absolute_scaled_error(y_test.values, test_pred, y_train.values)

            result['gap_ratio'] = result['test_smape'] / result['val_smape'] if result['val_smape'] > 0 else 0
            result['status'] = 'success'

            logger.info(f"  Features: {result['n_features']}")
            logger.info(f"  Val sMAPE:  {result['val_smape']:.2f}%")
            logger.info(f"  Test sMAPE: {result['test_smape']:.2f}%")
            logger.info(f"  Gap ratio:  {result['gap_ratio']:.2f}x")

        except Exception as e:
            result['status'] = 'failed'
            result['error'] = str(e)
            logger.error(f"  Failed: {e}")

        result['time_seconds'] = time.time() - start
        gc.collect()

        return result

    def _save_result(self, result: Dict):
        """Save result to CSV."""
        df = pd.DataFrame([result])
        if self.results_file.exists():
            df.to_csv(self.results_file, mode='a', header=False, index=False)
        else:
            df.to_csv(self.results_file, index=False)

    def run_search(
        self,
        targets: List[str] = None,
        models: List[str] = None,
        test_normalized: bool = True
    ):
        """Run full feature search."""
        logger.info("\n" + "="*70)
        logger.info("FEATURE SEARCH - Finding Best Feature Strategy")
        logger.info("="*70)

        if targets is None:
            targets = ['price_real']
        if models is None:
            models = ['lightgbm', 'xgboost', 'nhits']

        all_results = []
        total_start = time.time()

        for target in targets:
            strategies = FEATURE_STRATEGIES_PRICE if target == 'price_real' else FEATURE_STRATEGIES_CONSUMPTION
            available_models = [m for m in models if m in WINNING_CONFIGS.get(target, {})]

            logger.info(f"\n{'#'*70}")
            logger.info(f"TARGET: {target}")
            logger.info(f"Models: {available_models}")
            logger.info(f"Strategies: {strategies}")
            logger.info(f"{'#'*70}")

            for model_type in available_models:
                for strategy in strategies:
                    # Test absolute target
                    result = self.run_single_test(model_type, target, strategy, use_normalized=False)
                    self._save_result(result)
                    all_results.append(result)

                    # Test normalized target
                    if test_normalized and target == 'price_real':
                        result = self.run_single_test(model_type, target, strategy, use_normalized=True)
                        self._save_result(result)
                        all_results.append(result)

        # Summary
        total_time = time.time() - total_start
        successful = [r for r in all_results if r.get('status') == 'success']

        logger.info("\n" + "="*70)
        logger.info("FEATURE SEARCH COMPLETE")
        logger.info("="*70)
        logger.info(f"Total time: {total_time/60:.1f} minutes")
        logger.info(f"Successful: {len(successful)}/{len(all_results)}")

        if successful:
            # Best by test sMAPE
            best = min(successful, key=lambda x: x.get('test_smape', float('inf')))
            logger.info(f"\n🏆 BEST CONFIGURATION:")
            logger.info(f"   Model: {best['model_type']}")
            logger.info(f"   Strategy: {best['feature_strategy']}")
            logger.info(f"   Normalized: {best['use_normalized']}")
            logger.info(f"   Features: {best['n_features']}")
            logger.info(f"   Test sMAPE: {best['test_smape']:.2f}%")
            logger.info(f"   Gap ratio: {best['gap_ratio']:.2f}x")

            # Best by gap ratio (generalization)
            best_gap = min(successful, key=lambda x: x.get('gap_ratio', float('inf')))
            logger.info(f"\n🎯 BEST GENERALIZATION:")
            logger.info(f"   Model: {best_gap['model_type']}")
            logger.info(f"   Strategy: {best_gap['feature_strategy']}")
            logger.info(f"   Normalized: {best_gap['use_normalized']}")
            logger.info(f"   Gap ratio: {best_gap['gap_ratio']:.2f}x")

        # Save summary
        self._save_summary(all_results, total_time)

        logger.info(f"\nResults: {self.results_file}")

    def _save_summary(self, results: List[Dict], total_time: float):
        """Save summary JSON."""
        successful = [r for r in results if r.get('status') == 'success']

        summary = {
            'run_id': self.run_id,
            'timestamp': datetime.now().isoformat(),
            'total_tests': len(results),
            'successful': len(successful),
            'total_time_seconds': total_time,
        }

        if successful:
            # Group by target
            for target in set(r['target'] for r in successful):
                target_results = [r for r in successful if r['target'] == target]
                best = min(target_results, key=lambda x: x['test_smape'])

                summary[f'best_{target}'] = {
                    'model': best['model_type'],
                    'strategy': best['feature_strategy'],
                    'normalized': best['use_normalized'],
                    'n_features': best['n_features'],
                    'test_smape': best['test_smape'],
                    'gap_ratio': best['gap_ratio'],
                }

        with open(self.output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Feature Search')
    parser.add_argument('--targets', nargs='+', default=['price_real'],
                       choices=['price_real', 'consumption'])
    parser.add_argument('--models', nargs='+', default=['lightgbm', 'xgboost'],
                       choices=['lightgbm', 'xgboost', 'catboost', 'nhits'])
    parser.add_argument('--no-normalized', action='store_true',
                       help='Skip normalized target tests')
    parser.add_argument('--device', type=str, default=None)

    args = parser.parse_args()

    runner = FeatureSearchRunner(device=args.device)
    runner.run_search(
        targets=args.targets,
        models=args.models,
        test_normalized=not args.no_normalized
    )


if __name__ == "__main__":
    main()
