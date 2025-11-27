"""
Deep Learning Test Runner
=========================
Quick test runner to check for OOM or other errors before full grid search.
Does NOT save results or produce reports.

This test runner uses the EXACT same hyperparameter mappings as runner_v2.py
to ensure any errors caught here will also occur in the real grid search.

Usage:
    python -m src.models.new_experiment.deeplearning.test_runner
    python -m src.models.new_experiment.deeplearning.test_runner --models patchtst nhits
    python -m src.models.new_experiment.deeplearning.test_runner --target consumption
"""

import sys
import gc
import argparse
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
import warnings

import pandas as pd

warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_imports():
    """Test all required imports."""
    logger.info("=" * 60)
    logger.info("TESTING IMPORTS")
    logger.info("=" * 60)

    errors = []

    # Test PyTorch
    try:
        import torch
        logger.info(f"[OK] PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            logger.info(f"     CUDA available: {torch.cuda.get_device_name(0)}")
            logger.info(f"     VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("     MPS (Apple Silicon) available")
        else:
            logger.info("     CPU only")
    except ImportError as e:
        errors.append(f"PyTorch: {e}")
        logger.error(f"[FAIL] PyTorch: {e}")

    # Test NeuralForecast
    try:
        from neuralforecast import NeuralForecast
        from neuralforecast.models import PatchTST, NHITS, TFT
        import neuralforecast
        logger.info(f"[OK] NeuralForecast {neuralforecast.__version__}")
    except ImportError as e:
        errors.append(f"NeuralForecast: {e}")
        logger.error(f"[FAIL] NeuralForecast: {e}")

    # Test feature preparer
    try:
        from src.models.new_experiment.deeplearning.feature_preparer_v2 import (
            FundamentalFeaturePreparerV2,
            load_master_v2
        )
        logger.info("[OK] Feature preparer")
    except ImportError as e:
        errors.append(f"Feature preparer: {e}")
        logger.error(f"[FAIL] Feature preparer: {e}")

    # Test trainers
    try:
        from src.models.new_experiment.deeplearning.models.patchtst_trainer import PatchTSTTrainer
        logger.info("[OK] PatchTST trainer")
    except ImportError as e:
        errors.append(f"PatchTST trainer: {e}")
        logger.error(f"[FAIL] PatchTST trainer: {e}")

    try:
        from src.models.new_experiment.deeplearning.models.nhits_trainer import NHiTSTrainer
        logger.info("[OK] NHiTS trainer")
    except ImportError as e:
        errors.append(f"NHiTS trainer: {e}")
        logger.error(f"[FAIL] NHiTS trainer: {e}")

    try:
        from src.models.new_experiment.deeplearning.models.tft_trainer import TFTTrainer
        logger.info("[OK] TFT trainer")
    except ImportError as e:
        errors.append(f"TFT trainer: {e}")
        logger.error(f"[FAIL] TFT trainer: {e}")

    # Test grid config generator
    try:
        from src.models.new_experiment.deeplearning.grid_config_generator_v2 import (
            GridConfigGeneratorV2, TARGET_STRATEGIES
        )
        logger.info("[OK] Grid config generator")
    except ImportError as e:
        errors.append(f"Grid config generator: {e}")
        logger.error(f"[FAIL] Grid config generator: {e}")

    if errors:
        logger.error(f"\n{len(errors)} import errors found!")
        return False

    logger.info("\nAll imports successful!")
    return True


def test_data_loading(target: str = 'price_real') -> Optional[Tuple]:
    """Test data loading and preparation."""
    logger.info("\n" + "=" * 60)
    logger.info(f"TESTING DATA LOADING: {target}")
    logger.info("=" * 60)

    try:
        from src.models.new_experiment.deeplearning.feature_preparer_v2 import (
            FundamentalFeaturePreparerV2,
            load_master_v2
        )
        from src.models.new_experiment.deeplearning.grid_config_generator_v2 import TARGET_STRATEGIES

        # Load data
        df = load_master_v2()
        logger.info(f"[OK] Loaded master data: {df.shape}")

        # Get strategy for target
        strategy = TARGET_STRATEGIES.get(target, 'fundamental_v2')
        logger.info(f"     Strategy: {strategy}")

        # Prepare features
        preparer = FundamentalFeaturePreparerV2(target=target, strategy=strategy)
        X, y, feature_names = preparer.prepare_features(df)

        # Drop NaN
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]

        logger.info(f"[OK] Prepared features: X={X.shape}, y={len(y)}")
        logger.info(f"     Features: {len(feature_names)}")
        logger.info(f"     Date range: {X.index.min()} to {X.index.max()}")

        # Split (same as runner_v2: 60/20/20)
        n = len(X)
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)

        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
        X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

        logger.info(f"[OK] Data splits: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    except Exception as e:
        logger.error(f"[FAIL] Data loading: {e}")
        traceback.print_exc()
        return None


def config_to_hyperparams(model_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert grid config to trainer hyperparameters.

    EXACT COPY from runner_v2.py._config_to_hyperparams to ensure correlation.
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


def get_test_config(model_type: str) -> Dict[str, Any]:
    """
    Get a minimal test config that matches the grid config structure.
    Uses EXACT same keys as the grid configs in grid_config_generator_v2.py,
    but with minimal steps for quick testing.
    """
    base = {
        'input_size': 168,
        'horizon': 24,
        'learning_rate': 1e-3,
        'batch_size': 64,
        'max_steps': 10,  # Minimal steps for testing
        'early_stop_patience_steps': 5,
    }

    if model_type == 'patchtst':
        # Uses same keys as PATCHTST_CONFIGS
        return {
            **base,
            'patch_len': 12,
            'stride': 12,
            'n_layers': 2,      # Grid uses n_layers (runner maps to encoder_layers)
            'd_model': 64,      # Grid uses d_model (runner maps to hidden_size)
            'n_heads': 4,
            'dropout': 0.1,
        }
    elif model_type == 'nhits':
        # Uses same keys as NHITS_CONFIGS
        return {
            **base,
            'n_blocks': [1, 1, 1],
            'hidden_size': 64,
            'n_mlp_layers': 2,
            'n_pool_kernel_size': [2, 2, 1],
            'n_freq_downsample': [2, 1, 1],
        }
    elif model_type == 'tft':
        # Uses same keys as TFT_CONFIGS
        return {
            **base,
            'hidden_size': 32,
            'n_head': 2,
            'lstm_n_layers': 1,
            'dropout': 0.1,
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def test_model_training(
    model_type: str,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    target: str = 'price_real',
    device: Optional[str] = None
) -> bool:
    """
    Test training a single model with minimal steps.
    Uses the EXACT same flow as runner_v2.py._run_single_config.
    """
    logger.info("\n" + "=" * 60)
    logger.info(f"TESTING {model_type.upper()} TRAINING")
    logger.info("=" * 60)

    try:
        # Get test config (same structure as grid configs)
        config = get_test_config(model_type)

        # Convert to hyperparams (same as runner_v2)
        hyperparams = config_to_hyperparams(model_type, config)

        logger.info(f"     Grid config keys: {list(config.keys())}")
        logger.info(f"     Trainer hyperparams: {hyperparams}")

        # Create trainer (same as runner_v2._create_trainer)
        input_size = config.get('input_size', 168)
        horizon = config.get('horizon', 24)

        if model_type == 'patchtst':
            from src.models.new_experiment.deeplearning.models.patchtst_trainer import PatchTSTTrainer
            trainer = PatchTSTTrainer(
                target=target,
                horizon=horizon,
                input_size=input_size,
                random_seed=42,
                device=device
            )
        elif model_type == 'nhits':
            from src.models.new_experiment.deeplearning.models.nhits_trainer import NHiTSTrainer
            trainer = NHiTSTrainer(
                target=target,
                horizon=horizon,
                input_size=input_size,
                random_seed=42,
                device=device
            )
        elif model_type == 'tft':
            from src.models.new_experiment.deeplearning.models.tft_trainer import TFTTrainer
            trainer = TFTTrainer(
                target=target,
                horizon=horizon,
                input_size=input_size,
                random_seed=42,
                device=device
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        logger.info(f"[OK] Created {model_type} trainer")
        logger.info(f"     Starting training (max_steps={hyperparams['max_steps']})...")

        # Train model (same as runner_v2)
        model, val_metrics = trainer.train(
            X_train, y_train,
            X_val, y_val,
            hyperparams=hyperparams
        )

        logger.info(f"[OK] Training completed!")
        logger.info(f"     Val MAE: {val_metrics.get('MAE', 'N/A'):.2f}")
        logger.info(f"     Val sMAPE: {val_metrics.get('sMAPE', 'N/A'):.2f}%")

        # Test prediction (same as runner_v2)
        logger.info("     Testing prediction...")
        predictions = trainer.predict(X_test, y_test)
        logger.info(f"[OK] Prediction shape: {predictions.shape}")

        # Clean up
        del model, trainer
        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                mem_allocated = torch.cuda.memory_allocated() / 1e9
                mem_reserved = torch.cuda.memory_reserved() / 1e9
                logger.info(f"     GPU memory: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")
        except:
            pass

        return True

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error(f"[FAIL] OOM ERROR: {e}")
        else:
            logger.error(f"[FAIL] Runtime error: {e}")
        traceback.print_exc()
        return False
    except Exception as e:
        logger.error(f"[FAIL] {model_type} training: {e}")
        traceback.print_exc()
        return False


def run_tests(
    model_types: List[str] = None,
    target: str = 'price_real',
    device: Optional[str] = None
) -> bool:
    """Run all tests."""
    logger.info("\n" + "#" * 60)
    logger.info("# DEEP LEARNING TEST RUNNER")
    logger.info("# Testing for OOM and other errors before grid search")
    logger.info("# Uses EXACT same hyperparameter flow as runner_v2.py")
    logger.info("#" * 60)

    if model_types is None:
        model_types = ['patchtst', 'nhits', 'tft']

    results = {}

    # Test imports
    if not test_imports():
        logger.error("\nImport tests failed! Fix imports before proceeding.")
        return False

    # Test data loading
    data = test_data_loading(target=target)
    if data is None:
        logger.error("\nData loading failed! Fix data issues before proceeding.")
        return False

    X_train, X_val, X_test, y_train, y_val, y_test = data

    # Test each model
    for model_type in model_types:
        results[model_type] = test_model_training(
            model_type,
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            target=target,
            device=device
        )

        # Force cleanup between models
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)

    all_passed = True
    for model_type, passed in results.items():
        status = "PASS" if passed else "FAIL"
        logger.info(f"  {model_type.upper()}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        logger.info("\nAll tests passed! Ready for grid search.")
    else:
        logger.error("\nSome tests failed! Check errors above.")

    return all_passed


def main():
    parser = argparse.ArgumentParser(description='Test deep learning models for errors')
    parser.add_argument(
        '--models',
        nargs='+',
        choices=['patchtst', 'nhits', 'tft'],
        default=None,
        help='Model types to test (default: all)'
    )
    parser.add_argument(
        '--target',
        choices=['price_real', 'consumption'],
        default='price_real',
        help='Target variable'
    )
    parser.add_argument(
        '--device',
        choices=['cuda', 'mps', 'cpu'],
        default=None,
        help='Device to use'
    )

    args = parser.parse_args()

    run_tests(
        model_types=args.models,
        target=args.target,
        device=args.device
    )


if __name__ == "__main__":
    main()
