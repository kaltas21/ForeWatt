"""
Quick Test Script for Baseline Pipeline
========================================
Validates that all components are working correctly.

Author: ForeWatt Team
Date: November 2025
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all imports work."""
    logger.info("Testing imports...")

    try:
        from src.models.baseline import FeatureSelector, ModelTrainer, run_baseline_pipeline
        from src.models.baseline.data_loader import load_master_data, train_val_test_split
        from src.models.baseline.pipeline_runner import BaselinePipeline
        logger.info("✓ All imports successful")
        return True
    except Exception as e:
        logger.error(f"✗ Import failed: {e}")
        return False


def test_data_loading():
    """Test data loading."""
    logger.info("\nTesting data loading...")

    try:
        from src.models.baseline.data_loader import load_master_data

        df = load_master_data()
        logger.info(f"✓ Data loaded: {df.shape}")

        # Check for required columns
        required = ['consumption', 'price_real']
        missing = [col for col in required if col not in df.columns]

        if missing:
            logger.error(f"✗ Missing columns: {missing}")
            return False

        logger.info("✓ All required columns present")
        return True

    except Exception as e:
        logger.error(f"✗ Data loading failed: {e}")
        return False


def test_feature_selector():
    """Test feature selector."""
    logger.info("\nTesting feature selector...")

    try:
        from src.models.baseline import FeatureSelector

        # Test for demand
        selector_demand = FeatureSelector(target='consumption')
        features_boosting = selector_demand.get_features_for_model_type('catboost')
        features_statistical = selector_demand.get_features_for_model_type('prophet')

        logger.info(f"✓ Demand - Boosting: {len(features_boosting)} features")
        logger.info(f"✓ Demand - Statistical: {len(features_statistical)} features")

        # Test for price
        selector_price = FeatureSelector(target='price_real')
        features_price = selector_price.get_features_for_model_type('catboost')

        logger.info(f"✓ Price - Boosting: {len(features_price)} features")

        return True

    except Exception as e:
        logger.error(f"✗ Feature selector failed: {e}")
        return False


def test_data_split():
    """Test data splitting."""
    logger.info("\nTesting data split...")

    try:
        from src.models.baseline.data_loader import load_master_data, train_val_test_split

        df = load_master_data()
        train_df, val_df, test_df = train_val_test_split(df, val_size=0.1, test_size=0.2)

        logger.info(f"✓ Train: {len(train_df)} samples")
        logger.info(f"✓ Val: {len(val_df)} samples")
        logger.info(f"✓ Test: {len(test_df)} samples")

        # Check splits are correct
        total = len(train_df) + len(val_df) + len(test_df)
        if total != len(df):
            logger.error(f"✗ Split sizes don't match: {total} != {len(df)}")
            return False

        return True

    except Exception as e:
        logger.error(f"✗ Data split failed: {e}")
        return False


def test_model_trainer_init():
    """Test model trainer initialization."""
    logger.info("\nTesting model trainer initialization...")

    try:
        from src.models.baseline import ModelTrainer

        models = ['catboost', 'xgboost', 'lightgbm', 'prophet', 'sarimax']

        for model_type in models:
            trainer = ModelTrainer(model_type=model_type, target='consumption')
            logger.info(f"✓ {model_type.upper()} trainer initialized")

        return True

    except Exception as e:
        logger.error(f"✗ Model trainer init failed: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    logger.info("="*80)
    logger.info("BASELINE PIPELINE VALIDATION")
    logger.info("="*80)

    tests = [
        ("Imports", test_imports),
        ("Data Loading", test_data_loading),
        ("Feature Selector", test_feature_selector),
        ("Data Split", test_data_split),
        ("Model Trainer Init", test_model_trainer_init)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {e}")
            results.append((test_name, False))

    # Summary
    logger.info("\n" + "="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{test_name:30s}: {status}")

    logger.info("="*80)
    logger.info(f"Results: {passed}/{total} tests passed")

    if passed == total:
        logger.info("✓ All tests passed! Pipeline is ready to use.")
        return 0
    else:
        logger.error("✗ Some tests failed. Please check errors above.")
        return 1


if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)
