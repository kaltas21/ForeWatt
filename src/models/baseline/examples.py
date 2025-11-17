"""
Example Usage of Baseline Pipeline
===================================
Demonstrates various ways to use the baseline pipeline.

Author: ForeWatt Team
Date: November 2025
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.baseline import run_baseline_pipeline, FeatureSelector, ModelTrainer
from src.models.baseline.data_loader import load_master_data, train_val_test_split, prepare_target_data


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 1: Run Complete Pipeline (Both Targets, All Models)
# ═══════════════════════════════════════════════════════════════════════════════

def example_1_full_pipeline():
    """Run complete pipeline for both targets with all models."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Full Pipeline (Both Targets, All Models)")
    print("="*80)

    results = run_baseline_pipeline(
        targets=['consumption', 'price_real'],
        models=['catboost', 'xgboost', 'lightgbm', 'prophet', 'sarimax'],
        val_size=0.1,
        test_size=0.2
    )

    # Print results
    for target, target_results in results.items():
        print(f"\n{target.upper()} Results:")
        for model, metrics in target_results.items():
            if 'error' not in metrics:
                print(f"  {model}: MASE={metrics['MASE']:.4f}, sMAPE={metrics['sMAPE']:.2f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 2: Demand Forecasting Only with Boosting Models
# ═══════════════════════════════════════════════════════════════════════════════

def example_2_demand_boosting():
    """Run demand forecasting with boosting models only."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Demand Forecasting (Boosting Models)")
    print("="*80)

    results = run_baseline_pipeline(
        targets=['consumption'],
        models=['catboost', 'xgboost', 'lightgbm'],
        val_size=0.1,
        test_size=0.2
    )

    # Find best model
    consumption_results = results['consumption']
    best_model = min(
        consumption_results.keys(),
        key=lambda m: consumption_results[m].get('MASE', float('inf'))
    )

    print(f"\n🏆 Best Model: {best_model.upper()}")
    print(f"   MASE: {consumption_results[best_model]['MASE']:.4f}")
    print(f"   sMAPE: {consumption_results[best_model]['sMAPE']:.2f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 3: Price Forecasting with Custom Hyperparameters
# ═══════════════════════════════════════════════════════════════════════════════

def example_3_price_custom_hyperparams():
    """Run price forecasting with custom hyperparameters."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Price Forecasting (Custom Hyperparameters)")
    print("="*80)

    from src.models.baseline.pipeline_runner import BaselinePipeline

    # Load and split data
    df = load_master_data()
    train_df, val_df, test_df = train_val_test_split(df, val_size=0.1, test_size=0.2)
    train_df, val_df, test_df = prepare_target_data(train_df, val_df, test_df, target='price_real')

    # Initialize pipeline
    pipeline = BaselinePipeline(target='price_real')

    # Custom CatBoost hyperparameters
    custom_hyperparams = {
        'iterations': 2000,
        'learning_rate': 0.05,
        'depth': 8,
        'l2_leaf_reg': 5.0,
        'early_stopping_rounds': 100
    }

    # Run with custom hyperparameters
    results = pipeline.run_model(
        model_type='catboost',
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        hyperparams=custom_hyperparams
    )

    print(f"\nCustom CatBoost Results:")
    print(f"  MAE: {results['MAE']:.2f} TL")
    print(f"  RMSE: {results['RMSE']:.2f} TL")
    print(f"  sMAPE: {results['sMAPE']:.2f}%")
    print(f"  MASE: {results['MASE']:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 4: Feature Selection Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def example_4_feature_analysis():
    """Analyze feature selection for different models."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Feature Selection Analysis")
    print("="*80)

    selector = FeatureSelector(target='consumption')

    # Compare features for different models
    models = ['prophet', 'catboost']

    for model in models:
        print(f"\n{model.upper()} Feature Selection:")
        features = selector.get_features_for_model_type(model, 'consumption')
        print(f"  Total features: {len(features)}")

        # Count by category
        feature_groups = selector.get_feature_importance_groups()
        for group_name, group_features in feature_groups.items():
            count = len([f for f in features if f in group_features])
            if count > 0:
                print(f"    {group_name:30s}: {count:3d} features")


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 5: Train Single Model and Analyze Feature Importance
# ═══════════════════════════════════════════════════════════════════════════════

def example_5_single_model_importance():
    """Train single model and analyze feature importance."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Single Model with Feature Importance")
    print("="*80)

    # Load and split data
    df = load_master_data()
    train_df, val_df, test_df = train_val_test_split(df)
    train_df, val_df, test_df = prepare_target_data(train_df, val_df, test_df, target='consumption')

    # Initialize trainer
    trainer = ModelTrainer(model_type='catboost', target='consumption')

    # Prepare features
    X_train, y_train, feature_names = trainer.prepare_features(train_df)
    X_val, y_val, _ = trainer.prepare_features(val_df)
    X_test, y_test, _ = trainer.prepare_features(test_df)

    # Train model
    model = trainer.train(X_train, y_train, X_val, y_val)

    # Get predictions
    predictions = trainer.predict(X_test)

    # Get feature importance
    feature_importance = trainer.get_feature_importance()

    print("\nTop 20 Most Important Features:")
    print(feature_importance.head(20).to_string(index=False))

    # Calculate error
    import numpy as np
    mae = np.mean(np.abs(y_test.values - predictions))
    print(f"\nTest MAE: {mae:.2f} MWh")


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE 6: Quick Test (Fast Sanity Check)
# ═══════════════════════════════════════════════════════════════════════════════

def example_6_quick_test():
    """Quick test with single model and target."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Quick Test (CatBoost + Demand)")
    print("="*80)

    results = run_baseline_pipeline(
        targets=['consumption'],
        models=['catboost'],
        val_size=0.1,
        test_size=0.2
    )

    metrics = results['consumption']['catboost']
    print(f"\nQuick Test Results:")
    print(f"  MAE: {metrics['MAE']:.2f} MWh")
    print(f"  sMAPE: {metrics['sMAPE']:.2f}%")
    print(f"  MASE: {metrics['MASE']:.4f}")
    print(f"  Training time: {metrics['training_time_seconds']:.2f}s")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Baseline Pipeline Examples')
    parser.add_argument('--example', type=int, choices=range(1, 7),
                        help='Example number to run (1-6)')

    args = parser.parse_args()

    examples = {
        1: ("Full Pipeline", example_1_full_pipeline),
        2: ("Demand Forecasting (Boosting)", example_2_demand_boosting),
        3: ("Price Forecasting (Custom)", example_3_price_custom_hyperparams),
        4: ("Feature Analysis", example_4_feature_analysis),
        5: ("Feature Importance", example_5_single_model_importance),
        6: ("Quick Test", example_6_quick_test)
    }

    if args.example:
        # Run specific example
        name, func = examples[args.example]
        print(f"\n{'#'*80}")
        print(f"# Running Example {args.example}: {name}")
        print(f"{'#'*80}")
        func()
    else:
        # List all examples
        print("\n" + "="*80)
        print("Available Examples:")
        print("="*80)
        for num, (name, _) in examples.items():
            print(f"  {num}. {name}")
        print("\nUsage: python examples.py --example <number>")
        print("="*80)
