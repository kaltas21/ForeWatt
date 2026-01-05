"""
Calculate Train vs Test Error for Overfitting Analysis
=======================================================
Loads trained models and calculates metrics on train/test splits.
"""

import sys
from pathlib import Path
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

import pandas as pd
import numpy as np
import json
from catboost import CatBoostRegressor
import lightgbm as lgb

# Import the training module's data preparation
from src.models.price_train import load_data as load_price_data, prepare_data as prepare_price_data
from src.models.consumption_train import load_data as load_consumption_data, prepare_features as prepare_consumption_features

# Paths
DATA_PATH = BASE_DIR / 'data' / 'gold' / 'master' / 'master_v2_fundamental.parquet'
PRICE_MODEL_DIR = BASE_DIR / 'models' / 'price'
CONSUMPTION_MODEL_DIR = BASE_DIR / 'models' / 'consumption'


def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error"""
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))


def mae(y_true, y_pred):
    """Mean Absolute Error"""
    return np.mean(np.abs(y_pred - y_true))


def rmse(y_true, y_pred):
    """Root Mean Squared Error"""
    return np.sqrt(np.mean((y_pred - y_true) ** 2))


def analyze_price_model():
    """Analyze price model overfitting."""
    print("\n" + "="*60)
    print("PRICE MODEL OVERFITTING ANALYSIS")
    print("="*60)

    # Load data using training module's loader (no arguments)
    df = load_price_data()

    # Load features config
    with open(PRICE_MODEL_DIR / 'features.json') as f:
        feature_config = json.load(f)
    features = feature_config.get('features', [])

    # Prepare data using training module's function
    data = prepare_price_data(df, features, test_start='2024-06-01', finetune_months=6)

    X_train = data['X_base']
    y_train = data['y_base']
    X_test = data['X_test']
    y_test = data['y_test']

    print(f"\nTrain: {len(X_train):,} samples")
    print(f"Test:  {len(X_test):,} samples")
    print(f"Features: {X_train.shape[1]}")

    # Load CatBoost model
    cb_model = CatBoostRegressor()
    cb_model.load_model(str(PRICE_MODEL_DIR / 'catboost_v14.cbm'))

    # Load LightGBM model
    lgb_model = lgb.Booster(model_file=str(PRICE_MODEL_DIR / 'lightgbm_v14.txt'))

    # Load ensemble weights
    with open(PRICE_MODEL_DIR / 'ensemble_config.json') as f:
        config = json.load(f)
    cb_weight = config['catboost_weight']
    lgb_weight = config['lightgbm_weight']

    # Predictions
    cb_train_pred = cb_model.predict(X_train)
    cb_test_pred = cb_model.predict(X_test)

    lgb_train_pred = lgb_model.predict(X_train)
    lgb_test_pred = lgb_model.predict(X_test)

    # Ensemble predictions
    train_pred = cb_weight * cb_train_pred + lgb_weight * lgb_train_pred
    test_pred = cb_weight * cb_test_pred + lgb_weight * lgb_test_pred

    # Calculate metrics
    results = {
        'model': 'Price (CHybrid V14)',
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'catboost': {
            'train_smape': float(smape(y_train, cb_train_pred)),
            'test_smape': float(smape(y_test, cb_test_pred)),
            'train_mae': float(mae(y_train, cb_train_pred)),
            'test_mae': float(mae(y_test, cb_test_pred)),
        },
        'lightgbm': {
            'train_smape': float(smape(y_train, lgb_train_pred)),
            'test_smape': float(smape(y_test, lgb_test_pred)),
            'train_mae': float(mae(y_train, lgb_train_pred)),
            'test_mae': float(mae(y_test, lgb_test_pred)),
        },
        'ensemble': {
            'train_smape': float(smape(y_train, train_pred)),
            'test_smape': float(smape(y_test, test_pred)),
            'train_mae': float(mae(y_train, train_pred)),
            'test_mae': float(mae(y_test, test_pred)),
            'train_rmse': float(rmse(y_train, train_pred)),
            'test_rmse': float(rmse(y_test, test_pred)),
        }
    }

    # Calculate overfit ratio
    results['catboost']['overfit_ratio'] = results['catboost']['test_smape'] / results['catboost']['train_smape']
    results['lightgbm']['overfit_ratio'] = results['lightgbm']['test_smape'] / results['lightgbm']['train_smape']
    results['ensemble']['overfit_ratio_smape'] = results['ensemble']['test_smape'] / results['ensemble']['train_smape']
    results['ensemble']['overfit_ratio_mae'] = results['ensemble']['test_mae'] / results['ensemble']['train_mae']

    print("\n--- CatBoost ---")
    print(f"Train sMAPE: {results['catboost']['train_smape']:.2f}%")
    print(f"Test sMAPE:  {results['catboost']['test_smape']:.2f}%")
    print(f"Overfit Ratio: {results['catboost']['overfit_ratio']:.2f}x")

    print("\n--- LightGBM ---")
    print(f"Train sMAPE: {results['lightgbm']['train_smape']:.2f}%")
    print(f"Test sMAPE:  {results['lightgbm']['test_smape']:.2f}%")
    print(f"Overfit Ratio: {results['lightgbm']['overfit_ratio']:.2f}x")

    print("\n--- Ensemble ---")
    print(f"Train sMAPE: {results['ensemble']['train_smape']:.2f}%")
    print(f"Test sMAPE:  {results['ensemble']['test_smape']:.2f}%")
    print(f"Overfit Ratio: {results['ensemble']['overfit_ratio_smape']:.2f}x")
    print(f"\nTrain MAE: {results['ensemble']['train_mae']:.2f} TL/MWh")
    print(f"Test MAE:  {results['ensemble']['test_mae']:.2f} TL/MWh")

    return results


def analyze_consumption_model():
    """Analyze consumption model overfitting."""
    print("\n" + "="*60)
    print("CONSUMPTION MODEL OVERFITTING ANALYSIS")
    print("="*60)

    # Load data (no arguments)
    df = load_consumption_data()

    # Load features config
    with open(CONSUMPTION_MODEL_DIR / 'features.json') as f:
        feature_config = json.load(f)
    features = feature_config.get('features', [])

    # Prepare data using training module's function
    # This uses temporal split with val_size=0.2, test_size=0.2
    data = prepare_consumption_features(df, features, target='consumption')
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']

    print(f"\nTrain: {len(X_train):,} samples")
    print(f"Val:   {len(X_val):,} samples")
    print(f"Test:  {len(X_test):,} samples")

    # Load model
    model = CatBoostRegressor()
    model.load_model(str(CONSUMPTION_MODEL_DIR / 'model.cbm'))

    # Predictions
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    results = {
        'model': 'Consumption (CatBoost)',
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'train_smape': float(smape(y_train, train_pred)),
        'val_smape': float(smape(y_val, val_pred)),
        'test_smape': float(smape(y_test, test_pred)),
        'train_mae': float(mae(y_train, train_pred)),
        'val_mae': float(mae(y_val, val_pred)),
        'test_mae': float(mae(y_test, test_pred)),
        'train_rmse': float(rmse(y_train, train_pred)),
        'val_rmse': float(rmse(y_val, val_pred)),
        'test_rmse': float(rmse(y_test, test_pred)),
    }

    results['overfit_ratio_smape'] = results['test_smape'] / results['train_smape']
    results['overfit_ratio_mae'] = results['test_mae'] / results['train_mae']

    print("\n--- Results ---")
    print(f"Train sMAPE: {results['train_smape']:.2f}%")
    print(f"Val sMAPE:   {results['val_smape']:.2f}%")
    print(f"Test sMAPE:  {results['test_smape']:.2f}%")
    print(f"\nOverfit Ratio (Test/Train): {results['overfit_ratio_smape']:.2f}x")
    print(f"\nTrain MAE: {results['train_mae']:.2f} MWh")
    print(f"Val MAE:   {results['val_mae']:.2f} MWh")
    print(f"Test MAE:  {results['test_mae']:.2f} MWh")

    return results


def main():
    print("\n" + "#"*60)
    print("# FOREWATT MODEL OVERFITTING ANALYSIS")
    print("#"*60)

    price_results = analyze_price_model()
    consumption_results = analyze_consumption_model()

    # Save results
    output = {
        'price': price_results,
        'consumption': consumption_results,
        'timestamp': pd.Timestamp.now().isoformat()
    }

    output_path = BASE_DIR / 'models' / 'training_metrics.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n\nResults saved to: {output_path}")

    # Summary
    print("\n" + "="*60)
    print("OVERFITTING SUMMARY")
    print("="*60)
    print("\n| Model | Train sMAPE | Test sMAPE | Overfit Ratio |")
    print("|-------|-------------|------------|---------------|")
    print(f"| Price (Ensemble) | {price_results['ensemble']['train_smape']:.2f}% | {price_results['ensemble']['test_smape']:.2f}% | {price_results['ensemble']['overfit_ratio_smape']:.2f}x |")
    print(f"| Consumption | {consumption_results['train_smape']:.2f}% | {consumption_results['test_smape']:.2f}% | {consumption_results['overfit_ratio_smape']:.2f}x |")

    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    print("- Overfit Ratio < 1.2: ✅ Good generalization")
    print("- Overfit Ratio 1.2-1.5: ⚠️ Slight overfitting")
    print("- Overfit Ratio > 1.5: ❌ Significant overfitting")

    # Diagnosis
    price_ratio = price_results['ensemble']['overfit_ratio_smape']
    cons_ratio = consumption_results['overfit_ratio_smape']

    print("\n" + "="*60)
    print("DIAGNOSIS")
    print("="*60)

    if price_ratio < 1.2:
        print(f"✅ Price Model: Good generalization (ratio={price_ratio:.2f}x)")
    elif price_ratio < 1.5:
        print(f"⚠️ Price Model: Slight overfitting (ratio={price_ratio:.2f}x)")
    else:
        print(f"❌ Price Model: Significant overfitting (ratio={price_ratio:.2f}x)")

    if cons_ratio < 1.2:
        print(f"✅ Consumption Model: Good generalization (ratio={cons_ratio:.2f}x)")
    elif cons_ratio < 1.5:
        print(f"⚠️ Consumption Model: Slight overfitting (ratio={cons_ratio:.2f}x)")
    else:
        print(f"❌ Consumption Model: Significant overfitting (ratio={cons_ratio:.2f}x)")


if __name__ == '__main__':
    main()
