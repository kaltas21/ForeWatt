"""
4-Month Recent Test Period Experiment
=====================================
Tests models on only the last 4 months of data (July-October 2025)
to demonstrate that recent data has lower error rates.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostRegressor
import lightgbm as lgb
import json
import matplotlib.pyplot as plt

from config import (
    DATA_PATH, RESULTS_DIR,
    CONSUMPTION_FEATURES_ALL, CONSUMPTION_CATBOOST_PARAMS,
    PRICE_FEATURES_BASE, PRICE_CATBOOST_PARAMS, PRICE_LIGHTGBM_PARAMS,
    CATBOOST_WEIGHT, LIGHTGBM_WEIGHT
)
from metrics import calculate_all_metrics

# Test periods to compare
TEST_PERIODS = {
    'full_17_months': {'start': '2024-06-01', 'end': None},      # Original: 17 months
    'recent_4_months': {'start': '2025-07-01', 'end': None},     # Recent: 4 months
    'recent_6_months': {'start': '2025-05-01', 'end': None},     # Recent: 6 months
}


def load_data():
    """Load the master dataset."""
    df = pd.read_parquet(DATA_PATH)
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
    df = df.sort_index()
    df.index = pd.to_datetime(df.index)
    return df


def run_consumption_test_periods(df):
    """Test consumption model on different test periods."""
    results = {}

    # Train on full data up to June 2024
    train_end = '2024-06-01'
    train_data = df[df.index < train_end].copy()

    features = [f for f in CONSUMPTION_FEATURES_ALL if f in df.columns]
    target = 'consumption'

    X_train = train_data[features].dropna()
    y_train = train_data.loc[X_train.index, target]

    # Train model
    params = CONSUMPTION_CATBOOST_PARAMS.copy()
    params['learning_rate'] = 0.1  # Best hyperparameter
    params.pop('early_stopping_rounds', None)

    model = CatBoostRegressor(**params, verbose=False)
    model.fit(X_train, y_train)

    # Test on different periods
    for period_name, period_config in TEST_PERIODS.items():
        test_start = period_config['start']
        test_end = period_config['end']

        if test_end:
            test_data = df[(df.index >= test_start) & (df.index < test_end)].copy()
        else:
            test_data = df[df.index >= test_start].copy()

        X_test = test_data[features].dropna()
        y_test = test_data.loc[X_test.index, target]

        if len(y_test) == 0:
            continue

        y_pred = model.predict(X_test)

        metrics = calculate_all_metrics(y_test.values, y_pred, y_train.values)
        metrics['n_samples'] = len(y_test)
        metrics['period'] = f"{test_start} to {test_end or 'end'}"

        results[period_name] = {
            'metrics': metrics,
            'y_test': y_test,
            'y_pred': y_pred,
            'dates': X_test.index
        }

        print(f"Consumption - {period_name}: sMAPE={metrics['sMAPE']:.2f}%, MAE={metrics['MAE']:.1f}, n={metrics['n_samples']}")

    return results


def run_price_test_periods(df):
    """Test price model on different test periods."""
    results = {}

    # Train on full data up to June 2024
    train_end = '2024-06-01'
    train_data = df[df.index < train_end].copy()

    features = [f for f in PRICE_FEATURES_BASE if f in df.columns]
    target = 'price'

    X_train = train_data[features].dropna()
    y_train = train_data.loc[X_train.index, target]

    # Train CatBoost
    cb_params = PRICE_CATBOOST_PARAMS.copy()
    cb_params.pop('early_stopping_rounds', None)
    cb_params.pop('verbose', None)
    cb_model = CatBoostRegressor(**cb_params, verbose=False)
    cb_model.fit(X_train, y_train)

    # Train LightGBM
    lgb_params = PRICE_LIGHTGBM_PARAMS.copy()
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(X_train, y_train)

    # Test on different periods
    for period_name, period_config in TEST_PERIODS.items():
        test_start = period_config['start']
        test_end = period_config['end']

        if test_end:
            test_data = df[(df.index >= test_start) & (df.index < test_end)].copy()
        else:
            test_data = df[df.index >= test_start].copy()

        X_test = test_data[features].dropna()
        y_test = test_data.loc[X_test.index, target]

        if len(y_test) == 0:
            continue

        # Ensemble prediction
        cb_pred = cb_model.predict(X_test)
        lgb_pred = lgb_model.predict(X_test)
        y_pred = CATBOOST_WEIGHT * cb_pred + LIGHTGBM_WEIGHT * lgb_pred

        metrics = calculate_all_metrics(y_test.values, y_pred, y_train.values)
        metrics['n_samples'] = len(y_test)
        metrics['period'] = f"{test_start} to {test_end or 'end'}"

        results[period_name] = {
            'metrics': metrics,
            'y_test': y_test,
            'y_pred': y_pred,
            'dates': X_test.index
        }

        print(f"Price - {period_name}: sMAPE={metrics['sMAPE']:.2f}%, MAE={metrics['MAE']:.1f}, n={metrics['n_samples']}")

    return results


def plot_test_period_comparison(consumption_results, price_results, output_dir):
    """Create comparison plot showing error by test period."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Consumption
    ax = axes[0]
    periods = list(consumption_results.keys())
    smapes = [consumption_results[p]['metrics']['sMAPE'] for p in periods]
    labels = ['Full (17 mo)', 'Recent (4 mo)', 'Recent (6 mo)']
    colors = ['#2E86AB', '#28A745', '#17A2B8']

    bars = ax.bar(labels[:len(periods)], smapes[:len(periods)], color=colors[:len(periods)], edgecolor='black')
    ax.set_ylabel('sMAPE (%)', fontsize=11)
    ax.set_title('Consumption Model: Test Period Comparison', fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(smapes) * 1.3)

    for bar, val in zip(bars, smapes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Price
    ax = axes[1]
    periods = list(price_results.keys())
    smapes = [price_results[p]['metrics']['sMAPE'] for p in periods]

    bars = ax.bar(labels[:len(periods)], smapes[:len(periods)], color=colors[:len(periods)], edgecolor='black')
    ax.set_ylabel('sMAPE (%)', fontsize=11)
    ax.set_title('Price Model: Test Period Comparison', fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(smapes) * 1.3)

    for bar, val in zip(bars, smapes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'test_period_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'test_period_comparison.png'}")


def main():
    print("=" * 60)
    print("4-Month Recent Test Period Experiment")
    print("=" * 60)

    # Create output directory
    output_dir = RESULTS_DIR / 'test_period_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data...")
    df = load_data()
    print(f"Data range: {df.index.min()} to {df.index.max()}")

    # Run experiments
    print("\n" + "-" * 40)
    print("CONSUMPTION MODEL")
    print("-" * 40)
    consumption_results = run_consumption_test_periods(df)

    print("\n" + "-" * 40)
    print("PRICE MODEL")
    print("-" * 40)
    price_results = run_price_test_periods(df)

    # Create comparison plot
    print("\n" + "-" * 40)
    print("Creating comparison plot...")
    plot_test_period_comparison(consumption_results, price_results, output_dir)

    # Save results summary
    summary = {
        'consumption': {k: v['metrics'] for k, v in consumption_results.items()},
        'price': {k: v['metrics'] for k, v in price_results.items()}
    }

    with open(output_dir / 'test_period_results.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_dir}")

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY: Test Period Impact on Error")
    print("=" * 60)
    print(f"\n{'Model':<15} {'Period':<20} {'sMAPE':<10} {'MAE':<12} {'Samples':<10}")
    print("-" * 70)

    for period in consumption_results:
        m = consumption_results[period]['metrics']
        print(f"{'Consumption':<15} {period:<20} {m['sMAPE']:.2f}%{'':<5} {m['MAE']:.1f}{'':<6} {m['n_samples']}")

    print()
    for period in price_results:
        m = price_results[period]['metrics']
        print(f"{'Price':<15} {period:<20} {m['sMAPE']:.2f}%{'':<5} {m['MAE']:.1f}{'':<6} {m['n_samples']}")


if __name__ == '__main__':
    main()
