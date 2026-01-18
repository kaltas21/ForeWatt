"""
Generate Training History Plots
================================
Re-trains models briefly with verbose output to capture training curves.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostRegressor, Pool
import lightgbm as lgb
import matplotlib.pyplot as plt

from config import (
    DATA_PATH, RESULTS_DIR,
    CONSUMPTION_FEATURES_ALL, CONSUMPTION_CATBOOST_PARAMS,
    PRICE_FEATURES_BASE, PRICE_CATBOOST_PARAMS, PRICE_LIGHTGBM_PARAMS,
    TEST_START
)

REPORT_DIR = Path(__file__).parent / 'report_plots'


def load_data():
    """Load the master dataset."""
    df = pd.read_parquet(DATA_PATH)
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
    df = df.sort_index()
    df.index = pd.to_datetime(df.index)
    return df


def train_consumption_with_history(df):
    """Train consumption model and capture training history."""
    print("Training consumption model with history tracking...")

    train_data = df[df.index < TEST_START].copy()
    test_data = df[df.index >= TEST_START].copy()

    features = [f for f in CONSUMPTION_FEATURES_ALL if f in df.columns]
    target = 'consumption'

    X_train = train_data[features].dropna()
    y_train = train_data.loc[X_train.index, target]

    # Split for validation
    val_size = int(len(X_train) * 0.15)
    X_val = X_train.iloc[-val_size:]
    y_val = y_train.iloc[-val_size:]
    X_train_fit = X_train.iloc[:-val_size]
    y_train_fit = y_train.iloc[:-val_size]

    # Create pools
    train_pool = Pool(X_train_fit, y_train_fit)
    val_pool = Pool(X_val, y_val)

    # Train with history
    params = CONSUMPTION_CATBOOST_PARAMS.copy()
    params['learning_rate'] = 0.1
    params['iterations'] = 500  # Enough for good curve
    params.pop('early_stopping_rounds', None)

    model = CatBoostRegressor(**params, verbose=False)

    # Use eval_set to track validation metrics
    model.fit(train_pool, eval_set=val_pool, verbose=False, plot=False)

    # Get training history from evals_result
    train_mae = model.get_evals_result()

    print(f"  Final train iterations: {model.get_best_iteration()}")

    return model, train_mae


def train_price_with_history(df):
    """Train price models and capture training history."""
    print("Training price models with history tracking...")

    train_data = df[df.index < TEST_START].copy()

    features = [f for f in PRICE_FEATURES_BASE if f in df.columns]
    target = 'price'

    X_train = train_data[features].dropna()
    y_train = train_data.loc[X_train.index, target]

    # Split for validation
    val_size = int(len(X_train) * 0.15)
    X_val = X_train.iloc[-val_size:]
    y_val = y_train.iloc[-val_size:]
    X_train_fit = X_train.iloc[:-val_size]
    y_train_fit = y_train.iloc[:-val_size]

    # CatBoost
    print("  Training CatBoost...")
    train_pool = Pool(X_train_fit, y_train_fit)
    val_pool = Pool(X_val, y_val)

    cb_params = PRICE_CATBOOST_PARAMS.copy()
    cb_params['iterations'] = 500
    cb_params.pop('early_stopping_rounds', None)
    cb_params.pop('verbose', None)

    cb_model = CatBoostRegressor(**cb_params, verbose=False)
    cb_model.fit(train_pool, eval_set=val_pool, verbose=False)
    cb_history = cb_model.get_evals_result()

    # LightGBM
    print("  Training LightGBM...")
    lgb_params = PRICE_LIGHTGBM_PARAMS.copy()
    lgb_params['n_estimators'] = 500

    lgb_train = lgb.Dataset(X_train_fit, y_train_fit)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

    lgb_history = {}
    lgb_model = lgb.train(
        {k: v for k, v in lgb_params.items() if k not in ['n_estimators']},
        lgb_train,
        num_boost_round=lgb_params['n_estimators'],
        valid_sets=[lgb_train, lgb_val],
        valid_names=['train', 'valid'],
        callbacks=[lgb.record_evaluation(lgb_history)]
    )

    return cb_model, cb_history, lgb_model, lgb_history


def plot_consumption_training_history(history):
    """Plot consumption model training curve."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Extract validation MAE (this is what CatBoost records)
    if 'validation' in history:
        val_mae = history['validation'].get('MAE', [])
        if val_mae:
            ax.plot(val_mae, label='Validation MAE', color='#E74C3C', linewidth=2)

    if 'learn' in history:
        train_mae = history['learn'].get('MAE', [])
        if train_mae:
            ax.plot(train_mae, label='Training MAE', color='#2E86AB', linewidth=2)

    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('MAE (MWh)', fontsize=11)
    ax.set_title('Consumption Model: Training History (CatBoost)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Add annotation for convergence
    if 'validation' in history and history['validation'].get('MAE'):
        final_val = history['validation']['MAE'][-1]
        ax.axhline(y=final_val, color='#E74C3C', linestyle='--', alpha=0.5)
        ax.text(0.02, final_val + 20, f'Final: {final_val:.1f}', transform=ax.get_yaxis_transform(),
               fontsize=9, color='#E74C3C')

    plt.tight_layout()
    output_path = REPORT_DIR / '08_training_history' / 'consumption_training_history.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


def plot_price_training_history(cb_history, lgb_history):
    """Plot price model training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # CatBoost
    ax = axes[0]
    if 'validation' in cb_history:
        val_mae = cb_history['validation'].get('MAE', [])
        if val_mae:
            ax.plot(val_mae, label='Validation MAE', color='#E74C3C', linewidth=2)
    if 'learn' in cb_history:
        train_mae = cb_history['learn'].get('MAE', [])
        if train_mae:
            ax.plot(train_mae, label='Training MAE', color='#2E86AB', linewidth=2)

    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('MAE (TL/MWh)', fontsize=11)
    ax.set_title('CatBoost Training History', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # LightGBM
    ax = axes[1]
    if 'train' in lgb_history:
        train_mae = lgb_history['train'].get('l1', [])
        if train_mae:
            ax.plot(train_mae, label='Training MAE', color='#2E86AB', linewidth=2)
    if 'valid' in lgb_history:
        val_mae = lgb_history['valid'].get('l1', [])
        if val_mae:
            ax.plot(val_mae, label='Validation MAE', color='#E74C3C', linewidth=2)

    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('MAE (TL/MWh)', fontsize=11)
    ax.set_title('LightGBM Training History', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Price Model: Training History', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path = REPORT_DIR / '08_training_history' / 'price_training_history.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path.name}")


def main():
    print("=" * 60)
    print("Generating Training History Plots")
    print("=" * 60)

    # Ensure output directory exists
    (REPORT_DIR / '08_training_history').mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data...")
    df = load_data()

    # Consumption model
    print("\n" + "-" * 40)
    print("CONSUMPTION MODEL")
    print("-" * 40)
    _, cons_history = train_consumption_with_history(df)
    plot_consumption_training_history(cons_history)

    # Price model
    print("\n" + "-" * 40)
    print("PRICE MODEL")
    print("-" * 40)
    _, cb_history, _, lgb_history = train_price_with_history(df)
    plot_price_training_history(cb_history, lgb_history)

    print("\n" + "=" * 60)
    print("Training history plots complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
