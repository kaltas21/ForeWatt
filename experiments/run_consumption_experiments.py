"""
Consumption Model Experiments
=============================
Run all experiments for consumption forecasting model.

Experiments:
1. Data Size Ablation (1yr, 2yr, 3yr, 4yr, full)
2. Feature Ablation (lag_only, weather_only, calendar_only, combinations, all)
3. Model Comparison (CatBoost configurations)
4. Baseline Comparisons
"""

import sys
import os
import json
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.config import (
    DATA_PATH, RESULTS_DIR, PLOTS_DIR, LOGS_DIR,
    CONSUMPTION_FEATURES_ALL, CONSUMPTION_FEATURE_GROUPS,
    CONSUMPTION_CATBOOST_PARAMS, DATA_SIZE_EXPERIMENTS,
    TEST_START, VAL_RATIO, RUN_TIMESTAMP
)
from experiments.metrics import calculate_all_metrics, format_metrics_table, compare_metrics
from experiments.plotting import (
    plot_predictions_vs_actual, plot_error_distribution,
    plot_hourly_errors, plot_feature_importance,
    plot_metrics_comparison, plot_data_size_impact,
    plot_scatter_actual_vs_pred, create_summary_dashboard
)
from experiments.baselines import run_baseline_experiments

# Setup logging
log_file = LOGS_DIR / f'consumption_experiments_{RUN_TIMESTAMP}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_data() -> pd.DataFrame:
    """Load master dataset."""
    logger.info(f"Loading data from: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)

    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date' in df.columns:
            df = df.set_index('date')
        elif 'datetime' in df.columns:
            df = df.set_index('datetime')

    logger.info(f"Loaded data: {df.shape}, {df.index.min()} to {df.index.max()}")
    return df


def prepare_data(
    df: pd.DataFrame,
    features: List[str],
    target: str = 'consumption',
    train_start: Optional[str] = None,
    train_end: Optional[str] = None,
    test_start: str = TEST_START,
    val_ratio: float = VAL_RATIO
) -> Dict:
    """Prepare data with train/val/test split."""
    # Filter available features
    available = [f for f in features if f in df.columns]
    missing = set(features) - set(available)
    if missing:
        logger.warning(f"Missing {len(missing)} features: {list(missing)[:5]}...")

    # Extract features and target
    X = df[available].copy()
    y = df[target].copy()

    # Add hour for analysis
    X['hour'] = X.index.hour
    X['dow'] = X.index.dayofweek

    # Drop NaN rows
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]

    # Time-based filtering
    tz = X.index.tz
    test_start_dt = pd.Timestamp(test_start, tz=tz)

    if train_start:
        train_start_dt = pd.Timestamp(train_start, tz=tz)
        X = X[X.index >= train_start_dt]
        y = y[y.index >= train_start_dt]

    if train_end:
        train_end_dt = pd.Timestamp(train_end, tz=tz)
    else:
        train_end_dt = test_start_dt

    # Split
    train_mask = X.index < train_end_dt
    test_mask = X.index >= test_start_dt

    X_train_full = X[train_mask]
    y_train_full = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    # Validation split from training
    n_train = len(X_train_full)
    val_size = int(n_train * val_ratio)
    train_size = n_train - val_size

    X_train = X_train_full.iloc[:train_size]
    y_train = y_train_full.iloc[:train_size]
    X_val = X_train_full.iloc[train_size:]
    y_val = y_train_full.iloc[train_size:]

    # Extract hours and dow for later analysis
    hours_train = X_train['hour'].values
    hours_val = X_val['hour'].values
    hours_test = X_test['hour'].values
    dow_train = X_train['dow'].values
    dow_test = X_test['dow'].values

    # Drop hour/dow from features for model training
    feature_cols = [c for c in X_train.columns if c not in ['hour', 'dow']]

    return {
        'X_train': X_train[feature_cols],
        'X_val': X_val[feature_cols],
        'X_test': X_test[feature_cols],
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'y_train_full': y_train_full,
        'hours_train': hours_train,
        'hours_val': hours_val,
        'hours_test': hours_test,
        'dow_train': dow_train,
        'dow_test': dow_test,
        'feature_names': feature_cols,
        'dates_test': X_test.index,
        'n_train': len(X_train),
        'n_val': len(X_val),
        'n_test': len(X_test),
    }


def train_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: Dict = None
) -> Tuple[object, Dict]:
    """Train CatBoost model and return model with training info."""
    from catboost import CatBoostRegressor

    params = params or CONSUMPTION_CATBOOST_PARAMS

    model = CatBoostRegressor(**params, verbose=0)
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
        verbose=False
    )

    train_info = {
        'best_iteration': model.best_iteration_,
        'tree_count': model.tree_count_,
        'params': params,
    }

    return model, train_info


def run_single_experiment(
    experiment_name: str,
    df: pd.DataFrame,
    features: List[str],
    output_dir: Path,
    train_start: Optional[str] = None,
    train_end: Optional[str] = None,
    params: Dict = None
) -> Dict:
    """Run a single experiment and save results."""
    logger.info(f"\n{'='*60}")
    logger.info(f"EXPERIMENT: {experiment_name}")
    logger.info(f"{'='*60}")

    # Create output directory
    exp_dir = output_dir / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data
    data = prepare_data(
        df, features,
        train_start=train_start,
        train_end=train_end
    )

    logger.info(f"  Train: {data['n_train']:,} samples")
    logger.info(f"  Val:   {data['n_val']:,} samples")
    logger.info(f"  Test:  {data['n_test']:,} samples")
    logger.info(f"  Features: {len(data['feature_names'])}")

    # Train model
    model, train_info = train_catboost(
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val'],
        params
    )
    logger.info(f"  Best iteration: {train_info['best_iteration']}")

    # Predictions
    train_pred = model.predict(data['X_train'])
    val_pred = model.predict(data['X_val'])
    test_pred = model.predict(data['X_test'])

    # Calculate metrics
    train_metrics = calculate_all_metrics(
        data['y_train'].values, train_pred, data['y_train'].values
    )
    val_metrics = calculate_all_metrics(
        data['y_val'].values, val_pred, data['y_train'].values
    )
    test_metrics = calculate_all_metrics(
        data['y_test'].values, test_pred, data['y_train'].values
    )

    logger.info(f"\n  Train sMAPE: {train_metrics['sMAPE']:.4f}%")
    logger.info(f"  Val sMAPE:   {val_metrics['sMAPE']:.4f}%")
    logger.info(f"  Test sMAPE:  {test_metrics['sMAPE']:.4f}%")

    # Feature importance
    importance = model.get_feature_importance()
    importance_df = pd.DataFrame({
        'feature': data['feature_names'],
        'importance': importance
    }).sort_values('importance', ascending=False)

    # Save model
    model_path = exp_dir / 'model.cbm'
    model.save_model(str(model_path))

    # Save feature importance
    importance_df.to_csv(exp_dir / 'feature_importance.csv', index=False)

    # Save metrics
    results = {
        'experiment_name': experiment_name,
        'timestamp': datetime.now().isoformat(),
        'data': {
            'n_train': data['n_train'],
            'n_val': data['n_val'],
            'n_test': data['n_test'],
            'n_features': len(data['feature_names']),
            'features': data['feature_names'],
        },
        'training': train_info,
        'metrics': {
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics,
        },
        'overfitting_ratio': test_metrics['sMAPE'] / train_metrics['sMAPE'] if train_metrics['sMAPE'] > 0 else 0,
    }

    with open(exp_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Generate plots
    logger.info("  Generating plots...")

    # 1. Predictions vs Actual
    plot_predictions_vs_actual(
        data['y_test'].values, test_pred,
        data['dates_test'],
        f'{experiment_name}: Predictions vs Actual',
        exp_dir / 'predictions_vs_actual.png'
    )

    # 2. Error distribution
    plot_error_distribution(
        data['y_test'].values, test_pred,
        f'{experiment_name}: Error Distribution',
        exp_dir / 'error_distribution.png'
    )

    # 3. Hourly errors
    plot_hourly_errors(
        data['y_test'].values, test_pred,
        data['hours_test'],
        f'{experiment_name}: Hourly Errors',
        exp_dir / 'hourly_errors.png'
    )

    # 4. Feature importance
    plot_feature_importance(
        importance_df,
        f'{experiment_name}: Feature Importance',
        exp_dir / 'feature_importance.png'
    )

    # 5. Scatter plot
    plot_scatter_actual_vs_pred(
        data['y_test'].values, test_pred,
        f'{experiment_name}: Actual vs Predicted',
        exp_dir / 'scatter_actual_vs_pred.png'
    )

    # 6. Summary dashboard
    create_summary_dashboard(
        experiment_name,
        test_metrics,
        data['y_test'].values,
        test_pred,
        data['dates_test'],
        data['hours_test'],
        exp_dir / 'summary_dashboard.png',
        importance_df
    )

    return {
        'experiment_name': experiment_name,
        'test_metrics': test_metrics,
        'val_metrics': val_metrics,
        'train_metrics': train_metrics,
        'predictions': test_pred,
        'feature_importance': importance_df,
        'data': data,
    }


def run_all_consumption_experiments():
    """Run all consumption model experiments."""
    logger.info("="*70)
    logger.info("CONSUMPTION MODEL EXPERIMENTS")
    logger.info("="*70)

    # Create output directories
    output_dir = RESULTS_DIR / 'consumption' / RUN_TIMESTAMP
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_data()

    all_results = {}

    # =========================================================================
    # EXPERIMENT 1: Data Size Ablation
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("EXPERIMENT GROUP 1: DATA SIZE ABLATION")
    logger.info("="*70)

    data_size_results = {}
    for size_name, period in DATA_SIZE_EXPERIMENTS.items():
        result = run_single_experiment(
            experiment_name=f'data_size_{size_name}',
            df=df,
            features=CONSUMPTION_FEATURES_ALL,
            output_dir=output_dir,
            train_start=period['start'],
            train_end=period['end']
        )
        data_size_results[size_name] = result['test_metrics']
        all_results[f'data_size_{size_name}'] = result

    # Plot data size comparison
    plot_data_size_impact(
        data_size_results,
        output_dir / 'data_size_comparison.png',
        model_type='Consumption'
    )

    # =========================================================================
    # EXPERIMENT 2: Feature Ablation
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("EXPERIMENT GROUP 2: FEATURE ABLATION")
    logger.info("="*70)

    feature_results = {}
    for group_name, features in CONSUMPTION_FEATURE_GROUPS.items():
        result = run_single_experiment(
            experiment_name=f'features_{group_name}',
            df=df,
            features=features,
            output_dir=output_dir
        )
        feature_results[group_name] = result['test_metrics']
        all_results[f'features_{group_name}'] = result

    # Plot feature comparison
    plot_metrics_comparison(
        feature_results,
        ['MAE', 'sMAPE', 'RMSE'],
        'Consumption: Feature Ablation',
        output_dir / 'feature_ablation_comparison.png'
    )

    # =========================================================================
    # EXPERIMENT 3: Hyperparameter Variations
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("EXPERIMENT GROUP 3: HYPERPARAMETER VARIATIONS")
    logger.info("="*70)

    hyperparam_configs = {
        'default': CONSUMPTION_CATBOOST_PARAMS,
        'deeper': {**CONSUMPTION_CATBOOST_PARAMS, 'depth': 8},
        'shallower': {**CONSUMPTION_CATBOOST_PARAMS, 'depth': 3},
        'more_trees': {**CONSUMPTION_CATBOOST_PARAMS, 'iterations': 2000},
        'higher_lr': {**CONSUMPTION_CATBOOST_PARAMS, 'learning_rate': 0.1},
        'lower_lr': {**CONSUMPTION_CATBOOST_PARAMS, 'learning_rate': 0.01},
        'more_reg': {**CONSUMPTION_CATBOOST_PARAMS, 'l2_leaf_reg': 30.0},
        'less_reg': {**CONSUMPTION_CATBOOST_PARAMS, 'l2_leaf_reg': 3.0},
    }

    hyperparam_results = {}
    for config_name, params in hyperparam_configs.items():
        result = run_single_experiment(
            experiment_name=f'hyperparam_{config_name}',
            df=df,
            features=CONSUMPTION_FEATURES_ALL,
            output_dir=output_dir,
            params=params
        )
        hyperparam_results[config_name] = result['test_metrics']
        all_results[f'hyperparam_{config_name}'] = result

    # Plot hyperparameter comparison
    plot_metrics_comparison(
        hyperparam_results,
        ['MAE', 'sMAPE'],
        'Consumption: Hyperparameter Variations',
        output_dir / 'hyperparam_comparison.png'
    )

    # =========================================================================
    # EXPERIMENT 4: Baselines
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("EXPERIMENT GROUP 4: BASELINE MODELS")
    logger.info("="*70)

    # Get data for baselines
    data = prepare_data(df, CONSUMPTION_FEATURES_ALL)

    # Combine train, val and test for baseline calculations
    y_train_full = pd.concat([data['y_train'], data['y_val']]).values
    y_full = pd.concat([data['y_train'], data['y_val'], data['y_test']]).values
    hours_train_full = np.concatenate([data['hours_train'], data['hours_val']])
    hours_full = np.concatenate([data['hours_train'], data['hours_val'], data['hours_test']])
    dow_train_full = np.concatenate([data['dow_train'], np.zeros(len(data['hours_val']), dtype=int)])
    dow_full = np.concatenate([data['dow_train'], np.zeros(len(data['hours_val']), dtype=int), data['dow_test']])

    test_start_idx = len(data['y_train']) + len(data['y_val'])

    baseline_results = run_baseline_experiments(
        y_full=y_full,
        hours_full=hours_full,
        dow_full=dow_full,
        test_start_idx=test_start_idx,
        y_train=y_train_full,
        hours_train=hours_train_full,
        dow_train=dow_train_full
    )

    # Save baseline results
    baseline_dir = output_dir / 'baselines'
    baseline_dir.mkdir(exist_ok=True)

    baseline_metrics = {}
    for name, result in baseline_results.items():
        baseline_metrics[result['name']] = result['metrics']

        with open(baseline_dir / f'{name}_results.json', 'w') as f:
            json.dump({
                'name': result['name'],
                'metrics': result['metrics']
            }, f, indent=2, default=str)

        logger.info(f"  {result['name']}: sMAPE={result['metrics']['sMAPE']:.4f}%")

    # Plot baseline comparison
    plot_metrics_comparison(
        baseline_metrics,
        ['MAE', 'sMAPE'],
        'Consumption: Baseline Models',
        output_dir / 'baseline_comparison.png'
    )

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("FINAL SUMMARY")
    logger.info("="*70)

    # Combine all results for comparison
    all_metrics = {}
    for exp_name, result in all_results.items():
        all_metrics[exp_name] = result['test_metrics']

    # Add baselines
    for name, result in baseline_results.items():
        all_metrics[f'baseline_{name}'] = result['metrics']

    # Print comparison
    comparison_str = compare_metrics(all_metrics, 'sMAPE')
    logger.info(comparison_str)

    # Save final summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'n_experiments': len(all_results),
        'n_baselines': len(baseline_results),
        'all_metrics': all_metrics,
        'best_model': min(all_metrics.items(), key=lambda x: x[1]['sMAPE'])[0],
        'best_smape': min(all_metrics.items(), key=lambda x: x[1]['sMAPE'])[1]['sMAPE'],
    }

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Create overall comparison plot
    plot_metrics_comparison(
        {k: v for k, v in list(all_metrics.items())[:15]},  # Top 15 for visibility
        ['sMAPE'],
        'Consumption: All Experiments Comparison',
        output_dir / 'all_experiments_comparison.png'
    )

    logger.info(f"\nResults saved to: {output_dir}")
    logger.info(f"Best model: {summary['best_model']} with sMAPE={summary['best_smape']:.4f}%")

    return all_results, baseline_results


if __name__ == "__main__":
    run_all_consumption_experiments()
