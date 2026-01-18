"""
Price Model Experiments
=======================
Run all experiments for price forecasting model.

Experiments:
1. Data Size Ablation (1yr, 2yr, 3yr, 4yr, full)
2. Feature Ablation (price_lags, market_signals, calendar, combinations, all)
3. Model Comparison (CatBoost, LightGBM, Ensemble variations)
4. Error Correction Impact (Raw, Simple AEC, KNN-EC, Hybrid)
5. Baseline Comparisons
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
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.config import (
    DATA_PATH, RESULTS_DIR, PLOTS_DIR, LOGS_DIR,
    PRICE_FEATURES_BASE, PRICE_FEATURE_GROUPS,
    PRICE_CATBOOST_PARAMS, PRICE_LIGHTGBM_PARAMS,
    DATA_SIZE_EXPERIMENTS, CATBOOST_WEIGHT, LIGHTGBM_WEIGHT,
    TEST_START, RUN_TIMESTAMP
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
log_file = LOGS_DIR / f'price_experiments_{RUN_TIMESTAMP}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# V13 Hourly AEC Parameters
HOURLY_AEC_PARAMS = {
    h: {'lookback': 14 if h in [0, 1, 20] else 21 if h in [2, 4, 6, 7, 8, 14, 19, 21] else 7,
        'damping': 0.7 if h in [5, 11, 21] else 0.6 if h == 14 else 0.5}
    for h in range(24)
}

# Context features for KNN
CONTEXT_FEATURES = ['load_factor', 'renewable_saturation', 'thermal_gap']


def load_data() -> pd.DataFrame:
    """Load master dataset."""
    logger.info(f"Loading data from: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)

    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date' in df.columns:
            df = df.set_index('date')
        elif 'datetime' in df.columns:
            df = df.set_index('datetime')

    if 'price_real' not in df.columns:
        df['price_real'] = df['price']

    logger.info(f"Loaded data: {df.shape}, {df.index.min()} to {df.index.max()}")
    return df


def create_profile_features(df: pd.DataFrame, price_col: str = 'price_real') -> Tuple[pd.DataFrame, List[str]]:
    """Create profile evolution features."""
    df = df.copy()
    new_features = []

    df['hour'] = df.index.hour

    # Daily average price
    df['daily_avg_price'] = df[price_col].shift(1).rolling(24, min_periods=12).mean()
    df['hourly_ratio'] = (df[price_col].shift(1) / df['daily_avg_price'].shift(1)).clip(0.2, 5.0)
    new_features.append('hourly_ratio')

    # Profile features by hour
    profile_14d_list = []
    profile_28d_list = []

    for hour in range(24):
        hour_mask = df['hour'] == hour
        hour_ratios = df.loc[hour_mask, 'hourly_ratio']
        p14 = hour_ratios.rolling(14, min_periods=7).mean().shift(1)
        p28 = hour_ratios.rolling(28, min_periods=14).mean().shift(1)
        profile_14d_list.append(p14)
        profile_28d_list.append(p28)

    df['profile_14d'] = pd.concat(profile_14d_list).sort_index()
    df['profile_28d'] = pd.concat(profile_28d_list).sort_index()
    new_features.extend(['profile_14d', 'profile_28d'])

    df['profile_momentum'] = df['profile_14d'] - df['profile_28d']
    df['daily_avg_momentum'] = df['daily_avg_price'] - df['daily_avg_price'].shift(24)
    new_features.extend(['profile_momentum', 'daily_avg_momentum'])

    # Solar profile
    if 'renewable_saturation' in df.columns and 'load_factor' in df.columns:
        load = df['load_factor'].clip(lower=0.1)
        df['solar_ratio'] = (df['renewable_saturation'].shift(1) / load.shift(1)).clip(0, 5)
        new_features.append('solar_ratio')

    # Fill NaN
    for feat in new_features:
        if feat in df.columns:
            df[feat] = df[feat].fillna(df[feat].median())

    return df, new_features


def prepare_data(
    df: pd.DataFrame,
    features: List[str],
    target: str = 'price_real',
    train_start: Optional[str] = None,
    train_end: Optional[str] = None,
    test_start: str = TEST_START,
    finetune_months: int = 6
) -> Dict:
    """Prepare data with base/finetune/test split for transfer learning."""
    # Filter available features
    available = [f for f in features if f in df.columns]

    X = df[available].copy()
    y = df[target].copy()

    # Add profile features
    X_with_price = X.copy()
    X_with_price[target] = y
    X_with_price, profile_features = create_profile_features(X_with_price, target)
    X = X_with_price.drop(columns=[target])

    X['hour'] = X.index.hour
    X['dow'] = X.index.dayofweek

    # Drop NaN
    mask = ~(X.isna().any(axis=1) | y.isna())
    X, y = X[mask], y[mask]

    # Time splits
    tz = X.index.tz
    test_start_dt = pd.Timestamp(test_start, tz=tz)
    finetune_start_dt = test_start_dt - pd.DateOffset(months=finetune_months)

    if train_start:
        train_start_dt = pd.Timestamp(train_start, tz=tz)
        X = X[X.index >= train_start_dt]
        y = y[y.index >= train_start_dt]

    base_mask = X.index < finetune_start_dt
    finetune_mask = (X.index >= finetune_start_dt) & (X.index < test_start_dt)
    test_mask = X.index >= test_start_dt

    hours = X['hour']
    dow = X['dow']

    feature_cols = [c for c in X.columns if c not in ['hour', 'dow']]

    return {
        'X_base': X[base_mask][feature_cols],
        'y_base': y[base_mask],
        'X_finetune': X[finetune_mask][feature_cols],
        'y_finetune': y[finetune_mask],
        'X_test': X[test_mask][feature_cols],
        'y_test': y[test_mask],
        'hours_base': hours[base_mask].values,
        'hours_finetune': hours[finetune_mask].values,
        'hours_test': hours[test_mask].values,
        'dow_base': dow[base_mask].values,
        'dow_test': dow[test_mask].values,
        'dates_test': X[test_mask].index,
        'features': feature_cols,
        'n_base': base_mask.sum(),
        'n_finetune': finetune_mask.sum(),
        'n_test': test_mask.sum(),
    }


def train_catboost_transfer(data: Dict, params: Dict = None) -> Tuple[object, np.ndarray, np.ndarray]:
    """Train CatBoost with transfer learning."""
    from catboost import CatBoostRegressor

    params = params or PRICE_CATBOOST_PARAMS.copy()
    params.pop('verbose', None)  # Remove verbose if present

    X_base, y_base = data['X_base'], data['y_base']
    split_idx = int(len(X_base) * 0.85)

    # Base model
    base_model = CatBoostRegressor(**params, verbose=False)
    base_model.fit(
        X_base.iloc[:split_idx], y_base.iloc[:split_idx],
        eval_set=(X_base.iloc[split_idx:], y_base.iloc[split_idx:]),
        verbose=False
    )

    # Fine-tune
    X_ft, y_ft = data['X_finetune'], data['y_finetune']
    split_idx = int(len(X_ft) * 0.8)

    finetune_params = {**params, 'iterations': 500, 'learning_rate': 0.005}
    finetune_model = CatBoostRegressor(**finetune_params, verbose=False)
    finetune_model.fit(
        X_ft.iloc[:split_idx], y_ft.iloc[:split_idx],
        eval_set=(X_ft.iloc[split_idx:], y_ft.iloc[split_idx:]),
        init_model=base_model,
        verbose=False
    )

    finetune_pred = finetune_model.predict(data['X_finetune'])
    test_pred = finetune_model.predict(data['X_test'])

    return finetune_model, finetune_pred, test_pred


def train_lightgbm_transfer(data: Dict, params: Dict = None) -> Tuple[object, np.ndarray, np.ndarray]:
    """Train LightGBM with transfer learning."""
    import lightgbm as lgb

    params = params or PRICE_LIGHTGBM_PARAMS

    X_base, y_base = data['X_base'], data['y_base']
    split_idx = int(len(X_base) * 0.85)

    # Base model
    base_model = lgb.LGBMRegressor(**params)
    base_model.fit(
        X_base.iloc[:split_idx], y_base.iloc[:split_idx],
        eval_set=[(X_base.iloc[split_idx:], y_base.iloc[split_idx:])],
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )

    # Fine-tune
    X_ft, y_ft = data['X_finetune'], data['y_finetune']
    split_idx = int(len(X_ft) * 0.8)

    finetune_params = {**params, 'n_estimators': 500, 'learning_rate': 0.005}
    finetune_model = lgb.LGBMRegressor(**finetune_params)
    finetune_model.fit(
        X_ft.iloc[:split_idx], y_ft.iloc[:split_idx],
        eval_set=[(X_ft.iloc[split_idx:], y_ft.iloc[split_idx:])],
        callbacks=[lgb.early_stopping(50, verbose=False)],
        init_model=base_model.booster_
    )

    finetune_pred = finetune_model.predict(data['X_finetune'])
    test_pred = finetune_model.predict(data['X_test'])

    return finetune_model, finetune_pred, test_pred


def apply_simple_aec(df_preds: pd.DataFrame, hourly_params: Dict) -> np.ndarray:
    """Apply V13-style hourly Adaptive Error Correction."""
    df = df_preds.copy().sort_values('datetime').reset_index(drop=True)
    df['error'] = df['y_raw'] - df['y_true']
    df['y_corrected'] = df['y_raw'].copy()

    for hour in range(24):
        hour_mask = df['hour'] == hour
        if hour_mask.sum() == 0:
            continue

        params = hourly_params.get(hour, {'lookback': 7, 'damping': 0.5})
        lookback = params['lookback']
        damping = params['damping']

        hour_df = df[hour_mask].copy().reset_index(drop=True)
        errors = hour_df['error'].values
        raw = hour_df['y_raw'].values

        corrections = np.zeros(len(errors))
        for i in range(1, len(errors)):
            start_idx = max(0, i - lookback)
            past_errors = errors[start_idx:i]
            if len(past_errors) > 0:
                corrections[i] = damping * np.mean(past_errors)

        df.loc[hour_mask, 'y_corrected'] = raw - corrections

    return df['y_corrected'].values


def apply_knn_correction(df_preds: pd.DataFrame, X_context: pd.DataFrame,
                         scaler: StandardScaler, context_features: List[str],
                         k: int = 5, lookback_days: int = 45, damping: float = 0.8) -> np.ndarray:
    """Apply Context-Aware KNN Error Correction."""
    df = df_preds.copy().sort_values('datetime').reset_index(drop=True)
    df['error'] = df['y_raw'] - df['y_true']

    available_features = [f for f in context_features if f in X_context.columns]
    context_data = X_context[available_features].fillna(0).values
    context_normalized = scaler.transform(context_data)

    n = len(df)
    corrections = np.zeros(n)

    for hour in range(24):
        hour_mask = (df['hour'] == hour).values
        hour_indices = np.where(hour_mask)[0]

        if len(hour_indices) < k + 1:
            continue

        for i, idx in enumerate(hour_indices):
            if i < k:
                continue

            lookback_hours = lookback_days
            start_i = max(0, i - lookback_hours)
            history_indices = hour_indices[start_i:i]

            if len(history_indices) < 2:
                continue

            history_contexts = context_normalized[history_indices]
            current_context = context_normalized[idx].reshape(1, -1)

            k_actual = min(k, len(history_indices))
            knn = NearestNeighbors(n_neighbors=k_actual, metric='euclidean')
            knn.fit(history_contexts)

            distances, neighbor_idx = knn.kneighbors(current_context)
            distances = distances[0]
            neighbor_idx = neighbor_idx[0]

            neighbor_original_idx = history_indices[neighbor_idx]
            neighbor_errors = df.loc[neighbor_original_idx, 'error'].values

            epsilon = 1e-6
            weights = 1.0 / (distances + epsilon)
            weights = weights / weights.sum()

            weighted_bias = np.sum(weights * neighbor_errors)
            corrections[idx] = damping * weighted_bias

    return df['y_raw'].values - corrections


def run_single_experiment(
    experiment_name: str,
    df: pd.DataFrame,
    features: List[str],
    output_dir: Path,
    train_start: Optional[str] = None,
    model_type: str = 'ensemble',  # 'catboost', 'lightgbm', 'ensemble'
    apply_correction: str = 'hybrid',  # 'none', 'simple', 'knn', 'hybrid'
    catboost_weight: float = CATBOOST_WEIGHT,
    lightgbm_weight: float = LIGHTGBM_WEIGHT
) -> Dict:
    """Run a single price experiment."""
    logger.info(f"\n{'='*60}")
    logger.info(f"EXPERIMENT: {experiment_name}")
    logger.info(f"{'='*60}")

    exp_dir = output_dir / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data
    data = prepare_data(df, features, train_start=train_start)

    logger.info(f"  Base:     {data['n_base']:,} samples")
    logger.info(f"  Finetune: {data['n_finetune']:,} samples")
    logger.info(f"  Test:     {data['n_test']:,} samples")
    logger.info(f"  Features: {len(data['features'])}")

    # Train models
    catboost_model, lgb_model = None, None
    cat_test_pred, lgb_test_pred = None, None

    if model_type in ['catboost', 'ensemble']:
        catboost_model, _, cat_test_pred = train_catboost_transfer(data)
        logger.info(f"  CatBoost trained: {catboost_model.tree_count_} trees")

    if model_type in ['lightgbm', 'ensemble']:
        lgb_model, _, lgb_test_pred = train_lightgbm_transfer(data)
        logger.info(f"  LightGBM trained: {lgb_model.n_estimators_} trees")

    # Combine predictions
    if model_type == 'ensemble':
        total_w = catboost_weight + lightgbm_weight
        raw_test_pred = (catboost_weight * cat_test_pred + lightgbm_weight * lgb_test_pred) / total_w
    elif model_type == 'catboost':
        raw_test_pred = cat_test_pred
    else:
        raw_test_pred = lgb_test_pred

    # Apply error correction
    final_pred = raw_test_pred.copy()

    if apply_correction != 'none':
        df_preds = pd.DataFrame({
            'datetime': data['dates_test'],
            'hour': data['hours_test'],
            'y_true': data['y_test'].values,
            'y_raw': raw_test_pred,
        })

        if apply_correction in ['simple', 'hybrid']:
            simple_pred = apply_simple_aec(df_preds, HOURLY_AEC_PARAMS)

        if apply_correction in ['knn', 'hybrid']:
            # Fit scaler on finetune data
            available_context = [f for f in CONTEXT_FEATURES if f in data['X_finetune'].columns]
            if len(available_context) > 0:
                scaler = StandardScaler()
                scaler.fit(data['X_finetune'][available_context].fillna(0))

                knn_pred = apply_knn_correction(
                    df_preds, data['X_test'].reset_index(drop=True),
                    scaler, available_context,
                    k=5, lookback_days=45, damping=0.8
                )
            else:
                # No context features available, use raw predictions
                knn_pred = raw_test_pred.copy()

        if apply_correction == 'simple':
            final_pred = simple_pred
        elif apply_correction == 'knn':
            final_pred = knn_pred
        elif apply_correction == 'hybrid':
            final_pred = 0.5 * simple_pred + 0.5 * knn_pred

    # Calculate metrics
    y_train_full = pd.concat([data['y_base'], data['y_finetune']]).values

    raw_metrics = calculate_all_metrics(data['y_test'].values, raw_test_pred, y_train_full)
    final_metrics = calculate_all_metrics(data['y_test'].values, final_pred, y_train_full)

    logger.info(f"\n  Raw sMAPE:   {raw_metrics['sMAPE']:.4f}%")
    logger.info(f"  Final sMAPE: {final_metrics['sMAPE']:.4f}%")

    # Feature importance
    importance_df = None
    if catboost_model:
        importance = catboost_model.get_feature_importance()
        importance_df = pd.DataFrame({
            'feature': data['features'],
            'importance': importance
        }).sort_values('importance', ascending=False)
        importance_df.to_csv(exp_dir / 'feature_importance.csv', index=False)

    # Save models
    if catboost_model:
        catboost_model.save_model(str(exp_dir / 'catboost.cbm'))
    if lgb_model:
        lgb_model.booster_.save_model(str(exp_dir / 'lightgbm.txt'))

    # Save results
    results = {
        'experiment_name': experiment_name,
        'timestamp': datetime.now().isoformat(),
        'config': {
            'model_type': model_type,
            'correction': apply_correction,
            'catboost_weight': catboost_weight,
            'lightgbm_weight': lightgbm_weight,
        },
        'data': {
            'n_base': data['n_base'],
            'n_finetune': data['n_finetune'],
            'n_test': data['n_test'],
            'n_features': len(data['features']),
            'features': data['features'],
        },
        'metrics': {
            'raw': raw_metrics,
            'final': final_metrics,
        },
    }

    with open(exp_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Generate plots
    logger.info("  Generating plots...")

    plot_predictions_vs_actual(
        data['y_test'].values, final_pred,
        data['dates_test'],
        f'{experiment_name}: Predictions vs Actual',
        exp_dir / 'predictions_vs_actual.png'
    )

    plot_error_distribution(
        data['y_test'].values, final_pred,
        f'{experiment_name}: Error Distribution',
        exp_dir / 'error_distribution.png'
    )

    plot_hourly_errors(
        data['y_test'].values, final_pred,
        data['hours_test'],
        f'{experiment_name}: Hourly Errors',
        exp_dir / 'hourly_errors.png'
    )

    if importance_df is not None:
        plot_feature_importance(
            importance_df,
            f'{experiment_name}: Feature Importance',
            exp_dir / 'feature_importance.png'
        )

    plot_scatter_actual_vs_pred(
        data['y_test'].values, final_pred,
        f'{experiment_name}: Actual vs Predicted',
        exp_dir / 'scatter_actual_vs_pred.png'
    )

    create_summary_dashboard(
        experiment_name,
        final_metrics,
        data['y_test'].values,
        final_pred,
        data['dates_test'],
        data['hours_test'],
        exp_dir / 'summary_dashboard.png',
        importance_df
    )

    return {
        'experiment_name': experiment_name,
        'test_metrics': final_metrics,
        'raw_metrics': raw_metrics,
        'predictions': final_pred,
        'feature_importance': importance_df,
        'data': data,
    }


def run_all_price_experiments():
    """Run all price model experiments."""
    logger.info("="*70)
    logger.info("PRICE MODEL EXPERIMENTS")
    logger.info("="*70)

    output_dir = RESULTS_DIR / 'price' / RUN_TIMESTAMP
    output_dir.mkdir(parents=True, exist_ok=True)

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
            features=PRICE_FEATURES_BASE,
            output_dir=output_dir,
            train_start=period['start']
        )
        data_size_results[size_name] = result['test_metrics']
        all_results[f'data_size_{size_name}'] = result

    plot_data_size_impact(
        data_size_results,
        output_dir / 'data_size_comparison.png',
        model_type='Price'
    )

    # =========================================================================
    # EXPERIMENT 2: Feature Ablation
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("EXPERIMENT GROUP 2: FEATURE ABLATION")
    logger.info("="*70)

    feature_results = {}
    for group_name, features in PRICE_FEATURE_GROUPS.items():
        result = run_single_experiment(
            experiment_name=f'features_{group_name}',
            df=df,
            features=features,
            output_dir=output_dir
        )
        feature_results[group_name] = result['test_metrics']
        all_results[f'features_{group_name}'] = result

    plot_metrics_comparison(
        feature_results,
        ['MAE', 'sMAPE', 'RMSE'],
        'Price: Feature Ablation',
        output_dir / 'feature_ablation_comparison.png'
    )

    # =========================================================================
    # EXPERIMENT 3: Model Type Comparison
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("EXPERIMENT GROUP 3: MODEL TYPE COMPARISON")
    logger.info("="*70)

    model_configs = {
        'catboost_only': {'model_type': 'catboost', 'apply_correction': 'hybrid'},
        'lightgbm_only': {'model_type': 'lightgbm', 'apply_correction': 'hybrid'},
        'ensemble_default': {'model_type': 'ensemble', 'apply_correction': 'hybrid'},
        'ensemble_equal': {'model_type': 'ensemble', 'apply_correction': 'hybrid',
                          'catboost_weight': 0.5, 'lightgbm_weight': 0.5},
    }

    model_results = {}
    for config_name, config in model_configs.items():
        result = run_single_experiment(
            experiment_name=f'model_{config_name}',
            df=df,
            features=PRICE_FEATURES_BASE,
            output_dir=output_dir,
            **config
        )
        model_results[config_name] = result['test_metrics']
        all_results[f'model_{config_name}'] = result

    plot_metrics_comparison(
        model_results,
        ['MAE', 'sMAPE'],
        'Price: Model Comparison',
        output_dir / 'model_comparison.png'
    )

    # =========================================================================
    # EXPERIMENT 4: Error Correction Impact
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("EXPERIMENT GROUP 4: ERROR CORRECTION IMPACT")
    logger.info("="*70)

    correction_configs = {
        'raw_no_correction': {'apply_correction': 'none'},
        'simple_aec': {'apply_correction': 'simple'},
        'knn_ec': {'apply_correction': 'knn'},
        'hybrid_correction': {'apply_correction': 'hybrid'},
    }

    correction_results = {}
    for config_name, config in correction_configs.items():
        result = run_single_experiment(
            experiment_name=f'correction_{config_name}',
            df=df,
            features=PRICE_FEATURES_BASE,
            output_dir=output_dir,
            **config
        )
        correction_results[config_name] = result['test_metrics']
        all_results[f'correction_{config_name}'] = result

    plot_metrics_comparison(
        correction_results,
        ['MAE', 'sMAPE'],
        'Price: Error Correction Impact',
        output_dir / 'error_correction_comparison.png'
    )

    # =========================================================================
    # EXPERIMENT 5: Baselines
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("EXPERIMENT GROUP 5: BASELINE MODELS")
    logger.info("="*70)

    data = prepare_data(df, PRICE_FEATURES_BASE)

    y_train_full = pd.concat([data['y_base'], data['y_finetune']]).values
    y_full = pd.concat([data['y_base'], data['y_finetune'], data['y_test']]).values
    hours_train_full = np.concatenate([data['hours_base'], data['hours_finetune']])
    hours_full = np.concatenate([data['hours_base'], data['hours_finetune'], data['hours_test']])
    dow_train_full = np.concatenate([data['dow_base'], np.zeros(len(data['hours_finetune']), dtype=int)])
    dow_full = np.concatenate([data['dow_base'], np.zeros(len(data['hours_finetune']), dtype=int), data['dow_test']])

    test_start_idx = len(data['y_base']) + len(data['y_finetune'])

    baseline_results = run_baseline_experiments(
        y_full=y_full,
        hours_full=hours_full,
        dow_full=dow_full,
        test_start_idx=test_start_idx,
        y_train=y_train_full,
        hours_train=hours_train_full,
        dow_train=dow_train_full
    )

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

    plot_metrics_comparison(
        baseline_metrics,
        ['MAE', 'sMAPE'],
        'Price: Baseline Models',
        output_dir / 'baseline_comparison.png'
    )

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("FINAL SUMMARY")
    logger.info("="*70)

    all_metrics = {}
    for exp_name, result in all_results.items():
        all_metrics[exp_name] = result['test_metrics']

    for name, result in baseline_results.items():
        all_metrics[f'baseline_{name}'] = result['metrics']

    comparison_str = compare_metrics(all_metrics, 'sMAPE')
    logger.info(comparison_str)

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

    plot_metrics_comparison(
        {k: v for k, v in list(all_metrics.items())[:15]},
        ['sMAPE'],
        'Price: All Experiments Comparison',
        output_dir / 'all_experiments_comparison.png'
    )

    logger.info(f"\nResults saved to: {output_dir}")
    logger.info(f"Best model: {summary['best_model']} with sMAPE={summary['best_smape']:.4f}%")

    return all_results, baseline_results


if __name__ == "__main__":
    run_all_price_experiments()
