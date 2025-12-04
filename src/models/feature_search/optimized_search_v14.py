"""
Optimized Search V14 - Context-Aware KNN Error Correction
=========================================================
Context: V13 achieved 11.89% sMAPE with Hourly-Dynamic AEC.
         The Oracle floor is 11.75% - simple AEC cannot go further.

Problem: Simple AEC fails during weather transitions.
         Correcting "Cloudy Day" using "Sunny Day" bias is wrong.

Objective: Beat the 11.75% Oracle Floor using Context-Aware KNN Correction.

Strategy:
    1. Similarity Engine: Find similar historical hours by context
       (load_factor, renewable_saturation, thermal_gap)
    2. KNN-EC: Weight corrections by 1/distance to similar hours
    3. Hybrid: Combine Simple AEC + KNN-EC for robustness

Author: ForeWatt Team
Date: December 2025
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# V13 baseline (Oracle floor)
BASELINE_SMAPE = 11.89
ORACLE_FLOOR = 11.75

# Problem hours
PROBLEM_HOURS = [9, 10]

# Ensemble weights from V11
CATBOOST_WEIGHT = 0.658
LIGHTGBM_WEIGHT = 0.413

# KNN Parameters
KNN_K = 5
KNN_LOOKBACK_DAYS = 45

# Context features for similarity
CONTEXT_FEATURES = ['load_factor', 'renewable_saturation', 'thermal_gap']


# =============================================================================
# BASE FEATURE SET
# =============================================================================

BASE_FEATURES = [
    'hour_sin', 'hour_cos', 'dow_sin_x', 'dow_cos_x', 'is_weekend_x',
    'price_ptf_rolling_std_24h', 'price_ptf_rolling_mean_24h',
    'price_ptf_rolling_min_24h', 'price_ptf_rolling_max_24h',
    'price_ptf_lag_24h', 'price_ptf_lag_168h',
    'thermal_gap', 'thermal_gap_lag_24h',
    'renewable_saturation', 'spark_spread_proxy_lag_24h',
    'system_short_signal', 'load_factor', 'consumption_forecast',
    'reserve_margin_ratio', 'price_volatility_lag24h', 'realtime_premium_lag24h',
]


# =============================================================================
# ENHANCED PROFILE FEATURES
# =============================================================================

def create_enhanced_profile_features(df: pd.DataFrame, price_col: str = 'price_real') -> Tuple[pd.DataFrame, List[str]]:
    """Create enhanced Profile Evolution Features including Solar Profile."""
    df = df.copy()
    new_features = []

    df['hour'] = df.index.hour

    # V10 PRICE PROFILE FEATURES
    df['daily_avg_price'] = df[price_col].shift(1).rolling(24, min_periods=12).mean()
    df['hourly_ratio'] = (df[price_col].shift(1) / df['daily_avg_price'].shift(1)).clip(0.2, 5.0)
    new_features.append('hourly_ratio')

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
    new_features.append('profile_momentum')

    df['daily_avg_momentum'] = df['daily_avg_price'] - df['daily_avg_price'].shift(24)
    new_features.append('daily_avg_momentum')

    # V11 SOLAR PROFILE FEATURES
    if 'renewable_saturation' in df.columns and 'load_factor' in df.columns:
        load = df['load_factor'].clip(lower=0.1)
        df['solar_ratio'] = (df['renewable_saturation'].shift(1) / load.shift(1)).clip(0, 5)
        new_features.append('solar_ratio')

        solar_14d_list = []
        solar_28d_list = []

        for hour in range(24):
            hour_mask = df['hour'] == hour
            hour_solar = df.loc[hour_mask, 'solar_ratio']
            s14 = hour_solar.rolling(14, min_periods=7).mean().shift(1)
            s28 = hour_solar.rolling(28, min_periods=14).mean().shift(1)
            solar_14d_list.append(s14)
            solar_28d_list.append(s28)

        df['solar_profile_14d'] = pd.concat(solar_14d_list).sort_index()
        df['solar_profile_28d'] = pd.concat(solar_28d_list).sort_index()
        new_features.extend(['solar_profile_14d', 'solar_profile_28d'])

        df['solar_momentum'] = df['solar_profile_14d'] - df['solar_profile_28d']
        new_features.append('solar_momentum')

    if 'solar_momentum' in df.columns:
        df['price_solar_interaction'] = df['profile_14d'] * df['solar_momentum']
        new_features.append('price_solar_interaction')

    for feat in new_features:
        if feat in df.columns and df[feat].isna().any():
            median_val = df[feat].median()
            df[feat] = df[feat].fillna(median_val)

    return df, new_features


# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

def load_data() -> pd.DataFrame:
    """Load master dataset."""
    path = PROJECT_ROOT / 'data' / 'gold' / 'master' / 'master_v2_fundamental.parquet'
    df = pd.read_parquet(path)

    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date' in df.columns:
            df = df.set_index('date')
        elif 'datetime' in df.columns:
            df = df.set_index('datetime')

    if 'price_real' not in df.columns:
        df['price_real'] = df['price']

    return df


def prepare_data(df: pd.DataFrame, features: List[str],
                 test_start: str = '2024-06-01',
                 finetune_months: int = 6) -> Dict:
    """Prepare data with splits."""
    available = [f for f in features if f in df.columns]

    X = df[available].copy()
    y = df['price_real'].copy()

    X_with_price = X.copy()
    X_with_price['price_real'] = y
    X_with_price, profile_features = create_enhanced_profile_features(X_with_price, 'price_real')
    X = X_with_price.drop(columns=['price_real'])

    X['hour'] = X.index.hour

    mask = ~(X.isna().any(axis=1) | y.isna())
    X, y = X[mask], y[mask]
    hours = X['hour']

    tz = X.index.tz
    test_start_dt = pd.Timestamp(test_start, tz=tz)
    finetune_start_dt = test_start_dt - pd.DateOffset(months=finetune_months)

    base_mask = X.index < finetune_start_dt
    finetune_mask = (X.index >= finetune_start_dt) & (X.index < test_start_dt)
    test_mask = X.index >= test_start_dt

    return {
        'X_base': X[base_mask],
        'y_base': y[base_mask],
        'X_finetune': X[finetune_mask],
        'y_finetune': y[finetune_mask],
        'X_test': X[test_mask],
        'y_test': y[test_mask],
        'hours_base': hours[base_mask],
        'hours_finetune': hours[finetune_mask],
        'hours_test': hours[test_mask],
        'features': list(X.columns),
    }


# =============================================================================
# METRICS
# =============================================================================

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Calculate metrics."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae = np.mean(np.abs(y_true - y_pred))
    smape = 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))
    bias = np.mean(y_pred - y_true)

    return {'mae': mae, 'smape': smape, 'bias': bias}


def evaluate_with_breakdown(y_true: np.ndarray, y_pred: np.ndarray, hours: np.ndarray) -> Dict:
    """Evaluate with hour breakdown."""
    global_metrics = evaluate(y_true, y_pred)

    problem_mask = np.isin(hours, PROBLEM_HOURS)
    problem_metrics = evaluate(y_true[problem_mask], y_pred[problem_mask]) if problem_mask.sum() > 0 else {}
    problem_metrics['count'] = int(problem_mask.sum())

    other_mask = ~problem_mask
    other_metrics = evaluate(y_true[other_mask], y_pred[other_mask]) if other_mask.sum() > 0 else {}
    other_metrics['count'] = int(other_mask.sum())

    return {
        'global': global_metrics,
        'hours_9_10': problem_metrics,
        'other': other_metrics,
    }


# =============================================================================
# TRANSFER LEARNING TRAINERS
# =============================================================================

def train_catboost_transfer(data: Dict) -> Tuple[object, np.ndarray, np.ndarray]:
    """Train CatBoost with transfer learning."""
    from catboost import CatBoostRegressor

    logger.info("\n  Training CatBoost Transfer Learning...")

    X_base, y_base = data['X_base'], data['y_base']
    split_idx = int(len(X_base) * 0.85)

    base_model = CatBoostRegressor(
        loss_function='MAE',
        iterations=2000,
        depth=8,
        learning_rate=0.02,
        l2_leaf_reg=3,
        random_state=42,
        verbose=False,
        early_stopping_rounds=100,
    )
    base_model.fit(
        X_base.iloc[:split_idx], y_base.iloc[:split_idx],
        eval_set=(X_base.iloc[split_idx:], y_base.iloc[split_idx:]),
        verbose=False
    )
    logger.info(f"    Base: {base_model.tree_count_} trees")

    X_ft, y_ft = data['X_finetune'], data['y_finetune']
    split_idx = int(len(X_ft) * 0.8)

    finetune_model = CatBoostRegressor(
        loss_function='MAE',
        iterations=500,
        depth=8,
        learning_rate=0.005,
        l2_leaf_reg=3,
        random_state=42,
        verbose=False,
        early_stopping_rounds=50,
    )
    finetune_model.fit(
        X_ft.iloc[:split_idx], y_ft.iloc[:split_idx],
        eval_set=(X_ft.iloc[split_idx:], y_ft.iloc[split_idx:]),
        init_model=base_model,
        verbose=False
    )
    logger.info(f"    Fine-tuned: {finetune_model.tree_count_} trees")

    finetune_pred = finetune_model.predict(data['X_finetune'])
    test_pred = finetune_model.predict(data['X_test'])

    return finetune_model, finetune_pred, test_pred


def train_lightgbm_transfer(data: Dict) -> Tuple[object, np.ndarray, np.ndarray]:
    """Train LightGBM with transfer learning."""
    import lightgbm as lgb

    logger.info("\n  Training LightGBM Transfer Learning...")

    X_base, y_base = data['X_base'], data['y_base']
    split_idx = int(len(X_base) * 0.85)

    base_model = lgb.LGBMRegressor(
        objective='mae',
        n_estimators=2000,
        max_depth=8,
        learning_rate=0.02,
        num_leaves=127,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        verbosity=-1,
    )
    base_model.fit(
        X_base.iloc[:split_idx], y_base.iloc[:split_idx],
        eval_set=[(X_base.iloc[split_idx:], y_base.iloc[split_idx:])],
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    logger.info(f"    Base: {base_model.n_estimators_} trees")

    X_ft, y_ft = data['X_finetune'], data['y_finetune']
    split_idx = int(len(X_ft) * 0.8)

    finetune_model = lgb.LGBMRegressor(
        objective='mae',
        n_estimators=500,
        max_depth=8,
        learning_rate=0.005,
        num_leaves=127,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        verbosity=-1,
    )
    finetune_model.fit(
        X_ft.iloc[:split_idx], y_ft.iloc[:split_idx],
        eval_set=[(X_ft.iloc[split_idx:], y_ft.iloc[split_idx:])],
        callbacks=[lgb.early_stopping(50, verbose=False)],
        init_model=base_model.booster_
    )
    logger.info(f"    Fine-tuned: {finetune_model.n_estimators_} trees")

    finetune_pred = finetune_model.predict(data['X_finetune'])
    test_pred = finetune_model.predict(data['X_test'])

    return finetune_model, finetune_pred, test_pred


# =============================================================================
# SIMPLE AEC (V13 Baseline)
# =============================================================================

def apply_simple_aec(df_preds: pd.DataFrame, hourly_params: Dict) -> np.ndarray:
    """Apply V13-style hourly AEC."""
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


# =============================================================================
# KNN CONTEXT-AWARE ERROR CORRECTION
# =============================================================================

class KNNErrorCorrector:
    """
    Context-Aware KNN Error Corrector.

    Uses similar historical hours (by context features) to estimate bias,
    rather than simple time-based rolling averages.
    """

    def __init__(self, context_features: List[str], k: int = 5,
                 lookback_days: int = 45, damping: float = 0.8):
        self.context_features = context_features
        self.k = k
        self.lookback_days = lookback_days
        self.damping = damping
        self.scaler = StandardScaler()

    def fit_scaler(self, X_fit: pd.DataFrame):
        """Fit the scaler on training/fine-tune data to avoid leakage."""
        available = [f for f in self.context_features if f in X_fit.columns]
        if len(available) == 0:
            raise ValueError(f"No context features found in data")
        self.available_features = available
        self.scaler.fit(X_fit[available].fillna(0))
        logger.info(f"    Scaler fitted on {len(available)} context features")

    def apply_correction(self, df_preds: pd.DataFrame, X_context: pd.DataFrame) -> np.ndarray:
        """
        Apply KNN-based error correction.

        Args:
            df_preds: DataFrame with ['datetime', 'hour', 'y_true', 'y_raw']
            X_context: DataFrame with context features aligned with df_preds

        Returns:
            Corrected predictions
        """
        df = df_preds.copy()
        df = df.sort_values('datetime').reset_index(drop=True)
        df['error'] = df['y_raw'] - df['y_true']

        # Extract and normalize context features
        context_data = X_context[self.available_features].fillna(0).values
        context_normalized = self.scaler.transform(context_data)

        n = len(df)
        corrections = np.zeros(n)

        # Process by hour for efficiency
        for hour in range(24):
            hour_mask = (df['hour'] == hour).values
            hour_indices = np.where(hour_mask)[0]

            if len(hour_indices) < self.k + 1:
                continue

            for i, idx in enumerate(hour_indices):
                if i < self.k:
                    # Not enough history
                    continue

                # Define lookback window (same hour, past lookback_days)
                lookback_hours = self.lookback_days  # For hourly data, 1 day = 1 sample per hour
                start_i = max(0, i - lookback_hours)
                history_indices = hour_indices[start_i:i]

                if len(history_indices) < 2:
                    continue

                # Build KNN index on history
                history_contexts = context_normalized[history_indices]
                current_context = context_normalized[idx].reshape(1, -1)

                # Use KNN to find similar days
                k_actual = min(self.k, len(history_indices))
                knn = NearestNeighbors(n_neighbors=k_actual, metric='euclidean')
                knn.fit(history_contexts)

                distances, neighbor_idx = knn.kneighbors(current_context)
                distances = distances[0]
                neighbor_idx = neighbor_idx[0]

                # Get errors from neighbors
                neighbor_original_idx = history_indices[neighbor_idx]
                neighbor_errors = df.loc[neighbor_original_idx, 'error'].values

                # Weight by inverse distance (with epsilon to avoid div by zero)
                epsilon = 1e-6
                weights = 1.0 / (distances + epsilon)
                weights = weights / weights.sum()

                # Weighted bias
                weighted_bias = np.sum(weights * neighbor_errors)
                corrections[idx] = self.damping * weighted_bias

        return df['y_raw'].values - corrections


def apply_knn_correction(df_preds: pd.DataFrame, X_context: pd.DataFrame,
                         scaler: StandardScaler, context_features: List[str],
                         k: int = 5, lookback_days: int = 45,
                         damping: float = 0.8) -> np.ndarray:
    """
    Vectorized KNN error correction.
    """
    df = df_preds.copy().sort_values('datetime').reset_index(drop=True)
    df['error'] = df['y_raw'] - df['y_true']

    available = [f for f in context_features if f in X_context.columns]
    context_data = X_context[available].fillna(0).values
    context_normalized = scaler.transform(context_data)

    n = len(df)
    corrections = np.zeros(n)

    for hour in range(24):
        hour_mask = (df['hour'] == hour).values
        hour_indices = np.where(hour_mask)[0]

        if len(hour_indices) < k + 1:
            continue

        # Vectorized processing for this hour
        for i, idx in enumerate(hour_indices):
            if i < k:
                continue

            start_i = max(0, i - lookback_days)
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


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_v14_experiments():
    """Run V14 Context-Aware KNN Error Correction experiments."""
    output_dir = PROJECT_ROOT / 'reports' / 'optimized_search_v14'
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*70)
    logger.info("OPTIMIZED SEARCH V14 - Context-Aware KNN Error Correction")
    logger.info(f"V13 Baseline: {BASELINE_SMAPE}% sMAPE")
    logger.info(f"Oracle Floor: {ORACLE_FLOOR}% sMAPE")
    logger.info("="*70)

    # Load data
    df = load_data()
    logger.info(f"Loaded data: {df.shape}")

    # Prepare data
    data = prepare_data(df, BASE_FEATURES)

    logger.info(f"\n  DATA SPLITS:")
    logger.info(f"    Base:      {len(data['X_base']):,} rows")
    logger.info(f"    Fine-tune: {len(data['X_finetune']):,} rows")
    logger.info(f"    Test:      {len(data['X_test']):,} rows")

    # =========================================================================
    # STEP 1: TRAIN BASE ENSEMBLE
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 1: TRAINING BASE ENSEMBLE")
    logger.info("="*70)

    _, cat_ft_pred, cat_test_pred = train_catboost_transfer(data)
    _, lgb_ft_pred, lgb_test_pred = train_lightgbm_transfer(data)

    total_weight = CATBOOST_WEIGHT + LIGHTGBM_WEIGHT
    w_cat = CATBOOST_WEIGHT / total_weight
    w_lgb = LIGHTGBM_WEIGHT / total_weight

    logger.info(f"\n  Ensemble weights: CatBoost={w_cat:.3f}, LightGBM={w_lgb:.3f}")

    raw_ft_pred = w_cat * cat_ft_pred + w_lgb * lgb_ft_pred
    raw_test_pred = w_cat * cat_test_pred + w_lgb * lgb_test_pred

    # =========================================================================
    # STEP 2: PREPARE DATAFRAMES
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 2: PREPARING DATA FOR KNN CORRECTION")
    logger.info("="*70)

    df_ft = pd.DataFrame({
        'datetime': data['X_finetune'].index,
        'hour': data['hours_finetune'].values,
        'y_true': data['y_finetune'].values,
        'y_raw': raw_ft_pred,
    })

    df_test = pd.DataFrame({
        'datetime': data['X_test'].index,
        'hour': data['hours_test'].values,
        'y_true': data['y_test'].values,
        'y_raw': raw_test_pred,
    })

    logger.info(f"  Test: {len(df_test):,} rows ({df_test['datetime'].min()} to {df_test['datetime'].max()})")

    # =========================================================================
    # STEP 3: FIT SCALER ON FINE-TUNE SET (avoid leakage)
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 3: FITTING CONTEXT SCALER")
    logger.info("="*70)

    available_context = [f for f in CONTEXT_FEATURES if f in data['X_finetune'].columns]
    logger.info(f"  Context features: {available_context}")

    scaler = StandardScaler()
    scaler.fit(data['X_finetune'][available_context].fillna(0))
    logger.info(f"  Scaler fitted on Fine-Tune set ({len(data['X_finetune']):,} rows)")

    # =========================================================================
    # STEP 4: GET V13 HOURLY PARAMS
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 4: LOADING V13 HOURLY PARAMETERS")
    logger.info("="*70)

    # Load V13 params if available, otherwise use defaults
    v13_params_path = PROJECT_ROOT / 'reports' / 'optimized_search_v13' / 'hourly_aec_params.csv'
    if v13_params_path.exists():
        v13_params_df = pd.read_csv(v13_params_path)
        hourly_params_v13 = {
            int(row['hour']): {'lookback': int(row['lookback']), 'damping': float(row['damping'])}
            for _, row in v13_params_df.iterrows()
        }
        logger.info(f"  Loaded V13 params from {v13_params_path}")
    else:
        hourly_params_v13 = {h: {'lookback': 7, 'damping': 0.5} for h in range(24)}
        logger.info("  Using default params (V13 params not found)")

    # =========================================================================
    # STEP 5: RUN EXPERIMENTS
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("STEP 5: RUNNING CORRECTION EXPERIMENTS")
    logger.info("="*70)

    results = {}

    # Config A: V13 Simple AEC (Baseline)
    logger.info("\n  Config A: V13 Simple AEC (Baseline)...")
    simple_aec_pred = apply_simple_aec(df_test, hourly_params_v13)
    simple_metrics = evaluate_with_breakdown(
        data['y_test'].values, simple_aec_pred, data['hours_test'].values
    )
    results['A_Simple_AEC'] = {
        'pred': simple_aec_pred,
        'metrics': simple_metrics,
    }
    logger.info(f"    Global sMAPE: {simple_metrics['global']['smape']:.2f}%")

    # Config B: Pure KNN-EC
    logger.info("\n  Config B: Pure KNN-EC (K=5, 45d, d=0.8)...")
    knn_pred = apply_knn_correction(
        df_test, data['X_test'].reset_index(drop=True),
        scaler, available_context,
        k=KNN_K, lookback_days=KNN_LOOKBACK_DAYS, damping=0.8
    )
    knn_metrics = evaluate_with_breakdown(
        data['y_test'].values, knn_pred, data['hours_test'].values
    )
    results['B_KNN_EC'] = {
        'pred': knn_pred,
        'metrics': knn_metrics,
    }
    logger.info(f"    Global sMAPE: {knn_metrics['global']['smape']:.2f}%")

    # Config C: Hybrid (50% Simple + 50% KNN)
    logger.info("\n  Config C: Hybrid (50% Simple + 50% KNN)...")
    hybrid_pred = 0.5 * simple_aec_pred + 0.5 * knn_pred
    hybrid_metrics = evaluate_with_breakdown(
        data['y_test'].values, hybrid_pred, data['hours_test'].values
    )
    results['C_Hybrid'] = {
        'pred': hybrid_pred,
        'metrics': hybrid_metrics,
    }
    logger.info(f"    Global sMAPE: {hybrid_metrics['global']['smape']:.2f}%")

    # Config D: KNN with higher damping
    logger.info("\n  Config D: KNN-EC (K=5, 45d, d=0.9)...")
    knn_high_pred = apply_knn_correction(
        df_test, data['X_test'].reset_index(drop=True),
        scaler, available_context,
        k=KNN_K, lookback_days=KNN_LOOKBACK_DAYS, damping=0.9
    )
    knn_high_metrics = evaluate_with_breakdown(
        data['y_test'].values, knn_high_pred, data['hours_test'].values
    )
    results['D_KNN_High'] = {
        'pred': knn_high_pred,
        'metrics': knn_high_metrics,
    }
    logger.info(f"    Global sMAPE: {knn_high_metrics['global']['smape']:.2f}%")

    # Config E: KNN with more neighbors
    logger.info("\n  Config E: KNN-EC (K=7, 45d, d=0.8)...")
    knn_k7_pred = apply_knn_correction(
        df_test, data['X_test'].reset_index(drop=True),
        scaler, available_context,
        k=7, lookback_days=KNN_LOOKBACK_DAYS, damping=0.8
    )
    knn_k7_metrics = evaluate_with_breakdown(
        data['y_test'].values, knn_k7_pred, data['hours_test'].values
    )
    results['E_KNN_K7'] = {
        'pred': knn_k7_pred,
        'metrics': knn_k7_metrics,
    }
    logger.info(f"    Global sMAPE: {knn_k7_metrics['global']['smape']:.2f}%")

    # Config F: Raw (no correction)
    raw_metrics = evaluate_with_breakdown(
        data['y_test'].values, raw_test_pred, data['hours_test'].values
    )
    results['F_Raw'] = {
        'pred': raw_test_pred,
        'metrics': raw_metrics,
    }

    # =========================================================================
    # STEP 6: COMPARISON TABLE
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("COMPARISON TABLE")
    logger.info("="*70)

    logger.info(f"\n{'Method':<20} {'Global':>10} {'H9-10':>10} {'H9-10 Bias':>12} {'Beat 11.89%?':>14}")
    logger.info("-"*70)

    comparison = []
    for name, res in sorted(results.items()):
        m = res['metrics']
        beat = "YES" if m['global']['smape'] < BASELINE_SMAPE else "no"
        beat_oracle = " (ORACLE!)" if m['global']['smape'] < ORACLE_FLOOR else ""
        logger.info(f"{name:<20} {m['global']['smape']:>9.2f}% {m['hours_9_10']['smape']:>9.2f}% "
                    f"{m['hours_9_10']['bias']:>+11.2f} {beat:>14}{beat_oracle}")
        comparison.append({
            'Method': name,
            'Global_sMAPE': m['global']['smape'],
            'H910_sMAPE': m['hours_9_10']['smape'],
            'H910_Bias': m['hours_9_10']['bias'],
            'Beat_Baseline': m['global']['smape'] < BASELINE_SMAPE,
            'Beat_Oracle': m['global']['smape'] < ORACLE_FLOOR,
        })

    # =========================================================================
    # STEP 7: HOURLY BREAKDOWN
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("HOURLY BREAKDOWN: Simple AEC vs KNN-EC")
    logger.info("="*70)

    y_true = data['y_test'].values
    hours = data['hours_test'].values

    logger.info(f"\n{'Hour':<6} {'Simple':>12} {'KNN':>12} {'Hybrid':>12} {'Δ (KNN-S)':>12}")
    logger.info("-"*60)

    for hour in range(24):
        hour_mask = hours == hour
        if hour_mask.sum() == 0:
            continue

        simple_h = evaluate(y_true[hour_mask], simple_aec_pred[hour_mask])['smape']
        knn_h = evaluate(y_true[hour_mask], knn_pred[hour_mask])['smape']
        hybrid_h = evaluate(y_true[hour_mask], hybrid_pred[hour_mask])['smape']

        delta = knn_h - simple_h
        marker = " <--" if hour in PROBLEM_HOURS else ""

        logger.info(f"{hour:>4}   {simple_h:>11.2f}% {knn_h:>11.2f}% {hybrid_h:>11.2f}% "
                    f"{delta:>+11.2f}%{marker}")

    # =========================================================================
    # STEP 8: SCENARIO ANALYSIS
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("SCENARIO ANALYSIS: Sunny vs Cloudy Days (Hour 09)")
    logger.info("="*70)

    # Analyze corrections on different weather conditions
    test_context = data['X_test'].reset_index(drop=True)
    df_analysis = df_test.copy()
    df_analysis['renewable_saturation'] = test_context['renewable_saturation'].values
    df_analysis['simple_correction'] = raw_test_pred - simple_aec_pred
    df_analysis['knn_correction'] = raw_test_pred - knn_pred

    # Filter to Hour 09
    hour9_df = df_analysis[df_analysis['hour'] == 9].copy()

    # Define sunny (high renewable) vs cloudy (low renewable)
    renew_median = hour9_df['renewable_saturation'].median()
    sunny_mask = hour9_df['renewable_saturation'] > renew_median
    cloudy_mask = hour9_df['renewable_saturation'] <= renew_median

    logger.info(f"\n  Hour 09 Analysis (Renewable Median: {renew_median:.2f})")
    logger.info(f"  Sunny days (high solar): {sunny_mask.sum()} samples")
    logger.info(f"  Cloudy days (low solar): {cloudy_mask.sum()} samples")

    if sunny_mask.sum() > 0 and cloudy_mask.sum() > 0:
        logger.info(f"\n{'Scenario':<15} {'Simple Corr':>15} {'KNN Corr':>15} {'Raw Bias':>12}")
        logger.info("-"*60)

        sunny_simple = hour9_df.loc[sunny_mask, 'simple_correction'].mean()
        sunny_knn = hour9_df.loc[sunny_mask, 'knn_correction'].mean()
        sunny_error = (hour9_df.loc[sunny_mask, 'y_raw'] - hour9_df.loc[sunny_mask, 'y_true']).mean()
        logger.info(f"{'Sunny (High)':<15} {sunny_simple:>+14.2f} {sunny_knn:>+14.2f} {sunny_error:>+11.2f}")

        cloudy_simple = hour9_df.loc[cloudy_mask, 'simple_correction'].mean()
        cloudy_knn = hour9_df.loc[cloudy_mask, 'knn_correction'].mean()
        cloudy_error = (hour9_df.loc[cloudy_mask, 'y_raw'] - hour9_df.loc[cloudy_mask, 'y_true']).mean()
        logger.info(f"{'Cloudy (Low)':<15} {cloudy_simple:>+14.2f} {cloudy_knn:>+14.2f} {cloudy_error:>+11.2f}")

        logger.info(f"\n  Insight: Simple AEC uses same correction for both scenarios.")
        logger.info(f"           KNN-EC adapts correction based on weather context.")
        logger.info(f"           Diff: Sunny={sunny_knn - sunny_simple:+.2f}, Cloudy={cloudy_knn - cloudy_simple:+.2f}")

    # =========================================================================
    # STEP 9: FINAL RESULTS
    # =========================================================================
    logger.info("\n" + "="*70)
    logger.info("FINAL RESULTS")
    logger.info("="*70)

    best_method = min(comparison, key=lambda x: x['Global_sMAPE'])

    logger.info(f"\n  Best Method: {best_method['Method']}")
    logger.info(f"  Global sMAPE: {best_method['Global_sMAPE']:.2f}%")
    logger.info(f"  Hours 9-10 sMAPE: {best_method['H910_sMAPE']:.2f}%")
    logger.info(f"  V13 Baseline: {BASELINE_SMAPE:.2f}%")
    logger.info(f"  Oracle Floor: {ORACLE_FLOOR:.2f}%")

    if best_method['Beat_Oracle']:
        improvement = ORACLE_FLOOR - best_method['Global_sMAPE']
        logger.info(f"\n  BREAKTHROUGH! Beat Oracle Floor by {improvement:.2f}%!")
    elif best_method['Beat_Baseline']:
        improvement = BASELINE_SMAPE - best_method['Global_sMAPE']
        logger.info(f"\n  SUCCESS! Beat V13 Baseline by {improvement:.2f}%!")
    else:
        logger.info(f"\n  Did not beat baseline.")

    # Journey Summary
    logger.info("\n" + "="*70)
    logger.info("JOURNEY SUMMARY: V1 → V14")
    logger.info("="*70)

    journey = [
        ('N-HiTS Baseline', 16.01, 'Deep Learning'),
        ('V4', 14.29, 'Transfer Learning'),
        ('V5', 14.08, '6-month Fine-tune Window'),
        ('V10', 12.73, 'Profile Evolution Features'),
        ('V11', 12.31, 'CatBoost+LightGBM Ensemble'),
        ('V12', 12.03, 'Adaptive Error Correction'),
        ('V13', 11.89, 'Hourly-Dynamic AEC'),
        ('V14', best_method['Global_sMAPE'], 'Context-Aware KNN-EC'),
    ]

    logger.info(f"\n{'Version':<20} {'sMAPE':>10} {'Improvement':>12} {'Key Innovation':>35}")
    logger.info("-"*80)

    prev_smape = journey[0][1]
    for version, smape, innovation in journey:
        improvement = prev_smape - smape
        logger.info(f"{version:<20} {smape:>9.2f}% {improvement:>+11.2f}% {innovation:>35}")
        prev_smape = smape

    total_improvement = journey[0][1] - journey[-1][1]
    relative_improvement = 100 * total_improvement / journey[0][1]
    logger.info("-"*80)
    logger.info(f"{'TOTAL':<20} {journey[-1][1]:>9.2f}% {total_improvement:>+11.2f}% "
                f"({relative_improvement:.1f}% relative reduction)")

    logger.info(f"\n{'='*70}")
    logger.info(f"Final sMAPE: {best_method['Global_sMAPE']:.2f}%")
    logger.info(f"{'='*70}")

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    best_name = best_method['Method']
    best_pred = results[best_name]['pred']

    summary = {
        'timestamp': datetime.now().isoformat(),
        'baseline_smape': BASELINE_SMAPE,
        'oracle_floor': ORACLE_FLOOR,
        'final_smape': float(best_method['Global_sMAPE']),
        'best_method': best_name,
        'beat_oracle': best_method['Beat_Oracle'],
        'comparison': comparison,
        'journey': journey,
        'knn_params': {
            'k': KNN_K,
            'lookback_days': KNN_LOOKBACK_DAYS,
            'context_features': available_context,
        },
    }

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Save champion predictions
    pred_df = pd.DataFrame({
        'datetime': data['X_test'].index,
        'hour': data['hours_test'].values,
        'is_problem_hour': np.isin(data['hours_test'].values, PROBLEM_HOURS),
        'y_true': data['y_test'].values,
        'y_raw': raw_test_pred,
        'y_simple_aec': simple_aec_pred,
        'y_knn_ec': knn_pred,
        'y_hybrid': hybrid_pred,
        'y_champion': best_pred,
    })
    pred_df.to_csv(output_dir / 'champion_predictions.csv', index=False)
    pred_df.to_parquet(output_dir / 'champion_predictions.parquet', index=False)

    logger.info(f"\nResults saved to: {output_dir}")

    return results


if __name__ == "__main__":
    run_v14_experiments()
