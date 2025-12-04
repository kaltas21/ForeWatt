"""
Feature Importance Analysis & Essential Feature Discovery
==========================================================
Find the best ~20 features for price_real prediction.

Uses winning LightGBM config to get feature importance,
then tests compact feature sets.

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
from typing import Dict, List, Tuple
import logging
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# D-1 SAFE FEATURE POOL (from your column list, excluding forbidden)
# =============================================================================

FORBIDDEN = [
    'datetime', 'timestamp', 'price', 'priceUsd', 'priceEur', 'DID_index',
    'price_real', 'priceUsd_real', 'priceEur_real',  # targets
    'price_ptf_lag_1h', 'consumption_lag_1h', 'consumption_lag_2h',
    'consumption_lag_3h', 'consumption_lag_6h', 'consumption_lag_12h',
    'temp_lag_1h', 'temp_lag_2h', 'temp_lag_3h', 'temperature_lag_1h',
    'temperature_lag_2h', 'temperature_lag_3h',  # <24h lags
    'temp_change_1h', 'temp_change_3h',  # <24h changes
    'price_smf', 'price_ptf_raw',  # current prices
    'hour_y', 'month_y', 'is_weekend_y', 'dow_sin_y', 'dow_cos_y',  # duplicates
    'holiday_name', 'consumption',  # string/target
    'reserve_margin_ratio_lag_1h', 'renewable_saturation_lag_1h',
    'thermal_gap_lag_1h', 'import_cost_proxy_lag_1h',
    'spark_spread_proxy_lag_1h', 'load_factor_lag_1h',  # <24h fundamental lags
]

# All available D-1 safe features
ALL_SAFE_FEATURES = [
    # Calendar (6)
    'hour_sin', 'hour_cos', 'dow_sin_x', 'dow_cos_x', 'is_weekend_x', 'is_holiday_day',
    'month_sin', 'month_cos', 'day_of_week', 'day_of_year', 'dom', 'weekofyear',

    # Weather (14)
    'temp_national', 'humidity_national', 'wind_speed_national',
    'apparent_temp_national', 'precipitation_national', 'cloud_cover_national',
    'HDD', 'CDD', 'HDD_15', 'CDD_21', 'heat_index', 'wind_chill',
    'is_hot', 'is_very_hot', 'is_cold', 'is_very_cold',
    'is_raining', 'is_heavy_rain', 'is_cloudy',

    # Temperature lags/rolling (10)
    'temp_lag_24h', 'temp_lag_168h', 'temperature_lag_24h', 'temperature_lag_168h',
    'temp_rolling_24h', 'temp_rolling_7d', 'temp_rolling_168h',
    'temp_std', 'temp_std_24h', 'temp_change_24h', 'temp_shock', 'temp_range_24h',
    'temperature_rolling_mean_24h', 'temperature_rolling_std_24h',
    'temperature_rolling_mean_168h', 'temperature_rolling_std_168h',

    # Consumption lags/rolling (10)
    'consumption_lag_24h', 'consumption_lag_48h', 'consumption_lag_168h',
    'consumption_rolling_mean_24h', 'consumption_rolling_std_24h',
    'consumption_rolling_min_24h', 'consumption_rolling_max_24h',
    'consumption_rolling_mean_168h', 'consumption_rolling_std_168h',
    'consumption_range_24h', 'consumption_cv_24h',
    'consumption_forecast',

    # Price lags/rolling (10)
    'price_ptf_lag_24h', 'price_ptf_lag_168h',
    'price_ptf_rolling_mean_24h', 'price_ptf_rolling_std_24h',
    'price_ptf_rolling_min_24h', 'price_ptf_rolling_max_24h',
    'price_ptf_rolling_mean_168h', 'price_ptf_rolling_std_168h',
    'price_ptf_rolling_min_168h', 'price_ptf_rolling_max_168h',
    'price_smf_lag_24h', 'price_ptf_lag_24h_raw',

    # Fundamental ratios (8)
    'reserve_margin_ratio', 'renewable_saturation', 'thermal_gap',
    'load_factor', 'spark_spread_proxy', 'import_cost_proxy',
    'system_short_signal', 'renewable_capacity_share',
    'realtime_premium_lag24h', 'price_volatility_lag24h',

    # Fundamental lags 24h+ (18)
    'reserve_margin_ratio_lag_24h', 'reserve_margin_ratio_lag_48h', 'reserve_margin_ratio_lag_168h',
    'renewable_saturation_lag_24h', 'renewable_saturation_lag_48h', 'renewable_saturation_lag_168h',
    'thermal_gap_lag_24h', 'thermal_gap_lag_48h', 'thermal_gap_lag_168h',
    'load_factor_lag_24h', 'load_factor_lag_48h', 'load_factor_lag_168h',
    'spark_spread_proxy_lag_24h', 'spark_spread_proxy_lag_48h', 'spark_spread_proxy_lag_168h',
    'import_cost_proxy_lag_24h', 'import_cost_proxy_lag_48h', 'import_cost_proxy_lag_168h',

    # Fundamental rolling (20)
    'reserve_margin_ratio_rolling_mean_24h', 'reserve_margin_ratio_rolling_std_24h',
    'reserve_margin_ratio_rolling_mean_168h', 'reserve_margin_ratio_rolling_std_168h',
    'renewable_saturation_rolling_mean_24h', 'renewable_saturation_rolling_std_24h',
    'renewable_saturation_rolling_mean_168h', 'renewable_saturation_rolling_std_168h',
    'thermal_gap_rolling_mean_24h', 'thermal_gap_rolling_std_24h',
    'thermal_gap_rolling_mean_168h', 'thermal_gap_rolling_std_168h',
    'system_short_signal_rolling_mean_24h', 'system_short_signal_rolling_std_24h',
    'system_short_signal_rolling_mean_168h', 'system_short_signal_rolling_std_168h',
    'load_factor_rolling_mean_24h', 'load_factor_rolling_std_24h',
    'load_factor_rolling_mean_168h', 'load_factor_rolling_std_168h',

    # Capacity/FX (8)
    'capacity_eak', 'wind_forecast', 'hydro_energy',
    'USD_TRY', 'EUR_TRY', 'FX_basket', 'FX_volatility',
]

# =============================================================================
# CURATED FEATURE SETS (max 20 each)
# =============================================================================

FEATURE_SETS = {
    # Set 1: Minimal fundamental (12 features)
    'minimal_fundamental': [
        'hour_sin', 'hour_cos', 'dow_sin_x', 'dow_cos_x', 'is_weekend_x', 'is_holiday_day',
        'price_ptf_lag_24h', 'price_ptf_rolling_mean_24h',
        'reserve_margin_ratio', 'renewable_saturation', 'spark_spread_proxy',
        'temp_national',
    ],

    # Set 2: Price-focused (15 features)
    'price_focused': [
        'hour_sin', 'hour_cos', 'dow_sin_x', 'dow_cos_x', 'is_weekend_x',
        'price_ptf_lag_24h', 'price_ptf_lag_168h',
        'price_ptf_rolling_mean_24h', 'price_ptf_rolling_std_24h',
        'price_ptf_rolling_mean_168h',
        'reserve_margin_ratio', 'load_factor',
        'system_short_signal', 'realtime_premium_lag24h', 'price_volatility_lag24h',
    ],

    # Set 3: Supply-demand (16 features)
    'supply_demand': [
        'hour_sin', 'hour_cos', 'dow_sin_x', 'dow_cos_x', 'is_weekend_x', 'is_holiday_day',
        'price_ptf_lag_24h',
        'consumption_forecast', 'consumption_lag_24h',
        'capacity_eak', 'wind_forecast',
        'reserve_margin_ratio', 'renewable_saturation', 'thermal_gap', 'load_factor',
        'temp_national',
    ],

    # Set 4: Weather-heavy (18 features)
    'weather_heavy': [
        'hour_sin', 'hour_cos', 'dow_sin_x', 'dow_cos_x', 'is_weekend_x',
        'price_ptf_lag_24h', 'price_ptf_rolling_mean_24h',
        'temp_national', 'HDD', 'CDD', 'humidity_national',
        'is_hot', 'is_cold', 'temp_lag_24h',
        'reserve_margin_ratio', 'renewable_saturation',
        'consumption_forecast', 'wind_forecast',
    ],

    # Set 5: Ratios only (14 features)
    'ratios_only': [
        'hour_sin', 'hour_cos', 'dow_sin_x', 'dow_cos_x', 'is_weekend_x',
        'reserve_margin_ratio', 'renewable_saturation', 'load_factor',
        'spark_spread_proxy', 'realtime_premium_lag24h',
        'system_short_signal', 'price_volatility_lag24h',
        'price_ptf_lag_24h', 'price_ptf_lag_168h',
    ],

    # Set 6: Lag-heavy (18 features)
    'lag_heavy': [
        'hour_sin', 'hour_cos', 'dow_sin_x', 'dow_cos_x', 'is_weekend_x',
        'price_ptf_lag_24h', 'price_ptf_lag_168h',
        'consumption_lag_24h', 'consumption_lag_168h',
        'temp_lag_24h', 'temp_lag_168h',
        'reserve_margin_ratio_lag_24h', 'renewable_saturation_lag_24h',
        'load_factor_lag_24h', 'thermal_gap_lag_24h',
        'reserve_margin_ratio', 'temp_national', 'consumption_forecast',
    ],

    # Set 7: Rolling stats (18 features)
    'rolling_stats': [
        'hour_sin', 'hour_cos', 'dow_sin_x', 'dow_cos_x', 'is_weekend_x',
        'price_ptf_rolling_mean_24h', 'price_ptf_rolling_std_24h',
        'price_ptf_rolling_mean_168h', 'price_ptf_rolling_std_168h',
        'consumption_rolling_mean_24h', 'consumption_rolling_std_24h',
        'reserve_margin_ratio_rolling_mean_24h',
        'reserve_margin_ratio', 'load_factor',
        'temp_national', 'temp_rolling_24h',
        'consumption_forecast', 'capacity_eak',
    ],

    # Set 8: Compact essential (10 features) - absolute minimum
    'compact_essential': [
        'hour_sin', 'hour_cos', 'dow_sin_x', 'dow_cos_x',
        'price_ptf_lag_24h',
        'reserve_margin_ratio', 'load_factor',
        'temp_national', 'consumption_forecast', 'is_weekend_x',
    ],
}

# =============================================================================
# WINNING CONFIG (LightGBM - fastest for feature importance)
# =============================================================================

LIGHTGBM_CONFIG = {
    'n_estimators': 800,
    'max_depth': 8,
    'learning_rate': 0.05,
    'num_leaves': 63,
    'min_child_samples': 30,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
}


def load_data():
    """Load master dataset."""
    data_path = PROJECT_ROOT / 'data' / 'gold' / 'master' / 'master_v2_fundamental.parquet'
    df = pd.read_parquet(data_path)
    return df


def get_feature_importance(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Train LightGBM and get feature importance."""
    import lightgbm as lgb

    # Filter available features
    available = [f for f in features if f in df.columns]
    logger.info(f"Available features: {len(available)}/{len(features)}")

    X = df[available].copy()
    y = df['price_real'].copy()

    # Drop NaN
    mask = ~(X.isna().any(axis=1) | y.isna())
    X, y = X[mask], y[mask]

    # Split
    n = len(X)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]

    # Train
    model = lgb.LGBMRegressor(
        objective='regression',
        metric='mae',
        verbosity=-1,
        random_state=42,
        **LIGHTGBM_CONFIG
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )

    # Get importance
    importance = pd.DataFrame({
        'feature': available,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    return importance, model


def evaluate_feature_set(
    df: pd.DataFrame,
    features: List[str],
    set_name: str
) -> Dict:
    """Evaluate a feature set."""
    import lightgbm as lgb

    available = [f for f in features if f in df.columns]

    if len(available) < 5:
        return {'status': 'skipped', 'reason': 'Too few features'}

    X = df[available].copy()
    y = df['price_real'].copy()

    mask = ~(X.isna().any(axis=1) | y.isna())
    X, y = X[mask], y[mask]

    n = len(X)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

    model = lgb.LGBMRegressor(
        objective='regression',
        metric='mae',
        verbosity=-1,
        random_state=42,
        **LIGHTGBM_CONFIG
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    def smape(y_true, y_pred):
        return 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

    def mae(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    val_smape = smape(y_val.values, val_pred)
    test_smape = smape(y_test.values, test_pred)

    return {
        'set_name': set_name,
        'n_features': len(available),
        'features': available,
        'val_smape': val_smape,
        'test_smape': test_smape,
        'val_mae': mae(y_val.values, val_pred),
        'test_mae': mae(y_test.values, test_pred),
        'gap_ratio': test_smape / val_smape if val_smape > 0 else 0,
        'status': 'success'
    }


def run_feature_search():
    """Run complete feature importance analysis and search."""
    output_dir = PROJECT_ROOT / 'reports' / 'feature_search'
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("\n" + "="*70)
    logger.info("FEATURE IMPORTANCE & ESSENTIAL FEATURE DISCOVERY")
    logger.info("="*70)

    # Load data
    df = load_data()
    logger.info(f"Loaded data: {df.shape}")

    # Step 1: Get feature importance from all safe features
    logger.info("\n" + "-"*70)
    logger.info("STEP 1: Feature Importance Analysis")
    logger.info("-"*70)

    safe_features = [f for f in ALL_SAFE_FEATURES if f in df.columns]
    importance, _ = get_feature_importance(df, safe_features)

    logger.info(f"\nTop 30 Features by Importance:")
    for i, row in importance.head(30).iterrows():
        logger.info(f"  {row['feature']:40s} {row['importance']:>8.0f}")

    # Save importance
    importance.to_csv(output_dir / 'feature_importance.csv', index=False)

    # Step 2: Create importance-based feature sets
    logger.info("\n" + "-"*70)
    logger.info("STEP 2: Testing Curated Feature Sets")
    logger.info("-"*70)

    # Add importance-based sets
    top_10 = importance.head(10)['feature'].tolist()
    top_15 = importance.head(15)['feature'].tolist()
    top_20 = importance.head(20)['feature'].tolist()

    all_sets = {
        **FEATURE_SETS,
        'importance_top10': top_10,
        'importance_top15': top_15,
        'importance_top20': top_20,
    }

    results = []
    for set_name, features in all_sets.items():
        logger.info(f"\nTesting: {set_name} ({len(features)} features)")
        result = evaluate_feature_set(df, features, set_name)
        results.append(result)

        if result['status'] == 'success':
            logger.info(f"  Val sMAPE:  {result['val_smape']:.2f}%")
            logger.info(f"  Test sMAPE: {result['test_smape']:.2f}%")
            logger.info(f"  Gap ratio:  {result['gap_ratio']:.2f}x")

    # Step 3: Summary
    logger.info("\n" + "="*70)
    logger.info("RESULTS SUMMARY")
    logger.info("="*70)

    successful = [r for r in results if r['status'] == 'success']
    results_df = pd.DataFrame(successful)

    # Sort by test sMAPE
    results_df = results_df.sort_values('test_smape')

    logger.info(f"\n{'Set Name':<25} {'#Feat':>6} {'Val%':>8} {'Test%':>8} {'Gap':>6}")
    logger.info("-"*60)
    for _, row in results_df.iterrows():
        logger.info(f"{row['set_name']:<25} {row['n_features']:>6} {row['val_smape']:>8.2f} {row['test_smape']:>8.2f} {row['gap_ratio']:>6.2f}x")

    # Best results
    best_test = results_df.iloc[0]
    best_gap = results_df.loc[results_df['gap_ratio'].idxmin()]

    logger.info(f"\n🏆 BEST TEST sMAPE: {best_test['set_name']}")
    logger.info(f"   Features: {best_test['n_features']}")
    logger.info(f"   Test sMAPE: {best_test['test_smape']:.2f}%")
    logger.info(f"   Features: {best_test['features']}")

    logger.info(f"\n🎯 BEST GENERALIZATION: {best_gap['set_name']}")
    logger.info(f"   Gap ratio: {best_gap['gap_ratio']:.2f}x")
    logger.info(f"   Features: {best_gap['features']}")

    # Save results
    results_df.to_csv(output_dir / 'feature_set_results.csv', index=False)

    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'best_test_smape': {
            'set_name': best_test['set_name'],
            'n_features': int(best_test['n_features']),
            'test_smape': float(best_test['test_smape']),
            'features': best_test['features'],
        },
        'best_generalization': {
            'set_name': best_gap['set_name'],
            'gap_ratio': float(best_gap['gap_ratio']),
            'features': best_gap['features'],
        },
        'top_20_important_features': importance.head(20)['feature'].tolist(),
    }

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n✅ Results saved to: {output_dir}")

    return results_df, importance


if __name__ == "__main__":
    run_feature_search()
