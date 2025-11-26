"""
Baseline Feature Preparer for Tree-Based Models
================================================
Feature preparation for CatBoost, XGBoost, LightGBM, and Prophet.

D-1 SAFE: All features verified available at prediction time for day-ahead forecasting.
Uses the same forbidden columns as deep learning module.

Author: ForeWatt Team - New Experiment V2
Date: November 2025
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse D-1 safety from deep learning module
from src.models.new_experiment.deeplearning.feature_preparer_v2 import (
    FORBIDDEN_COLUMNS,
    D1_SAFE_COLUMNS,
    load_master_v2
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# BASELINE FEATURE STRATEGIES
# =============================================================================
# Tree-based models can handle more features without overfitting as easily,
# so we provide different tier distributions.

BASELINE_FEATURE_STRATEGIES = {
    # =========================================================================
    # PRICE FORECASTING STRATEGIES
    # =========================================================================

    # -------------------------------------------------------------------------
    # TIER 1: MINIMAL (~15 features) - Fast iteration
    # -------------------------------------------------------------------------
    'baseline_minimal': {
        'description': 'Minimal curated features for fast iteration',
        'target': 'price_real',
        'features': [
            # Core fundamentals
            'reserve_margin_ratio',
            'renewable_saturation',
            'spark_spread_proxy',
            'system_short_signal',
            # Price lags
            'price_ptf_lag_24h',
            'price_ptf_lag_168h',
            # Calendar
            'hour_sin', 'hour_cos',
            'dow_sin_x', 'dow_cos_x',
            'is_weekend_x',
            'is_holiday_day',
            # Weather
            'temp_national',
            'HDD',
            'CDD',
        ]
    },

    # -------------------------------------------------------------------------
    # TIER 2: CORE (~30 features) - Recommended default
    # -------------------------------------------------------------------------
    'baseline_core': {
        'description': 'Core balanced features - recommended default',
        'target': 'price_real',
        'features': [
            # Fundamentals
            'reserve_margin_ratio',
            'renewable_saturation',
            'load_factor',
            'spark_spread_proxy',
            'realtime_premium_lag24h',
            'system_short_signal',
            'thermal_gap',
            'price_volatility_lag24h',
            # Price features
            'price_ptf_lag_24h',
            'price_ptf_lag_168h',
            'price_ptf_rolling_mean_24h',
            'price_ptf_rolling_std_24h',
            # Consumption
            'consumption_forecast',
            'consumption_lag_24h',
            # Calendar
            'hour_sin', 'hour_cos',
            'dow_sin_x', 'dow_cos_x',
            'month_sin', 'month_cos',
            'is_weekend_x',
            'is_holiday_day',
            'day_of_week',
            # Weather
            'temp_national',
            'HDD',
            'CDD',
            # FX
            'USD_TRY',
        ]
    },

    # -------------------------------------------------------------------------
    # TIER 3: EXTENDED (~50 features) - More signals
    # -------------------------------------------------------------------------
    'baseline_extended': {
        'description': 'Extended features with more lags and rolling stats',
        'target': 'price_real',
        'features': [
            # Fundamentals
            'reserve_margin_ratio',
            'renewable_saturation',
            'load_factor',
            'spark_spread_proxy',
            'realtime_premium_lag24h',
            'renewable_capacity_share',
            'system_short_signal',
            'thermal_gap',
            'price_volatility_lag24h',
            'import_cost_proxy',
            # Fundamental lags
            'reserve_margin_ratio_lag_24h',
            'reserve_margin_ratio_lag_168h',
            'renewable_saturation_lag_24h',
            'load_factor_lag_24h',
            # Fundamental rolling
            'reserve_margin_ratio_rolling_mean_24h',
            'system_short_signal_rolling_mean_24h',
            # Price features
            'price_ptf_lag_24h',
            'price_ptf_lag_168h',
            'price_ptf_rolling_mean_24h',
            'price_ptf_rolling_std_24h',
            'price_ptf_rolling_mean_168h',
            'price_ptf_rolling_min_24h',
            'price_ptf_rolling_max_24h',
            # Consumption
            'consumption_forecast',
            'consumption_lag_24h',
            'consumption_lag_168h',
            'consumption_rolling_mean_24h',
            # Calendar
            'hour_sin', 'hour_cos',
            'dow_sin_x', 'dow_cos_x',
            'month_sin', 'month_cos',
            'is_weekend_x',
            'is_holiday_day',
            'is_holiday_hour',
            'day_of_week',
            'day_of_year',
            # Weather
            'temp_national',
            'HDD',
            'CDD',
            'heat_index',
            'is_hot',
            'is_cold',
            'temp_lag_24h',
            # FX
            'USD_TRY',
            'EUR_TRY',
        ]
    },

    # -------------------------------------------------------------------------
    # TIER 4: FULL (~70 features) - Maximum signal
    # -------------------------------------------------------------------------
    'baseline_full': {
        'description': 'Full feature set for maximum signal',
        'target': 'price_real',
        'features': [
            # Fundamentals
            'reserve_margin_ratio',
            'renewable_saturation',
            'load_factor',
            'spark_spread_proxy',
            'realtime_premium_lag24h',
            'renewable_capacity_share',
            'system_short_signal',
            'thermal_gap',
            'price_volatility_lag24h',
            'import_cost_proxy',
            # Fundamental lags
            'reserve_margin_ratio_lag_24h',
            'reserve_margin_ratio_lag_48h',
            'reserve_margin_ratio_lag_168h',
            'renewable_saturation_lag_24h',
            'renewable_saturation_lag_168h',
            'thermal_gap_lag_24h',
            'load_factor_lag_24h',
            'load_factor_lag_168h',
            'spark_spread_proxy_lag_24h',
            'import_cost_proxy_lag_24h',
            # Fundamental rolling
            'reserve_margin_ratio_rolling_mean_24h',
            'reserve_margin_ratio_rolling_std_24h',
            'renewable_saturation_rolling_mean_24h',
            'thermal_gap_rolling_mean_24h',
            'system_short_signal_rolling_mean_24h',
            'load_factor_rolling_mean_24h',
            # Price features
            'price_ptf_lag_24h',
            'price_ptf_lag_168h',
            'price_smf_lag_24h',
            'price_ptf_rolling_mean_24h',
            'price_ptf_rolling_std_24h',
            'price_ptf_rolling_min_24h',
            'price_ptf_rolling_max_24h',
            'price_ptf_rolling_mean_168h',
            'price_ptf_rolling_std_168h',
            # Consumption
            'consumption_forecast',
            'consumption_lag_24h',
            'consumption_lag_48h',
            'consumption_lag_168h',
            'consumption_rolling_mean_24h',
            'consumption_rolling_std_24h',
            'consumption_rolling_mean_168h',
            # D-1 forecasts
            'capacity_eak',
            'wind_forecast',
            'hydro_energy',
            # Calendar
            'hour_sin', 'hour_cos',
            'dow_sin_x', 'dow_cos_x',
            'month_sin', 'month_cos',
            'is_weekend_x',
            'is_holiday_day',
            'is_holiday_hour',
            'day_of_week',
            'day_of_year',
            'weekofyear',
            # Weather
            'temp_national',
            'humidity_national',
            'HDD',
            'CDD',
            'heat_index',
            'is_hot',
            'is_very_hot',
            'is_cold',
            'is_very_cold',
            'temp_lag_24h',
            'temp_lag_168h',
            'temp_rolling_24h',
            # FX
            'USD_TRY',
            'EUR_TRY',
            'FX_basket',
        ]
    },

    # =========================================================================
    # CONSUMPTION FORECASTING STRATEGIES
    # =========================================================================

    # -------------------------------------------------------------------------
    # CONSUMPTION MINIMAL (~15 features)
    # -------------------------------------------------------------------------
    'consumption_baseline_minimal': {
        'description': 'Minimal features for consumption forecasting',
        'target': 'consumption',
        'features': [
            # Consumption lags
            'consumption_lag_24h',
            'consumption_lag_168h',
            # Weather
            'temp_national',
            'HDD',
            'CDD',
            'is_hot',
            'is_cold',
            # Calendar
            'hour_sin', 'hour_cos',
            'dow_sin_x', 'dow_cos_x',
            'is_weekend_x',
            'is_holiday_day',
        ]
    },

    # -------------------------------------------------------------------------
    # CONSUMPTION CORE (~25 features)
    # -------------------------------------------------------------------------
    'consumption_baseline_core': {
        'description': 'Core features for consumption forecasting',
        'target': 'consumption',
        'features': [
            # Consumption lags
            'consumption_lag_24h',
            'consumption_lag_48h',
            'consumption_lag_168h',
            # Consumption rolling
            'consumption_rolling_mean_24h',
            'consumption_rolling_std_24h',
            # Weather
            'temp_national',
            'humidity_national',
            'HDD',
            'CDD',
            'heat_index',
            'is_hot',
            'is_cold',
            # Temperature lags
            'temp_lag_24h',
            # Calendar
            'hour_sin', 'hour_cos',
            'dow_sin_x', 'dow_cos_x',
            'month_sin', 'month_cos',
            'is_weekend_x',
            'is_holiday_day',
            'is_holiday_hour',
            # Price signal
            'price_ptf_lag_24h',
        ]
    },

    # -------------------------------------------------------------------------
    # CONSUMPTION FULL (~45 features)
    # -------------------------------------------------------------------------
    'consumption_baseline_full': {
        'description': 'Full features for consumption forecasting',
        'target': 'consumption',
        'features': [
            # Consumption lags
            'consumption_lag_24h',
            'consumption_lag_48h',
            'consumption_lag_168h',
            # Consumption rolling
            'consumption_rolling_mean_24h',
            'consumption_rolling_std_24h',
            'consumption_rolling_min_24h',
            'consumption_rolling_max_24h',
            'consumption_rolling_mean_168h',
            'consumption_rolling_std_168h',
            # Weather
            'temp_national',
            'humidity_national',
            'wind_speed_national',
            'HDD',
            'CDD',
            'heat_index',
            'is_hot',
            'is_very_hot',
            'is_cold',
            'is_very_cold',
            # Temperature features
            'temp_lag_24h',
            'temp_lag_168h',
            'temp_rolling_24h',
            'temperature_rolling_mean_24h',
            # Calendar
            'hour_sin', 'hour_cos',
            'dow_sin_x', 'dow_cos_x',
            'month_sin', 'month_cos',
            'is_weekend_x',
            'is_holiday_day',
            'is_holiday_hour',
            'day_of_week',
            'day_of_year',
            # Fundamental
            'load_factor',
            'load_factor_lag_24h',
            'reserve_margin_ratio',
            # Price signals
            'price_ptf_lag_24h',
            'price_ptf_lag_168h',
            'price_ptf_rolling_mean_24h',
            # D-1 forecasts
            'capacity_eak',
            'wind_forecast',
        ]
    },
}

# Feature tier mapping for price forecasting
PRICE_FEATURE_TIERS = {
    'minimal': 'baseline_minimal',
    'core': 'baseline_core',
    'extended': 'baseline_extended',
    'full': 'baseline_full',
}

# Feature tier mapping for consumption forecasting
CONSUMPTION_FEATURE_TIERS = {
    'minimal': 'consumption_baseline_minimal',
    'core': 'consumption_baseline_core',
    'full': 'consumption_baseline_full',
}


class BaselineFeaturePreparer:
    """
    Feature preparer for baseline models (CatBoost, XGBoost, LightGBM, Prophet).

    All features are D-1 safe (available at day-ahead prediction time).
    """

    def __init__(
        self,
        target: str = 'price_real',
        strategy: str = 'baseline_core',
        custom_features: List[str] = None
    ):
        """
        Initialize baseline feature preparer.

        Args:
            target: Target variable (default: 'price_real')
            strategy: Feature selection strategy from BASELINE_FEATURE_STRATEGIES
            custom_features: Optional list of custom features to include
        """
        self.target = target
        self.strategy = strategy
        self.custom_features = custom_features or []

        # Validate strategy
        if strategy not in BASELINE_FEATURE_STRATEGIES:
            raise ValueError(
                f"Unknown strategy: {strategy}. "
                f"Available: {list(BASELINE_FEATURE_STRATEGIES.keys())}"
            )

        self.strategy_config = BASELINE_FEATURE_STRATEGIES[strategy]
        self.feature_names = None

        logger.info(f"Initialized BaselineFeaturePreparer")
        logger.info(f"  Strategy: {strategy}")
        logger.info(f"  Target: {target}")

    def get_feature_list(self) -> List[str]:
        """Get the complete list of features for the selected strategy."""
        features = list(self.strategy_config['features'])
        features.extend(self.custom_features)
        return features

    def get_available_features(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of features that are available in the dataframe.
        Automatically filters out FORBIDDEN_COLUMNS to ensure D-1 safety.
        """
        requested = self.get_feature_list()

        # Filter: feature exists in df
        available = [f for f in requested if f in df.columns]

        # CRITICAL: Filter out any forbidden columns for D-1 safety
        safe_features = [f for f in available if f not in FORBIDDEN_COLUMNS]

        # Log if any forbidden columns were filtered
        forbidden_found = set(available) - set(safe_features)
        if forbidden_found:
            logger.warning(f"Filtered {len(forbidden_found)} forbidden columns: {sorted(forbidden_found)}")

        return safe_features

    def prepare_features(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Prepare features for baseline models.

        Args:
            df: Master dataset (master_v2_fundamental.parquet)

        Returns:
            Tuple of (X, y, feature_names)
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"PREPARING BASELINE FEATURES (D-1 SAFE)")
        logger.info(f"Strategy: {self.strategy}")
        logger.info(f"{'='*80}")

        # Get available features (automatically filters forbidden)
        available_features = self.get_available_features(df)

        # Log missing features
        requested = self.get_feature_list()
        missing = set(requested) - set(available_features)
        if missing:
            logger.warning(f"Missing {len(missing)} features:")
            for f in sorted(list(missing))[:10]:
                logger.warning(f"  - {f}")

        # Extract features and target
        X = df[available_features].copy()
        y = df[self.target].copy()

        # Store feature names
        self.feature_names = available_features

        # Log summary
        logger.info(f"\nFeature set prepared: {len(available_features)} features (D-1 SAFE)")
        logger.info(f"  Target: {self.target}")
        logger.info(f"  Samples: {len(X)}")

        return X, y, available_features

    def prepare_train_val_test(
        self,
        df: pd.DataFrame,
        val_size: float = 0.2,
        test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
               pd.Series, pd.Series, pd.Series, List[str]]:
        """
        Prepare features with train/val/test split.

        Args:
            df: Master dataset
            val_size: Validation set fraction
            test_size: Test set fraction

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, feature_names)
        """
        X, y, feature_names = self.prepare_features(df)

        # Drop NaN rows
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        logger.info(f"After dropping NaN: {len(X)} samples")

        # Temporal split
        n = len(X)
        train_end = int(n * (1 - val_size - test_size))
        val_end = int(n * (1 - test_size))

        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        X_val = X.iloc[train_end:val_end]
        y_val = y.iloc[train_end:val_end]
        X_test = X.iloc[val_end:]
        y_test = y.iloc[val_end:]

        logger.info(f"\nData splits:")
        logger.info(f"  Train: {len(X_train)} ({len(X_train)/n*100:.1f}%)")
        logger.info(f"  Val:   {len(X_val)} ({len(X_val)/n*100:.1f}%)")
        logger.info(f"  Test:  {len(X_test)} ({len(X_test)/n*100:.1f}%)")

        return X_train, X_val, X_test, y_train, y_val, y_test, feature_names


def get_feature_strategy_for_tier(tier: str, target: str = 'price_real') -> str:
    """Get the feature strategy name for a given tier and target."""
    if target == 'price_real':
        return PRICE_FEATURE_TIERS.get(tier, 'baseline_core')
    else:
        return CONSUMPTION_FEATURE_TIERS.get(tier, 'consumption_baseline_core')
