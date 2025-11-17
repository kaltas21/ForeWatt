"""
Intelligent Feature Selection for ForeWatt Baseline Models
==========================================================
Automatically selects optimal features for different model types:
- Statistical models (Prophet, SARIMAX): Core features without lags
- Boosting models (CatBoost, XGBoost, LightGBM): All features

Author: ForeWatt Team
Date: November 2025
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureSelector:
    """
    Intelligent feature selector for different model types and targets.
    """

    def __init__(self, target: str = 'consumption'):
        """
        Initialize feature selector.

        Args:
            target: Target variable ('consumption' or 'price_real')
        """
        self.target = target
        self.feature_groups = self._define_feature_groups()

    def _define_feature_groups(self) -> Dict[str, List[str]]:
        """
        Define feature groups for intelligent selection.

        Returns:
            Dictionary of feature group patterns
        """
        return {
            # Core temporal features (all models)
            'temporal_core': [
                'hour_x', 'hour_y', 'hour_sin', 'hour_cos',
                'dow', 'dom', 'day_of_week', 'day_of_year',
                'month_x', 'month_y', 'weekofyear',
                'dow_sin_x', 'dow_cos_x', 'dow_sin_y', 'dow_cos_y',
                'month_sin', 'month_cos'
            ],

            # Calendar/holiday features (all models)
            'calendar': [
                'is_weekend_x', 'is_weekend_y',
                'is_holiday_day', 'is_holiday_hour',
                'holiday_name'  # Categorical for boosting
            ],

            # Weather features (all models)
            'weather_core': [
                'temp_national', 'humidity_national',
                'wind_speed_national', 'apparent_temp_national',
                'precipitation_national', 'cloud_cover_national',
                'temp_std'
            ],

            # Weather derived (all models)
            'weather_derived': [
                'HDD', 'CDD', 'HDD_15', 'CDD_21',
                'heat_index', 'wind_chill',
                'is_hot', 'is_very_hot', 'is_cold', 'is_very_cold',
                'is_raining', 'is_heavy_rain', 'is_cloudy'
            ],

            # Temperature change features (good for all)
            'temp_dynamics': [
                'temp_change_1h', 'temp_change_3h', 'temp_change_24h',
                'temp_shock', 'temp_range_24h'
            ],

            # Lag features (BOOSTING ONLY - statistical models handle autocorrelation internally)
            'consumption_lags': [
                'consumption_lag_1h', 'consumption_lag_2h', 'consumption_lag_3h',
                'consumption_lag_6h', 'consumption_lag_12h', 'consumption_lag_24h',
                'consumption_lag_48h', 'consumption_lag_168h'
            ],

            'temperature_lags': [
                'temp_lag_1h', 'temp_lag_2h', 'temp_lag_3h',
                'temp_lag_24h', 'temp_lag_168h',
                'temperature_lag_1h', 'temperature_lag_2h', 'temperature_lag_3h',
                'temperature_lag_24h', 'temperature_lag_168h'
            ],

            'price_lags': [
                'price_ptf_lag_1h', 'price_ptf_lag_24h', 'price_ptf_lag_168h'
            ],

            # Rolling statistics (BOOSTING ONLY)
            'consumption_rolling': [
                'consumption_rolling_mean_24h', 'consumption_rolling_std_24h',
                'consumption_rolling_min_24h', 'consumption_rolling_max_24h',
                'consumption_rolling_mean_168h', 'consumption_rolling_std_168h',
                'consumption_rolling_min_168h', 'consumption_rolling_max_168h',
                'consumption_range_24h', 'consumption_cv_24h'
            ],

            'temperature_rolling': [
                'temp_rolling_24h', 'temp_rolling_7d', 'temp_std_24h',
                'temperature_rolling_mean_24h', 'temperature_rolling_std_24h',
                'temperature_rolling_min_24h', 'temperature_rolling_max_24h',
                'temperature_rolling_mean_168h', 'temperature_rolling_std_168h',
                'temperature_rolling_min_168h', 'temperature_rolling_max_168h'
            ],

            'price_rolling': [
                'price_ptf_rolling_mean_24h', 'price_ptf_rolling_std_24h',
                'price_ptf_rolling_min_24h', 'price_ptf_rolling_max_24h',
                'price_ptf_rolling_mean_168h', 'price_ptf_rolling_std_168h',
                'price_ptf_rolling_min_168h', 'price_ptf_rolling_max_168h'
            ],

            # Price features (for price prediction target)
            'price_current': [
                'price', 'priceUsd', 'priceEur',
                'DID_index'  # Deflation index
            ],

            # Cross-domain features
            'cross_consumption_price': [
                'consumption',  # Current consumption (for price prediction)
                'price_real', 'priceUsd_real', 'priceEur_real'  # Real prices (for demand prediction)
            ]
        }

    def get_features_for_model_type(
        self,
        model_type: str,
        target: Optional[str] = None
    ) -> List[str]:
        """
        Get optimal features for a specific model type and target.

        Args:
            model_type: One of ['prophet', 'sarimax', 'catboost', 'xgboost', 'lightgbm']
            target: Target variable (override self.target if provided)

        Returns:
            List of feature names
        """
        if target is None:
            target = self.target

        model_type = model_type.lower()

        # Statistical models: No lags, no rolling (they model autocorrelation internally)
        if model_type in ['prophet', 'sarimax']:
            features = self._get_statistical_features(target)

        # Boosting models: All features
        elif model_type in ['catboost', 'xgboost', 'lightgbm']:
            features = self._get_boosting_features(target)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        logger.info(f"\n{'='*80}")
        logger.info(f"FEATURE SELECTION: {model_type.upper()} for {target}")
        logger.info(f"{'='*80}")
        logger.info(f"Selected {len(features)} features")

        return features

    def _get_statistical_features(self, target: str) -> List[str]:
        """
        Get features for statistical models (Prophet, SARIMAX).

        Statistical models:
        - Handle autocorrelation internally (no need for lags)
        - Use external regressors for weather, calendar effects
        - Keep it simple with core predictive features

        Args:
            target: Target variable

        Returns:
            List of feature names
        """
        feature_groups = [
            'temporal_core',
            'calendar',
            'weather_core',
            'weather_derived',
            'temp_dynamics'
        ]

        # For price prediction, add current consumption as regressor
        if target == 'price_real':
            feature_groups.append('cross_consumption_price')

        # Collect all features from selected groups
        features = []
        for group in feature_groups:
            features.extend(self.feature_groups[group])

        # Remove holiday_name for statistical models (text categorical)
        if 'holiday_name' in features:
            features.remove('holiday_name')

        logger.info(f"Statistical model features: {len(features)} total")
        logger.info(f"  - Temporal/Calendar: ✓")
        logger.info(f"  - Weather: ✓")
        logger.info(f"  - Lag features: ✗ (models handle autocorrelation)")
        logger.info(f"  - Rolling stats: ✗ (models handle autocorrelation)")

        return list(set(features))  # Remove duplicates

    def _get_boosting_features(self, target: str) -> List[str]:
        """
        Get features for gradient boosting models.

        Boosting models:
        - Can handle many features effectively
        - Benefit from lag features
        - Benefit from rolling statistics
        - Can handle feature interactions

        Args:
            target: Target variable

        Returns:
            List of feature names
        """
        # Start with all feature groups except target-specific ones
        feature_groups = [
            'temporal_core',
            'calendar',
            'weather_core',
            'weather_derived',
            'temp_dynamics',
            'temperature_lags',
            'temperature_rolling'
        ]

        if target == 'consumption':
            # For demand prediction
            feature_groups.extend([
                'consumption_lags',
                'consumption_rolling',
                'price_lags',
                'price_rolling',
                'price_current'
            ])

        elif target == 'price_real':
            # For price prediction
            feature_groups.extend([
                'consumption_lags',
                'consumption_rolling',
                'price_lags',
                'price_rolling',
                'cross_consumption_price'
            ])

        # Collect all features from selected groups
        features = []
        for group in feature_groups:
            features.extend(self.feature_groups[group])

        features = list(set(features))  # Remove duplicates

        logger.info(f"Boosting model features: {len(features)} total")
        logger.info(f"  - Temporal/Calendar: ✓")
        logger.info(f"  - Weather: ✓")
        logger.info(f"  - Lag features: ✓")
        logger.info(f"  - Rolling stats: ✓")

        return features

    def prepare_features(
        self,
        df: pd.DataFrame,
        model_type: str,
        target: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Prepare features for a specific model type.

        Args:
            df: Master dataset
            model_type: Model type
            target: Target variable (override self.target if provided)

        Returns:
            Tuple of (X, y, selected_features)
        """
        if target is None:
            target = self.target

        # Get optimal features for this model type
        selected_features = self.get_features_for_model_type(model_type, target)

        # Filter to available features in dataframe
        available_features = [f for f in selected_features if f in df.columns]
        missing_features = set(selected_features) - set(available_features)

        if missing_features:
            logger.warning(f"Missing {len(missing_features)} features: {missing_features}")

        # Exclude target and datetime columns
        exclude_cols = [
            target, 'datetime', 'timestamp',
            'date_only', 'date', 'time'
        ]

        feature_cols = [f for f in available_features if f not in exclude_cols]

        # Handle holiday_name for boosting models (fill NaN)
        df = df.copy()
        if 'holiday_name' in feature_cols:
            df['holiday_name'] = df['holiday_name'].fillna('no_holiday')

        # Extract features and target
        X = df[feature_cols].copy()
        y = df[target].copy()

        logger.info(f"Final feature set: {len(feature_cols)} features")
        logger.info(f"Target: {target} ({len(y)} samples)")
        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")

        return X, y, feature_cols

    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """
        Get feature groups for analysis.

        Returns:
            Dictionary of feature groups
        """
        return self.feature_groups

    def print_feature_summary(self, model_type: str, target: Optional[str] = None):
        """
        Print feature selection summary.

        Args:
            model_type: Model type
            target: Target variable
        """
        if target is None:
            target = self.target

        features = self.get_features_for_model_type(model_type, target)

        print(f"\n{'='*80}")
        print(f"FEATURE SUMMARY: {model_type.upper()} → {target}")
        print(f"{'='*80}")
        print(f"Total features: {len(features)}")
        print(f"\nFeature breakdown:")

        for group_name, group_features in self.feature_groups.items():
            count = len([f for f in features if f in group_features])
            if count > 0:
                print(f"  {group_name:30s}: {count:3d} features")

        print(f"{'='*80}\n")


def get_categorical_features(feature_list: List[str]) -> List[str]:
    """
    Identify categorical features from feature list.

    Args:
        feature_list: List of feature names

    Returns:
        List of categorical feature names
    """
    categorical = []

    # Binary flags
    binary_prefixes = ['is_', 'has_']
    for feature in feature_list:
        if any(feature.startswith(prefix) for prefix in binary_prefixes):
            categorical.append(feature)

    # Explicit categorical
    explicit_categorical = ['holiday_name', 'dow', 'dom', 'month_x', 'month_y']
    for feature in feature_list:
        if feature in explicit_categorical and feature not in categorical:
            categorical.append(feature)

    return categorical
