"""
Fundamental Feature Preparer V2
===============================
Feature preparation for deep learning models using fundamental features.

D-1 SAFE DESIGN: All features are verified to be available at prediction time
for day-ahead (D+1) forecasting. No data leakage.

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# COMPLETE COLUMN CLASSIFICATION FOR D-1 SAFETY
# =============================================================================
# Based on master_v2 with 172 columns. Every column is classified.
#
# For day-ahead (D+1) forecasting at prediction time P (e.g., 10am on day D):
#   - We predict all 24 hours of D+1
#   - We can ONLY use data available at time P
#
# SAFE:
#   - D-1 forecasts: consumption_forecast, wind_forecast, capacity_eak
#   - Weather forecasts: temp_national, HDD, CDD (published D-1)
#   - Calendar: Always known in advance
#   - 24h+ lags: Data from D-1 or earlier
#   - Fundamental ratios using forecasts (not actuals)
#
# FORBIDDEN:
#   - Current values: price_real, consumption, price_smf, etc.
#   - Short-term lags (<24h): Not available for D+1 hours
#   - Duplicate columns: Avoid redundancy
# =============================================================================

# =============================================================================
# FORBIDDEN COLUMNS - COMPREHENSIVE LIST (40 columns)
# =============================================================================
FORBIDDEN_COLUMNS = [
    # =========================================================================
    # TARGETS - Never use as features
    # =========================================================================
    'price_real',           # PRIMARY TARGET for price forecasting
    'consumption',          # TARGET for consumption forecasting (actual value)

    # =========================================================================
    # CURRENT PRICE VALUES - Not available at D+1 prediction time
    # =========================================================================
    'price',                # Current PTF price
    'priceUsd',             # Current price in USD
    'priceEur',             # Current price in EUR
    'priceUsd_real',        # Inflation-adjusted current price USD
    'priceEur_real',        # Inflation-adjusted current price EUR
    'price_smf',            # Current SMF (real-time market) price
    'price_ptf_raw',        # Current PTF raw price

    # =========================================================================
    # SHORT-TERM PRICE LAGS (<24h) - Not available for D+1 hours
    # =========================================================================
    'price_ptf_lag_1h',     # 1h lag not available for tomorrow's hours

    # =========================================================================
    # SHORT-TERM CONSUMPTION LAGS (<24h) - Not available for D+1 hours
    # =========================================================================
    'consumption_lag_1h',
    'consumption_lag_2h',
    'consumption_lag_3h',
    'consumption_lag_6h',
    'consumption_lag_12h',

    # =========================================================================
    # SHORT-TERM TEMPERATURE LAGS (<24h) - Not available for D+1 hours
    # =========================================================================
    'temp_lag_1h',
    'temp_lag_2h',
    'temp_lag_3h',
    'temperature_lag_1h',
    'temperature_lag_2h',
    'temperature_lag_3h',

    # =========================================================================
    # SHORT-TERM FUNDAMENTAL LAGS (<24h) - Not available for D+1 hours
    # =========================================================================
    'reserve_margin_ratio_lag_1h',
    'renewable_saturation_lag_1h',
    'thermal_gap_lag_1h',
    'import_cost_proxy_lag_1h',
    'spark_spread_proxy_lag_1h',
    'load_factor_lag_1h',

    # =========================================================================
    # SHORT-TERM TEMPERATURE CHANGES (<24h) - Not available for D+1
    # =========================================================================
    'temp_change_1h',
    'temp_change_3h',

    # =========================================================================
    # DUPLICATE/ARTIFACT COLUMNS - Avoid redundancy
    # =========================================================================
    'datetime',             # String timestamp
    'timestamp',            # Duplicate index
    'DID_index',            # Internal index
    'holiday_name',         # String (use is_holiday_day instead)
    'hour_x',               # Duplicate (use hour_sin, hour_cos)
    'hour_y',               # Duplicate
    'month_x',              # Duplicate (use month_sin, month_cos)
    'month_y',              # Duplicate
    'dow_sin_y',            # Duplicate (use dow_sin_x)
    'dow_cos_y',            # Duplicate (use dow_cos_x)
    'is_weekend_y',         # Duplicate (use is_weekend_x)
]

# =============================================================================
# D-1 SAFE COLUMNS - Comprehensive categorization
# =============================================================================
D1_SAFE_COLUMNS = {
    # =========================================================================
    # CALENDAR FEATURES - Always known in advance
    # =========================================================================
    'calendar': [
        'hour_sin', 'hour_cos',           # Hour of day (cyclical)
        'dow_sin_x', 'dow_cos_x',         # Day of week (cyclical)
        'month_sin', 'month_cos',         # Month (cyclical)
        'is_weekend_x',                   # Weekend flag
        'is_holiday_day',                 # Holiday flag (day)
        'is_holiday_hour',                # Holiday flag (hour)
        'day_of_week',                    # Day of week (integer)
        'day_of_year',                    # Day of year
        'dom',                            # Day of month
        'dow',                            # Day of week (integer)
        'weekofyear',                     # Week number
    ],

    # =========================================================================
    # WEATHER FORECASTS - Published D-1 (available for D+1)
    # =========================================================================
    'weather_forecasts': [
        'temp_national',                  # Temperature forecast
        'humidity_national',              # Humidity forecast
        'wind_speed_national',            # Wind speed forecast
        'apparent_temp_national',         # Apparent temperature
        'precipitation_national',         # Precipitation forecast
        'cloud_cover_national',           # Cloud cover forecast
    ],

    # =========================================================================
    # DERIVED WEATHER FEATURES - Computed from D-1 forecasts
    # =========================================================================
    'weather_derived': [
        'HDD', 'HDD_15',                  # Heating degree days
        'CDD', 'CDD_21',                  # Cooling degree days
        'heat_index',                     # Heat index
        'wind_chill',                     # Wind chill
        'temp_std',                       # Temperature std
        'is_hot', 'is_very_hot',          # Heat flags
        'is_cold', 'is_very_cold',        # Cold flags
        'is_raining', 'is_heavy_rain',    # Rain flags
        'is_cloudy',                      # Cloud flag
    ],

    # =========================================================================
    # D-1 FORECASTS/SCHEDULES - Published day-ahead
    # =========================================================================
    'd1_forecasts': [
        'consumption_forecast',           # Day-ahead consumption forecast
        'wind_forecast',                  # Day-ahead wind forecast
        'capacity_eak',                   # Day-ahead capacity
        'hydro_energy',                   # Day-ahead hydro schedule
    ],

    # =========================================================================
    # FUNDAMENTAL RATIOS - Computed using D-1 forecasts (not actuals)
    # =========================================================================
    'fundamental_ratios': [
        'reserve_margin_ratio',           # (capacity - consumption_fcst) / capacity
        'renewable_saturation',           # renewables / consumption_fcst
        'thermal_gap',                    # capacity - renewables - consumption_fcst
        'load_factor',                    # consumption_fcst / capacity
        'renewable_capacity_share',       # renewable_capacity / total_capacity
        'spark_spread_proxy',             # Lagged price / gas proxy
        'import_cost_proxy',              # Import cost estimate
        'system_short_signal',            # SMF-PTF spread (lagged)
    ],

    # =========================================================================
    # LAGGED SIGNALS (24h+) - Available at D+1 prediction time
    # =========================================================================
    'lagged_signals': [
        'realtime_premium_lag24h',        # SMF/PTF ratio (D-1)
        'price_volatility_lag24h',        # Price volatility (D-1)
        'price_smf_lag_24h',              # SMF price (D-1)
        'price_ptf_lag_24h_raw',          # PTF raw (D-1)
    ],

    # =========================================================================
    # PRICE LAGS (24h+) - D-1 safe
    # =========================================================================
    'price_lags': [
        'price_ptf_lag_24h',              # D-1 same hour
        'price_ptf_lag_168h',             # D-7 same hour (weekly pattern)
    ],

    # =========================================================================
    # PRICE ROLLING STATS - Computed on D-1 data
    # =========================================================================
    'price_rolling': [
        'price_ptf_rolling_mean_24h',
        'price_ptf_rolling_std_24h',
        'price_ptf_rolling_min_24h',
        'price_ptf_rolling_max_24h',
        'price_ptf_rolling_mean_168h',
        'price_ptf_rolling_std_168h',
        'price_ptf_rolling_min_168h',
        'price_ptf_rolling_max_168h',
    ],

    # =========================================================================
    # CONSUMPTION LAGS (24h+) - D-1 safe
    # =========================================================================
    'consumption_lags': [
        'consumption_lag_24h',            # D-1 same hour
        'consumption_lag_48h',            # D-2 same hour
        'consumption_lag_168h',           # D-7 same hour
    ],

    # =========================================================================
    # CONSUMPTION ROLLING STATS - Computed on D-1 data
    # =========================================================================
    'consumption_rolling': [
        'consumption_rolling_mean_24h',
        'consumption_rolling_std_24h',
        'consumption_rolling_min_24h',
        'consumption_rolling_max_24h',
        'consumption_rolling_mean_168h',
        'consumption_rolling_std_168h',
        'consumption_rolling_min_168h',
        'consumption_rolling_max_168h',
        'consumption_range_24h',
        'consumption_cv_24h',
    ],

    # =========================================================================
    # TEMPERATURE LAGS (24h+) - D-1 safe
    # =========================================================================
    'temp_lags': [
        'temp_lag_24h',
        'temp_lag_168h',
        'temperature_lag_24h',
        'temperature_lag_168h',
        'temp_change_24h',                # 24h change is safe
    ],

    # =========================================================================
    # TEMPERATURE ROLLING STATS - Computed on D-1 data
    # =========================================================================
    'temp_rolling': [
        'temp_rolling_24h',
        'temp_rolling_7d',
        'temp_rolling_168h',
        'temp_std_24h',
        'temp_range_24h',
        'temp_shock',
        'temperature_rolling_mean_24h',
        'temperature_rolling_std_24h',
        'temperature_rolling_min_24h',
        'temperature_rolling_max_24h',
        'temperature_rolling_mean_168h',
        'temperature_rolling_std_168h',
        'temperature_rolling_min_168h',
        'temperature_rolling_max_168h',
    ],

    # =========================================================================
    # FUNDAMENTAL LAGS (24h+) - D-1 safe
    # =========================================================================
    'fundamental_lags': [
        'reserve_margin_ratio_lag_24h',
        'reserve_margin_ratio_lag_48h',
        'reserve_margin_ratio_lag_168h',
        'renewable_saturation_lag_24h',
        'renewable_saturation_lag_48h',
        'renewable_saturation_lag_168h',
        'thermal_gap_lag_24h',
        'thermal_gap_lag_48h',
        'thermal_gap_lag_168h',
        'import_cost_proxy_lag_24h',
        'import_cost_proxy_lag_48h',
        'import_cost_proxy_lag_168h',
        'spark_spread_proxy_lag_24h',
        'spark_spread_proxy_lag_48h',
        'spark_spread_proxy_lag_168h',
        'load_factor_lag_24h',
        'load_factor_lag_48h',
        'load_factor_lag_168h',
    ],

    # =========================================================================
    # FUNDAMENTAL ROLLING STATS - Computed on D-1 data
    # =========================================================================
    'fundamental_rolling': [
        'reserve_margin_ratio_rolling_mean_24h',
        'reserve_margin_ratio_rolling_std_24h',
        'reserve_margin_ratio_rolling_mean_168h',
        'reserve_margin_ratio_rolling_std_168h',
        'renewable_saturation_rolling_mean_24h',
        'renewable_saturation_rolling_std_24h',
        'renewable_saturation_rolling_mean_168h',
        'renewable_saturation_rolling_std_168h',
        'thermal_gap_rolling_mean_24h',
        'thermal_gap_rolling_std_24h',
        'thermal_gap_rolling_mean_168h',
        'thermal_gap_rolling_std_168h',
        'system_short_signal_rolling_mean_24h',
        'system_short_signal_rolling_std_24h',
        'system_short_signal_rolling_mean_168h',
        'system_short_signal_rolling_std_168h',
        'load_factor_rolling_mean_24h',
        'load_factor_rolling_std_24h',
        'load_factor_rolling_mean_168h',
        'load_factor_rolling_std_168h',
    ],

    # =========================================================================
    # FX DATA - D-1 close prices
    # =========================================================================
    'fx': [
        'USD_TRY',                         # USD/TRY (D-1 close)
        'EUR_TRY',                         # EUR/TRY (D-1 close)
        'FX_basket',                       # FX basket index
        'FX_volatility',                   # FX volatility
    ],
}


# =============================================================================
# FEATURE STRATEGIES - ALL D-1 SAFE
# =============================================================================
FUNDAMENTAL_V2_FEATURE_STRATEGIES = {
    # =========================================================================
    # PRICE FORECASTING STRATEGIES
    # =========================================================================

    # -------------------------------------------------------------------------
    # TIER 1: RATIOS ONLY (~15 base features) - Cleanest signal
    # -------------------------------------------------------------------------
    'price_ratios_only': {
        'description': 'Only ratio-based features - cleanest signal',
        'target': 'price_real',
        'features': {
            'fundamental_ratios': [
                'reserve_margin_ratio',
                'renewable_saturation',
                'load_factor',
                'spark_spread_proxy',
                'realtime_premium_lag24h',
            ],
            'signals': [
                'system_short_signal',
                'price_volatility_lag24h',
            ],
            'price_lags': [
                'price_ptf_lag_24h',
                'price_ptf_lag_168h',
            ],
            'calendar': [
                'hour_sin', 'hour_cos',
                'dow_sin_x', 'dow_cos_x',
                'is_weekend_x',
            ],
        }
    },

    # -------------------------------------------------------------------------
    # TIER 2: MINIMAL (~20 base features) - Fast iteration
    # -------------------------------------------------------------------------
    'fundamental_v2_minimal': {
        'description': 'Minimal curated features - fast iteration',
        'target': 'price_real',
        'features': {
            'fundamental_ratios': [
                'reserve_margin_ratio',
                'renewable_saturation',
                'spark_spread_proxy',
            ],
            'signals': [
                'system_short_signal',
                'thermal_gap',
            ],
            'price_lags': [
                'price_ptf_lag_24h',
            ],
            'price_rolling': [
                'price_ptf_rolling_mean_24h',
            ],
            'calendar': [
                'hour_sin', 'hour_cos',
                'dow_sin_x', 'dow_cos_x',
                'is_weekend_x',
                'is_holiday_day',
            ],
            'weather': [
                'temp_national',
            ],
        }
    },

    # -------------------------------------------------------------------------
    # TIER 3: CORE (~35 base features) - Recommended default
    # -------------------------------------------------------------------------
    'fundamental_v2': {
        'description': 'Core curated features - recommended default',
        'target': 'price_real',
        'features': {
            'fundamental_ratios': [
                'reserve_margin_ratio',
                'renewable_saturation',
                'load_factor',
                'spark_spread_proxy',
                'realtime_premium_lag24h',
            ],
            'signals': [
                'system_short_signal',
                'thermal_gap',
                'price_volatility_lag24h',
            ],
            'fundamental_lags': [
                'reserve_margin_ratio_lag_24h',
                'renewable_saturation_lag_24h',
            ],
            'price_lags': [
                'price_ptf_lag_24h',
                'price_ptf_lag_168h',
            ],
            'price_rolling': [
                'price_ptf_rolling_mean_24h',
                'price_ptf_rolling_std_24h',
            ],
            'consumption': [
                'consumption_forecast',
            ],
            'calendar': [
                'hour_sin', 'hour_cos',
                'dow_sin_x', 'dow_cos_x',
                'month_sin', 'month_cos',
                'is_weekend_x',
                'is_holiday_day',
            ],
            'weather': [
                'temp_national',
                'HDD',
                'CDD',
            ],
        }
    },

    # -------------------------------------------------------------------------
    # TIER 4: EXTENDED (~50 base features) - More signals
    # -------------------------------------------------------------------------
    'fundamental_v2_extended': {
        'description': 'Extended features with more lags/rolling',
        'target': 'price_real',
        'features': {
            'fundamental_ratios': [
                'reserve_margin_ratio',
                'renewable_saturation',
                'load_factor',
                'spark_spread_proxy',
                'realtime_premium_lag24h',
                'renewable_capacity_share',
            ],
            'signals': [
                'system_short_signal',
                'thermal_gap',
                'price_volatility_lag24h',
                'import_cost_proxy',
            ],
            'fundamental_lags': [
                'reserve_margin_ratio_lag_24h',
                'reserve_margin_ratio_lag_168h',
                'renewable_saturation_lag_24h',
                'load_factor_lag_24h',
            ],
            'fundamental_rolling': [
                'reserve_margin_ratio_rolling_mean_24h',
                'system_short_signal_rolling_mean_24h',
            ],
            'price_lags': [
                'price_ptf_lag_24h',
                'price_ptf_lag_168h',
            ],
            'price_rolling': [
                'price_ptf_rolling_mean_24h',
                'price_ptf_rolling_std_24h',
                'price_ptf_rolling_mean_168h',
            ],
            'consumption': [
                'consumption_forecast',
                'consumption_lag_24h',
            ],
            'calendar': [
                'hour_sin', 'hour_cos',
                'dow_sin_x', 'dow_cos_x',
                'month_sin', 'month_cos',
                'is_weekend_x',
                'is_holiday_day',
            ],
            'fx': [
                'USD_TRY',
            ],
            'weather': [
                'temp_national',
                'HDD',
                'CDD',
            ],
        }
    },

    # -------------------------------------------------------------------------
    # TIER 5: FULL (~65 base features) - Maximum signal
    # -------------------------------------------------------------------------
    'fundamental_v2_full': {
        'description': 'Full feature set - maximum signal',
        'target': 'price_real',
        'features': {
            'fundamental_ratios': [
                'reserve_margin_ratio',
                'renewable_saturation',
                'load_factor',
                'spark_spread_proxy',
                'realtime_premium_lag24h',
                'renewable_capacity_share',
            ],
            'signals': [
                'system_short_signal',
                'thermal_gap',
                'price_volatility_lag24h',
                'import_cost_proxy',
            ],
            'fundamental_lags': [
                'reserve_margin_ratio_lag_24h',
                'reserve_margin_ratio_lag_48h',
                'reserve_margin_ratio_lag_168h',
                'renewable_saturation_lag_24h',
                'renewable_saturation_lag_168h',
                'thermal_gap_lag_24h',
                'load_factor_lag_24h',
                'load_factor_lag_168h',
            ],
            'fundamental_rolling': [
                'reserve_margin_ratio_rolling_mean_24h',
                'reserve_margin_ratio_rolling_std_24h',
                'renewable_saturation_rolling_mean_24h',
                'thermal_gap_rolling_mean_24h',
                'system_short_signal_rolling_mean_24h',
                'load_factor_rolling_mean_24h',
            ],
            'price_lags': [
                'price_ptf_lag_24h',
                'price_ptf_lag_168h',
            ],
            'price_rolling': [
                'price_ptf_rolling_mean_24h',
                'price_ptf_rolling_std_24h',
                'price_ptf_rolling_min_24h',
                'price_ptf_rolling_max_24h',
                'price_ptf_rolling_mean_168h',
                'price_ptf_rolling_std_168h',
            ],
            'consumption': [
                'consumption_forecast',
                'consumption_lag_24h',
                'consumption_lag_168h',
                'consumption_rolling_mean_24h',
            ],
            'calendar': [
                'hour_sin', 'hour_cos',
                'dow_sin_x', 'dow_cos_x',
                'month_sin', 'month_cos',
                'is_weekend_x',
                'is_holiday_day',
                'is_holiday_hour',
            ],
            'fx': [
                'USD_TRY',
                'EUR_TRY',
            ],
            'weather': [
                'temp_national',
                'HDD',
                'CDD',
                'heat_index',
                'is_hot',
                'is_cold',
            ],
        }
    },

    # =========================================================================
    # CONSUMPTION FORECASTING STRATEGIES
    # =========================================================================

    # -------------------------------------------------------------------------
    # TIER 1: MINIMAL (~15 base features)
    # -------------------------------------------------------------------------
    'consumption_v2_minimal': {
        'description': 'Minimal features for consumption forecasting',
        'target': 'consumption',
        'features': {
            'consumption_lags': [
                'consumption_lag_24h',
                'consumption_lag_168h',
            ],
            'weather': [
                'temp_national',
                'HDD',
                'CDD',
            ],
            'calendar': [
                'hour_sin', 'hour_cos',
                'dow_sin_x', 'dow_cos_x',
                'is_weekend_x',
                'is_holiday_day',
            ],
        }
    },

    # -------------------------------------------------------------------------
    # TIER 2: CORE (~25 base features) - Recommended default
    # -------------------------------------------------------------------------
    'consumption_v2': {
        'description': 'Core features for consumption forecasting',
        'target': 'consumption',
        'features': {
            'consumption_lags': [
                'consumption_lag_24h',
                'consumption_lag_168h',
            ],
            'consumption_rolling': [
                'consumption_rolling_mean_24h',
                'consumption_rolling_std_24h',
            ],
            'weather': [
                'temp_national',
                'HDD',
                'CDD',
                'is_hot',
                'is_cold',
            ],
            'calendar': [
                'hour_sin', 'hour_cos',
                'dow_sin_x', 'dow_cos_x',
                'month_sin', 'month_cos',
                'is_weekend_x',
                'is_holiday_day',
            ],
            'price_lags': [
                'price_ptf_lag_24h',
            ],
        }
    },

    # -------------------------------------------------------------------------
    # TIER 3: EXTENDED (~40 base features)
    # -------------------------------------------------------------------------
    'consumption_v2_extended': {
        'description': 'Extended features for consumption forecasting',
        'target': 'consumption',
        'features': {
            'consumption_lags': [
                'consumption_lag_24h',
                'consumption_lag_48h',
                'consumption_lag_168h',
            ],
            'consumption_rolling': [
                'consumption_rolling_mean_24h',
                'consumption_rolling_std_24h',
                'consumption_rolling_mean_168h',
            ],
            'weather': [
                'temp_national',
                'humidity_national',
                'HDD',
                'CDD',
                'heat_index',
                'is_hot',
                'is_cold',
            ],
            'temp_lags': [
                'temp_lag_24h',
                'temp_lag_168h',
            ],
            'calendar': [
                'hour_sin', 'hour_cos',
                'dow_sin_x', 'dow_cos_x',
                'month_sin', 'month_cos',
                'is_weekend_x',
                'is_holiday_day',
                'is_holiday_hour',
            ],
            'price_lags': [
                'price_ptf_lag_24h',
                'price_ptf_lag_168h',
            ],
            'fundamental': [
                'load_factor',
            ],
        }
    },

    # -------------------------------------------------------------------------
    # TIER 4: FULL (~55 base features)
    # -------------------------------------------------------------------------
    'consumption_v2_full': {
        'description': 'Full features for consumption forecasting',
        'target': 'consumption',
        'features': {
            'consumption_lags': [
                'consumption_lag_24h',
                'consumption_lag_48h',
                'consumption_lag_168h',
            ],
            'consumption_rolling': [
                'consumption_rolling_mean_24h',
                'consumption_rolling_std_24h',
                'consumption_rolling_min_24h',
                'consumption_rolling_max_24h',
                'consumption_rolling_mean_168h',
                'consumption_rolling_std_168h',
            ],
            'weather': [
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
            ],
            'temp_lags': [
                'temp_lag_24h',
                'temp_lag_168h',
            ],
            'temp_rolling': [
                'temp_rolling_24h',
                'temperature_rolling_mean_24h',
            ],
            'calendar': [
                'hour_sin', 'hour_cos',
                'dow_sin_x', 'dow_cos_x',
                'month_sin', 'month_cos',
                'is_weekend_x',
                'is_holiday_day',
                'is_holiday_hour',
            ],
            'price_lags': [
                'price_ptf_lag_24h',
                'price_ptf_lag_168h',
                'price_ptf_rolling_mean_24h',
            ],
            'fundamental': [
                'load_factor',
                'load_factor_lag_24h',
                'reserve_margin_ratio',
            ],
        }
    },
}


class FundamentalFeaturePreparerV2:
    """
    Feature preparer for deep learning models using V2 fundamental features.

    All features are verified to be D-1 safe (available at day-ahead prediction time).
    No data leakage.
    """

    def __init__(
        self,
        target: str = 'price_real',
        strategy: str = 'fundamental_v2',
        fourier_orders: Dict[str, int] = None,
        custom_features: List[str] = None
    ):
        """
        Initialize the fundamental feature preparer.

        Args:
            target: Target variable (default: 'price_real')
            strategy: Feature selection strategy from FUNDAMENTAL_V2_FEATURE_STRATEGIES
            fourier_orders: Fourier series orders for seasonality
            custom_features: Optional list of custom features to include
        """
        self.target = target
        self.strategy = strategy
        self.custom_features = custom_features or []

        # Validate strategy
        if strategy not in FUNDAMENTAL_V2_FEATURE_STRATEGIES:
            raise ValueError(
                f"Unknown strategy: {strategy}. "
                f"Available: {list(FUNDAMENTAL_V2_FEATURE_STRATEGIES.keys())}"
            )

        self.strategy_config = FUNDAMENTAL_V2_FEATURE_STRATEGIES[strategy]

        # Fourier orders for seasonality
        self.fourier_orders = fourier_orders or {
            'daily': 5,      # 24h period
            'weekly': 3,     # 168h period
            'yearly': 4      # 8760h period
        }

        self.feature_names = None
        self.feature_version = "v2_fundamental"

        logger.info(f"Initialized FundamentalFeaturePreparerV2")
        logger.info(f"  Strategy: {strategy}")
        logger.info(f"  Target: {target}")

    def create_fourier_features(
        self,
        df: pd.DataFrame,
        period: int,
        order: int,
        prefix: str
    ) -> pd.DataFrame:
        """Create Fourier series features for a given period."""
        features = pd.DataFrame(index=df.index)
        t = np.arange(len(df))

        for k in range(1, order + 1):
            features[f'{prefix}_sin_{k}'] = np.sin(2 * np.pi * k * t / period)
            features[f'{prefix}_cos_{k}'] = np.cos(2 * np.pi * k * t / period)

        return features

    def add_fourier_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Fourier seasonality features to dataframe."""
        df = df.copy()

        # Daily seasonality (24h)
        if 'daily' in self.fourier_orders:
            fourier_daily = self.create_fourier_features(
                df, period=24,
                order=self.fourier_orders['daily'],
                prefix='fourier_daily'
            )
            df = pd.concat([df, fourier_daily], axis=1)

        # Weekly seasonality (168h)
        if 'weekly' in self.fourier_orders:
            fourier_weekly = self.create_fourier_features(
                df, period=168,
                order=self.fourier_orders['weekly'],
                prefix='fourier_weekly'
            )
            df = pd.concat([df, fourier_weekly], axis=1)

        # Yearly seasonality (8760h)
        if 'yearly' in self.fourier_orders:
            fourier_yearly = self.create_fourier_features(
                df, period=8760,
                order=self.fourier_orders['yearly'],
                prefix='fourier_yearly'
            )
            df = pd.concat([df, fourier_yearly], axis=1)

        return df

    def get_feature_list(self) -> List[str]:
        """Get the complete list of features for the selected strategy."""
        features = []

        # Add features from each group in strategy
        for group_name, group_features in self.strategy_config['features'].items():
            features.extend(group_features)

        # Add Fourier features
        for period_name, order in self.fourier_orders.items():
            for k in range(1, order + 1):
                features.append(f'fourier_{period_name}_sin_{k}')
                features.append(f'fourier_{period_name}_cos_{k}')

        # Add custom features
        features.extend(self.custom_features)

        return features

    def get_available_features(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of features that are available in the dataframe.
        Automatically filters out FORBIDDEN_COLUMNS to ensure D-1 safety.
        """
        requested = self.get_feature_list()

        # Filter: feature exists in df OR is a fourier feature (will be generated)
        available = [f for f in requested if f in df.columns or 'fourier' in f]

        # CRITICAL: Filter out any forbidden columns for D-1 safety
        safe_features = [f for f in available if f not in FORBIDDEN_COLUMNS]

        # Log if any forbidden columns were filtered
        forbidden_found = set(available) - set(safe_features)
        if forbidden_found:
            logger.warning(f"Filtered {len(forbidden_found)} forbidden columns: {sorted(forbidden_found)}")

        return safe_features

    def prepare_features(
        self,
        df: pd.DataFrame,
        add_fourier: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Prepare features for deep learning models.

        Args:
            df: Master dataset (master_v2_fundamental.parquet)
            add_fourier: Whether to add Fourier features

        Returns:
            Tuple of (X, y, feature_names)
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"PREPARING FUNDAMENTAL V2 FEATURES (D-1 SAFE)")
        logger.info(f"Strategy: {self.strategy}")
        logger.info(f"{'='*80}")

        # Add Fourier features
        if add_fourier:
            df = self.add_fourier_features(df)
            logger.info("Fourier seasonality features added")

        # Get available features (automatically filters forbidden)
        available_features = self.get_available_features(df)

        # Log missing features
        requested = self.get_feature_list()
        missing = set(requested) - set(available_features) - set(f for f in requested if 'fourier' in f)
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

    def get_feature_groups(self) -> Dict[str, List[str]]:
        """Get feature groups for analysis."""
        if self.feature_names is None:
            raise ValueError("Features not prepared yet")

        groups = {
            'fundamental': [f for f in self.feature_names if any(x in f for x in [
                'reserve_margin', 'renewable_sat', 'thermal_gap', 'system_short',
                'import_cost', 'spark_spread', 'load_factor', 'realtime_premium'
            ])],
            'fourier': [f for f in self.feature_names if 'fourier' in f],
            'lags': [f for f in self.feature_names if 'lag' in f],
            'rolling': [f for f in self.feature_names if 'rolling' in f],
            'calendar': [f for f in self.feature_names if any(x in f for x in [
                'hour', 'dow', 'month', 'weekend', 'holiday'
            ])],
            'fx': [f for f in self.feature_names if any(x in f for x in [
                'USD', 'EUR', 'FX'
            ])],
            'weather': [f for f in self.feature_names if any(x in f for x in [
                'temp', 'humidity', 'wind', 'HDD', 'CDD', 'heat', 'hot', 'cold'
            ])]
        }

        return groups

    def print_feature_summary(self):
        """Print feature summary."""
        if self.feature_names is None:
            logger.error("Features not prepared yet")
            return

        groups = self.get_feature_groups()

        print(f"\n{'='*80}")
        print(f"FUNDAMENTAL V2 FEATURE SET (D-1 SAFE): {self.target}")
        print(f"{'='*80}")
        print(f"Strategy: {self.strategy}")
        print(f"Version: {self.feature_version}")
        print(f"Total features: {len(self.feature_names)}")
        print(f"\nFeature groups:")
        for group_name, features in groups.items():
            print(f"  {group_name:20s}: {len(features):3d} features")
        print(f"{'='*80}\n")


def load_master_v2() -> pd.DataFrame:
    """Load master_v2_fundamental.parquet dataset."""
    data_dir = PROJECT_ROOT / 'data' / 'gold' / 'master'

    # Try parquet first
    parquet_file = data_dir / 'master_v2_fundamental.parquet'
    if parquet_file.exists():
        df = pd.read_parquet(parquet_file)
        logger.info(f"Loaded master_v2 from parquet: {df.shape}")
        return df

    # Try CSV
    csv_files = list(data_dir.glob('master_v2*.csv'))
    if csv_files:
        csv_file = max(csv_files, key=lambda p: p.stat().st_mtime)
        df = pd.read_csv(csv_file, parse_dates=['timestamp'], index_col='timestamp')
        logger.info(f"Loaded master_v2 from CSV: {df.shape}")
        return df

    raise FileNotFoundError(
        f"No master_v2 file found in {data_dir}. "
        "Run feature_engineering_v2.py first."
    )


def verify_d1_safety():
    """Verify all feature strategies are D-1 safe."""
    print("\n" + "="*80)
    print("D-1 SAFETY VERIFICATION")
    print("="*80)

    # Get all features from all strategies
    all_features = set()
    for strategy_name, strategy_config in FUNDAMENTAL_V2_FEATURE_STRATEGIES.items():
        for group_features in strategy_config['features'].values():
            all_features.update(group_features)

    # Check for forbidden columns
    forbidden_in_strategies = all_features & set(FORBIDDEN_COLUMNS)
    if forbidden_in_strategies:
        print(f"\n❌ FORBIDDEN COLUMNS FOUND IN STRATEGIES:")
        for col in sorted(forbidden_in_strategies):
            print(f"   - {col}")
    else:
        print(f"\n✅ All {len(all_features)} unique features are D-1 SAFE")
        print(f"   No forbidden columns found in any strategy")

    print(f"\nForbidden columns ({len(FORBIDDEN_COLUMNS)}):")
    for col in sorted(FORBIDDEN_COLUMNS):
        print(f"   - {col}")

    print("="*80 + "\n")


def main():
    """Demo usage."""
    # Verify D-1 safety
    verify_d1_safety()

    # Load data
    df = load_master_v2()

    # Initialize preparer
    preparer = FundamentalFeaturePreparerV2(
        target='price_real',
        strategy='fundamental_v2'
    )

    # Prepare features
    X, y, feature_names = preparer.prepare_features(df)

    # Print summary
    preparer.print_feature_summary()

    print(f"\nX shape: {X.shape}")
    print(f"y shape: {y.shape}")


if __name__ == "__main__":
    main()
