"""
ForeWatt Real-Time Scheduler Service
====================================
Orchestrates hourly data fetching, prediction, and storage.

Architecture:
    1. Fetch latest weather data from Open-Meteo API
    2. Fetch latest EPİAŞ data (consumption, prices) if credentials available
    3. Engineer all 23 features for CatBoost model
    4. Generate 24h forecast with prediction intervals
    5. Store predictions to JSON and optionally InfluxDB
    6. Dashboard auto-refreshes to display

Usage:
    python services/scheduler.py              # Run continuously
    python services/scheduler.py --once       # Run once and exit
    python services/scheduler.py --interval 5 # Run every 5 minutes (for testing)

Author: ForeWatt Team
Date: December 2025
"""

import sys
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List
import json
import os

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

# Import the data fetchers
from src.data.weather_fetcher import DemandWeatherFetcher
try:
    from src.data.epias_fetcher import EpiasDataFetcher
    EPIAS_AVAILABLE = True
except (ImportError, ValueError) as e:
    EPIAS_AVAILABLE = False
    logger_init = logging.getLogger(__name__)
    logger_init.warning(f"EPİAŞ fetcher not available: {e}")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================
CONFIG = {
    'model_path': PROJECT_ROOT / 'reports' / 'new_experiment' / 'baseline' / 'models',
    'predictions_path': PROJECT_ROOT / 'data' / 'predictions',
    'best_model': 'catboost_consumption_8327b57030a0',  # Best CatBoost model for consumption (23 features)
    'horizon': 24,  # 24-hour forecast
    'target': 'consumption',
    'influxdb': {
        'url': 'http://localhost:8086',
        'token': None,  # Set from environment
        'org': 'forewatt',
        'bucket': 'predictions'
    }
}

# CatBoost model expected features (24 features in order)
CATBOOST_FEATURES = [
    'consumption_lag_24h',
    'consumption_lag_168h',
    'consumption_rolling_mean_24h',
    'dow_cos_x',
    'hour_cos',
    'hour_sin',
    'dow_sin_x',
    'consumption_rolling_std_24h',
    'is_weekend_x',
    'is_holiday_day',
    'consumption_lag_48h',
    'is_holiday_hour',
    'temp_lag_24h',
    'month_cos',
    'humidity_national',
    'HDD',
    'month_sin',
    'price_ptf_lag_24h',
    'temp_national',
    'heat_index',
    'is_cold',
    'CDD',
    'is_hot'
]


class RealTimeForecaster:
    """
    Real-time forecasting service that fetches data and generates predictions.
    """

    def __init__(self, config: Dict = None):
        self.config = config or CONFIG
        self.model = None
        self.feature_columns = None
        self.influx_client = None

        # Ensure predictions directory exists
        self.predictions_path = Path(self.config['predictions_path'])
        self.predictions_path.mkdir(parents=True, exist_ok=True)

        logger.info("RealTimeForecaster initialized")

    def fetch_epias_direct(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Fetch consumption data directly from EPİAŞ API (matches eptr2 library approach).
        TGT comes from response body, not Location header.
        """
        import requests
        from urllib.parse import quote
        from dotenv import load_dotenv
        load_dotenv()

        username = os.getenv('EPTR_USERNAME')
        password = os.getenv('EPTR_PASSWORD')

        if not username or not password:
            logger.warning("EPİAŞ credentials not set")
            return None

        try:
            logger.info(f"Authenticating with EPİAŞ as {username}...")

            # Step 1: Get TGT (Ticket Granting Ticket) - eptr2 approach
            # TGT comes from response BODY, not Location header
            auth_url = "https://giris.epias.com.tr/cas/v1/tickets"
            body_str = f"username={quote(username)}&password={quote(password)}"

            auth_response = requests.post(
                auth_url,
                data=body_str,
                headers={
                    'Content-Type': 'application/x-www-form-urlencoded',
                    'Accept': 'text/plain'
                },
                timeout=30
            )

            if auth_response.status_code not in [200, 201]:
                logger.error(f"EPİAŞ TGT auth failed: {auth_response.status_code} - {auth_response.text[:200]}")
                return None

            # TGT is directly in response body (starts with "TGT-")
            tgt = auth_response.text.strip()
            if not tgt.startswith("TGT-"):
                logger.error(f"Invalid TGT format: {tgt[:50]}")
                return None

            logger.info(f"Got TGT: {tgt[:20]}...")

            # Step 2: Fetch real-time consumption data using TGT header
            consumption_url = "https://seffaflik.epias.com.tr/electricity-service/v1/consumption/data/realtime-consumption"

            payload = {
                "startDate": f"{start_date}T00:00:00+03:00",
                "endDate": f"{end_date}T23:00:00+03:00"
            }

            headers = {
                "TGT": tgt,  # Use TGT directly, not service ticket
                "Content-Type": "application/json",
                "Accept": "application/json"
            }

            logger.info(f"Fetching consumption from {start_date} to {end_date}...")
            data_response = requests.post(
                consumption_url,
                json=payload,
                headers=headers,
                timeout=60
            )

            if data_response.status_code == 200:
                data = data_response.json()
                if 'items' in data and data['items']:
                    df = pd.DataFrame(data['items'])
                    logger.info(f"EPİAŞ API response columns: {list(df.columns)}")

                    # Parse datetime from 'date' field
                    if 'date' in df.columns:
                        df['datetime'] = pd.to_datetime(df['date'])
                        df = df.set_index('datetime')
                        # EPİAŞ returns Turkey timezone (+03:00)
                        if df.index.tz is None:
                            df.index = df.index.tz_localize('Europe/Istanbul')

                    # Ensure 'consumption' column exists
                    if 'consumption' not in df.columns and 'Consumption' in df.columns:
                        df['consumption'] = df['Consumption']

                    # Sort by datetime index
                    df = df.sort_index()

                    logger.info(f"EPİAŞ direct API: fetched {len(df)} records")
                    logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
                    logger.info(f"Consumption range: {df['consumption'].min():.0f} - {df['consumption'].max():.0f} MW")
                    return df

                elif 'body' in data and 'realtimeConsumptionList' in data['body']:
                    # Alternative response format
                    items = data['body']['realtimeConsumptionList']
                    df = pd.DataFrame(items)
                    if 'date' in df.columns:
                        df['datetime'] = pd.to_datetime(df['date'])
                        df = df.set_index('datetime')
                        if df.index.tz is None:
                            df.index = df.index.tz_localize('Europe/Istanbul')
                    if 'consumption' not in df.columns:
                        for c in df.columns:
                            if c.lower() == 'consumption':
                                df['consumption'] = df[c]
                                break
                    df = df.sort_index()
                    logger.info(f"EPİAŞ direct API: fetched {len(df)} records (alt format)")
                    return df
                else:
                    logger.warning(f"EPİAŞ API returned empty items: {list(data.keys())}")

            logger.warning(f"EPİAŞ API returned: {data_response.status_code} - {data_response.text[:200]}")
            return None

        except Exception as e:
            logger.error(f"EPİAŞ direct API error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def load_model(self) -> bool:
        """Load the best trained model (CatBoost for consumption)."""
        try:
            model_dir = self.config['model_path']
            model_name = self.config['best_model']

            # 1. Try to load specific model directory first
            specific_model_dir = model_dir / model_name
            if specific_model_dir.exists():
                model_file = specific_model_dir / 'model.cbm'
                if model_file.exists():
                    from catboost import CatBoostRegressor
                    self.model = CatBoostRegressor()
                    self.model.load_model(str(model_file))
                    self._model_type = 'catboost'
                    logger.info(f"CatBoost model loaded from {model_file}")

                    # Load feature columns from feature_importance.csv
                    fi_file = specific_model_dir / 'feature_importance.csv'
                    if fi_file.exists():
                        fi_df = pd.read_csv(fi_file)
                        self.feature_columns = fi_df['feature'].tolist()
                        logger.info(f"Loaded {len(self.feature_columns)} features from {fi_file}")
                    else:
                        self.feature_columns = CATBOOST_FEATURES
                        logger.info(f"Using default {len(self.feature_columns)} CatBoost features")

                    return True

            # 2. Fallback: CatBoost .cbm format (any consumption model)
            catboost_files = list(model_dir.glob("catboost_consumption*/model.cbm"))
            if catboost_files:
                from catboost import CatBoostRegressor
                model_file = sorted(catboost_files)[-1]
                self.model = CatBoostRegressor()
                self.model.load_model(str(model_file))
                self._model_type = 'catboost'
                logger.info(f"CatBoost model loaded from {model_file}")

                fi_file = model_file.parent / 'feature_importance.csv'
                if fi_file.exists():
                    fi_df = pd.read_csv(fi_file)
                    self.feature_columns = fi_df['feature'].tolist()
                    logger.info(f"Loaded {len(self.feature_columns)} features from {fi_file}")
                else:
                    self.feature_columns = CATBOOST_FEATURES
                    logger.info(f"Using default {len(self.feature_columns)} CatBoost features")

                return True

            # 2. LightGBM .txt format (fallback)
            lgbm_files = list(model_dir.glob(f"{model_name}*/model.txt"))
            if not lgbm_files:
                lgbm_files = list(model_dir.glob("lightgbm_consumption*/model.txt"))

            if lgbm_files:
                import lightgbm as lgb
                model_file = lgbm_files[0]
                self.model = lgb.Booster(model_file=str(model_file))
                self._model_type = 'lightgbm'
                logger.info(f"LightGBM model loaded from {model_file}")

                # Load feature columns from feature_importance.csv
                fi_file = model_file.parent / 'feature_importance.csv'
                if fi_file.exists():
                    fi_df = pd.read_csv(fi_file)
                    self.feature_columns = fi_df['feature'].tolist()
                    logger.info(f"Loaded {len(self.feature_columns)} features")

                return True

            # 3. Joblib .pkl format (fallback)
            pkl_files = list(model_dir.glob(f"{model_name}*/*.pkl"))
            if pkl_files:
                import joblib
                model_file = pkl_files[0]
                self.model = joblib.load(model_file)
                self._model_type = 'sklearn'
                logger.info(f"Sklearn model loaded from {model_file}")
                return True

            logger.error(f"No model files found in {model_dir}")
            return False

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def fetch_latest_data(self) -> Optional[pd.DataFrame]:
        """
        Fetch the latest data from APIs:
        1. EPİAŞ data (consumption) - REAL TODAY's data
        2. Weather data from Open-Meteo (free, no credentials)
        3. Falls back to cached data if APIs fail
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=8)  # 8 days for weekly lag features

        logger.info(f"Fetching real-time data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

        weather_df = None
        epias_df = None

        # Step 1: Try to fetch EPİAŞ REAL consumption data using direct API
        try:
            logger.info("Fetching EPİAŞ consumption data (direct API)...")
            epias_df = self.fetch_epias_direct(
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            if epias_df is not None and not epias_df.empty:
                logger.info(f"EPİAŞ real data: {len(epias_df)} records from {epias_df.index.min()} to {epias_df.index.max()}")
        except Exception as e:
            logger.warning(f"EPİAŞ direct API failed: {e}")

        # Step 2: Fetch weather data from Open-Meteo API (always available)
        try:
            logger.info("Fetching weather data from Open-Meteo API...")
            weather_fetcher = DemandWeatherFetcher(cache_dir=str(PROJECT_ROOT / '.cache'))

            # Fetch for all 10 Turkish cities and create national features
            city_data = weather_fetcher.fetch_all_cities(
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                delay_between_requests=2.0  # Faster for real-time
            )

            if city_data:
                # Create national population-weighted features
                national_weather = weather_fetcher.create_national_features(city_data)
                # Engineer demand features (HDD, CDD, heat_index, etc.)
                weather_df = weather_fetcher.create_demand_features(national_weather)
                logger.info(f"Weather data fetched: {len(weather_df)} hours, {len(weather_df.columns)} features")
            else:
                logger.warning("No weather data from API, will use cached")

        except Exception as e:
            logger.error(f"Weather API fetch failed: {e}")

        # Step 3: Combine data sources
        # IMPORTANT: Filter to only include data up to CURRENT hour (no future data)
        current_time = pd.Timestamp.now(tz='Europe/Istanbul')
        logger.info(f"Current time: {current_time}")

        if epias_df is not None and not epias_df.empty:
            # We have REAL EPİAŞ data - use it as base
            df = epias_df.copy()
            # Filter to current time
            if df.index.tz is None:
                df.index = df.index.tz_localize('Europe/Istanbul')
            df = df[df.index <= current_time]
            logger.info(f"Filtered EPİAŞ to {len(df)} records (up to {current_time})")

            # Merge weather data if available
            if weather_df is not None:
                # Join weather to EPİAŞ data by datetime
                weather_cols = [c for c in weather_df.columns if c not in df.columns]
                if weather_cols:
                    df = df.join(weather_df[weather_cols], how='left')
                    logger.info(f"Merged EPİAŞ with {len(weather_cols)} weather features")

            return df

        elif weather_df is not None:
            # No EPİAŞ data - use weather with cached consumption
            df = weather_df.copy()

            # Filter weather to only include data up to CURRENT hour
            if df.index.tz is None:
                df.index = df.index.tz_localize('Europe/Istanbul')
            df = df[df.index <= current_time]
            logger.info(f"Filtered weather to {len(df)} records (up to {current_time})")

            # Try to merge with cached consumption data
            cached_df = self._load_cached_data()
            if cached_df is not None and 'consumption' in cached_df.columns:
                # Get last rows of cached data (use for lag features)
                cached_last = cached_df.tail(len(df))

                # Try to join by index first (if dates overlap)
                overlap = df.index.intersection(cached_last.index)
                if len(overlap) > 24:
                    # Dates overlap - join normally
                    cols_to_join = [c for c in ['consumption', 'price_ptf'] if c in cached_df.columns]
                    df = df.join(cached_df[cols_to_join], how='left')
                    logger.info(f"Joined fresh weather with cached data ({len(overlap)} overlapping hours)")
                else:
                    # No overlap - use cached data's last values aligned by position
                    logger.info("No date overlap - using cached values aligned by position")
                    # Reset both to use numeric index for alignment
                    df_reset = df.reset_index()
                    cached_reset = cached_last.reset_index()

                    # Take consumption values from cached (aligned by row position)
                    if len(cached_reset) >= len(df_reset):
                        df_reset['consumption'] = cached_reset['consumption'].values[-len(df_reset):]
                        if 'price_ptf' in cached_reset.columns:
                            df_reset['price_ptf'] = cached_reset['price_ptf'].values[-len(df_reset):]
                    else:
                        # Cached has fewer rows - pad with NaN
                        consumption_values = np.full(len(df_reset), np.nan)
                        consumption_values[-len(cached_reset):] = cached_reset['consumption'].values
                        df_reset['consumption'] = consumption_values

                    df = df_reset.set_index('datetime' if 'datetime' in df_reset.columns else df_reset.columns[0])
                    logger.info("Merged fresh weather with cached consumption (positional alignment)")

            return df
        else:
            # Fall back completely to cached data
            logger.warning("Using cached data as fallback")
            return self._load_cached_data()

    def _load_cached_data(self) -> Optional[pd.DataFrame]:
        """Load cached/historical data as fallback."""
        try:
            master_path = PROJECT_ROOT / 'data' / 'gold' / 'master'

            # Try to load master_v2_fundamental first
            v2_file = master_path / 'master_v2_fundamental.parquet'
            if v2_file.exists():
                df = pd.read_parquet(v2_file)
                logger.info(f"Loaded {len(df)} rows from master_v2_fundamental.parquet")
                # Return last 168 hours
                return df.tail(168)

            # Fallback to any parquet file
            parquet_files = list(master_path.glob('*.parquet'))
            if parquet_files:
                df = pd.read_parquet(parquet_files[0])
                return df.tail(168)

            csv_files = list(master_path.glob('*.csv'))
            if csv_files:
                df = pd.read_csv(csv_files[0], parse_dates=['timestamp'], index_col='timestamp')
                return df.tail(168)

            return None
        except Exception as e:
            logger.error(f"Failed to load cached data: {e}")
            return None

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer all 23 features required by CatBoost model:
        - Consumption lags and rolling stats
        - Calendar features (cyclical encoding)
        - Weather features (temp_national, humidity_national, HDD, CDD, heat_index)
        - Holiday flags
        """
        try:
            # Ensure we have datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'timestamp' in df.columns:
                    df = df.set_index('timestamp')
                elif 'date' in df.columns:
                    df = df.set_index('date')
                elif 'datetime' in df.columns:
                    df = df.set_index('datetime')

            logger.info(f"Engineering features for {len(df)} rows...")

            # Check if features are already engineered (master_v2 has 173 columns)
            existing_features = [c for c in CATBOOST_FEATURES if c in df.columns or c.rstrip('_x') in df.columns]
            if len(existing_features) >= 20:
                logger.info(f"Most features already present ({len(existing_features)}/23), minimal engineering needed")
                df = df.dropna(subset=['consumption']) if 'consumption' in df.columns else df
                return df

            # ==========================================
            # CONSUMPTION FEATURES (from cached/live data)
            # ==========================================
            if 'consumption' in df.columns:
                # Lag features
                if 'consumption_lag_24h' not in df.columns:
                    df['consumption_lag_24h'] = df['consumption'].shift(24)
                if 'consumption_lag_48h' not in df.columns:
                    df['consumption_lag_48h'] = df['consumption'].shift(48)
                if 'consumption_lag_168h' not in df.columns:
                    df['consumption_lag_168h'] = df['consumption'].shift(168)

                # Rolling statistics
                if 'consumption_rolling_mean_24h' not in df.columns:
                    df['consumption_rolling_mean_24h'] = df['consumption'].rolling(24, min_periods=12).mean()
                if 'consumption_rolling_std_24h' not in df.columns:
                    df['consumption_rolling_std_24h'] = df['consumption'].rolling(24, min_periods=12).std()

            # ==========================================
            # CALENDAR FEATURES
            # ==========================================
            if 'hour' not in df.columns:
                df['hour'] = df.index.hour
            if 'dow' not in df.columns:
                df['dow'] = df.index.dayofweek
            if 'month' not in df.columns:
                df['month'] = df.index.month

            # Cyclical encoding for hour
            if 'hour_sin' not in df.columns:
                df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            if 'hour_cos' not in df.columns:
                df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

            # Cyclical encoding for day of week (with _x suffix for CatBoost compatibility)
            if 'dow_sin_x' not in df.columns and 'dow_sin' not in df.columns:
                df['dow_sin_x'] = np.sin(2 * np.pi * df['dow'] / 7)
            elif 'dow_sin' in df.columns and 'dow_sin_x' not in df.columns:
                df['dow_sin_x'] = df['dow_sin']

            if 'dow_cos_x' not in df.columns and 'dow_cos' not in df.columns:
                df['dow_cos_x'] = np.cos(2 * np.pi * df['dow'] / 7)
            elif 'dow_cos' in df.columns and 'dow_cos_x' not in df.columns:
                df['dow_cos_x'] = df['dow_cos']

            # Cyclical encoding for month
            if 'month_sin' not in df.columns:
                df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            if 'month_cos' not in df.columns:
                df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

            # Weekend flag (with _x suffix)
            if 'is_weekend_x' not in df.columns and 'is_weekend' not in df.columns:
                df['is_weekend_x'] = (df['dow'] >= 5).astype(int)
            elif 'is_weekend' in df.columns and 'is_weekend_x' not in df.columns:
                df['is_weekend_x'] = df['is_weekend']

            # ==========================================
            # HOLIDAY FEATURES
            # ==========================================
            if 'is_holiday_day' not in df.columns:
                # Turkish public holidays (basic list)
                turkish_holidays = [
                    '01-01',  # New Year
                    '04-23',  # National Sovereignty Day
                    '05-01',  # Labor Day
                    '05-19',  # Youth Day
                    '07-15',  # Democracy Day
                    '08-30',  # Victory Day
                    '10-29',  # Republic Day
                ]
                df['is_holiday_day'] = df.index.strftime('%m-%d').isin(turkish_holidays).astype(int)

            if 'is_holiday_hour' not in df.columns:
                # Holiday hours (weekends + holidays)
                df['is_holiday_hour'] = ((df['dow'] >= 5) | (df['is_holiday_day'] == 1)).astype(int)

            # ==========================================
            # WEATHER FEATURES (from Open-Meteo API)
            # ==========================================
            # These should be present from weather_fetcher.create_demand_features()
            # But create defaults if not present

            if 'temp_national' not in df.columns:
                # Use apparent_temp_national if available, otherwise default
                if 'apparent_temp_national' in df.columns:
                    df['temp_national'] = df['apparent_temp_national']
                else:
                    df['temp_national'] = 15.0  # Default mild temperature

            if 'humidity_national' not in df.columns:
                df['humidity_national'] = 60.0  # Default humidity

            # Temperature lag
            if 'temp_lag_24h' not in df.columns:
                df['temp_lag_24h'] = df['temp_national'].shift(24)

            # Degree days (if not from weather fetcher)
            if 'HDD' not in df.columns:
                df['HDD'] = np.maximum(18 - df['temp_national'], 0)
            if 'CDD' not in df.columns:
                df['CDD'] = np.maximum(df['temp_national'] - 18, 0)

            # Heat index (if not from weather fetcher)
            if 'heat_index' not in df.columns:
                df['heat_index'] = df['temp_national']  # Simplified

            # Temperature flags
            if 'is_cold' not in df.columns:
                df['is_cold'] = (df['temp_national'] < 5).astype(int)
            if 'is_hot' not in df.columns:
                df['is_hot'] = (df['temp_national'] > 30).astype(int)

            # ==========================================
            # PRICE FEATURES
            # ==========================================
            if 'price_ptf' in df.columns and 'price_ptf_lag_24h' not in df.columns:
                df['price_ptf_lag_24h'] = df['price_ptf'].shift(24)
            elif 'price_ptf_lag_24h' not in df.columns:
                df['price_ptf_lag_24h'] = 0  # Default if no price data

            # Fill NaN in lag features with forward/backward fill for continuity
            lag_cols = [c for c in df.columns if 'lag' in c.lower() or 'rolling' in c.lower()]
            for col in lag_cols:
                if col in df.columns:
                    df[col] = df[col].ffill().bfill()

            # Only drop rows where we have NO valid data at all
            required_cols = ['temp_national', 'humidity_national']
            if any(c in df.columns for c in required_cols):
                df = df.dropna(subset=[c for c in required_cols if c in df.columns], how='all')

            # Fill remaining NaN with sensible defaults
            if 'consumption' in df.columns:
                df['consumption'] = df['consumption'].ffill().bfill()
            if 'consumption_lag_24h' in df.columns:
                df['consumption_lag_24h'] = df['consumption_lag_24h'].ffill().bfill()
            if 'consumption_lag_48h' in df.columns:
                df['consumption_lag_48h'] = df['consumption_lag_48h'].ffill().bfill()
            if 'consumption_lag_168h' in df.columns:
                df['consumption_lag_168h'] = df['consumption_lag_168h'].ffill().bfill()

            logger.info(f"Engineered features: {len(df)} rows, {len(df.columns)} columns")
            logger.info(f"CatBoost features present: {len([c for c in CATBOOST_FEATURES if c in df.columns or c.rstrip('_x') in df.columns])}/23")

            return df

        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            import traceback
            traceback.print_exc()
            return df

    def _get_feature_value(self, row: pd.Series, feat: str) -> float:
        """Get feature value from row, handling suffix variations."""
        if feat in row.index:
            return float(row[feat])
        elif feat.rstrip('_x') in row.index:
            return float(row[feat.rstrip('_x')])
        elif feat + '_x' in row.index:
            return float(row[feat + '_x'])
        return 0.0

    def generate_predictions(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Generate autoregressive predictions for next 12 hours.
        Each hour uses updated features (calendar, weather forecast).
        Also includes last 12 hours of historical data for visualization.
        """
        try:
            if self.model is None:
                if not self.load_model():
                    return None

            horizon = 12  # Predict 12 hours ahead
            history_hours = 12  # Show 12 hours of history

            # Ensure all timestamps are in Turkey timezone
            turkey_tz = 'Europe/Istanbul'
            if df.index.tz is None:
                df.index = df.index.tz_localize(turkey_tz)
            elif str(df.index.tz) != turkey_tz:
                df.index = df.index.tz_convert(turkey_tz)

            # Get historical data (last 12 hours)
            hist_df = df.tail(history_hours).copy()
            historical_timestamps = hist_df.index.tolist()
            historical_values = hist_df['consumption'].tolist() if 'consumption' in hist_df.columns else [30000] * history_hours

            # Prepare for autoregressive prediction
            predictions = []
            prediction_timestamps = []

            # Get the last row as starting point
            current_features = df.iloc[-1].copy()
            base_time = df.index[-1]

            # Get recent consumption values for updating lags
            recent_consumption = list(df['consumption'].tail(168).values) if 'consumption' in df.columns else [30000] * 168

            logger.info(f"Starting autoregressive prediction for {horizon} hours...")

            for h in range(1, horizon + 1):
                # Calculate future timestamp
                future_time = base_time + pd.Timedelta(hours=h)
                prediction_timestamps.append(future_time)

                # Update calendar features for the future hour
                future_hour = future_time.hour
                future_dow = future_time.dayofweek
                future_month = future_time.month

                # Update cyclical calendar features
                current_features['hour'] = future_hour
                current_features['hour_sin'] = np.sin(2 * np.pi * future_hour / 24)
                current_features['hour_cos'] = np.cos(2 * np.pi * future_hour / 24)
                current_features['dow'] = future_dow
                current_features['dow_sin_x'] = np.sin(2 * np.pi * future_dow / 7)
                current_features['dow_cos_x'] = np.cos(2 * np.pi * future_dow / 7)
                if 'dow_sin' in current_features.index:
                    current_features['dow_sin'] = current_features['dow_sin_x']
                if 'dow_cos' in current_features.index:
                    current_features['dow_cos'] = current_features['dow_cos_x']
                current_features['month'] = future_month
                current_features['month_sin'] = np.sin(2 * np.pi * future_month / 12)
                current_features['month_cos'] = np.cos(2 * np.pi * future_month / 12)
                current_features['is_weekend_x'] = 1 if future_dow >= 5 else 0
                if 'is_weekend' in current_features.index:
                    current_features['is_weekend'] = current_features['is_weekend_x']

                # Update consumption lag features using recent values + predictions
                if len(recent_consumption) >= 24:
                    current_features['consumption_lag_24h'] = recent_consumption[-24]
                if len(recent_consumption) >= 48:
                    current_features['consumption_lag_48h'] = recent_consumption[-48]
                if len(recent_consumption) >= 168:
                    current_features['consumption_lag_168h'] = recent_consumption[-168]

                # Update rolling features
                if len(recent_consumption) >= 24:
                    current_features['consumption_rolling_mean_24h'] = np.mean(recent_consumption[-24:])
                    current_features['consumption_rolling_std_24h'] = np.std(recent_consumption[-24:])

                # Update temperature lag (use current temp as approximation for forecast)
                if 'temp_national' in current_features.index:
                    current_features['temp_lag_24h'] = current_features['temp_national']

                # Prepare feature vector for model
                feature_data = []
                for feat in self.feature_columns:
                    feature_data.append(self._get_feature_value(current_features, feat))

                X = np.array([feature_data])

                # Make prediction
                pred = self.model.predict(X)[0]
                predictions.append(float(pred))

                # Update recent consumption with prediction for next iteration
                recent_consumption.append(pred)
                if len(recent_consumption) > 200:
                    recent_consumption = recent_consumption[-200:]

            predictions = np.array(predictions)

            # Calculate prediction intervals (wider for further horizons)
            base_std = df['consumption'].std() * 0.05 if 'consumption' in df.columns else 300
            uncertainties = np.array([base_std * (1 + 0.1 * h) for h in range(horizon)])
            lower_bound = predictions - 1.645 * uncertainties
            upper_bound = predictions + 1.645 * uncertainties

            # Log prediction range
            logger.info(f"Predictions range: {predictions.min():.0f} - {predictions.max():.0f} MW")
            logger.info(f"Generated {len(predictions)} unique hourly predictions")

            result = {
                'generated_at': datetime.now().isoformat(),
                'base_time': base_time.isoformat(),
                'horizon': horizon,
                'model': self.config['best_model'],
                # Historical data (last 12 hours)
                'historical_timestamps': [t.isoformat() for t in historical_timestamps],
                'historical_values': [float(v) if not np.isnan(v) else 30000 for v in historical_values],
                # Predictions (next 12 hours)
                'timestamps': [t.isoformat() for t in prediction_timestamps],
                'predictions': predictions.tolist(),
                'lower_bound': lower_bound.tolist(),
                'upper_bound': upper_bound.tolist()
            }

            logger.info(f"Generated {len(predictions)} predictions from {prediction_timestamps[0]} to {prediction_timestamps[-1]}")
            return result

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def save_predictions(self, predictions: Dict) -> bool:
        """Save predictions to file and optionally to InfluxDB."""
        try:
            # Save to JSON file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            json_path = self.predictions_path / f'forecast_{timestamp}.json'

            with open(json_path, 'w') as f:
                json.dump(predictions, f, indent=2)

            logger.info(f"Predictions saved to {json_path}")

            # Also save as "latest" for easy dashboard access
            latest_path = self.predictions_path / 'latest_forecast.json'
            with open(latest_path, 'w') as f:
                json.dump(predictions, f, indent=2)

            # Try to write to InfluxDB
            self._write_to_influxdb(predictions)

            return True

        except Exception as e:
            logger.error(f"Failed to save predictions: {e}")
            return False

    def _write_to_influxdb(self, predictions: Dict) -> bool:
        """Write predictions to InfluxDB."""
        try:
            from influxdb_client import InfluxDBClient, Point
            from influxdb_client.client.write_api import SYNCHRONOUS
            import os

            url = os.getenv('INFLUXDB_URL', self.config['influxdb']['url'])
            token = os.getenv('INFLUXDB_TOKEN', self.config['influxdb']['token'])
            org = os.getenv('INFLUXDB_ORG', self.config['influxdb']['org'])
            bucket = os.getenv('INFLUXDB_BUCKET', self.config['influxdb']['bucket'])

            if not token:
                logger.warning("InfluxDB token not set, skipping InfluxDB write")
                return False

            client = InfluxDBClient(url=url, token=token, org=org)
            write_api = client.write_api(write_options=SYNCHRONOUS)

            points = []
            for i, ts in enumerate(predictions['timestamps']):
                point = Point("forecast") \
                    .tag("model", predictions['model']) \
                    .field("prediction", predictions['predictions'][i]) \
                    .field("lower_bound", predictions['lower_bound'][i]) \
                    .field("upper_bound", predictions['upper_bound'][i]) \
                    .time(ts)
                points.append(point)

            write_api.write(bucket=bucket, record=points)
            client.close()

            logger.info(f"Wrote {len(points)} points to InfluxDB")
            return True

        except Exception as e:
            logger.warning(f"InfluxDB write failed (non-critical): {e}")
            return False

    def run_cycle(self) -> bool:
        """Run one complete fetch-predict-store cycle."""
        logger.info("=" * 60)
        logger.info("Starting forecasting cycle")
        logger.info("=" * 60)

        # Step 1: Fetch latest data
        df = self.fetch_latest_data()
        if df is None or df.empty:
            logger.error("No data available for prediction")
            return False

        # Step 2: Engineer features
        df = self.engineer_features(df)
        if df.empty:
            logger.error("Feature engineering produced no valid rows")
            return False

        # Step 3: Generate predictions
        predictions = self.generate_predictions(df)
        if predictions is None:
            logger.error("Failed to generate predictions")
            return False

        # Step 4: Save predictions
        success = self.save_predictions(predictions)

        logger.info(f"Cycle completed: {'SUCCESS' if success else 'FAILED'}")
        logger.info("=" * 60)

        return success


def main():
    """Main entry point for the scheduler."""
    parser = argparse.ArgumentParser(description='ForeWatt Real-Time Forecasting Scheduler')
    parser.add_argument('--once', action='store_true', help='Run once and exit')
    parser.add_argument('--interval', type=int, default=60, help='Interval in minutes (default: 60)')
    parser.add_argument('--cron', type=str, help='Cron expression (e.g., "0 * * * *" for hourly)')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("ForeWatt Real-Time Forecasting Service")
    logger.info("=" * 60)

    forecaster = RealTimeForecaster()

    if args.once:
        # Run once and exit
        logger.info("Running single forecasting cycle")
        success = forecaster.run_cycle()
        sys.exit(0 if success else 1)

    # Setup scheduler
    scheduler = BlockingScheduler()

    if args.cron:
        # Use cron expression
        trigger = CronTrigger.from_crontab(args.cron)
        logger.info(f"Scheduled with cron: {args.cron}")
    else:
        # Use interval
        trigger = IntervalTrigger(minutes=args.interval)
        logger.info(f"Scheduled to run every {args.interval} minutes")

    scheduler.add_job(
        forecaster.run_cycle,
        trigger=trigger,
        id='forecast_job',
        name='Hourly Forecast Generation',
        max_instances=1,
        coalesce=True
    )

    # Run immediately on startup
    logger.info("Running initial forecasting cycle...")
    forecaster.run_cycle()

    # Start scheduler
    logger.info("Scheduler started. Press Ctrl+C to exit.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")


if __name__ == '__main__':
    main()
