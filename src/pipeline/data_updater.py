"""
Incremental Data Updater for ForeWatt
======================================
Fetches new EPIAS and weather data, generates features, and updates the master parquet.

This module is designed to run daily via Cloud Scheduler to keep the forecast
data fresh and enable model predictions for recent dates.

Pipeline:
1. Read existing master parquet to find last timestamp
2. Fetch new EPIAS data (price, consumption) via eptr2
3. Fetch new weather data from Open-Meteo
4. Generate features for the new data
5. Append to master parquet

Author: ForeWatt Team
Date: January 2026
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, Dict
import logging
import os

logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MASTER_PARQUET_PATH = PROJECT_ROOT / 'data' / 'gold' / 'master' / 'master_v2_fundamental.parquet'


class IncrementalDataUpdater:
    """
    Updates the master parquet with new EPIAS and weather data.

    Features generated:
    - Calendar features (hour, day of week, month, etc.)
    - Weather features (temperature, humidity, HDD/CDD, etc.)
    - Lag features (consumption, price, temperature lags)
    - Rolling features (24h, 168h windows)
    """

    def __init__(self, master_path: str = None):
        """
        Initialize the data updater.

        Args:
            master_path: Path to master parquet file
        """
        self.master_path = Path(master_path) if master_path else MASTER_PARQUET_PATH
        self.epias_fetcher = None
        self.weather_fetcher = None

    def _init_epias_fetcher(self):
        """Initialize EPIAS fetcher with credentials."""
        if self.epias_fetcher is not None:
            return True

        try:
            from src.data.epias_fetcher import EpiasDataFetcher
            self.epias_fetcher = EpiasDataFetcher()
            logger.info("EPIAS fetcher initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize EPIAS fetcher: {e}")
            return False

    def get_last_timestamp(self) -> Optional[datetime]:
        """Get the last timestamp in the master parquet."""
        if not self.master_path.exists():
            logger.warning(f"Master parquet not found: {self.master_path}")
            return None

        try:
            df = pd.read_parquet(self.master_path)
            if 'timestamp' in df.columns:
                last_ts = pd.to_datetime(df['timestamp']).max()
                if pd.notna(last_ts):
                    # Remove timezone if present
                    if hasattr(last_ts, 'tz') and last_ts.tz is not None:
                        last_ts = last_ts.tz_localize(None)
                    logger.info(f"Last timestamp in master: {last_ts}")
                    return last_ts
        except Exception as e:
            logger.error(f"Failed to read master parquet: {e}")

        return None

    def fetch_epias_data(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch consumption and price data from EPIAS.

        Args:
            start_date: Start datetime
            end_date: End datetime

        Returns:
            Tuple of (consumption_df, price_df)
        """
        if not self._init_epias_fetcher():
            return pd.DataFrame(), pd.DataFrame()

        consumption_df = pd.DataFrame()
        price_df = pd.DataFrame()

        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        try:
            # Fetch consumption using EpiasDataFetcher
            logger.info(f"Fetching EPIAS consumption: {start_str} to {end_str}")
            consumption_result = self.epias_fetcher.fetch_dataset(
                'consumption_actual',
                start_str,
                end_str
            )

            if consumption_result is not None and len(consumption_result) > 0:
                consumption_df = consumption_result.copy()

                # Find and normalize timestamp column
                ts_cols = [c for c in consumption_df.columns if any(kw in c.lower() for kw in ['date', 'time', 'tarih'])]
                if ts_cols:
                    consumption_df['timestamp'] = pd.to_datetime(consumption_df[ts_cols[0]])
                    # Remove timezone if present
                    if consumption_df['timestamp'].dt.tz is not None:
                        consumption_df['timestamp'] = consumption_df['timestamp'].dt.tz_localize(None)

                # Find consumption column
                cons_cols = [c for c in consumption_df.columns if any(kw in c.lower() for kw in ['consumption', 'tuketim', 'load', 'value'])]
                if cons_cols:
                    consumption_df['consumption'] = pd.to_numeric(consumption_df[cons_cols[0]], errors='coerce')

                logger.info(f"Fetched {len(consumption_df)} consumption records")

        except Exception as e:
            logger.error(f"Failed to fetch EPIAS consumption: {e}")

        try:
            # Fetch price using EpiasDataFetcher
            logger.info(f"Fetching EPIAS price: {start_str} to {end_str}")
            price_result = self.epias_fetcher.fetch_dataset(
                'price_ptf',
                start_str,
                end_str
            )

            if price_result is not None and len(price_result) > 0:
                price_df = price_result.copy()

                # Find and normalize timestamp column
                ts_cols = [c for c in price_df.columns if any(kw in c.lower() for kw in ['date', 'time', 'tarih'])]
                if ts_cols:
                    price_df['timestamp'] = pd.to_datetime(price_df[ts_cols[0]])
                    # Remove timezone if present
                    if price_df['timestamp'].dt.tz is not None:
                        price_df['timestamp'] = price_df['timestamp'].dt.tz_localize(None)

                # Find price column
                price_cols = [c for c in price_df.columns if any(kw in c.lower() for kw in ['price', 'mcp', 'ptf', 'fiyat'])]
                if price_cols:
                    price_df['price_ptf'] = pd.to_numeric(price_df[price_cols[0]], errors='coerce')

                logger.info(f"Fetched {len(price_df)} price records")

        except Exception as e:
            logger.error(f"Failed to fetch EPIAS price: {e}")

        return consumption_df, price_df

    def fetch_weather_data(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch weather data from Open-Meteo.

        Args:
            start_date: Start datetime
            end_date: End datetime

        Returns:
            DataFrame with weather features
        """
        try:
            from src.data.weather_fetcher import DemandWeatherFetcher

            fetcher = DemandWeatherFetcher()

            # Fetch weather data
            logger.info(f"Fetching weather data: {start_date.date()} to {end_date.date()}")
            weather_df = fetcher.run_pipeline(
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                output_dir=str(PROJECT_ROOT / 'data'),
                force_refetch=True,
                delay_between_requests=3.0  # Faster for incremental updates
            )

            logger.info(f"Fetched {len(weather_df)} weather records")
            return weather_df

        except Exception as e:
            logger.error(f"Failed to fetch weather data: {e}")
            return pd.DataFrame()

    def generate_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate calendar features for the data."""
        df = df.copy()

        if 'timestamp' not in df.columns:
            return df

        ts = pd.to_datetime(df['timestamp'])

        # Basic calendar features
        df['hour'] = ts.dt.hour
        df['day_of_week'] = ts.dt.dayofweek
        df['day_of_year'] = ts.dt.dayofyear
        df['month'] = ts.dt.month
        df['is_weekend'] = (ts.dt.dayofweek >= 5).astype(int)

        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        return df

    def generate_lag_features(
        self,
        new_df: pd.DataFrame,
        existing_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate lag features using both new and existing data.

        Args:
            new_df: New data to generate features for
            existing_df: Existing master data for lookback

        Returns:
            DataFrame with lag features
        """
        # Combine existing and new for proper lag calculation
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.sort_values('timestamp').reset_index(drop=True)

        # Find where new data starts
        new_start_idx = len(existing_df)

        # Generate lag features
        lag_configs = {
            'consumption': [1, 2, 3, 24, 48, 168],  # hours
            'price_ptf': [1, 24, 168],
            'temp_national': [1, 24, 168]
        }

        for col, lags in lag_configs.items():
            if col in combined.columns:
                for lag in lags:
                    lag_col = f'{col}_lag_{lag}h'
                    combined[lag_col] = combined[col].shift(lag)

        # Return only the new data portion
        return combined.iloc[new_start_idx:].reset_index(drop=True)

    def generate_rolling_features(
        self,
        new_df: pd.DataFrame,
        existing_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate rolling features using both new and existing data.

        Args:
            new_df: New data to generate features for
            existing_df: Existing master data for lookback

        Returns:
            DataFrame with rolling features
        """
        # Combine existing and new for proper rolling calculation
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.sort_values('timestamp').reset_index(drop=True)

        # Find where new data starts
        new_start_idx = len(existing_df)

        # Generate rolling features
        rolling_configs = {
            'consumption': {'windows': [24, 168], 'funcs': ['mean', 'std']},
            'price_ptf': {'windows': [24, 168], 'funcs': ['mean', 'std']},
            'temp_national': {'windows': [24, 168], 'funcs': ['mean', 'std']}
        }

        for col, config in rolling_configs.items():
            if col in combined.columns:
                for window in config['windows']:
                    for func in config['funcs']:
                        roll_col = f'{col}_rolling_{window}h_{func}'
                        if func == 'mean':
                            combined[roll_col] = combined[col].rolling(window, min_periods=window//2).mean()
                        elif func == 'std':
                            combined[roll_col] = combined[col].rolling(window, min_periods=window//2).std()

        # Return only the new data portion
        return combined.iloc[new_start_idx:].reset_index(drop=True)

    def merge_data_sources(
        self,
        consumption_df: pd.DataFrame,
        price_df: pd.DataFrame,
        weather_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge consumption, price, and weather data.

        Args:
            consumption_df: EPIAS consumption data
            price_df: EPIAS price data
            weather_df: Open-Meteo weather data

        Returns:
            Merged DataFrame
        """
        # Start with consumption
        if consumption_df.empty:
            logger.warning("No consumption data available")
            return pd.DataFrame()

        merged = consumption_df[['timestamp', 'consumption']].copy()

        # Merge price
        if not price_df.empty and 'price_ptf' in price_df.columns:
            price_cols = ['timestamp', 'price_ptf']
            merged = merged.merge(price_df[price_cols], on='timestamp', how='left')

        # Merge weather
        if not weather_df.empty:
            # Reset index if weather is indexed by datetime
            if isinstance(weather_df.index, pd.DatetimeIndex):
                weather_df = weather_df.reset_index()
                weather_df = weather_df.rename(columns={'datetime': 'timestamp', 'index': 'timestamp'})

            # Ensure timestamp column exists and is datetime
            if 'timestamp' in weather_df.columns:
                weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])
                if weather_df['timestamp'].dt.tz is not None:
                    weather_df['timestamp'] = weather_df['timestamp'].dt.tz_localize(None)

                merged = merged.merge(weather_df, on='timestamp', how='left')

        logger.info(f"Merged data: {len(merged)} records, {len(merged.columns)} columns")
        return merged

    def update_master_parquet(
        self,
        days_back: int = 7,
        force: bool = False
    ) -> Dict:
        """
        Update the master parquet with new data.

        Args:
            days_back: Number of days to look back for new data
            force: Force update even if data is recent

        Returns:
            Dictionary with update statistics
        """
        result = {
            'success': False,
            'records_added': 0,
            'last_timestamp_before': None,
            'last_timestamp_after': None,
            'error': None
        }

        try:
            # Get last timestamp
            last_ts = self.get_last_timestamp()
            result['last_timestamp_before'] = str(last_ts) if last_ts else None

            if last_ts is None:
                result['error'] = "Could not determine last timestamp"
                return result

            # Check if update is needed
            now = datetime.now()
            hours_since_last = (now - last_ts).total_seconds() / 3600

            if hours_since_last < 24 and not force:
                result['error'] = f"Data is recent ({hours_since_last:.1f} hours old). Use force=True to update anyway."
                return result

            # Calculate date range for new data
            start_date = last_ts - timedelta(hours=24)  # Overlap for lag features
            end_date = now - timedelta(hours=2)  # EPIAS has ~2h delay

            logger.info(f"Fetching data from {start_date} to {end_date}")

            # Fetch new data
            consumption_df, price_df = self.fetch_epias_data(start_date, end_date)
            weather_df = self.fetch_weather_data(start_date, end_date)

            if consumption_df.empty:
                result['error'] = "No new consumption data available"
                return result

            # Merge data sources
            new_data = self.merge_data_sources(consumption_df, price_df, weather_df)

            if new_data.empty:
                result['error'] = "Merged data is empty"
                return result

            # Generate calendar features
            new_data = self.generate_calendar_features(new_data)

            # Load existing master for lag/rolling features
            existing_df = pd.read_parquet(self.master_path)
            if existing_df['timestamp'].dt.tz is not None:
                existing_df['timestamp'] = existing_df['timestamp'].dt.tz_localize(None)

            # Generate lag features
            new_data = self.generate_lag_features(new_data, existing_df)

            # Generate rolling features
            new_data = self.generate_rolling_features(new_data, existing_df)

            # Filter to only truly new records
            new_data['timestamp'] = pd.to_datetime(new_data['timestamp'])
            if new_data['timestamp'].dt.tz is not None:
                new_data['timestamp'] = new_data['timestamp'].dt.tz_localize(None)

            new_data = new_data[new_data['timestamp'] > last_ts]

            if new_data.empty:
                result['error'] = "No new records after filtering"
                return result

            # Ensure columns match existing data
            for col in existing_df.columns:
                if col not in new_data.columns:
                    new_data[col] = np.nan

            # Keep only columns that exist in the original
            new_data = new_data[existing_df.columns]

            # Append to master
            updated_df = pd.concat([existing_df, new_data], ignore_index=True)
            updated_df = updated_df.drop_duplicates(subset=['timestamp'], keep='last')
            updated_df = updated_df.sort_values('timestamp').reset_index(drop=True)

            # Save updated master
            updated_df.to_parquet(self.master_path, index=False, engine='pyarrow')

            result['success'] = True
            result['records_added'] = len(new_data)
            result['last_timestamp_after'] = str(updated_df['timestamp'].max())

            logger.info(f"Successfully added {len(new_data)} records")
            logger.info(f"New last timestamp: {result['last_timestamp_after']}")

        except Exception as e:
            logger.error(f"Update failed: {e}", exc_info=True)
            result['error'] = str(e)

        return result


def update_data(days_back: int = 7, force: bool = False) -> Dict:
    """
    Convenience function to update the master parquet.

    Args:
        days_back: Number of days to look back
        force: Force update even if data is recent

    Returns:
        Dictionary with update statistics
    """
    updater = IncrementalDataUpdater()
    return updater.update_master_parquet(days_back=days_back, force=force)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Run update
    result = update_data(force=True)
    print("\nUpdate Result:")
    for key, value in result.items():
        print(f"  {key}: {value}")
