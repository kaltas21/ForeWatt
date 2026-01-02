"""
Forecast Storage Manager
========================
Handles Parquet file storage for forecasts with monthly partitioning.

Storage Structure:
    data/forecasts/
    ├── price/
    │   ├── 2024-01.parquet
    │   ├── 2024-02.parquet
    │   └── ...
    └── consumption/
        ├── 2024-01.parquet
        └── ...

Each parquet file contains:
    - forecast_time: When the forecast was made (hourly)
    - target_time: The hour being forecasted
    - forecast_value: Predicted value
    - model_version: Model version used
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
import logging

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class ForecastStorage:
    """Manages forecast storage in Parquet files with monthly partitioning."""

    def __init__(self, base_path: str = None):
        """
        Initialize storage manager.

        Args:
            base_path: Base directory for forecast storage.
                      Defaults to PROJECT_ROOT/data/forecasts/
        """
        if base_path is None:
            project_root = Path(__file__).resolve().parents[2]
            base_path = project_root / 'data' / 'forecasts'

        self.base_path = Path(base_path)
        self.price_path = self.base_path / 'price'
        self.consumption_path = self.base_path / 'consumption'

        # Create directories
        self.price_path.mkdir(parents=True, exist_ok=True)
        self.consumption_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"ForecastStorage initialized at: {self.base_path}")

    def _get_partition_path(self, forecast_type: str, forecast_time: datetime) -> Path:
        """Get path to monthly partition file."""
        year_month = forecast_time.strftime('%Y-%m')

        if forecast_type == 'price':
            return self.price_path / f'{year_month}.parquet'
        elif forecast_type == 'consumption':
            return self.consumption_path / f'{year_month}.parquet'
        else:
            raise ValueError(f"Unknown forecast type: {forecast_type}")

    def save_forecast(
        self,
        forecast_type: str,
        forecast_time: datetime,
        target_times: List[datetime],
        values: List[float],
        model_version: str = 'v14'
    ) -> Path:
        """
        Save a 24-hour forecast to storage.

        Args:
            forecast_type: 'price' or 'consumption'
            forecast_time: When the forecast was made
            target_times: List of 24 target hours
            values: List of 24 forecast values
            model_version: Model version string

        Returns:
            Path to the parquet file
        """
        if len(target_times) != len(values):
            raise ValueError("target_times and values must have same length")

        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'forecast_time': [forecast_time] * len(target_times),
            'target_time': target_times,
            'forecast_value': values,
            'model_version': [model_version] * len(target_times),
        })

        # Get partition file path
        partition_path = self._get_partition_path(forecast_type, forecast_time)

        # Append to existing file or create new
        if partition_path.exists():
            existing_df = pd.read_parquet(partition_path)
            combined_df = pd.concat([existing_df, forecast_df], ignore_index=True)
            # Remove duplicates (same forecast_time + target_time)
            combined_df = combined_df.drop_duplicates(
                subset=['forecast_time', 'target_time'],
                keep='last'
            )
        else:
            combined_df = forecast_df

        # Sort by forecast_time, target_time
        combined_df = combined_df.sort_values(['forecast_time', 'target_time'])

        # Save to parquet with compression
        combined_df.to_parquet(
            partition_path,
            compression='snappy',
            index=False
        )

        logger.info(f"Saved {forecast_type} forecast: {len(values)} hours to {partition_path}")
        return partition_path

    def load_forecasts(
        self,
        forecast_type: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Load forecasts from storage.

        Args:
            forecast_type: 'price' or 'consumption'
            start_date: Filter forecasts from this date
            end_date: Filter forecasts until this date

        Returns:
            DataFrame with all forecasts in range
        """
        if forecast_type == 'price':
            base_dir = self.price_path
        elif forecast_type == 'consumption':
            base_dir = self.consumption_path
        else:
            raise ValueError(f"Unknown forecast type: {forecast_type}")

        # Find all parquet files
        parquet_files = sorted(base_dir.glob('*.parquet'))

        if not parquet_files:
            logger.warning(f"No forecast files found for {forecast_type}")
            return pd.DataFrame()

        # Load and filter
        dfs = []
        for file_path in parquet_files:
            df = pd.read_parquet(file_path)

            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(df['forecast_time']):
                df['forecast_time'] = pd.to_datetime(df['forecast_time'])
            if not pd.api.types.is_datetime64_any_dtype(df['target_time']):
                df['target_time'] = pd.to_datetime(df['target_time'])

            # Apply date filters
            if start_date is not None:
                df = df[df['forecast_time'] >= start_date]
            if end_date is not None:
                df = df[df['forecast_time'] <= end_date]

            if len(df) > 0:
                dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        result = pd.concat(dfs, ignore_index=True)
        result = result.sort_values(['forecast_time', 'target_time'])

        logger.info(f"Loaded {len(result)} {forecast_type} forecast rows")
        return result

    def get_latest_forecast(self, forecast_type: str) -> pd.DataFrame:
        """
        Get the most recent 24-hour forecast.

        Args:
            forecast_type: 'price' or 'consumption'

        Returns:
            DataFrame with latest 24-hour forecast
        """
        all_forecasts = self.load_forecasts(forecast_type)

        if all_forecasts.empty:
            return pd.DataFrame()

        # Get latest forecast_time
        latest_time = all_forecasts['forecast_time'].max()

        return all_forecasts[all_forecasts['forecast_time'] == latest_time]

    def get_forecast_history(
        self,
        forecast_type: str,
        target_hour: datetime
    ) -> pd.DataFrame:
        """
        Get all forecasts that predicted a specific target hour.
        Useful for analyzing forecast accuracy over time.

        Args:
            forecast_type: 'price' or 'consumption'
            target_hour: The hour to look up

        Returns:
            DataFrame with all forecasts for that hour
        """
        all_forecasts = self.load_forecasts(forecast_type)

        if all_forecasts.empty:
            return pd.DataFrame()

        # Filter to target hour
        target_hour = pd.Timestamp(target_hour)
        mask = all_forecasts['target_time'] == target_hour

        return all_forecasts[mask].sort_values('forecast_time')

    def get_storage_stats(self) -> Dict:
        """Get storage statistics."""
        stats = {
            'base_path': str(self.base_path),
            'price': {'files': 0, 'total_rows': 0, 'size_mb': 0},
            'consumption': {'files': 0, 'total_rows': 0, 'size_mb': 0},
        }

        for forecast_type in ['price', 'consumption']:
            base_dir = self.price_path if forecast_type == 'price' else self.consumption_path
            files = list(base_dir.glob('*.parquet'))

            stats[forecast_type]['files'] = len(files)

            total_size = sum(f.stat().st_size for f in files)
            stats[forecast_type]['size_mb'] = round(total_size / (1024 * 1024), 2)

            total_rows = 0
            for f in files:
                try:
                    df = pd.read_parquet(f)
                    total_rows += len(df)
                except Exception:
                    pass
            stats[forecast_type]['total_rows'] = total_rows

        return stats


if __name__ == "__main__":
    # Test storage
    logging.basicConfig(level=logging.INFO)

    storage = ForecastStorage()

    # Test save
    from datetime import timedelta

    now = datetime.now()
    target_times = [now + timedelta(hours=i) for i in range(1, 25)]
    values = [100 + i * 10 for i in range(24)]

    storage.save_forecast(
        forecast_type='price',
        forecast_time=now,
        target_times=target_times,
        values=values,
        model_version='v14'
    )

    # Test load
    latest = storage.get_latest_forecast('price')
    print(f"Latest forecast: {len(latest)} rows")
    print(latest.head())

    # Stats
    stats = storage.get_storage_stats()
    print(f"Stats: {stats}")
