"""
ForeWatt Forecast Pipeline
===========================
Main pipeline that runs hourly to generate and store forecasts.

This pipeline:
1. Loads the latest market data
2. Generates 24-hour price forecasts
3. Generates 24-hour consumption forecasts
4. Stores forecasts in Parquet files

Designed to run on Cloud Run, triggered by Cloud Scheduler every hour.
"""

import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional
import logging

import pandas as pd
import numpy as np

from .price_inference import PriceForecaster
from .consumption_inference import ConsumptionForecaster
from .storage import ForecastStorage

logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / 'data' / 'gold' / 'master' / 'master_v2_fundamental.parquet'


class ForecastPipeline:
    """
    Main forecast pipeline for hourly predictions.

    Handles both price and consumption forecasting with Parquet storage.
    """

    def __init__(
        self,
        data_path: str = None,
        storage_path: str = None,
        price_models_dir: str = None,
        consumption_models_dir: str = None
    ):
        """
        Initialize the forecast pipeline.

        Args:
            data_path: Path to master data file
            storage_path: Base path for forecast storage
            price_models_dir: Path to price models
            consumption_models_dir: Path to consumption models
        """
        self.data_path = Path(data_path) if data_path else DATA_PATH

        # Initialize components
        self.storage = ForecastStorage(storage_path)
        self.price_forecaster = PriceForecaster(price_models_dir)
        self.consumption_forecaster = ConsumptionForecaster(consumption_models_dir)

        logger.info("ForecastPipeline initialized")
        logger.info(f"  Data source: {self.data_path}")
        logger.info(f"  Storage: {self.storage.base_path}")

    def load_latest_data(self, min_hours: int = 200) -> pd.DataFrame:
        """
        Load the latest market data for forecasting.

        In production, this would fetch from a live data source.
        For now, reads from the master parquet file.

        Args:
            min_hours: Minimum hours of history needed

        Returns:
            DataFrame with recent market data
        """
        logger.info(f"Loading data from: {self.data_path}")

        df = pd.read_parquet(self.data_path)

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'date' in df.columns:
                df = df.set_index('date')
            elif 'datetime' in df.columns:
                df = df.set_index('datetime')

        # Get last N hours
        df = df.tail(min_hours)

        logger.info(f"Loaded {len(df)} rows: {df.index.min()} to {df.index.max()}")

        return df

    def run_price_forecast(
        self,
        data: pd.DataFrame,
        forecast_time: datetime
    ) -> pd.DataFrame:
        """
        Generate and store price forecast.

        Args:
            data: Historical data for features
            forecast_time: When the forecast is made

        Returns:
            DataFrame with 24-hour forecast
        """
        logger.info("Generating price forecast...")

        # Generate forecast
        forecast = self.price_forecaster.forecast_next_24h(data, forecast_time)

        # Store to parquet
        self.storage.save_forecast(
            forecast_type='price',
            forecast_time=forecast_time,
            target_times=forecast['target_time'].tolist(),
            values=forecast['forecast_value'].tolist(),
            model_version='v14'
        )

        logger.info(f"Price forecast complete: {len(forecast)} hours")

        return forecast

    def run_consumption_forecast(
        self,
        data: pd.DataFrame,
        forecast_time: datetime
    ) -> pd.DataFrame:
        """
        Generate and store consumption forecast.

        Args:
            data: Historical data for features
            forecast_time: When the forecast is made

        Returns:
            DataFrame with 24-hour forecast
        """
        logger.info("Generating consumption forecast...")

        # Generate forecast
        forecast = self.consumption_forecaster.forecast_next_24h(data, forecast_time)

        # Store to parquet
        self.storage.save_forecast(
            forecast_type='consumption',
            forecast_time=forecast_time,
            target_times=forecast['target_time'].tolist(),
            values=forecast['forecast_value'].tolist(),
            model_version='v1'
        )

        logger.info(f"Consumption forecast complete: {len(forecast)} hours")

        return forecast

    def run(self, forecast_time: Optional[datetime] = None) -> Dict:
        """
        Run the complete forecast pipeline.

        This is the main entry point for hourly forecasting.

        Args:
            forecast_time: Base time for forecasts (defaults to now)

        Returns:
            Dictionary with forecast results and metadata
        """
        if forecast_time is None:
            forecast_time = datetime.now().replace(minute=0, second=0, microsecond=0)

        logger.info("=" * 60)
        logger.info(f"FORECAST PIPELINE - {forecast_time.isoformat()}")
        logger.info("=" * 60)

        start_time = datetime.now()

        # Load data
        data = self.load_latest_data()

        # Run price forecast
        price_forecast = self.run_price_forecast(data, forecast_time)

        # Run consumption forecast
        consumption_forecast = self.run_consumption_forecast(data, forecast_time)

        # Calculate runtime
        runtime = (datetime.now() - start_time).total_seconds()

        # Prepare results
        results = {
            'forecast_time': forecast_time.isoformat(),
            'runtime_seconds': runtime,
            'price': {
                'count': len(price_forecast),
                'min': float(price_forecast['forecast_value'].min()),
                'max': float(price_forecast['forecast_value'].max()),
                'mean': float(price_forecast['forecast_value'].mean()),
            },
            'consumption': {
                'count': len(consumption_forecast),
                'min': float(consumption_forecast['forecast_value'].min()),
                'max': float(consumption_forecast['forecast_value'].max()),
                'mean': float(consumption_forecast['forecast_value'].mean()),
            },
            'storage': self.storage.get_storage_stats(),
        }

        logger.info("=" * 60)
        logger.info(f"PIPELINE COMPLETE - {runtime:.2f}s")
        logger.info(f"  Price: {results['price']['min']:.1f} - {results['price']['max']:.1f} TL/MWh")
        logger.info(f"  Consumption: {results['consumption']['min']:.0f} - {results['consumption']['max']:.0f} MWh")
        logger.info("=" * 60)

        return results

    def backfill(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict:
        """
        Backfill historical forecasts.

        Useful for populating forecast history from historical data.

        Args:
            start_date: Start of backfill period
            end_date: End of backfill period

        Returns:
            Summary of backfill operation
        """
        logger.info(f"Backfilling forecasts: {start_date} to {end_date}")

        # Load full data
        df = pd.read_parquet(self.data_path)
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'date' in df.columns:
                df = df.set_index('date')

        # Ensure timezone consistency
        if df.index.tz is not None:
            start_date = pd.Timestamp(start_date).tz_localize(df.index.tz)
            end_date = pd.Timestamp(end_date).tz_localize(df.index.tz)

        # Generate hourly timestamps
        hours = pd.date_range(start_date, end_date, freq='H')

        total = len(hours)
        success = 0
        failed = 0

        for i, forecast_time in enumerate(hours):
            try:
                # Get data up to this point
                mask = df.index <= forecast_time
                available_data = df[mask].tail(200)

                if len(available_data) < 50:
                    logger.warning(f"Insufficient data for {forecast_time}")
                    failed += 1
                    continue

                # Generate forecasts
                price_forecast = self.price_forecaster.forecast_next_24h(
                    available_data, forecast_time.to_pydatetime()
                )
                consumption_forecast = self.consumption_forecaster.forecast_next_24h(
                    available_data, forecast_time.to_pydatetime()
                )

                # Store
                self.storage.save_forecast(
                    'price', forecast_time.to_pydatetime(),
                    price_forecast['target_time'].tolist(),
                    price_forecast['forecast_value'].tolist()
                )
                self.storage.save_forecast(
                    'consumption', forecast_time.to_pydatetime(),
                    consumption_forecast['target_time'].tolist(),
                    consumption_forecast['forecast_value'].tolist()
                )

                success += 1

                if (i + 1) % 100 == 0:
                    logger.info(f"Progress: {i + 1}/{total} ({success} success, {failed} failed)")

            except Exception as e:
                logger.error(f"Failed for {forecast_time}: {e}")
                failed += 1

        return {
            'total': total,
            'success': success,
            'failed': failed,
            'storage': self.storage.get_storage_stats(),
        }


def run_hourly_forecast():
    """Entry point for hourly forecast job."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    pipeline = ForecastPipeline()
    results = pipeline.run()

    return results


if __name__ == "__main__":
    results = run_hourly_forecast()
    print(f"\nResults: {results}")
