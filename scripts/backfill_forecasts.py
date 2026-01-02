"""
Backfill Historical Forecasts
==============================
Generates historical forecasts for the entire dataset.

This creates forecast records as if the model had been running hourly
throughout the historical period. Useful for:
- Dashboard historical views
- Backtesting forecast accuracy
- Populating initial data

Usage:
    python scripts/backfill_forecasts.py
    python scripts/backfill_forecasts.py --start-date 2023-06-01  # Resume from date
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import logging

import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import ForecastPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def backfill_all(start_date=None):
    """Backfill all historical forecasts.

    Args:
        start_date: Optional start date to resume from (YYYY-MM-DD format string or datetime)
    """

    # Load data to get date range
    data_path = PROJECT_ROOT / 'data' / 'gold' / 'master' / 'master_v2_fundamental.parquet'
    df = pd.read_parquet(data_path)

    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date' in df.columns:
            df = df.set_index('date')

    # Need at least 200 hours of history before starting
    # Start from hour 200 by default
    start_idx = 200

    # If start_date provided, find the corresponding index
    if start_date:
        if isinstance(start_date, str):
            start_date = pd.Timestamp(start_date, tz='UTC')
        elif not hasattr(start_date, 'tzinfo') or start_date.tzinfo is None:
            start_date = pd.Timestamp(start_date, tz='UTC')

        # Find the index for the start date
        matching_idx = df.index.get_indexer([start_date], method='bfill')[0]
        if matching_idx >= 0:
            start_idx = max(start_idx, matching_idx)
            logger.info(f"Resuming from {start_date} (index {start_idx})")

    logger.info("=" * 70)
    logger.info("BACKFILL HISTORICAL FORECASTS")
    logger.info("=" * 70)
    logger.info(f"Data range: {df.index.min()} to {df.index.max()}")
    logger.info(f"Total hours to process: {len(df) - start_idx:,}")

    # Initialize pipeline
    pipeline = ForecastPipeline()

    # Process in batches for efficiency
    batch_size = 100
    total = len(df) - start_idx
    processed = 0
    failed = 0

    start_time = datetime.now()

    for batch_start in range(start_idx, len(df), batch_size):
        batch_end = min(batch_start + batch_size, len(df))

        for i in range(batch_start, batch_end):
            try:
                # Get data up to this point
                historical_data = df.iloc[:i].tail(200)
                forecast_time = df.index[i]

                # Skip if not enough data
                if len(historical_data) < 50:
                    failed += 1
                    continue

                # Generate price forecast
                price_forecast = pipeline.price_forecaster.forecast_next_24h(
                    historical_data,
                    forecast_time.to_pydatetime() if hasattr(forecast_time, 'to_pydatetime') else forecast_time
                )

                # Generate consumption forecast
                consumption_forecast = pipeline.consumption_forecaster.forecast_next_24h(
                    historical_data,
                    forecast_time.to_pydatetime() if hasattr(forecast_time, 'to_pydatetime') else forecast_time
                )

                # Convert forecast_time to Python datetime
                ft = forecast_time.to_pydatetime() if hasattr(forecast_time, 'to_pydatetime') else forecast_time

                # Store forecasts
                pipeline.storage.save_forecast(
                    'price', ft,
                    price_forecast['target_time'].tolist(),
                    price_forecast['forecast_value'].tolist(),
                    'v14'
                )

                pipeline.storage.save_forecast(
                    'consumption', ft,
                    consumption_forecast['target_time'].tolist(),
                    consumption_forecast['forecast_value'].tolist(),
                    'v1'
                )

                processed += 1

            except Exception as e:
                failed += 1
                if failed <= 5:
                    logger.error(f"Failed at index {i}: {e}")

        # Progress update
        elapsed = (datetime.now() - start_time).total_seconds()
        rate = processed / elapsed if elapsed > 0 else 0
        eta = (total - processed) / rate if rate > 0 else 0

        logger.info(
            f"Progress: {processed:,}/{total:,} ({processed/total*100:.1f}%) | "
            f"Rate: {rate:.1f}/s | ETA: {eta/60:.1f} min | Failed: {failed}"
        )

    # Final stats
    elapsed = (datetime.now() - start_time).total_seconds()
    stats = pipeline.storage.get_storage_stats()

    logger.info("=" * 70)
    logger.info("BACKFILL COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Processed: {processed:,}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Runtime: {elapsed/60:.1f} minutes")
    logger.info(f"Storage stats: {stats}")

    return {
        'processed': processed,
        'failed': failed,
        'runtime_seconds': elapsed,
        'storage': stats
    }


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser(description='Backfill historical forecasts')
    parser.add_argument('--start-date', type=str, help='Start date to resume from (YYYY-MM-DD)')
    args = parser.parse_args()

    results = backfill_all(start_date=args.start_date)
    print(f"\nDone! Processed {results['processed']:,} forecasts in {results['runtime_seconds']/60:.1f} minutes")
