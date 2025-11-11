"""
Lag Features Generator for ForeWatt
====================================
Creates lagged features for target variable (consumption) and key exogenous variables.

Features Created:
-----------------
Target (Consumption):
- consumption_lag_1h, 2h, 3h, 6h, 12h, 24h, 48h, 168h (1 week)

Temperature (Key weather driver):
- temp_lag_1h, 2h, 3h, 24h, 168h

Electricity Price (Economic signal):
- price_ptf_real_lag_1h, 24h, 168h

Rationale:
----------
- Short lags (1-3h): Capture inertia and autoregressive patterns
- Daily lag (24h): Capture day-over-day changes and seasonality
- Weekly lag (168h): Capture week-over-week patterns and trends

Author: ForeWatt Team
Date: November 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LagFeaturesGenerator:
    """Generate lag features for forecasting models."""

    # Lag periods in hours
    CONSUMPTION_LAGS = [1, 2, 3, 6, 12, 24, 48, 168]
    TEMPERATURE_LAGS = [1, 2, 3, 24, 168]
    PRICE_LAGS = [1, 24, 168]

    def __init__(self, data_dir: str = './data'):
        """
        Initialize lag features generator.

        Args:
            data_dir: Root data directory
        """
        self.data_dir = Path(data_dir)
        self.silver_dir = self.data_dir / 'silver'
        self.gold_dir = self.data_dir / 'gold'
        self.output_dir = self.gold_dir / 'lag_features'
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_consumption(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load consumption (target variable) from Silver EPİAŞ."""
        logger.info("Loading consumption data...")
        file_path = self.silver_dir / 'epias' / f'consumption_actual_normalized_{start_date}_{end_date}.parquet'

        if not file_path.exists():
            raise FileNotFoundError(f"Consumption data not found: {file_path}")

        df = pd.read_parquet(file_path)

        # Handle different timestamp column names
        timestamp_col = None
        for col in ['timestamp', 'date', 'datetime']:
            if col in df.columns:
                timestamp_col = col
                break

        if timestamp_col is None:
            raise ValueError(f"No timestamp column found. Available columns: {df.columns.tolist()}")

        df['timestamp'] = pd.to_datetime(df[timestamp_col])
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Rename to standard column name
        if 'consumption' not in df.columns and 'value' in df.columns:
            df = df.rename(columns={'value': 'consumption'})

        logger.info(f"Loaded consumption: {len(df)} records")
        return df[['timestamp', 'consumption']]

    def load_temperature(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load national temperature from Gold weather features."""
        logger.info("Loading temperature data...")
        file_path = self.gold_dir / 'demand_features' / f'demand_features_{start_date}_{end_date}.parquet'

        if not file_path.exists():
            raise FileNotFoundError(f"Weather features not found: {file_path}")

        df = pd.read_parquet(file_path)

        # Check if timestamp is in index
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            # Rename index to timestamp if needed
            if df.columns[0] in ['datetime', 'date']:
                df = df.rename(columns={df.columns[0]: 'timestamp'})

        # Handle different timestamp column names
        timestamp_col = None
        for col in ['timestamp', 'date', 'datetime']:
            if col in df.columns:
                timestamp_col = col
                break

        if timestamp_col is None:
            raise ValueError(f"No timestamp column found. Available columns: {df.columns.tolist()}")

        df['timestamp'] = pd.to_datetime(df[timestamp_col])
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Extract temperature (primary weather variable)
        temp_col = None
        for col in ['temperature_2m', 'temp_weighted', 'temp_national', 'temperature']:
            if col in df.columns:
                temp_col = col
                break

        if temp_col is None:
            raise ValueError(f"No temperature column found. Available: {df.columns.tolist()}")

        logger.info(f"Using temperature column: {temp_col}")

        return df[['timestamp', temp_col]].rename(columns={temp_col: 'temperature'})

    def load_price(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load deflated electricity price from Gold EPİAŞ."""
        logger.info("Loading deflated price data...")
        file_path = self.gold_dir / 'epias' / f'price_ptf_deflated_{start_date}_{end_date}.parquet'

        if not file_path.exists():
            logger.warning(f"Deflated price not found: {file_path}")
            logger.warning("Falling back to normalized price from Silver...")

            # Fallback to silver normalized price
            file_path = self.silver_dir / 'epias' / f'price_ptf_normalized_{start_date}_{end_date}.parquet'
            if not file_path.exists():
                raise FileNotFoundError(f"Price data not found in Gold or Silver")

        df = pd.read_parquet(file_path)

        # Handle different timestamp column names
        timestamp_col = None
        for col in ['timestamp', 'date', 'datetime']:
            if col in df.columns:
                timestamp_col = col
                break

        if timestamp_col is None:
            raise ValueError(f"No timestamp column found. Available columns: {df.columns.tolist()}")

        df['timestamp'] = pd.to_datetime(df[timestamp_col])
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Find price column
        price_col = None
        for col in ['price_ptf_real', 'price_real', 'ptf', 'price', 'value']:
            if col in df.columns:
                price_col = col
                break

        if price_col is None:
            raise ValueError(f"Could not find price column. Available: {df.columns.tolist()}")

        logger.info(f"Using price column: {price_col}")
        return df[['timestamp', price_col]].rename(columns={price_col: 'price_ptf'})

    def create_lags(self, df: pd.DataFrame, column: str, lags: list) -> pd.DataFrame:
        """
        Create lag features for a given column.

        Args:
            df: DataFrame with timestamp and column
            column: Column name to lag
            lags: List of lag periods in hours

        Returns:
            DataFrame with original column and lag features
        """
        logger.info(f"Creating {len(lags)} lag features for {column}...")

        result = df.copy()

        for lag in lags:
            lag_col = f"{column}_lag_{lag}h"
            result[lag_col] = result[column].shift(lag)

            # Count missing values
            n_missing = result[lag_col].isna().sum()
            logger.info(f"  {lag_col}: {n_missing} missing values (from shift)")

        return result

    def generate_all_lags(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Generate all lag features and merge into single dataset.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with all lag features
        """
        logger.info(f"Generating lag features for {start_date} to {end_date}...")

        # Load data
        consumption_df = self.load_consumption(start_date, end_date)
        temperature_df = self.load_temperature(start_date, end_date)
        price_df = self.load_price(start_date, end_date)

        # Create lags
        consumption_lagged = self.create_lags(consumption_df, 'consumption', self.CONSUMPTION_LAGS)
        temperature_lagged = self.create_lags(temperature_df, 'temperature', self.TEMPERATURE_LAGS)
        price_lagged = self.create_lags(price_df, 'price_ptf', self.PRICE_LAGS)

        # Merge all features on timestamp
        logger.info("Merging lag features...")
        result = consumption_lagged.merge(
            temperature_lagged, on='timestamp', how='inner'
        ).merge(
            price_lagged, on='timestamp', how='inner'
        )

        logger.info(f"Total features created: {len(result.columns) - 1} (excl. timestamp)")
        logger.info(f"Final shape: {result.shape}")

        return result

    def save_lag_features(self, df: pd.DataFrame, start_date: str, end_date: str):
        """Save lag features to Gold layer."""
        # Parquet
        parquet_path = self.output_dir / f'lag_features_{start_date}_{end_date}.parquet'
        df.to_parquet(parquet_path, index=False, engine='pyarrow')
        logger.info(f"Saved Parquet: {parquet_path}")

        # CSV (secondary format for inspection)
        csv_path = self.output_dir / f'lag_features_{start_date}_{end_date}.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved CSV: {csv_path}")

        # Log summary statistics
        logger.info("\nLag Features Summary:")
        logger.info(f"Total records: {len(df)}")
        logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        logger.info(f"Missing values per column:")
        for col in df.columns:
            n_missing = df[col].isna().sum()
            if n_missing > 0:
                logger.info(f"  {col}: {n_missing} ({n_missing/len(df)*100:.2f}%)")

    def run_pipeline(self, start_date: str = '2020-01-01', end_date: str = '2024-12-31'):
        """Run complete lag features pipeline."""
        logger.info("="*60)
        logger.info("LAG FEATURES PIPELINE")
        logger.info("="*60)

        # Generate lags
        lag_df = self.generate_all_lags(start_date, end_date)

        # Save
        self.save_lag_features(lag_df, start_date, end_date)

        logger.info("="*60)
        logger.info("LAG FEATURES PIPELINE COMPLETE")
        logger.info("="*60)

        return lag_df


def main():
    """Main execution."""
    generator = LagFeaturesGenerator(data_dir='./data')
    lag_df = generator.run_pipeline(
        start_date='2020-01-01',
        end_date='2024-12-31'
    )

    print("\nLag features preview:")
    print(lag_df.head())
    print("\nLag features columns:")
    print(lag_df.columns.tolist())


if __name__ == '__main__':
    main()
