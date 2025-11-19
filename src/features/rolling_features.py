"""
Rolling Features Generator for ForeWatt
========================================
Creates rolling window statistics for target and key exogenous variables.

Features Created:
-----------------
For each variable (consumption, temperature, price_ptf):
- Rolling mean (24h, 168h windows)
- Rolling std (24h, 168h windows)
- Rolling min (24h, 168h windows)
- Rolling max (24h, 168h windows)

Additional Derived Features:
- consumption_range_24h: max - min (volatility measure)
- consumption_cv_24h: std / mean (coefficient of variation)
- temp_range_24h: diurnal temperature range

Rationale:
----------
- 24h window: Capture daily patterns and volatility
- 168h window: Capture weekly trends and stability
- Volatility measures: Identify periods of uncertainty

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


class RollingFeaturesGenerator:
    """Generate rolling window statistics for forecasting models."""

    # Rolling windows in hours
    WINDOWS = [24, 168]  # 1 day, 1 week

    # Statistics to compute
    STATS = ['mean', 'std', 'min', 'max']

    def __init__(self, data_dir: str = './data'):
        """
        Initialize rolling features generator.

        Args:
            data_dir: Root data directory
        """
        self.data_dir = Path(data_dir)
        self.silver_dir = self.data_dir / 'silver'
        self.gold_dir = self.data_dir / 'gold'
        self.output_dir = self.gold_dir / 'rolling_features'
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

    def create_rolling_stats(self, df: pd.DataFrame, column: str, windows: list, stats: list) -> pd.DataFrame:
        """
        Create rolling window statistics for a given column.

        Args:
            df: DataFrame with timestamp and column
            column: Column name to compute rolling stats
            windows: List of window sizes in hours
            stats: List of statistics to compute ('mean', 'std', 'min', 'max')

        Returns:
            DataFrame with original column and rolling features
        """
        logger.info(f"Creating rolling features for {column}...")

        result = df.copy()

        for window in windows:
            for stat in stats:
                feat_name = f"{column}_rolling_{stat}_{window}h"

                if stat == 'mean':
                    result[feat_name] = result[column].rolling(window=window, min_periods=1).mean()
                elif stat == 'std':
                    result[feat_name] = result[column].rolling(window=window, min_periods=window).std()
                elif stat == 'min':
                    result[feat_name] = result[column].rolling(window=window, min_periods=1).min()
                elif stat == 'max':
                    result[feat_name] = result[column].rolling(window=window, min_periods=1).max()

                n_missing = result[feat_name].isna().sum()
                logger.info(f"  {feat_name}: {n_missing} missing values")

        return result

    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features from rolling statistics.

        Features:
        - consumption_range_24h: Daily volatility (max - min)
        - consumption_cv_24h: Coefficient of variation (std / mean)
        - temp_range_24h: Diurnal temperature range
        """
        logger.info("Creating derived rolling features...")

        result = df.copy()

        # Consumption range (volatility)
        if 'consumption_rolling_max_24h' in df.columns and 'consumption_rolling_min_24h' in df.columns:
            result['consumption_range_24h'] = (
                df['consumption_rolling_max_24h'] - df['consumption_rolling_min_24h']
            )
            logger.info("  Created consumption_range_24h")

        # Consumption coefficient of variation
        if 'consumption_rolling_std_24h' in df.columns and 'consumption_rolling_mean_24h' in df.columns:
            result['consumption_cv_24h'] = (
                df['consumption_rolling_std_24h'] / (df['consumption_rolling_mean_24h'] + 1e-6)
            )
            logger.info("  Created consumption_cv_24h")

        # Temperature range (diurnal variation)
        if 'temperature_rolling_max_24h' in df.columns and 'temperature_rolling_min_24h' in df.columns:
            result['temp_range_24h'] = (
                df['temperature_rolling_max_24h'] - df['temperature_rolling_min_24h']
            )
            logger.info("  Created temp_range_24h")

        return result

    def generate_all_rolling(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Generate all rolling features and merge into single dataset.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with all rolling features
        """
        logger.info(f"Generating rolling features for {start_date} to {end_date}...")

        # Load data
        consumption_df = self.load_consumption(start_date, end_date)
        temperature_df = self.load_temperature(start_date, end_date)
        price_df = self.load_price(start_date, end_date)

        # Create rolling stats
        consumption_rolling = self.create_rolling_stats(
            consumption_df, 'consumption', self.WINDOWS, self.STATS
        )
        temperature_rolling = self.create_rolling_stats(
            temperature_df, 'temperature', self.WINDOWS, self.STATS
        )
        price_rolling = self.create_rolling_stats(
            price_df, 'price_ptf', self.WINDOWS, self.STATS
        )

        # Merge all features on timestamp
        logger.info("Merging rolling features...")
        result = consumption_rolling.merge(
            temperature_rolling, on='timestamp', how='inner'
        ).merge(
            price_rolling, on='timestamp', how='inner'
        )

        # Add derived features
        result = self.create_derived_features(result)

        logger.info(f"Total features created: {len(result.columns) - 1} (excl. timestamp)")
        logger.info(f"Final shape: {result.shape}")

        return result

    def save_rolling_features(self, df: pd.DataFrame, start_date: str, end_date: str):
        """Save rolling features to Gold layer."""
        # Parquet
        parquet_path = self.output_dir / f'rolling_features_{start_date}_{end_date}.parquet'
        df.to_parquet(parquet_path, index=False, engine='pyarrow')
        logger.info(f"Saved Parquet: {parquet_path}")

        # CSV (secondary format for inspection)
        csv_path = self.output_dir / f'rolling_features_{start_date}_{end_date}.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved CSV: {csv_path}")

        # Log summary statistics
        logger.info("\nRolling Features Summary:")
        logger.info(f"Total records: {len(df)}")
        logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        logger.info(f"Missing values per column:")
        for col in df.columns:
            n_missing = df[col].isna().sum()
            if n_missing > 0:
                logger.info(f"  {col}: {n_missing} ({n_missing/len(df)*100:.2f}%)")

    def run_pipeline(self, start_date: str = '2020-01-01', end_date: str = '2025-10-31'):
        """Run complete rolling features pipeline."""
        logger.info("="*60)
        logger.info("ROLLING FEATURES PIPELINE")
        logger.info("="*60)

        # Generate rolling features
        rolling_df = self.generate_all_rolling(start_date, end_date)

        # Save
        self.save_rolling_features(rolling_df, start_date, end_date)

        logger.info("="*60)
        logger.info("ROLLING FEATURES PIPELINE COMPLETE")
        logger.info("="*60)

        return rolling_df


def main():
    """Main execution."""
    generator = RollingFeaturesGenerator(data_dir='./data')
    rolling_df = generator.run_pipeline(
        start_date='2020-01-01',
        end_date='2025-10-31'
    )

    print("\nRolling features preview:")
    print(rolling_df.head())
    print("\nRolling features columns:")
    print(rolling_df.columns.tolist())


if __name__ == '__main__':
    main()
