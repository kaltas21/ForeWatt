"""
EPİAŞ Price Deflation Utility
==============================
Converts nominal Turkish Lira prices to real values using EVDS-based deflator indices.

Purpose:
--------
Electricity prices in Turkey are subject to high inflation. To enable meaningful
time-series modeling, we need to normalize nominal prices to real values.

This utility:
1. Loads EPİAŞ price data (bronze or silver layer)
2. Loads deflator index (DID_index from deflator_builder.py)
3. Interpolates monthly deflator to hourly (monthly → daily linear → hourly ffill)
4. Joins hourly price data with hourly deflator
5. Applies deflation: real_price = nominal_price / (DID_index / 100)
6. Saves deflated data to gold layer

EPİAŞ Datasets Requiring Deflation:
------------------------------------
All price datasets in Turkish Lira (TL/MWh):

1. price_ptf: Day-Ahead Market Clearing Price (Piyasa Takas Fiyatı)
   - Files: data/bronze/epias/price_ptf_*.parquet
   - Columns: Date, Price (TL/MWh)

2. price_smf: Balancing Power Market Price (Sistem Marjinal Fiyatı)
   - Files: data/bronze/epias/price_smf_*.parquet
   - Columns: Date, Price (TL/MWh)

3. price_idm: Intraday Market Quantity (Gün İçi Piyasa)
   - Files: data/bronze/epias/price_idm_*.parquet
   - Columns: Date, Quantity/Price (TL/MWh)

4. price_wap: Weighted Average Price
   - Files: data/bronze/epias/price_wap_*.parquet
   - Columns: Date, Price (TL/MWh)

Datasets NOT Requiring Deflation:
----------------------------------
- consumption_*: Measured in MW (physical units)
- generation_*: Measured in MW (physical units)
- capacity_*: Measured in MW (physical units)
- wind_forecast: Measured in MW (physical units)
- hydro_*: Measured in volume/energy (physical units)

Usage:
------
    python src/data/deflate_prices.py

Or import as module:
    from src.data.deflate_prices import deflate_price_dataset

    deflate_price_dataset(
        dataset_name='price_ptf',
        start_date='2020-01-01',
        end_date='2025-10-31',
        deflator_method='baseline'  # or 'dfm'
    )

Author: ForeWatt Team
Date: November 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Optional, Literal

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PriceDeflator:
    """
    Utility to deflate EPİAŞ price data using EVDS-based deflator indices.
    """

    # EPİAŞ price datasets that need deflation
    PRICE_DATASETS = ['price_ptf', 'price_smf', 'price_idm', 'price_wap']

    def __init__(self, deflator_method: Literal['baseline', 'dfm'] = 'baseline'):
        """
        Initialize deflator with chosen method.

        Args:
            deflator_method: 'baseline' (Factor Analysis) or 'dfm' (Dynamic Factor Model)
        """
        self.deflator_method = deflator_method
        self.deflator_df = None
        self._load_deflator()

    def _load_deflator(self):
        """
        Load and interpolate deflator index from silver layer.

        Methodology per spec:
        1. Load monthly DID_index (base=100 at 2022-01)
        2. Monthly → Daily: Linear interpolation (smooth transitions between months)
        3. Daily → Hourly: Forward fill (all 24 hours in day get same value)

        This preserves import/FX noise while providing smooth domestic inflation adjustment.
        """
        silver_dir = Path("data/silver/macro")

        if self.deflator_method == 'baseline':
            deflator_file = silver_dir / "deflator_did_baseline.parquet"
            if not deflator_file.exists():
                deflator_file = silver_dir / "deflator_did_baseline.csv"
        else:
            deflator_file = silver_dir / "deflator_did_dfm.parquet"
            if not deflator_file.exists():
                deflator_file = silver_dir / "deflator_did_dfm.csv"

        if not deflator_file.exists():
            raise FileNotFoundError(
                f"Deflator file not found: {deflator_file}\n"
                f"Run deflator_builder.py first to generate deflator indices."
            )

        # Load deflator
        if deflator_file.suffix == '.parquet':
            deflator_monthly = pd.read_parquet(deflator_file)
        else:
            deflator_monthly = pd.read_csv(deflator_file)

        # Validate required columns
        if 'DATE' not in deflator_monthly.columns or 'DID_index' not in deflator_monthly.columns:
            raise ValueError(
                f"Deflator file must contain 'DATE' and 'DID_index' columns. "
                f"Found: {list(deflator_monthly.columns)}"
            )

        logger.info(f"✓ Loaded monthly deflator ({self.deflator_method}): {len(deflator_monthly)} months")
        logger.info(f"  Date range: {deflator_monthly['DATE'].min()} to {deflator_monthly['DATE'].max()}")

        # Convert to hourly with interpolation (monthly → daily → hourly)
        self.deflator_df = self._interpolate_monthly_to_hourly(deflator_monthly)

        # Log sample values
        base_month = self.deflator_df[self.deflator_df['datetime'].dt.to_period('M') == '2022-01']
        if not base_month.empty:
            logger.info(f"  Base value (2022-01): {base_month['DID_index'].iloc[0]:.2f}")

    def _interpolate_monthly_to_hourly(self, monthly_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert monthly deflator to hourly frequency with linear interpolation.

        Steps per methodology:
        1. Monthly → Daily: Linear interpolation (smooth transitions between months)
        2. Daily → Hourly: Forward fill (24 hours per day get same value)

        This approach:
        - Avoids step changes at month boundaries (would create artificial volatility)
        - Preserves FX/import shock noise (not in deflator, will be in features)
        - Provides granular deflator for hourly price data

        Args:
            monthly_df: Monthly deflator with columns ['DATE', 'DID_index']
                       DATE in 'YYYY-MM' format

        Returns:
            Hourly deflator with columns ['datetime', 'DID_index']
        """
        # Parse DATE column (handles both 'YYYY-MM' and datetime)
        if not pd.api.types.is_datetime64_any_dtype(monthly_df['DATE']):
            # Convert YYYY-MM to datetime (first day of month)
            monthly_df['datetime'] = pd.to_datetime(monthly_df['DATE'], format='%Y-%m')
        else:
            monthly_df['datetime'] = pd.to_datetime(monthly_df['DATE'])

        # Keep only datetime and DID_index
        monthly_df = monthly_df[['datetime', 'DID_index']].sort_values('datetime').copy()

        # Step 1: Monthly → Daily (linear interpolation)
        # Create daily date range
        date_range = pd.date_range(
            start=monthly_df['datetime'].min(),
            end=monthly_df['datetime'].max() + pd.DateOffset(months=1) - pd.Timedelta(days=1),
            freq='D'
        )

        # Reindex to daily and interpolate linearly
        daily_df = monthly_df.set_index('datetime').reindex(date_range)
        daily_df['DID_index'] = daily_df['DID_index'].interpolate(method='linear')

        # Forward fill any remaining gaps (at edges)
        daily_df['DID_index'] = daily_df['DID_index'].ffill().bfill()

        # Step 2: Daily → Hourly (forward fill)
        # Create hourly date range
        hourly_range = pd.date_range(
            start=date_range[0],
            end=date_range[-1] + pd.Timedelta(days=1) - pd.Timedelta(hours=1),
            freq='h'  # Use 'h' instead of deprecated 'H'
        )

        # Reindex to hourly and forward fill (all 24 hours in day get same value)
        hourly_df = daily_df.reindex(hourly_range, method='ffill')
        hourly_df = hourly_df.reset_index().rename(columns={'index': 'datetime'})

        # Localize to Europe/Istanbul timezone (EPİAŞ data timezone)
        hourly_df['datetime'] = hourly_df['datetime'].dt.tz_localize('Europe/Istanbul')

        logger.info(f"  ✓ Interpolated to hourly: {len(hourly_df)} hours")
        logger.info(f"    Hourly range: {hourly_df['datetime'].min()} to {hourly_df['datetime'].max()}")

        return hourly_df

    def deflate_dataset(
        self,
        dataset_name: str,
        start_date: str,
        end_date: str,
        layer: Literal['bronze', 'silver'] = 'silver',
        output_layer: str = 'gold'
    ) -> pd.DataFrame:
        """
        Deflate a single EPİAŞ price dataset.

        Args:
            dataset_name: Name of dataset (e.g., 'price_ptf')
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
            layer: Input layer ('bronze' or 'silver')
            output_layer: Output layer (default 'gold')

        Returns:
            DataFrame with deflated prices
        """
        if dataset_name not in self.PRICE_DATASETS:
            raise ValueError(
                f"Dataset '{dataset_name}' is not a price dataset. "
                f"Valid options: {self.PRICE_DATASETS}"
            )

        logger.info(f"Deflating {dataset_name} ({start_date} to {end_date})...")

        # Load EPİAŞ price data
        input_dir = Path(f"data/{layer}/epias")

        if layer == 'bronze':
            input_file = input_dir / f"{dataset_name}_{start_date}_{end_date}.parquet"
        else:
            input_file = input_dir / f"{dataset_name}_normalized_{start_date}_{end_date}.parquet"

        if not input_file.exists():
            # Try CSV fallback
            input_file = input_file.with_suffix('.csv')
            if not input_file.exists():
                raise FileNotFoundError(f"Price data not found: {input_file}")

        # Load price data
        if input_file.suffix == '.parquet':
            df = pd.read_parquet(input_file)
        else:
            df = pd.read_csv(input_file)

        logger.info(f"  Loaded {len(df)} records from {input_file.name}")

        # Find datetime column
        datetime_col = self._find_datetime_column(df)
        if not datetime_col:
            raise ValueError(f"Could not find datetime column in {dataset_name}")

        # Ensure datetime column is datetime type
        if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
            df[datetime_col] = pd.to_datetime(df[datetime_col])

        # Standardize datetime column name for merging
        df = df.rename(columns={datetime_col: 'datetime'})

        # Find price columns (usually contain 'Price', 'Fiyat', or 'MCP', 'PTF', 'SMF')
        price_cols = self._find_price_columns(df)
        if not price_cols:
            raise ValueError(f"Could not find price columns in {dataset_name}")

        logger.info(f"  Found {len(price_cols)} price column(s): {price_cols}")

        # Join with hourly deflator (exact timestamp matching)
        df_merged = df.merge(
            self.deflator_df[['datetime', 'DID_index']],
            on='datetime',
            how='left'
        )

        # Check for missing deflator values
        missing_deflator = df_merged['DID_index'].isna().sum()
        if missing_deflator > 0:
            logger.warning(
                f"  ⚠ {missing_deflator} records missing deflator values "
                f"(dates outside deflator range). Will forward-fill."
            )
            df_merged['DID_index'] = df_merged['DID_index'].ffill().bfill()

        # Apply deflation to each price column
        for col in price_cols:
            # Formula: real_price = nominal_price / (DID_index / 100)
            # Equivalent to: real_price = nominal_price * 100 / DID_index
            df_merged[f"{col}_real"] = df_merged[col] / (df_merged['DID_index'] / 100.0)

            # Log deflation stats
            nominal_mean = df_merged[col].mean()
            real_mean = df_merged[f"{col}_real"].mean()
            logger.info(
                f"    {col}: {nominal_mean:.2f} TL/MWh (nominal) → "
                f"{real_mean:.2f} TL/MWh (real, base=2022-01)"
            )

        # datetime column is preserved (primary timestamp column)

        # Save to gold layer
        output_dir = Path(f"data/{output_layer}/epias")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_base = output_dir / f"{dataset_name}_deflated_{start_date}_{end_date}"

        # Save as Parquet (primary)
        parquet_path = output_base.with_suffix('.parquet')
        df_merged.to_parquet(parquet_path, engine='pyarrow', compression='snappy')
        logger.info(f"  ✓ Saved deflated Parquet: {parquet_path.name}")

        # Save as CSV (secondary)
        csv_path = output_base.with_suffix('.csv')
        df_merged.to_csv(csv_path, index=False)
        logger.info(f"  ✓ Saved deflated CSV: {csv_path.name}")

        return df_merged

    def _find_datetime_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the primary datetime column."""
        datetime_candidates = ['Date', 'date', 'Datetime', 'datetime', 'Time', 'time', 'Tarih']
        for col in datetime_candidates:
            if col in df.columns:
                return col

        # Check for datetime dtype
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return col

        return None

    def _find_price_columns(self, df: pd.DataFrame) -> list:
        """Find columns containing price data."""
        price_keywords = ['price', 'fiyat', 'mcp', 'ptf', 'smf', 'wap']
        price_cols = []

        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in price_keywords):
                # Exclude columns that are already deflated
                if 'real' not in col_lower and 'deflated' not in col_lower:
                    # Check if column is numeric
                    if pd.api.types.is_numeric_dtype(df[col]):
                        price_cols.append(col)

        return price_cols

    def deflate_all_price_datasets(
        self,
        start_date: str = "2020-01-01",
        end_date: str = "2025-10-31",
        layer: Literal['bronze', 'silver'] = 'silver'
    ):
        """
        Deflate all EPİAŞ price datasets.

        Args:
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
            layer: Input layer ('bronze' or 'silver')
        """
        logger.info("=" * 70)
        logger.info("EPİAŞ PRICE DEFLATION - Starting")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Deflator method: {self.deflator_method}")
        logger.info("=" * 70)

        results = {}
        failed = []

        for dataset_name in self.PRICE_DATASETS:
            try:
                df = self.deflate_dataset(dataset_name, start_date, end_date, layer=layer)
                results[dataset_name] = df
                logger.info(f"✓ {dataset_name} deflated successfully\n")
            except FileNotFoundError as e:
                logger.warning(f"⚠ Skipping {dataset_name}: {e}\n")
                failed.append(dataset_name)
            except Exception as e:
                logger.error(f"✗ Failed to deflate {dataset_name}: {e}\n")
                failed.append(dataset_name)

        # Summary
        logger.info("=" * 70)
        logger.info("EPİAŞ PRICE DEFLATION - Summary")
        logger.info("=" * 70)
        logger.info(f"Successfully deflated: {len(results)}/{len(self.PRICE_DATASETS)}")
        if failed:
            logger.warning(f"Failed/Skipped: {', '.join(failed)}")
        logger.info("=" * 70)

        return results


# ═══════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Example: Deflate all EPİAŞ price datasets
    """

    # Option 1: Use baseline deflator (Factor Analysis)
    deflator = PriceDeflator(deflator_method='baseline')

    # Deflate all price datasets
    results = deflator.deflate_all_price_datasets(
        start_date='2020-01-01',
        end_date='2025-10-31',
        layer='silver'  # Use normalized silver layer data
    )

    # Option 2: Use DFM deflator (more sophisticated)
    # deflator_dfm = PriceDeflator(deflator_method='dfm')
    # results_dfm = deflator_dfm.deflate_all_price_datasets(
    #     start_date='2020-01-01',
    #     end_date='2024-12-31',
    #     layer='silver'
    # )

    print("\n✓ Deflation complete!")
    print("Real-value price data saved to: data/gold/epias/")
    print("\nNext steps:")
    print("  1. Use deflated prices (*_real columns) for modeling")
    print("  2. Original nominal prices preserved for reference")
    print("  3. Deflator base: 2022-01 = 100")
