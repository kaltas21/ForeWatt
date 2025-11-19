"""
External Features Fetcher for ForeWatt
=======================================
Fetches FX rates and gold prices to capture external shocks affecting electricity prices.

Purpose:
--------
Preserve import/FX shock signals as explicit features for forecasting models.
These are NOT used in deflation - they're kept as predictive signals.

Why these features matter:
- Turkey imports ~70% of energy (natural gas, oil)
- Electricity prices correlate with FX rates (import cost pass-through)
- Gold reflects capital flight and inflation expectations

Methodology:
------------
1. Fetch daily FX/Gold data from EVDS (Turkish Central Bank - official source)
2. Convert daily → hourly via forward fill (all 24 hours get same daily value)
3. Compute derived features (FX basket, momentum, volatility)
4. Save to gold layer for feature engineering

Fetched Series:
---------------
- USD/TRY: US Dollar exchange rate (primary FX anchor)
- EUR/TRY: Euro exchange rate (Turkey's main trade partner, ~40% of trade)
- XAU/TRY: Gold price in Turkish Lira (capital flight indicator)

Derived Features:
-----------------
- FX_basket: Weighted average (0.5 × USD/TRY + 0.5 × EUR/TRY)
- USD_TRY_mom: 7-day momentum (rate of change)
- EUR_TRY_mom: 7-day momentum
- XAU_TRY_mom: 7-day momentum
- FX_volatility: 30-day rolling standard deviation

Usage:
------
    python src/data/external_features_fetcher.py

Or import as module:
    from src.data.external_features_fetcher import fetch_external_features

    df = fetch_external_features(
        start_date='2020-01-01',
        end_date='2025-10-31'
    )

Requirements:
-------------
- EVDS_API_KEY in .env file
- evdspy library (pip install evdspy)
- Register at: https://evds2.tcmb.gov.tr/

Author: ForeWatt Team
Date: November 2025
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from evds import evdsAPI
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

EVDS_API_KEY = os.getenv("EVDS_API_KEY")

# EVDS Series Codes (Turkish Central Bank)
# Reference: https://evds2.tcmb.gov.tr/index.php?/evds/serieMarket
FX_SERIES = {
    "USD_TRY": "TP.DK.USD.A.YTL",  # US Dollar buying rate
    "EUR_TRY": "TP.DK.EUR.A.YTL",  # Euro buying rate
    "GBP_TRY": "TP.DK.GBP.A.YTL",  # Pound Sterling (optional)
    "XAU_TRY": "TP.DK.XAU.A.YTL",  # Gold (USD/oz × USD/TRY, approx)
}

# Alternative series if above don't work
FX_SERIES_ALT = {
    "USD_TRY": "TP.DK.USD.A",
    "EUR_TRY": "TP.DK.EUR.A",
    "XAU_TRY": "TP.DK.XAU.A",
}


class ExternalFeaturesFetcher:
    """
    Fetches and processes external FX/Gold features from EVDS.
    """

    def __init__(self, api_key: str = None):
        """
        Initialize fetcher with EVDS API key.

        Args:
            api_key: EVDS API key (defaults to environment variable)
        """
        self.api_key = api_key or EVDS_API_KEY

        if not self.api_key:
            raise ValueError(
                "EVDS_API_KEY not found. Please set it in .env file.\n"
                "Get your key from: https://evds2.tcmb.gov.tr/"
            )

        self.evds = evdsAPI(self.api_key)
        logger.info("✓ Initialized EVDS API client")

    def fetch_daily_fx(
        self,
        start_date: str = "2020-01-01",
        end_date: str = "2025-10-31",
        include_derived: bool = True
    ) -> pd.DataFrame:
        """
        Fetch daily FX and gold data from EVDS.

        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            include_derived: Compute derived features (basket, momentum, volatility)

        Returns:
            DataFrame with daily FX/Gold rates
        """
        # Convert YYYY-MM-DD to DD-MM-YYYY (EVDS format)
        start_evds = datetime.strptime(start_date, "%Y-%m-%d").strftime("%d-%m-%Y")
        end_evds = datetime.strptime(end_date, "%Y-%m-%d").strftime("%d-%m-%Y")

        logger.info(f"Fetching external features from EVDS ({start_date} to {end_date})...")

        # Try primary series codes
        series_to_fetch = list(FX_SERIES.values())

        try:
            df = self.evds.get_data(
                series_to_fetch,
                startdate=start_evds,
                enddate=end_evds,
                frequency="1"  # Daily
            )

            # Rename columns to readable names
            df.columns = ["Date"] + list(FX_SERIES.keys())

        except Exception as e:
            logger.warning(f"Primary series codes failed: {e}")
            logger.info("Trying alternative series codes...")

            # Try alternative series codes
            series_to_fetch = list(FX_SERIES_ALT.values())

            try:
                df = self.evds.get_data(
                    series_to_fetch,
                    startdate=start_evds,
                    enddate=end_evds,
                    frequency="1"
                )

                df.columns = ["Date"] + list(FX_SERIES_ALT.keys())

            except Exception as e2:
                logger.error(f"Alternative series codes also failed: {e2}")
                raise RuntimeError(
                    "Could not fetch FX data from EVDS. Check series codes.\n"
                    "Visit https://evds2.tcmb.gov.tr/ to find correct codes."
                )

        # Parse date column
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

        # Sort by date
        df = df.sort_values('Date').reset_index(drop=True)

        # Convert to numeric (handle any string values)
        for col in df.columns:
            if col != 'Date':
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Basic validation
        logger.info(f"  ✓ Fetched {len(df)} daily records")
        logger.info(f"    Date range: {df['Date'].min()} to {df['Date'].max()}")

        # Check for missing values
        missing = df.isnull().sum()
        if missing.any():
            logger.warning(f"  ⚠ Missing values detected:\n{missing[missing > 0]}")
            logger.info("  Forward-filling missing values...")
            df = df.ffill().bfill()

        # Log sample values
        for col in ['USD_TRY', 'EUR_TRY', 'XAU_TRY']:
            if col in df.columns:
                mean_val = df[col].mean()
                logger.info(f"    {col}: mean={mean_val:.2f}, min={df[col].min():.2f}, max={df[col].max():.2f}")

        # Add derived features
        if include_derived:
            df = self._add_derived_features(df)

        return df

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features from raw FX/Gold data.

        Features:
        - FX_basket: Weighted average (0.5 × USD + 0.5 × EUR)
        - *_mom7d: 7-day momentum (rate of change)
        - *_mom30d: 30-day momentum
        - FX_volatility: 30-day rolling std of FX basket

        Args:
            df: DataFrame with USD_TRY, EUR_TRY, XAU_TRY columns

        Returns:
            DataFrame with added derived features
        """
        logger.info("  Computing derived features...")

        # FX Basket (weighted average of USD and EUR)
        if 'USD_TRY' in df.columns and 'EUR_TRY' in df.columns:
            df['FX_basket'] = 0.5 * df['USD_TRY'] + 0.5 * df['EUR_TRY']
            logger.info("    ✓ FX_basket (0.5×USD + 0.5×EUR)")

        # Momentum features (7-day rate of change)
        for col in ['USD_TRY', 'EUR_TRY', 'XAU_TRY', 'FX_basket']:
            if col in df.columns:
                # 7-day momentum (%)
                df[f'{col}_mom7d'] = df[col].pct_change(periods=7) * 100

                # 30-day momentum (%)
                df[f'{col}_mom30d'] = df[col].pct_change(periods=30) * 100

        # Volatility (30-day rolling std)
        if 'FX_basket' in df.columns:
            df['FX_volatility'] = df['FX_basket'].rolling(window=30).std()
            logger.info("    ✓ FX_volatility (30-day rolling std)")

        # Fill NaN created by rolling calculations
        df = df.ffill().bfill()

        derived_count = len([c for c in df.columns if '_mom' in c or 'volatility' in c])
        logger.info(f"    ✓ Added {derived_count} derived features")

        return df

    def convert_to_hourly(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert daily FX data to hourly frequency via forward fill.

        Methodology: All 24 hours in a day get the same daily value.
        This is appropriate because:
        - FX rates are daily snapshots (TCMB publishes once per day)
        - Hourly granularity allows joining with hourly electricity data
        - Forward fill preserves intraday stability (realistic assumption)

        Args:
            daily_df: Daily FX data with 'Date' column

        Returns:
            Hourly FX data with 'datetime' column
        """
        logger.info("  Converting daily → hourly (forward fill)...")

        # Create hourly datetime range
        start_date = daily_df['Date'].min()
        end_date = daily_df['Date'].max()

        hourly_range = pd.date_range(
            start=start_date,
            end=end_date + pd.Timedelta(days=1) - pd.Timedelta(hours=1),
            freq='H'
        )

        # Create hourly dataframe with date component for merging
        hourly_df = pd.DataFrame({'datetime': hourly_range})
        hourly_df['date'] = hourly_df['datetime'].dt.date

        # Merge with daily data (forward fill within each day)
        daily_df['date'] = daily_df['Date'].dt.date

        hourly_df = hourly_df.merge(
            daily_df.drop(columns=['Date']),
            on='date',
            how='left'
        )

        # Drop temporary date column
        hourly_df = hourly_df.drop(columns=['date'])

        # Forward fill any gaps (at boundaries)
        hourly_df = hourly_df.ffill().bfill()

        logger.info(f"    ✓ Created {len(hourly_df)} hourly records")
        logger.info(f"      Range: {hourly_df['datetime'].min()} to {hourly_df['datetime'].max()}")

        return hourly_df

    def save_features(
        self,
        daily_df: pd.DataFrame,
        hourly_df: pd.DataFrame,
        start_date: str,
        end_date: str
    ):
        """
        Save FX features to gold layer in dual format.

        Output structure:
        - data/gold/external/fx_features_daily_YYYY-MM-DD_YYYY-MM-DD.{parquet,csv}
        - data/gold/external/fx_features_hourly_YYYY-MM-DD_YYYY-MM-DD.{parquet,csv}

        Args:
            daily_df: Daily FX features
            hourly_df: Hourly FX features
            start_date: Start date for filename
            end_date: End date for filename
        """
        output_dir = Path("data/gold/external")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save daily features
        daily_base = output_dir / f"fx_features_daily_{start_date}_{end_date}"

        # Parquet (primary)
        daily_parquet = daily_base.with_suffix('.parquet')
        daily_df.to_parquet(daily_parquet, engine='pyarrow', compression='snappy')
        logger.info(f"  ✓ Saved daily Parquet: {daily_parquet.name}")

        # CSV (secondary)
        daily_csv = daily_base.with_suffix('.csv')
        daily_df.to_csv(daily_csv, index=False)
        logger.info(f"  ✓ Saved daily CSV: {daily_csv.name}")

        # Save hourly features
        hourly_base = output_dir / f"fx_features_hourly_{start_date}_{end_date}"

        # Parquet (primary)
        hourly_parquet = hourly_base.with_suffix('.parquet')
        hourly_df.to_parquet(hourly_parquet, engine='pyarrow', compression='snappy')
        logger.info(f"  ✓ Saved hourly Parquet: {hourly_parquet.name}")

        # CSV (secondary, but can be large)
        hourly_csv = hourly_base.with_suffix('.csv')
        hourly_df.to_csv(hourly_csv, index=False)
        logger.info(f"  ✓ Saved hourly CSV: {hourly_csv.name}")

        # Summary
        logger.info(f"\n{'='*70}")
        logger.info("External Features Summary")
        logger.info(f"{'='*70}")
        logger.info(f"Daily records:  {len(daily_df):,}")
        logger.info(f"Hourly records: {len(hourly_df):,}")
        logger.info(f"Features:       {len(daily_df.columns)} columns")
        logger.info(f"Date range:     {start_date} to {end_date}")
        logger.info(f"Output:         {output_dir}")
        logger.info(f"{'='*70}\n")


# ═══════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════

def fetch_external_features(
    start_date: str = "2020-01-01",
    end_date: str = "2025-10-31",
    include_derived: bool = True,
    save: bool = True
) -> tuple:
    """
    Fetch external FX/Gold features (convenience wrapper).

    Args:
        start_date: Start date 'YYYY-MM-DD'
        end_date: End date 'YYYY-MM-DD'
        include_derived: Include momentum and volatility features
        save: Save to gold layer

    Returns:
        (daily_df, hourly_df): Tuple of daily and hourly DataFrames
    """
    fetcher = ExternalFeaturesFetcher()

    # Fetch daily data
    daily_df = fetcher.fetch_daily_fx(
        start_date=start_date,
        end_date=end_date,
        include_derived=include_derived
    )

    # Convert to hourly
    hourly_df = fetcher.convert_to_hourly(daily_df)

    # Save
    if save:
        fetcher.save_features(daily_df, hourly_df, start_date, end_date)

    return daily_df, hourly_df


def load_external_features(
    start_date: str = "2020-01-01",
    end_date: str = "2025-10-31",
    frequency: str = "hourly"
) -> pd.DataFrame:
    """
    Load previously fetched external features.

    Args:
        start_date: Start date 'YYYY-MM-DD'
        end_date: End date 'YYYY-MM-DD'
        frequency: 'daily' or 'hourly'

    Returns:
        DataFrame with FX features
    """
    gold_dir = Path("data/gold/external")

    filename = f"fx_features_{frequency}_{start_date}_{end_date}.parquet"
    filepath = gold_dir / filename

    if not filepath.exists():
        # Try CSV fallback
        filepath = filepath.with_suffix('.csv')
        if not filepath.exists():
            raise FileNotFoundError(
                f"External features not found: {filename}\n"
                f"Run: python src/data/external_features_fetcher.py"
            )
        return pd.read_csv(filepath)

    return pd.read_parquet(filepath)


# ═══════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Fetch external FX/Gold features for full training period.
    """

    # Aligned with EPİAŞ and EVDS pipeline
    START_DATE = "2020-01-01"
    END_DATE = "2025-10-31"

    logger.info("="*70)
    logger.info("EXTERNAL FEATURES FETCHER - Starting")
    logger.info(f"Period: {START_DATE} to {END_DATE}")
    logger.info("="*70 + "\n")

    try:
        # Fetch and process features
        daily_df, hourly_df = fetch_external_features(
            start_date=START_DATE,
            end_date=END_DATE,
            include_derived=True,
            save=True
        )

        # Display sample
        logger.info("\nSample of hourly features (first 24 hours):")
        print(hourly_df.head(24)[['datetime', 'USD_TRY', 'EUR_TRY', 'FX_basket', 'USD_TRY_mom7d']])

        logger.info("\n✓ External features pipeline complete!")
        logger.info("\nNext steps:")
        logger.info("  1. Use hourly features in feature engineering")
        logger.info("  2. Add as exogenous covariates to N-HiTS/CatBoost")
        logger.info("  3. Validate correlation with electricity prices")

    except Exception as e:
        logger.error(f"\n✗ Failed to fetch external features: {e}")
        logger.error("\nTroubleshooting:")
        logger.error("  1. Check EVDS_API_KEY in .env file")
        logger.error("  2. Verify API key is valid: https://evds2.tcmb.gov.tr/")
        logger.error("  3. Check internet connection")
        raise
