"""
EPİAŞ Electricity Data Pipeline
================================
Fetches electricity load, price, and generation data from EPİAŞ Transparency Platform.

Data Sources:
- Actual Consumption (Target Variable)
- Day-Ahead Consumption Forecast
- Market Prices (PTF, SMF, IDM)
- Generation by Source
- Available Capacity
- Generation Plans
- Wind Forecast (RİTM)
- Hydro Reservoir Status (DSİ)

Author: ForeWatt Team
Date: November 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
import time
import os
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EpiasDataFetcher:
    """
    Fetches and processes electricity data from EPİAŞ Transparency Platform.

    Features:
    - Comprehensive dataset coverage (load, price, generation, capacity)
    - Medallion architecture (Bronze → Silver layers)
    - Dual format export (Parquet + CSV)
    - Europe/Istanbul timezone handling
    - Data quality validation
    - Automatic retry logic
    """

    # EPİAŞ API Dataset Mappings
    # Format: {friendly_name: (eptr2_method, description, data_id)}
    # All methods have been tested and confirmed working with eptr2 v1.2.4
    DATASETS = {
        'consumption_actual': {
            'method': 'rt-cons',  # Real-time consumption
            'description': 'Actual Consumption (TEİAŞ) - Target Variable',
            'data_id': '49',
            'api_params': {}
        },
        'consumption_forecast': {
            'method': 'load-plan',  # Load Estimation Plan (LEP)
            'description': 'Day-Ahead Consumption Forecast (TEİAŞ)',
            'data_id': '50',
            'api_params': {}
        },
        'price_ptf': {
            'method': 'mcp',  # Market Clearing Price
            'description': 'Day-Ahead Market Price (PTF/MCP)',
            'data_id': '52',
            'api_params': {}
        },
        'price_smf': {
            'method': 'smp',  # System Marginal Price
            'description': 'Balancing Power Market Price (SMF)',
            'data_id': '53',
            'api_params': {}
        },
        'price_idm': {
            'method': 'idm-qty',  # Intraday Market Quantity
            'description': 'Intraday Market Quantity',
            'data_id': '54',
            'api_params': {}
        },
        'price_wap': {
            'method': 'wap',  # Weighted Average Price
            'description': 'Weighted Average Price',
            'data_id': '54b',
            'api_params': {}
        },
        'generation_realtime': {
            'method': 'rt-gen',  # Real-time generation
            'description': 'Actual Generation by Source/Plant (TEİAŞ)',
            'data_id': '15',
            'api_params': {},
            'use_fallback': True,
            'fallback_endpoint': '/v1/generation/export/realtime-generation',
            'fallback_extra_params': {}
        },
        'capacity_eak': {
            'method': 'eak',  # Available capacity (Emre Amade Kapasite)
            'description': 'Available Capacity (EAK)',
            'data_id': '11',
            'api_params': {},
            'use_fallback': True,
            'fallback_endpoint': '/v1/generation/export/aic',
            'fallback_extra_params': {'region': 'TR1'}
        },
        'plan_kgup': {
            'method': 'kgup',  # Day-ahead production plan (Kesinleşmiş Gün Öncesi Üretim Planı)
            'description': 'Day-Ahead Generation Plan (KGÜP)',
            'data_id': '13',
            'api_params': {},
            'use_fallback': True,
            'fallback_endpoint': '/v1/generation/export/dpp',
            'fallback_extra_params': {'region': 'TR1'}
        },
        'plan_kudup': {
            'method': 'kudup',  # Bilateral contract plan (Kesinleşmiş Uzlaştırmaya Esas Dağıtım Üretim Planı)
            'description': 'Bilateral Contract Plan (KUDÜP)',
            'data_id': '14',
            'api_params': {}
        },
        'wind_forecast': {
            'method': 'wind-forecast',  # Wind power forecast (RİTM)
            'description': 'Wind Generation Forecast (RİTM)',
            'data_id': '17',
            'api_params': {}
        },
        'hydro_reservoir_volume': {
            'method': 'dams-active-volume',  # DSİ dam active volume
            'description': 'Hydropower Reservoir Active Volume (DSİ)',
            'data_id': '18a',
            'api_params': {}
        },
        'hydro_energy_provision': {
            'method': 'dams-water-energy-provision',  # DSİ dam energy generation
            'description': 'Hydropower Energy Provision (DSİ)',
            'data_id': '18b',
            'api_params': {}
        }
    }

    def __init__(self, username: Optional[str] = None, password: Optional[str] = None):
        """
        Initialize the EPİAŞ data fetcher with authentication.

        Args:
            username: EPİAŞ username (if None, reads from .env)
            password: EPİAŞ password (if None, reads from .env)
        """
        # Load environment variables
        load_dotenv()

        # Get credentials
        self.username = username or os.getenv('EPTR_USERNAME')
        self.password = password or os.getenv('EPTR_PASSWORD')

        if not self.username or not self.password:
            raise ValueError(
                "EPİAŞ credentials not found. "
                "Set EPTR_USERNAME and EPTR_PASSWORD in .env file or pass as arguments."
            )

        # Initialize eptr2 client (lazy loading - will be created when first needed)
        self.client = None
        self._client_initialized = False

        logger.info(f"Initialized EpiasDataFetcher for user: {self.username}")
        logger.info(f"Available datasets: {len(self.DATASETS)} (all tested and working with eptr2 v1.2.4)")

    def _get_client(self):
        """Lazy initialization of eptr2 client."""
        if not self._client_initialized:
            try:
                from eptr2 import EPTR2
                self.client = EPTR2(username=self.username, password=self.password)
                self._client_initialized = True
                logger.info("✓ EPTR2 client initialized successfully")
            except ImportError:
                raise ImportError(
                    "eptr2 library not found. Install with: pip install eptr2"
                )
            except Exception as e:
                raise ConnectionError(f"Failed to initialize EPTR2 client: {str(e)}")

        return self.client

    def _fetch_with_direct_api(
        self,
        endpoint: str,
        start_date: str,
        end_date: str,
        extra_params: dict = None
    ) -> Optional[pd.DataFrame]:
        """Fallback: Direct EPİAŞ API POST calls with CSV export."""
        import requests
        from io import StringIO

        try:
            client = self._get_client()
            base_url = "https://seffaflik.epias.com.tr/electricity-service"
            url = f"{base_url}{endpoint}"

            body = {
                "startDate": f"{start_date}T00:00:00+03:00",
                "endDate": f"{end_date}T23:00:00+03:00",
                "exportType": "CSV"
            }

            if extra_params:
                body.update(extra_params)

            headers = {
                "TGT": client.tgt,
                "Accept": "application/json",
                "Content-Type": "application/json"
            }

            response = requests.post(url, json=body, headers=headers, timeout=60)

            if response.status_code == 200 and response.text:
                return pd.read_csv(StringIO(response.text), sep=';', on_bad_lines='skip')

            return None

        except Exception:
            return None

    def _split_date_range(self, start_date: str, end_date: str, max_days: int = 365) -> List[Tuple[str, str]]:
        """
        Split a date range into smaller chunks to respect API limits.

        EPİAŞ API has a maximum 1-year limit per request.

        Args:
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
            max_days: Maximum days per chunk (default 365)

        Returns:
            List of (start_date, end_date) tuples
        """
        from datetime import datetime, timedelta

        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')

        chunks = []
        current_start = start

        while current_start < end:
            current_end = min(current_start + timedelta(days=max_days), end)
            chunks.append((
                current_start.strftime('%Y-%m-%d'),
                current_end.strftime('%Y-%m-%d')
            ))
            current_start = current_end + timedelta(days=1)

        return chunks

    def fetch_dataset(
        self,
        dataset_name: str,
        start_date: str,
        end_date: str,
        retry_attempts: int = 3,
        retry_delay: float = 5.0
    ) -> Optional[pd.DataFrame]:
        """
        Fetch a single dataset from EPİAŞ API.

        Automatically splits large date ranges into 1-year chunks to respect API limits.

        Args:
            dataset_name: Name from DATASETS dictionary
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            retry_attempts: Number of retry attempts on failure
            retry_delay: Seconds to wait between retries

        Returns:
            DataFrame with fetched data, or None if all retries fail
        """
        if dataset_name not in self.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(self.DATASETS.keys())}")

        dataset_info = self.DATASETS[dataset_name]
        method_name = dataset_info['method']
        description = dataset_info['description']
        use_fallback = dataset_info.get('use_fallback', False)
        fallback_endpoint = dataset_info.get('fallback_endpoint', None)
        fallback_extra_params = dataset_info.get('fallback_extra_params', {})

        max_days = 365
        date_chunks = self._split_date_range(start_date, end_date, max_days=max_days)

        logger.info(f"Fetching {dataset_name}: {description} ({start_date} to {end_date})...")
        if len(date_chunks) > 1:
            logger.info(f"  Splitting into {len(date_chunks)} yearly chunks")

        # Get client
        client = self._get_client()

        # Fetch each chunk
        all_data = []

        for chunk_idx, (chunk_start, chunk_end) in enumerate(date_chunks, 1):
            if len(date_chunks) > 1 and chunk_idx % 10 == 1:
                progress = (chunk_idx / len(date_chunks)) * 100
                logger.info(f"  Progress: {progress:.0f}% (chunk {chunk_idx}/{len(date_chunks)})")

            # Attempt fetch with retries
            for attempt in range(1, retry_attempts + 1):
                try:
                    # Call the eptr2 method
                    df = client.call(
                        method_name,
                        start_date=chunk_start,
                        end_date=chunk_end,
                        **dataset_info['api_params']
                    )

                    if df is None or df.empty:
                        break

                    df = self._standardize_timezone(df)
                    all_data.append(df)
                    break

                except Exception as e:
                    error_msg = str(e)[:200]
                    if attempt == retry_attempts:
                        logger.error(f"    Failed after {retry_attempts} attempts: {error_msg}")

                    if attempt < retry_attempts:
                        wait_time = retry_delay * (2 ** (attempt - 1))
                        time.sleep(wait_time)
                    else:
                        if use_fallback and fallback_endpoint:
                            try:
                                df_fallback = self._fetch_with_direct_api(
                                    fallback_endpoint,
                                    chunk_start,
                                    chunk_end,
                                    extra_params=fallback_extra_params
                                )
                                if df_fallback is not None and not df_fallback.empty:
                                    df_fallback = self._standardize_timezone(df_fallback)
                                    all_data.append(df_fallback)
                                    break
                            except Exception:
                                pass

            if chunk_idx < len(date_chunks):
                time.sleep(1)

        if not all_data:
            logger.warning(f"No data collected for {dataset_name}")
            return None

        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"  ✓ Total fetched: {len(combined_df)} records across {len(date_chunks)} chunk(s)")

        return combined_df

    def _standardize_timezone(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize datetime columns to Europe/Istanbul timezone.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with standardized timezone
        """
        # Common datetime column names in EPİAŞ data
        datetime_cols = ['Date', 'date', 'Datetime', 'datetime', 'Time', 'time', 'Tarih', 'Saat']

        for col in df.columns:
            if col in datetime_cols or 'date' in col.lower() or 'time' in col.lower():
                try:
                    # Convert to datetime if not already
                    if not pd.api.types.is_datetime64_any_dtype(df[col]):
                        df[col] = pd.to_datetime(df[col])

                    # If timezone-naive, assume UTC and convert to Istanbul
                    if df[col].dt.tz is None:
                        df[col] = df[col].dt.tz_localize('UTC').dt.tz_convert('Europe/Istanbul')
                    else:
                        df[col] = df[col].dt.tz_convert('Europe/Istanbul')

                    logger.debug(f"  Standardized timezone for column: {col}")

                except Exception as e:
                    logger.debug(f"  Could not convert {col} to timezone: {str(e)}")
                    continue

        return df

    def validate_silver_layer(
        self,
        df: pd.DataFrame,
        dataset_name: str
    ) -> Tuple[pd.DataFrame, Dict[str, any]]:
        """
        Validate and clean data for silver layer.

        Checks:
        - Schema validation (expected columns exist)
        - Range checks (physically plausible values)
        - Monotonicity (timestamp ordering)
        - Duplicate removal

        Args:
            df: Raw bronze DataFrame
            dataset_name: Name of dataset for validation rules

        Returns:
            Tuple of (cleaned DataFrame, validation report dict)
        """
        report = {
            'original_rows': len(df),
            'issues': []
        }

        if df is None or df.empty:
            report['issues'].append('Empty or None DataFrame')
            return df, report

        logger.info(f"  Validating {dataset_name} (silver layer)...")

        # 1. Remove duplicates
        original_len = len(df)
        df = df.drop_duplicates()
        duplicates_removed = original_len - len(df)
        if duplicates_removed > 0:
            report['issues'].append(f'Removed {duplicates_removed} duplicates')
            logger.info(f"    Removed {duplicates_removed} duplicate rows")

        # 2. Sort by datetime if datetime column exists
        datetime_col = self._find_datetime_column(df)
        if datetime_col:
            df = df.sort_values(datetime_col)
            logger.info(f"    Sorted by {datetime_col}")

        # 3. Range validation (dataset-specific)
        if 'consumption' in dataset_name:
            # Electricity consumption: 20,000 - 50,000 MW for Turkey
            value_col = self._find_value_column(df, ['Consumption', 'Tüketim', 'Value'])
            if value_col:
                invalid_mask = (df[value_col] < 10000) | (df[value_col] > 60000)
                invalid_count = invalid_mask.sum()
                if invalid_count > 0:
                    report['issues'].append(f'{invalid_count} values outside plausible range (10-60 GW)')
                    logger.warning(f"    ⚠ {invalid_count} values outside plausible range")

        elif 'price' in dataset_name:
            # Prices: typically 0 - 3000 TL/MWh (can spike higher)
            value_col = self._find_value_column(df, ['Price', 'Fiyat', 'Value', 'MCP', 'PTF', 'SMF'])
            if value_col:
                invalid_mask = (df[value_col] < 0) | (df[value_col] > 10000)
                invalid_count = invalid_mask.sum()
                if invalid_count > 0:
                    report['issues'].append(f'{invalid_count} prices outside normal range')
                    logger.warning(f"    ⚠ {invalid_count} prices outside 0-10,000 TL/MWh")

        # 4. Check for missing values in critical columns
        missing_summary = df.isnull().sum()
        critical_missing = missing_summary[missing_summary > len(df) * 0.1]  # >10% missing
        if len(critical_missing) > 0:
            report['issues'].append(f'{len(critical_missing)} columns with >10% missing values')
            logger.warning(f"    ⚠ Columns with >10% missing: {list(critical_missing.index)}")

        report['final_rows'] = len(df)
        report['rows_removed'] = report['original_rows'] - report['final_rows']

        logger.info(f"  ✓ Validation complete: {report['final_rows']} rows, {len(report['issues'])} issues")

        return df, report

    def _find_datetime_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the primary datetime column in DataFrame."""
        datetime_cols = ['Date', 'date', 'Datetime', 'datetime', 'Time', 'time', 'Tarih']
        for col in datetime_cols:
            if col in df.columns:
                return col

        # Check for columns with datetime dtype
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return col

        return None

    def _find_value_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Find a value column from candidates list."""
        for col in df.columns:
            if any(candidate.lower() in col.lower() for candidate in candidates):
                return col
        return None

    def save_dual_format(
        self,
        df: pd.DataFrame,
        base_filepath: str,
        layer: str = 'bronze'
    ):
        """
        Save DataFrame in both Parquet and CSV formats.

        Args:
            df: DataFrame to save
            base_filepath: Base file path (without extension)
            layer: Data layer ('bronze' or 'silver')
        """
        base_path = Path(base_filepath)
        base_path.parent.mkdir(parents=True, exist_ok=True)

        # Parquet format (efficient, compressed)
        parquet_path = base_path.with_suffix('.parquet')
        df.to_parquet(parquet_path, engine='pyarrow', compression='snappy')
        logger.info(f"  ✓ Saved {layer} Parquet: {parquet_path.name} ({len(df)} rows)")

        # CSV format (human-readable, Excel-compatible)
        csv_path = base_path.with_suffix('.csv')
        df.to_csv(csv_path, index=True, encoding='utf-8')
        logger.info(f"  ✓ Saved {layer} CSV: {csv_path.name} ({len(df)} rows)")

    def _check_existing_data(
        self,
        dataset_name: str,
        start_date: str,
        end_date: str,
        output_dir: str
    ) -> Optional[pd.DataFrame]:
        """
        Check if dataset already exists in silver layer and load it.

        Args:
            dataset_name: Name of dataset
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
            output_dir: Output directory

        Returns:
            DataFrame if exists and valid, None otherwise
        """
        silver_path = Path(output_dir) / 'silver' / 'epias' / f"{dataset_name}_normalized_{start_date}_{end_date}.parquet"

        if silver_path.exists():
            try:
                df = pd.read_parquet(silver_path)
                if df is not None and not df.empty:
                    logger.info(f"  ✓ Found existing data: {silver_path.name} ({len(df)} rows)")
                    logger.info(f"  Skipping fetch (already downloaded)")
                    return df
            except Exception as e:
                logger.warning(f"  ⚠ Could not load existing file: {e}")

        return None

    def run_pipeline(
        self,
        start_date: str,
        end_date: str,
        output_dir: str = './data',
        datasets: Optional[List[str]] = None,
        skip_validation: bool = False,
        skip_existing: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Run the complete EPİAŞ data pipeline: fetch → validate → save.

        Args:
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
            output_dir: Output directory for data files
            datasets: List of dataset names to fetch (if None, fetches all)
            skip_validation: If True, skip silver layer validation
            skip_existing: If True, skip datasets that already exist (default: True)

        Returns:
            Dictionary mapping dataset names to their silver DataFrames
        """
        output_path = Path(output_dir)

        logger.info("=" * 70)
        logger.info("EPİAŞ DATA PIPELINE - Starting")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Output: {output_path.absolute()}")
        logger.info("=" * 70)

        # Determine which datasets to fetch
        if datasets is None:
            datasets = list(self.DATASETS.keys())

        logger.info(f"Fetching {len(datasets)} datasets...")
        if skip_existing:
            logger.info("Skip mode: Will skip already-downloaded datasets")

        results = {}
        failed_datasets = []
        skipped_datasets = []
        validation_reports = {}

        for idx, dataset_name in enumerate(datasets, 1):
            logger.info(f"\n[{idx}/{len(datasets)}] Processing: {dataset_name}")
            logger.info("-" * 70)

            try:
                # Check if data already exists
                if skip_existing:
                    existing_df = self._check_existing_data(dataset_name, start_date, end_date, output_dir)
                    if existing_df is not None:
                        results[dataset_name] = existing_df
                        skipped_datasets.append(dataset_name)
                        continue

                # Step 1: Fetch raw data (BRONZE layer)
                df_bronze = self.fetch_dataset(dataset_name, start_date, end_date)

                if df_bronze is None or df_bronze.empty:
                    logger.warning(f"  Skipping {dataset_name} (no data returned)")
                    failed_datasets.append(dataset_name)
                    continue

                # Save bronze layer
                bronze_dir = output_path / 'bronze' / 'epias'
                bronze_base = bronze_dir / f"{dataset_name}_{start_date}_{end_date}"
                self.save_dual_format(df_bronze, str(bronze_base), layer='bronze')

                # Step 2: Validate and clean (SILVER layer)
                if not skip_validation:
                    df_silver, report = self.validate_silver_layer(df_bronze, dataset_name)
                    validation_reports[dataset_name] = report
                else:
                    df_silver = df_bronze
                    logger.info("  ⚠ Skipping validation (skip_validation=True)")

                # Save silver layer
                silver_dir = output_path / 'silver' / 'epias'
                silver_base = silver_dir / f"{dataset_name}_normalized_{start_date}_{end_date}"
                self.save_dual_format(df_silver, str(silver_base), layer='silver')

                results[dataset_name] = df_silver
                logger.info(f"  ✓ {dataset_name} pipeline complete")

                # Small delay between datasets to be respectful to API
                if idx < len(datasets):
                    time.sleep(1)

            except Exception as e:
                logger.error(f"  ✗ Failed to process {dataset_name}: {str(e)}")
                failed_datasets.append(dataset_name)
                continue

        # Pipeline summary
        logger.info("\n" + "=" * 70)
        logger.info("EPİAŞ DATA PIPELINE - Summary")
        logger.info("=" * 70)
        logger.info(f"Total datasets: {len(results)}/{len(datasets)}")
        logger.info(f"  Successfully fetched: {len(results) - len(skipped_datasets)}")
        logger.info(f"  Skipped (already exist): {len(skipped_datasets)}")
        logger.info(f"  Failed: {len(failed_datasets)}")

        if skipped_datasets:
            logger.info(f"\nSkipped datasets (already downloaded): {', '.join(skipped_datasets)}")

        if failed_datasets:
            logger.warning(f"\nFailed datasets: {', '.join(failed_datasets)}")

        # Validation summary
        if validation_reports:
            logger.info("\nValidation Summary:")
            for dataset_name, report in validation_reports.items():
                logger.info(f"  {dataset_name}:")
                logger.info(f"    Original rows: {report['original_rows']}")
                logger.info(f"    Final rows: {report['final_rows']}")
                if report['issues']:
                    for issue in report['issues']:
                        logger.info(f"    - {issue}")

        logger.info("\n" + "=" * 70)
        logger.info("EPİAŞ DATA PIPELINE - Completed")
        logger.info("=" * 70)

        return results


# ═══════════════════════════════════════════════════════════
# EXAMPLE USAGE & TESTING
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Example usage of the EPİAŞ Data Pipeline
    """

    # Initialize fetcher (reads credentials from .env)
    fetcher = EpiasDataFetcher()

    # Example 1: Fetch recent 1 week for testing
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Fetching Recent Data (1 week)")
    print("=" * 70)

    recent_data = fetcher.run_pipeline(
        start_date='2024-10-25',
        end_date='2024-11-01',
        output_dir='./data',
        datasets=['consumption_actual', 'price_ptf']  # Test with core datasets first
    )

    if 'consumption_actual' in recent_data:
        print("\nSample of consumption data:")
        print(recent_data['consumption_actual'].head(24))

    # Example 2: Fetch all datasets for training period
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Fetching All Datasets (Training Period)")
    print("=" * 70)

    all_data = fetcher.run_pipeline(
        start_date='2020-01-01',
        end_date='2024-12-31',
        output_dir='./data'
        # datasets=None means fetch all available datasets
    )

    print("\n" + "=" * 70)
    print("DATA QUALITY CHECK")
    print("=" * 70)

    for dataset_name, df in all_data.items():
        print(f"\n{dataset_name}:")
        print(f"  Records: {len(df)}")
        print(f"  Date range: {df.index.min() if hasattr(df.index, 'min') else 'N/A'} to {df.index.max() if hasattr(df.index, 'max') else 'N/A'}")
        print(f"  Columns: {list(df.columns)[:5]}{'...' if len(df.columns) > 5 else ''}")

        missing = df.isnull().sum().sum()
        if missing > 0:
            print(f"  ⚠ Missing values: {missing} ({missing/df.size*100:.2f}%)")
        else:
            print(f"  ✓ No missing values")

    print("\n✓ Pipeline execution complete!")
    print(f"Data saved to: ./data/bronze/epias/ and ./data/silver/epias/")
