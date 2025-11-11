"""
Master Feature Merger for ForeWatt
===================================
Merges all Gold layer features into unified ML-ready dataset.

Gold Layers Merged:
-------------------
1. Target variable: consumption_actual (Silver EPİAŞ)
2. Demand weather features: 60+ engineered features (Gold)
3. Deflated prices: Real TL/MWh (Gold)
4. External features: FX/Gold with momentum/volatility (Gold - optional)
5. Calendar features: Holidays + temporal (Gold - optional)
6. Lag features: Target + temp + price lags (Gold)
7. Rolling features: 24h/168h windows (Gold)

Versioning Strategy:
--------------------
Files named: master_v{version}_{date}_{hash}.parquet
- version: Manual version number (e.g., v1, v2)
- date: Creation date (YYYY-MM-DD)
- hash: First 8 chars of MD5 hash of sorted column names

Metadata JSON includes:
- Version info
- Feature list
- Data date range
- Missing value statistics
- Creation timestamp

Author: ForeWatt Team
Date: November 2025
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import hashlib
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MasterFeatureMerger:
    """Merge all Gold layer features into unified ML-ready dataset."""

    def __init__(self, data_dir: str = './data', version: str = 'v1'):
        """
        Initialize master feature merger.

        Args:
            data_dir: Root data directory
            version: Version string (e.g., 'v1', 'v2')
        """
        self.data_dir = Path(data_dir)
        self.silver_dir = self.data_dir / 'silver'
        self.gold_dir = self.data_dir / 'gold'
        self.output_dir = self.gold_dir / 'master'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.version = version

    def _standardize_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize timestamp column name and format."""
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
        return df

    def load_target(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load target variable (consumption) from Silver EPİAŞ."""
        logger.info("Loading target variable (consumption)...")
        file_path = self.silver_dir / 'epias' / f'consumption_actual_normalized_{start_date}_{end_date}.parquet'

        if not file_path.exists():
            raise FileNotFoundError(f"Target data not found: {file_path}")

        df = pd.read_parquet(file_path)
        df = self._standardize_timestamp(df)

        # Rename to standard target name
        if 'consumption' not in df.columns and 'value' in df.columns:
            df = df.rename(columns={'value': 'consumption'})

        logger.info(f"✓ Loaded target: {len(df)} records")
        return df[['timestamp', 'consumption']]

    def load_weather_features(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load demand weather features from Gold."""
        logger.info("Loading weather demand features...")
        file_path = self.gold_dir / 'demand_features' / f'demand_features_{start_date}_{end_date}.parquet'

        if not file_path.exists():
            raise FileNotFoundError(f"Weather features not found: {file_path}")

        df = pd.read_parquet(file_path)
        df = self._standardize_timestamp(df)

        logger.info(f"✓ Loaded weather features: {len(df.columns)-1} columns")
        return df

    def load_deflated_prices(self, start_date: str, end_date: str) -> pd.DataFrame | None:
        """Load deflated electricity prices from Gold."""
        logger.info("Loading deflated prices...")
        file_path = self.gold_dir / 'epias' / f'price_ptf_deflated_{start_date}_{end_date}.parquet'

        if not file_path.exists():
            logger.warning(f"Deflated prices not found: {file_path}")
            logger.warning("Skipping deflated prices...")
            return None

        df = pd.read_parquet(file_path)
        df = self._standardize_timestamp(df)

        logger.info(f"✓ Loaded deflated prices: {len(df.columns)-1} columns")
        return df

    def load_external_features(self, start_date: str, end_date: str) -> pd.DataFrame | None:
        """Load external features (FX/Gold) from Gold."""
        logger.info("Loading external features (FX/Gold)...")
        file_path = self.gold_dir / 'external_features' / f'external_features_hourly_{start_date}_{end_date}.parquet'

        if not file_path.exists():
            logger.warning(f"External features not found: {file_path}")
            logger.warning("Skipping external features (will be added in future versions)...")
            return None

        df = pd.read_parquet(file_path)
        df = self._standardize_timestamp(df)

        logger.info(f"✓ Loaded external features: {len(df.columns)-1} columns")
        return df

    def load_calendar_features(self, start_date: str, end_date: str) -> pd.DataFrame | None:
        """Load calendar features from Gold."""
        logger.info("Loading calendar features...")
        file_path = self.gold_dir / 'calendar_features' / f'calendar_features_{start_date}_{end_date}.parquet'

        if not file_path.exists():
            logger.warning(f"Calendar features not found: {file_path}")
            logger.warning("Skipping calendar features (will be added in future versions)...")
            return None

        df = pd.read_parquet(file_path)
        df = self._standardize_timestamp(df)

        logger.info(f"✓ Loaded calendar features: {len(df.columns)-1} columns")
        return df

    def load_lag_features(self, start_date: str, end_date: str) -> pd.DataFrame | None:
        """Load lag features from Gold."""
        logger.info("Loading lag features...")
        file_path = self.gold_dir / 'lag_features' / f'lag_features_{start_date}_{end_date}.parquet'

        if not file_path.exists():
            logger.warning(f"Lag features not found: {file_path}")
            logger.warning("Run: python src/features/lag_features.py")
            return None

        df = pd.read_parquet(file_path)
        df = self._standardize_timestamp(df)

        logger.info(f"✓ Loaded lag features: {len(df.columns)-1} columns")
        return df

    def load_rolling_features(self, start_date: str, end_date: str) -> pd.DataFrame | None:
        """Load rolling features from Gold."""
        logger.info("Loading rolling features...")
        file_path = self.gold_dir / 'rolling_features' / f'rolling_features_{start_date}_{end_date}.parquet'

        if not file_path.exists():
            logger.warning(f"Rolling features not found: {file_path}")
            logger.warning("Run: python src/features/rolling_features.py")
            return None

        df = pd.read_parquet(file_path)
        df = self._standardize_timestamp(df)

        logger.info(f"✓ Loaded rolling features: {len(df.columns)-1} columns")
        return df

    def merge_all_features(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Merge all Gold layer features.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Unified ML-ready DataFrame
        """
        logger.info("="*60)
        logger.info(f"MERGING FEATURES: {start_date} to {end_date}")
        logger.info("="*60)

        # Load all feature sets
        target_df = self.load_target(start_date, end_date)
        weather_df = self.load_weather_features(start_date, end_date)
        prices_df = self.load_deflated_prices(start_date, end_date)
        external_df = self.load_external_features(start_date, end_date)
        calendar_df = self.load_calendar_features(start_date, end_date)
        lag_df = self.load_lag_features(start_date, end_date)
        rolling_df = self.load_rolling_features(start_date, end_date)

        # Start with target
        logger.info("\nMerging all features on timestamp...")
        master = target_df.copy()

        # Merge weather features
        master = master.merge(weather_df, on='timestamp', how='inner')
        logger.info(f"  After weather: {master.shape}")

        # Merge prices (if available)
        if prices_df is not None:
            master = master.merge(prices_df, on='timestamp', how='left')
            logger.info(f"  After prices: {master.shape}")

        # Merge external features (if available)
        if external_df is not None:
            master = master.merge(external_df, on='timestamp', how='left')
            logger.info(f"  After external: {master.shape}")

        # Merge calendar features (if available)
        if calendar_df is not None:
            master = master.merge(calendar_df, on='timestamp', how='left')
            logger.info(f"  After calendar: {master.shape}")

        # Merge lag features (if available)
        if lag_df is not None:
            # Remove duplicate base columns (consumption, temperature, price_ptf)
            lag_cols = [c for c in lag_df.columns if c == 'timestamp' or '_lag_' in c]
            master = master.merge(lag_df[lag_cols], on='timestamp', how='left')
            logger.info(f"  After lags: {master.shape}")

        # Merge rolling features (if available)
        if rolling_df is not None:
            # Remove duplicate base columns
            rolling_cols = [c for c in rolling_df.columns if c == 'timestamp' or '_rolling_' in c or '_range_' in c or '_cv_' in c]
            master = master.merge(rolling_df[rolling_cols], on='timestamp', how='left')
            logger.info(f"  After rolling: {master.shape}")

        logger.info(f"\n✓ Final merged dataset: {master.shape}")
        return master

    def compute_feature_hash(self, df: pd.DataFrame) -> str:
        """Compute hash of feature column names (for versioning)."""
        feature_cols = sorted([c for c in df.columns if c != 'timestamp'])
        hash_input = '|'.join(feature_cols)
        hash_full = hashlib.md5(hash_input.encode()).hexdigest()
        return hash_full[:8]  # First 8 characters

    def save_master_dataset(self, df: pd.DataFrame, start_date: str, end_date: str):
        """Save master dataset with versioning and metadata."""
        # Generate filename
        today = datetime.now().strftime('%Y-%m-%d')
        feature_hash = self.compute_feature_hash(df)
        base_name = f'master_{self.version}_{today}_{feature_hash}'

        # Save Parquet
        parquet_path = self.output_dir / f'{base_name}.parquet'
        df.to_parquet(parquet_path, index=False, engine='pyarrow')
        logger.info(f"\n✓ Saved Parquet: {parquet_path}")

        # Save CSV (secondary format)
        csv_path = self.output_dir / f'{base_name}.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"✓ Saved CSV: {csv_path}")

        # Generate metadata
        metadata = {
            'version': self.version,
            'creation_date': today,
            'feature_hash': feature_hash,
            'data_date_range': {
                'start': start_date,
                'end': end_date,
                'first_timestamp': str(df['timestamp'].min()),
                'last_timestamp': str(df['timestamp'].max())
            },
            'shape': {
                'rows': len(df),
                'columns': len(df.columns)
            },
            'features': {
                'total': len(df.columns) - 1,  # Exclude timestamp
                'list': sorted([c for c in df.columns if c != 'timestamp'])
            },
            'missing_values': {
                col: int(df[col].isna().sum())
                for col in df.columns if df[col].isna().sum() > 0
            },
            'missing_value_percentage': {
                col: round(df[col].isna().sum() / len(df) * 100, 2)
                for col in df.columns if df[col].isna().sum() > 0
            },
            'target_column': 'consumption',
            'timestamp_column': 'timestamp'
        }

        # Save metadata JSON
        metadata_path = self.output_dir / f'{base_name}_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"✓ Saved metadata: {metadata_path}")

        # Print summary
        logger.info("\n" + "="*60)
        logger.info("MASTER DATASET SUMMARY")
        logger.info("="*60)
        logger.info(f"Version: {self.version}")
        logger.info(f"Feature hash: {feature_hash}")
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        logger.info(f"Total features: {len(df.columns) - 1}")
        logger.info(f"Missing values: {df.isna().sum().sum()} ({df.isna().sum().sum() / df.size * 100:.2f}%)")
        logger.info("="*60)

    def run_pipeline(self, start_date: str = '2020-01-01', end_date: str = '2024-12-31'):
        """Run complete master merge pipeline."""
        logger.info("="*60)
        logger.info("MASTER FEATURE MERGE PIPELINE")
        logger.info("="*60)

        # Merge all features
        master_df = self.merge_all_features(start_date, end_date)

        # Save with versioning
        self.save_master_dataset(master_df, start_date, end_date)

        logger.info("\n" + "="*60)
        logger.info("MASTER MERGE PIPELINE COMPLETE")
        logger.info("="*60)

        return master_df


def main():
    """Main execution."""
    merger = MasterFeatureMerger(data_dir='./data', version='v1')
    master_df = merger.run_pipeline(
        start_date='2020-01-01',
        end_date='2024-12-31'
    )

    print("\nMaster dataset preview:")
    print(master_df.head())
    print("\nMaster dataset info:")
    print(master_df.info())


if __name__ == '__main__':
    main()
