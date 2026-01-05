"""
GCS Storage for ForeWatt Master Parquet
========================================
Handles loading and saving the master parquet file from/to Google Cloud Storage.
This enables persistent storage that survives Cloud Run container restarts.

Author: ForeWatt Team
Date: January 2026
"""

import pandas as pd
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# GCS Configuration
GCS_BUCKET = os.getenv('GCS_BUCKET', 'forewatt-data')
GCS_MASTER_PATH = 'master/master_v2_fundamental.parquet'

# Local paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_MASTER_PATH = PROJECT_ROOT / 'data' / 'gold' / 'master' / 'master_v2_fundamental.parquet'
LOCAL_CACHE_PATH = Path(tempfile.gettempdir()) / 'forewatt_master.parquet'


class GCSMasterStorage:
    """
    Manages the master parquet file in Google Cloud Storage.

    Features:
    - Lazy loading with local caching
    - Automatic upload after updates
    - Fallback to bundled parquet if GCS unavailable
    """

    def __init__(self):
        self._client = None
        self._bucket = None
        self._df_cache = None
        self._cache_timestamp = None

    def _get_client(self):
        """Lazy initialization of GCS client."""
        if self._client is None:
            try:
                from google.cloud import storage
                self._client = storage.Client()
                self._bucket = self._client.bucket(GCS_BUCKET)
                logger.info(f"GCS client initialized for bucket: {GCS_BUCKET}")
            except Exception as e:
                logger.warning(f"GCS client initialization failed: {e}")
                self._client = False
        return self._client if self._client else None

    def load_master_df(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Load the master parquet from GCS with local caching.

        Args:
            force_refresh: Force reload from GCS instead of cache

        Returns:
            Master DataFrame with all features
        """
        # Check cache first (valid for 5 minutes)
        if not force_refresh and self._df_cache is not None:
            if self._cache_timestamp:
                cache_age = (datetime.now() - self._cache_timestamp).total_seconds()
                if cache_age < 300:  # 5 minutes
                    logger.debug("Using in-memory cache")
                    return self._df_cache.copy()

        # Try to load from GCS
        client = self._get_client()
        if client:
            try:
                blob = self._bucket.blob(GCS_MASTER_PATH)
                if blob.exists():
                    # Download to temp file
                    LOCAL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
                    blob.download_to_filename(str(LOCAL_CACHE_PATH))
                    logger.info(f"Downloaded master parquet from GCS ({blob.size / 1024 / 1024:.1f} MB)")

                    # Load and cache
                    df = pd.read_parquet(LOCAL_CACHE_PATH)
                    self._df_cache = df
                    self._cache_timestamp = datetime.now()

                    # Process timestamp
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        if df['timestamp'].dt.tz is not None:
                            df['timestamp'] = df['timestamp'].dt.tz_localize(None)

                    logger.info(f"Loaded {len(df)} records from GCS master parquet")
                    return df
                else:
                    logger.warning("Master parquet not found in GCS")
            except Exception as e:
                logger.error(f"Failed to load from GCS: {e}")

        # Fallback to local bundled file
        if LOCAL_MASTER_PATH.exists():
            logger.info("Falling back to bundled master parquet")
            df = pd.read_parquet(LOCAL_MASTER_PATH)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                if df['timestamp'].dt.tz is not None:
                    df['timestamp'] = df['timestamp'].dt.tz_localize(None)

            self._df_cache = df
            self._cache_timestamp = datetime.now()
            return df

        logger.error("No master parquet available (GCS or local)")
        return pd.DataFrame()

    def save_master_df(self, df: pd.DataFrame) -> bool:
        """
        Save the master parquet to GCS.

        Args:
            df: DataFrame to save

        Returns:
            True if successful, False otherwise
        """
        client = self._get_client()
        if not client:
            logger.error("GCS client not available, cannot save")
            return False

        try:
            # Save to temp file first
            LOCAL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(LOCAL_CACHE_PATH, index=False, engine='pyarrow')

            # Upload to GCS
            blob = self._bucket.blob(GCS_MASTER_PATH)
            blob.upload_from_filename(str(LOCAL_CACHE_PATH))

            # Update cache
            self._df_cache = df
            self._cache_timestamp = datetime.now()

            logger.info(f"Saved master parquet to GCS ({len(df)} records, {LOCAL_CACHE_PATH.stat().st_size / 1024 / 1024:.1f} MB)")
            return True

        except Exception as e:
            logger.error(f"Failed to save to GCS: {e}")
            return False

    def get_last_timestamp(self) -> Optional[datetime]:
        """Get the last timestamp in the master data."""
        df = self.load_master_df()
        if df.empty or 'timestamp' not in df.columns:
            return None

        last_ts = pd.to_datetime(df['timestamp']).max()
        if pd.notna(last_ts):
            return last_ts
        return None

    def clear_cache(self):
        """Clear the in-memory cache."""
        self._df_cache = None
        self._cache_timestamp = None
        logger.info("Cache cleared")


# Singleton instance
_gcs_storage = None


def get_gcs_storage() -> GCSMasterStorage:
    """Get the singleton GCS storage instance."""
    global _gcs_storage
    if _gcs_storage is None:
        _gcs_storage = GCSMasterStorage()
    return _gcs_storage


def load_master_from_gcs(force_refresh: bool = False) -> pd.DataFrame:
    """Convenience function to load master parquet."""
    return get_gcs_storage().load_master_df(force_refresh)


def save_master_to_gcs(df: pd.DataFrame) -> bool:
    """Convenience function to save master parquet."""
    return get_gcs_storage().save_master_df(df)
