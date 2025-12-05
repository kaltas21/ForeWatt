"""
ForeWatt Data Loader
Utilities for loading and processing data from the gold layer.
"""
import pandas as pd
import streamlit as st
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np

from .config import MASTER_DATA, TARGET_VARIABLE, VALIDATION_CONFIG


@st.cache_data
def load_master_data(nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Load the master dataset from gold layer.

    Args:
        nrows: Optional number of rows to load (for testing)

    Returns:
        DataFrame with timezone-aware datetime index
    """
    try:
        # Load from CSV or parquet based on file extension
        if str(MASTER_DATA).endswith('.csv'):
            df = pd.read_csv(MASTER_DATA, nrows=nrows)
        else:
            df = pd.read_parquet(MASTER_DATA)
            if nrows:
                df = df.tail(nrows)

        # Ensure datetime index (prioritize 'timestamp' column, fallback to 'datetime')
        time_col = None
        if 'timestamp' in df.columns:
            time_col = 'timestamp'
        elif 'datetime' in df.columns:
            time_col = 'datetime'

        if time_col:
            # Convert to datetime and ensure timezone awareness
            df[time_col] = pd.to_datetime(df[time_col])
            if df[time_col].dt.tz is None:
                # If timezone-naive, localize to Europe/Istanbul
                df[time_col] = df[time_col].dt.tz_localize('Europe/Istanbul')
            df = df.set_index(time_col)
        else:
            # If no time column, try to convert existing index
            df.index = pd.to_datetime(df.index)
            if df.index.tz is None:
                df.index = df.index.tz_localize('Europe/Istanbul')

        df = df.sort_index()

        return df
    except Exception as e:
        st.error(f"Error loading master data: {e}")
        return pd.DataFrame()


@st.cache_data
def get_data_summary(df: pd.DataFrame) -> dict:
    """Get summary statistics for the dataset."""
    return {
        "total_rows": len(df),
        "date_range": (df.index.min(), df.index.max()),
        "num_features": len(df.columns),
        "missing_values": df.isnull().sum().sum(),
        "target_mean": df[TARGET_VARIABLE].mean() if TARGET_VARIABLE in df.columns else None,
        "target_std": df[TARGET_VARIABLE].std() if TARGET_VARIABLE in df.columns else None,
        "target_min": df[TARGET_VARIABLE].min() if TARGET_VARIABLE in df.columns else None,
        "target_max": df[TARGET_VARIABLE].max() if TARGET_VARIABLE in df.columns else None
    }


def split_train_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets based on config.

    Returns:
        Tuple of (train_df, test_df)
    """
    # Get timezone from dataframe index if available
    tz = df.index.tz if hasattr(df.index, 'tz') and df.index.tz is not None else 'Europe/Istanbul'

    # Convert config dates to timezone-aware timestamps
    train_start = pd.to_datetime(VALIDATION_CONFIG["train_start"]).tz_localize(tz)
    train_end = pd.to_datetime(VALIDATION_CONFIG["train_end"]).tz_localize(tz)
    test_start = pd.to_datetime(VALIDATION_CONFIG["test_start"]).tz_localize(tz)
    test_end = pd.to_datetime(VALIDATION_CONFIG["test_end"]).tz_localize(tz)

    train_df = df.loc[train_start:train_end]
    test_df = df.loc[test_start:test_end]

    return train_df, test_df


def get_recent_data(df: pd.DataFrame, hours: int = 168) -> pd.DataFrame:
    """
    Get the most recent N hours of data.

    Args:
        df: Full dataframe
        hours: Number of hours to retrieve (default: 168 = 1 week)

    Returns:
        DataFrame with recent data
    """
    return df.tail(hours)


def get_feature_groups(df: pd.DataFrame) -> dict:
    """
    Group features by category.

    Returns:
        Dictionary with feature group names as keys and column lists as values
    """
    from .config import FEATURE_GROUPS

    grouped = {}
    columns = set(df.columns)

    for group_name, patterns in FEATURE_GROUPS.items():
        group_cols = []
        for pattern in patterns:
            if '*' in pattern:
                prefix = pattern.split('*')[0]
                matching = [col for col in columns if col.startswith(prefix)]
                group_cols.extend(matching)
            else:
                if pattern in columns:
                    group_cols.append(pattern)
        grouped[group_name] = sorted(list(set(group_cols)))

    # Add ungrouped features
    all_grouped = set([col for cols in grouped.values() for col in cols])
    ungrouped = sorted(list(columns - all_grouped - {TARGET_VARIABLE}))
    if ungrouped:
        grouped["Other"] = ungrouped

    return grouped


def calculate_naive_forecast(df: pd.DataFrame, horizon: int = 24) -> np.ndarray:
    """
    Calculate naive seasonal forecast (last week same hour).
    Used as baseline for MASE calculation.

    Args:
        df: DataFrame with target variable
        horizon: Forecast horizon

    Returns:
        Array of naive forecasts
    """
    target = df[TARGET_VARIABLE].values
    # Seasonal naive: use value from 168 hours ago (same hour last week)
    naive = np.roll(target, 168)
    return naive


def get_date_range_options() -> List[str]:
    """Get common date range options for filtering."""
    return [
        "Last 24 hours",
        "Last 7 days",
        "Last 30 days",
        "Last 90 days",
        "Last year",
        "2024 (Test set)",
        "2023",
        "2022",
        "2021",
        "2020",
        "All data",
        "Custom range"
    ]


def apply_date_filter(df: pd.DataFrame, filter_option: str,
                      custom_start: Optional[pd.Timestamp] = None,
                      custom_end: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Apply date range filter to dataframe with timezone-aware handling.

    Args:
        df: Input dataframe with timezone-aware index
        filter_option: Selected filter option
        custom_start: Start date for custom range
        custom_end: End date for custom range

    Returns:
        Filtered dataframe
    """
    # Get timezone from dataframe index
    tz = df.index.tz if hasattr(df.index, 'tz') and df.index.tz is not None else None

    def make_tz_aware(date_str: str) -> pd.Timestamp:
        """Convert date string to timezone-aware timestamp."""
        ts = pd.to_datetime(date_str)
        if tz and ts.tz is None:
            ts = ts.tz_localize(tz)
        return ts

    if filter_option == "Last 24 hours":
        return df.last("24H")
    elif filter_option == "Last 7 days":
        return df.last("7D")
    elif filter_option == "Last 30 days":
        return df.last("30D")
    elif filter_option == "Last 90 days":
        return df.last("90D")
    elif filter_option == "Last year":
        return df.last("365D")
    elif filter_option == "2024 (Test set)":
        start = make_tz_aware("2024-01-01")
        end = make_tz_aware("2024-12-31 23:59:59")
        return df.loc[start:end]
    elif filter_option == "2023":
        start = make_tz_aware("2023-01-01")
        end = make_tz_aware("2023-12-31 23:59:59")
        return df.loc[start:end]
    elif filter_option == "2022":
        start = make_tz_aware("2022-01-01")
        end = make_tz_aware("2022-12-31 23:59:59")
        return df.loc[start:end]
    elif filter_option == "2021":
        start = make_tz_aware("2021-01-01")
        end = make_tz_aware("2021-12-31 23:59:59")
        return df.loc[start:end]
    elif filter_option == "2020":
        start = make_tz_aware("2020-01-01")
        end = make_tz_aware("2020-12-31 23:59:59")
        return df.loc[start:end]
    elif filter_option == "Custom range":
        if custom_start and custom_end:
            # Ensure custom dates are timezone-aware
            if tz and custom_start.tz is None:
                custom_start = custom_start.tz_localize(tz)
            if tz and custom_end.tz is None:
                custom_end = custom_end.tz_localize(tz)
            return df.loc[custom_start:custom_end]
        return df
    else:  # "All data"
        return df


def get_hourly_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate hourly consumption patterns.

    Returns:
        DataFrame with hour as index and statistics
    """
    if TARGET_VARIABLE not in df.columns:
        return pd.DataFrame()

    hourly = df.groupby(df.index.hour)[TARGET_VARIABLE].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max'),
        ('median', 'median')
    ])

    return hourly


def get_daily_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily consumption patterns.

    Returns:
        DataFrame with day of week as index and statistics
    """
    if TARGET_VARIABLE not in df.columns:
        return pd.DataFrame()

    daily = df.groupby(df.index.dayofweek)[TARGET_VARIABLE].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max'),
        ('median', 'median')
    ])

    # Map day numbers to names (handles cases where not all days are present)
    day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
                 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    daily.index = daily.index.map(day_names)

    return daily


def detect_missing_periods(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect missing time periods in the dataset.

    Returns:
        DataFrame with missing periods
    """
    # Expected frequency is hourly
    expected_index = pd.date_range(start=df.index.min(),
                                   end=df.index.max(),
                                   freq='H')

    missing_timestamps = expected_index.difference(df.index)

    if len(missing_timestamps) == 0:
        return pd.DataFrame()

    # Group consecutive missing periods
    missing_df = pd.DataFrame({
        'missing_timestamp': missing_timestamps
    })

    return missing_df


@st.cache_data
def load_feature_importance() -> Optional[pd.DataFrame]:
    """
    Load feature importance from saved model reports.

    Returns:
        DataFrame with feature names and importance scores, or None
    """
    # Try to load from reports directory
    from .config import BASELINE_REPORTS

    try:
        # Look for feature importance files
        importance_files = list(BASELINE_REPORTS.glob("**/feature_importance*.csv"))
        if importance_files:
            # Load the most recent one
            latest = max(importance_files, key=lambda p: p.stat().st_mtime)
            return pd.read_csv(latest)
    except Exception as e:
        st.warning(f"Could not load feature importance: {e}")

    return None
