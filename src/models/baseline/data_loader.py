"""
Data Loader for Baseline Models
================================
Handles loading and splitting of master dataset.

Author: ForeWatt Team
Date: November 2025
"""

import sys
import pandas as pd
from pathlib import Path
from typing import Tuple
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_master_data(
    start_date: str = '2020-01-01',
    end_date: str = '2025-10-31'
) -> pd.DataFrame:
    """
    Load master dataset from Gold layer.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        Master dataset with datetime index
    """
    gold_master = PROJECT_ROOT / 'data' / 'gold' / 'master'
    master_files = list(gold_master.glob('master_v*.parquet'))

    if not master_files:
        raise FileNotFoundError(f"No master files found in {gold_master}")

    # Use latest master file
    latest_master = sorted(master_files)[-1]
    logger.info(f"Loading master data from {latest_master.name}")

    df = pd.read_parquet(latest_master)

    # Set datetime index
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        if 'timestamp' not in df.columns:
            df['timestamp'] = df['datetime']
        df = df.set_index('datetime')
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')

    # Filter date range
    df = df.loc[start_date:end_date]

    logger.info(f"Loaded {len(df)} samples")
    logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
    logger.info(f"Shape: {df.shape}")

    # Check for target columns
    targets = ['consumption', 'price_real']
    for target in targets:
        if target in df.columns:
            logger.info(f"  {target}: {df[target].isna().sum()} missing values")
        else:
            logger.warning(f"  {target}: NOT FOUND in dataset")

    return df


def train_val_test_split(
    df: pd.DataFrame,
    val_size: float = 0.1,
    test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/val/test temporally.

    Args:
        df: Full dataset
        val_size: Validation set fraction
        test_size: Test set fraction

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # Temporal split (no shuffling)
    test_idx = int(len(df) * (1 - test_size))
    val_idx = int(len(df) * (1 - test_size - val_size))

    train_df = df.iloc[:val_idx].copy()
    val_df = df.iloc[val_idx:test_idx].copy()
    test_df = df.iloc[test_idx:].copy()

    logger.info(f"\n{'='*80}")
    logger.info("DATA SPLIT (Temporal)")
    logger.info(f"{'='*80}")
    logger.info(f"Train: {len(train_df):6d} samples ({train_df.index.min()} to {train_df.index.max()})")
    logger.info(f"Val:   {len(val_df):6d} samples ({val_df.index.min()} to {val_df.index.max()})")
    logger.info(f"Test:  {len(test_df):6d} samples ({test_df.index.min()} to {test_df.index.max()})")
    logger.info(f"{'='*80}\n")

    return train_df, val_df, test_df


def prepare_target_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str = 'consumption'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepare data for specific target (remove missing values).

    Args:
        train_df: Training data
        val_df: Validation data
        test_df: Test data
        target: Target column name

    Returns:
        Tuple of (train_df, val_df, test_df) with target NaNs removed
    """
    if target not in train_df.columns:
        raise ValueError(f"Target '{target}' not found in dataset")

    # Remove rows with missing target
    train_df = train_df.dropna(subset=[target])
    val_df = val_df.dropna(subset=[target])
    test_df = test_df.dropna(subset=[target])

    logger.info(f"Target: {target}")
    logger.info(f"  Train: {len(train_df)} samples")
    logger.info(f"  Val:   {len(val_df)} samples")
    logger.info(f"  Test:  {len(test_df)} samples")

    return train_df, val_df, test_df
