"""
Feature Engineering Pipeline Runner
====================================
Orchestrates the complete feature engineering pipeline for ForeWatt.

Pipeline Steps:
---------------
1. Generate lag features (consumption, temperature, price)
2. Generate rolling features (24h, 168h windows)
3. Merge all gold layers into master ML-ready dataset

Optional (run separately if needed):
- External features fetcher (FX/Gold from EVDS)
- Calendar features integration

Usage:
------
    python src/features/run_feature_pipeline.py

Or with custom date range:
    python src/features/run_feature_pipeline.py --start 2020-01-01 --end 2025-10-31

Author: ForeWatt Team
Date: November 2025
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.lag_features import LagFeaturesGenerator
from features.rolling_features import RollingFeaturesGenerator
from features.calendar_features import CalendarFeaturesGenerator
from features.merge_features import MasterFeatureMerger

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_complete_pipeline(start_date: str, end_date: str, version: str = 'v1'):
    """
    Run complete feature engineering pipeline.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        version: Master dataset version string
    """
    logger.info("="*70)
    logger.info("FOREWATT FEATURE ENGINEERING PIPELINE")
    logger.info("="*70)
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Master version: {version}")
    logger.info("="*70)

    try:
        # Step 1: Generate lag features
        logger.info("\n" + "="*70)
        logger.info("STEP 1/4: GENERATING LAG FEATURES")
        logger.info("="*70)
        lag_gen = LagFeaturesGenerator(data_dir='./data')
        lag_df = lag_gen.run_pipeline(start_date, end_date)
        logger.info("✓ Lag features generation complete")

        # Step 2: Generate rolling features
        logger.info("\n" + "="*70)
        logger.info("STEP 2/4: GENERATING ROLLING FEATURES")
        logger.info("="*70)
        rolling_gen = RollingFeaturesGenerator(data_dir='./data')
        rolling_df = rolling_gen.run_pipeline(start_date, end_date)
        logger.info("✓ Rolling features generation complete")

        # Step 3: Generate calendar features
        logger.info("\n" + "="*70)
        logger.info("STEP 3/4: GENERATING CALENDAR FEATURES")
        logger.info("="*70)
        calendar_gen = CalendarFeaturesGenerator(data_dir='./data')
        calendar_df = calendar_gen.run_pipeline(start_date, end_date)
        logger.info("✓ Calendar features generation complete")

        # Step 4: Merge all features
        logger.info("\n" + "="*70)
        logger.info("STEP 4/4: MERGING ALL FEATURES")
        logger.info("="*70)
        merger = MasterFeatureMerger(data_dir='./data', version=version)
        master_df = merger.run_pipeline(start_date, end_date)
        logger.info("✓ Master dataset creation complete")

        # Final summary
        logger.info("\n" + "="*70)
        logger.info("PIPELINE EXECUTION COMPLETE")
        logger.info("="*70)
        logger.info(f"✓ Lag features: {lag_df.shape}")
        logger.info(f"✓ Rolling features: {rolling_df.shape}")
        logger.info(f"✓ Calendar features: {calendar_df.shape}")
        logger.info(f"✓ Master dataset: {master_df.shape}")
        logger.info("="*70)

        logger.info("\nNext steps:")
        logger.info("1. (Optional) Run external features: python src/data/external_features_fetcher.py")
        logger.info("2. Train models: python src/models/train.py (TODO)")

        return master_df

    except Exception as e:
        logger.error(f"\n{'='*70}")
        logger.error("PIPELINE EXECUTION FAILED")
        logger.error(f"{'='*70}")
        logger.error(f"Error: {str(e)}")
        logger.error(f"{'='*70}")
        raise


def main():
    """Main execution with CLI arguments."""
    parser = argparse.ArgumentParser(
        description='Run ForeWatt feature engineering pipeline'
    )
    parser.add_argument(
        '--start',
        type=str,
        default='2020-01-01',
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end',
        type=str,
        default='2025-10-31',
        help='End date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--version',
        type=str,
        default='v1',
        help='Master dataset version (e.g., v1, v2)'
    )

    args = parser.parse_args()

    # Run pipeline
    master_df = run_complete_pipeline(
        start_date=args.start,
        end_date=args.end,
        version=args.version
    )

    print("\n" + "="*70)
    print("Master dataset preview:")
    print(master_df.head())
    print("\nMaster dataset columns:")
    print(f"Total: {len(master_df.columns)}")
    print(master_df.columns.tolist()[:20], "...")


if __name__ == '__main__':
    main()
