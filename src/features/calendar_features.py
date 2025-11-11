"""
Calendar Features Generator for ForeWatt
=========================================
Creates calendar features for electricity demand forecasting.

Features Created:
-----------------
Temporal:
- dow (day of week: 0=Monday, 6=Sunday)
- dom (day of month: 1-31)
- month (1-12)
- weekofyear (1-53)
- is_weekend (0/1)

Holiday-related:
- is_holiday_day (0/1, day-level flag)
- is_holiday_hour (0/1, hour-level flag with half-day PM handling)
- holiday_name (string, e.g., "Ramazan Bayramı", "Cumhuriyet Bayramı")

Cyclical Encodings (for neural models):
- dow_sin, dow_cos (day of week circular)
- month_sin, month_cos (month circular)

Holiday Types:
- Turkish official holidays (New Year, Republic Day, etc.)
- Religious holidays (Ramadan, Kurban Bayramı) - multi-day
- Half-day holidays (e.g., Oct 28 PM only)

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


class CalendarFeaturesGenerator:
    """Generate calendar and holiday features for forecasting models."""

    TZ = "Europe/Istanbul"

    def __init__(self, data_dir: str = './data'):
        """
        Initialize calendar features generator.

        Args:
            data_dir: Root data directory
        """
        self.data_dir = Path(data_dir)
        self.silver_dir = self.data_dir / 'silver'
        self.gold_dir = self.data_dir / 'gold'
        self.output_dir = self.gold_dir / 'calendar_features'
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _to_datetime_index(self, idx: pd.Index) -> pd.DatetimeIndex:
        """Ensure a tz-aware Europe/Istanbul DatetimeIndex."""
        di = pd.to_datetime(idx)
        if isinstance(di, pd.DatetimeIndex):
            if di.tz is None:
                di = di.tz_localize(self.TZ)
            else:
                di = di.tz_convert(self.TZ)
            return di
        return pd.DatetimeIndex(di, tz=self.TZ)

    def load_calendar_days(self) -> pd.DataFrame:
        """Load holiday calendar from Silver layer."""
        logger.info("Loading calendar days...")
        file_path = self.silver_dir / 'calendar' / 'calendar_days.parquet'

        if not file_path.exists():
            # Fallback to CSV
            file_path = self.silver_dir / 'calendar' / 'calendar_days.csv'
            if not file_path.exists():
                raise FileNotFoundError(f"Calendar data not found in Silver layer")

        if file_path.suffix == '.parquet':
            cal_days = pd.read_parquet(file_path)
        else:
            cal_days = pd.read_csv(file_path, parse_dates=['date_only'])

        # Normalize to pure Python date
        cal_days['date_only'] = pd.to_datetime(cal_days['date_only']).dt.date

        logger.info(f"Loaded {len(cal_days)} holiday records")
        return cal_days

    def create_hourly_calendar(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Create hourly calendar features for date range.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with hourly calendar features
        """
        logger.info(f"Creating hourly calendar for {start_date} to {end_date}...")

        # Create hourly datetime range
        start = pd.to_datetime(start_date).tz_localize(self.TZ)
        end = pd.to_datetime(end_date).tz_localize(self.TZ) + pd.Timedelta(hours=23)
        hourly_index = pd.date_range(start=start, end=end, freq='H', tz=self.TZ)

        logger.info(f"  Hourly range: {len(hourly_index)} hours")

        # Initialize features DataFrame
        feats = pd.DataFrame(index=hourly_index)
        feats['timestamp'] = hourly_index

        # Temporal features
        feats['date_only'] = hourly_index.date
        feats['dow'] = hourly_index.dayofweek  # 0=Monday, 6=Sunday
        feats['dom'] = hourly_index.day
        feats['month'] = hourly_index.month
        feats['weekofyear'] = hourly_index.isocalendar().week.astype(int)
        feats['is_weekend'] = feats['dow'].isin([5, 6]).astype('int8')

        # Load holiday data
        cal_days = self.load_calendar_days()

        # Day-level holiday flag
        day_map = cal_days.set_index('date_only')['is_holiday_day'].to_dict()
        feats['is_holiday_day'] = feats['date_only'].map(
            lambda d: int(bool(day_map.get(d, 0)))
        ).astype('int8')

        # Hour-level holiday flag (with half-day handling)
        feats['is_holiday_hour'] = feats['is_holiday_day'].copy()

        if 'half_day' in cal_days.columns:
            half = cal_days[cal_days['half_day'].notna()][['date_only', 'half_day']].drop_duplicates()
            if not half.empty:
                half_map = half.set_index('date_only')['half_day'].to_dict()
                # For 'pm' half-days, AM hours (<13) are not holiday
                am_mask = hourly_index.hour < 13
                feats.loc[
                    feats['date_only'].map(lambda d: half_map.get(d) == 'pm') & am_mask,
                    'is_holiday_hour'
                ] = 0
                logger.info("  Applied half-day PM handling")

        # Holiday name (categorical feature)
        name_map = cal_days.set_index('date_only')['name'].to_dict() if 'name' in cal_days.columns else {}
        feats['holiday_name'] = feats['date_only'].map(lambda d: name_map.get(d, 'None'))

        # Cyclical encodings (for neural models)
        feats['dow_sin'] = np.sin(2 * np.pi * feats['dow'] / 7)
        feats['dow_cos'] = np.cos(2 * np.pi * feats['dow'] / 7)
        feats['month_sin'] = np.sin(2 * np.pi * feats['month'] / 12)
        feats['month_cos'] = np.cos(2 * np.pi * feats['month'] / 12)

        # Drop intermediate date_only column
        feats = feats.drop(columns=['date_only'])

        logger.info(f"Created {len(feats.columns)} calendar features")
        return feats

    def save_calendar_features(self, df: pd.DataFrame, start_date: str, end_date: str):
        """Save calendar features to Gold layer."""
        # Parquet
        parquet_path = self.output_dir / f'calendar_features_{start_date}_{end_date}.parquet'
        df.to_parquet(parquet_path, index=False, engine='pyarrow')
        logger.info(f"Saved Parquet: {parquet_path}")

        # CSV (secondary format for inspection)
        csv_path = self.output_dir / f'calendar_features_{start_date}_{end_date}.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved CSV: {csv_path}")

        # Log summary statistics
        logger.info("\nCalendar Features Summary:")
        logger.info(f"Total records: {len(df)}")
        logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        logger.info(f"Total holidays: {df['is_holiday_day'].sum()} days")
        logger.info(f"Holiday hours: {df['is_holiday_hour'].sum()} hours")
        logger.info(f"Weekend hours: {df['is_weekend'].sum()} hours")

        # List holidays
        holidays = df[df['is_holiday_day'] == 1][['timestamp', 'holiday_name']].drop_duplicates('holiday_name')
        if len(holidays) > 0:
            logger.info(f"\nHolidays in range:")
            for _, row in holidays.head(10).iterrows():
                logger.info(f"  - {row['holiday_name']}")
            if len(holidays) > 10:
                logger.info(f"  ... and {len(holidays) - 10} more")

    def run_pipeline(self, start_date: str = '2020-01-01', end_date: str = '2024-12-31'):
        """Run complete calendar features pipeline."""
        logger.info("="*60)
        logger.info("CALENDAR FEATURES PIPELINE")
        logger.info("="*60)

        # Generate calendar features
        calendar_df = self.create_hourly_calendar(start_date, end_date)

        # Save
        self.save_calendar_features(calendar_df, start_date, end_date)

        logger.info("="*60)
        logger.info("CALENDAR FEATURES PIPELINE COMPLETE")
        logger.info("="*60)

        return calendar_df


def main():
    """Main execution."""
    generator = CalendarFeaturesGenerator(data_dir='./data')
    calendar_df = generator.run_pipeline(
        start_date='2020-01-01',
        end_date='2024-12-31'
    )

    print("\nCalendar features preview:")
    print(calendar_df.head())
    print("\nCalendar features columns:")
    print(calendar_df.columns.tolist())


if __name__ == '__main__':
    main()
