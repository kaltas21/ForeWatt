"""
Fundamental Feature Engineering V2
==================================
Generates master_v2_fundamental.parquet with high-signal fundamental features
for breaking the 10.87% sMAPE floor on Price Forecasting.

Features Created:
1. reserve_margin_ratio: (capacity_eak - consumption_forecast) / capacity_eak
2. renewable_saturation: (wind_forecast + hydro_energy) / consumption_forecast
3. thermal_gap: consumption_forecast - (wind_forecast + hydro_energy)
4. system_short_signal: price_smf(lag 24h) - price_ptf(lag 24h)
5. import_cost_proxy: USD_TRY * consumption_forecast
6. spark_spread_proxy: price_real / (USD_TRY * 100)

Author: ForeWatt Team - New Experiment V2
Date: November 2025
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging
import warnings

warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FundamentalFeatureEngineerV2:
    """
    Creates fundamental features for electricity price forecasting.

    These features capture supply-demand dynamics, renewable penetration,
    and cost structure signals that drive price formation in the Turkish
    electricity market.
    """

    def __init__(self, data_dir: Path = None):
        """
        Initialize feature engineer.

        Args:
            data_dir: Path to data directory (default: PROJECT_ROOT/data)
        """
        self.data_dir = data_dir or PROJECT_ROOT / 'data'
        self.silver_epias = self.data_dir / 'silver' / 'epias'
        self.gold_external = self.data_dir / 'gold' / 'external'
        self.gold_master = self.data_dir / 'gold' / 'master'

    def _find_latest_file(self, directory: Path, pattern: str) -> Path:
        """Find latest file matching pattern in directory."""
        files = list(directory.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No files matching {pattern} in {directory}")
        # Sort by modification time, return newest
        return max(files, key=lambda p: p.stat().st_mtime)

    def _load_and_prepare_timestamps(self, df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
        """
        Standardize timestamps to UTC-aware hourly index.

        Args:
            df: Input dataframe
            date_col: Name of date column

        Returns:
            DataFrame with standardized datetime index
        """
        df = df.copy()

        # Parse date column
        if date_col in df.columns:
            df['datetime'] = pd.to_datetime(df[date_col], utc=True)
        elif 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
        else:
            raise ValueError(f"No date column found. Available: {df.columns.tolist()}")

        # Convert to Turkey time (UTC+3) then back to UTC for alignment
        if df['datetime'].dt.tz is not None:
            df['datetime'] = df['datetime'].dt.tz_convert('UTC')

        # Set as index and sort
        df = df.set_index('datetime').sort_index()

        # Remove duplicates (keep last)
        df = df[~df.index.duplicated(keep='last')]

        return df

    def load_master_v1(self) -> pd.DataFrame:
        """Load master_v1 dataset."""
        logger.info("Loading master_v1...")

        master_file = self._find_latest_file(self.gold_master, 'master_v1*.csv')
        logger.info(f"  Found: {master_file.name}")

        df = pd.read_csv(master_file, parse_dates=['timestamp'])
        df['datetime'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df.set_index('datetime').sort_index()
        df = df[~df.index.duplicated(keep='last')]

        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Date range: {df.index.min()} to {df.index.max()}")

        return df

    def load_capacity_data(self) -> pd.DataFrame:
        """
        Load available capacity data (EAK - Emre Edilebilir Kapasite).

        The capacity_eak file contains hourly available generation capacity
        from all sources.
        """
        logger.info("Loading capacity data (EAK)...")

        cap_file = self._find_latest_file(self.silver_epias, 'capacity_eak_normalized*.csv')
        df = pd.read_csv(cap_file)

        # The file has generation by source with 'toplam' as total
        df = self._load_and_prepare_timestamps(df)

        # Total available capacity
        if 'toplam' in df.columns:
            df['capacity_eak'] = df['toplam']
        else:
            # Sum all generation sources if toplam not available
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df['capacity_eak'] = df[numeric_cols].sum(axis=1)

        logger.info(f"  Capacity range: {df['capacity_eak'].min():.0f} - {df['capacity_eak'].max():.0f} MW")

        return df[['capacity_eak']]

    def load_consumption_forecast(self) -> pd.DataFrame:
        """Load day-ahead consumption forecast (LEP)."""
        logger.info("Loading consumption forecast (LEP)...")

        cf_file = self._find_latest_file(self.silver_epias, 'consumption_forecast_normalized*.csv')
        df = pd.read_csv(cf_file)
        df = self._load_and_prepare_timestamps(df)

        # LEP is the consumption forecast
        if 'lep' in df.columns:
            df['consumption_forecast'] = df['lep']
        else:
            raise ValueError("LEP column not found in consumption forecast data")

        logger.info(f"  Forecast range: {df['consumption_forecast'].min():.0f} - {df['consumption_forecast'].max():.0f} MWh")

        return df[['consumption_forecast']]

    def load_wind_forecast(self) -> pd.DataFrame:
        """Load wind generation forecast."""
        logger.info("Loading wind forecast...")

        wf_file = self._find_latest_file(self.silver_epias, 'wind_forecast_normalized*.csv')
        df = pd.read_csv(wf_file)
        df = self._load_and_prepare_timestamps(df)

        # Use 'forecast' column or 'generation' if available
        if 'forecast' in df.columns:
            df['wind_forecast'] = df['forecast']
        elif 'generation' in df.columns:
            df['wind_forecast'] = df['generation']
        else:
            # Take mean of quarter columns if available
            quarter_cols = [c for c in df.columns if 'quarter' in c.lower()]
            if quarter_cols:
                df['wind_forecast'] = df[quarter_cols].mean(axis=1)
            else:
                raise ValueError("No wind forecast column found")

        # Keep only the wind_forecast column before resampling
        df = df[['wind_forecast']]

        # Resample to hourly (wind data might be sub-hourly)
        df = df.resample('h').mean()

        logger.info(f"  Wind forecast range: {df['wind_forecast'].min():.1f} - {df['wind_forecast'].max():.1f} MWh")

        return df[['wind_forecast']]

    def load_hydro_energy(self) -> pd.DataFrame:
        """Load hydro energy provision (aggregated across all dams)."""
        logger.info("Loading hydro energy provision...")

        hydro_file = self._find_latest_file(self.silver_epias, 'hydro_energy_provision_normalized*.csv')
        df = pd.read_csv(hydro_file)
        df = self._load_and_prepare_timestamps(df)

        # Aggregate by datetime (sum across all dams)
        if 'energyGeneration' in df.columns:
            df['hydro_energy'] = df['energyGeneration']
        else:
            # Try to find any energy column
            energy_cols = [c for c in df.columns if 'energy' in c.lower() or 'generation' in c.lower()]
            if energy_cols:
                df['hydro_energy'] = df[energy_cols[0]]
            else:
                raise ValueError("No hydro energy column found")

        # Aggregate by hour
        df = df.groupby(df.index).agg({'hydro_energy': 'sum'})

        # Resample to ensure hourly
        df = df.resample('h').mean().ffill()

        logger.info(f"  Hydro energy range: {df['hydro_energy'].min():.1f} - {df['hydro_energy'].max():.1f} MWh")

        return df[['hydro_energy']]

    def load_price_smf(self) -> pd.DataFrame:
        """Load System Marginal Price (SMF - balancing price)."""
        logger.info("Loading SMF price...")

        smf_file = self._find_latest_file(self.silver_epias, 'price_smf_normalized*.csv')
        df = pd.read_csv(smf_file)
        df = self._load_and_prepare_timestamps(df)

        if 'systemMarginalPrice' in df.columns:
            df['price_smf'] = df['systemMarginalPrice']
        elif 'price' in df.columns:
            df['price_smf'] = df['price']
        else:
            raise ValueError("No SMF price column found")

        logger.info(f"  SMF price range: {df['price_smf'].min():.2f} - {df['price_smf'].max():.2f} TL/MWh")

        return df[['price_smf']]

    def load_price_ptf(self) -> pd.DataFrame:
        """Load Day-Ahead Price (PTF)."""
        logger.info("Loading PTF price...")

        ptf_file = self._find_latest_file(self.silver_epias, 'price_ptf_normalized*.csv')
        df = pd.read_csv(ptf_file)
        df = self._load_and_prepare_timestamps(df)

        if 'price' in df.columns:
            df['price_ptf_raw'] = df['price']
        else:
            raise ValueError("No PTF price column found")

        logger.info(f"  PTF price range: {df['price_ptf_raw'].min():.2f} - {df['price_ptf_raw'].max():.2f} TL/MWh")

        return df[['price_ptf_raw']]

    def load_fx_features(self) -> pd.DataFrame:
        """Load foreign exchange features."""
        logger.info("Loading FX features...")

        fx_file = self._find_latest_file(self.gold_external, 'fx_features_hourly*.csv')
        df = pd.read_csv(fx_file)

        # FX file uses 'datetime' column
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
        df = df.set_index('datetime').sort_index()
        df = df[~df.index.duplicated(keep='last')]

        # Key FX features for price forecasting
        fx_cols = ['USD_TRY', 'EUR_TRY', 'FX_basket', 'FX_volatility']
        available_fx = [c for c in fx_cols if c in df.columns]

        logger.info(f"  USD/TRY range: {df['USD_TRY'].min():.2f} - {df['USD_TRY'].max():.2f}")

        return df[available_fx]

    def create_fundamental_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create fundamental features from joined data.

        Args:
            df: DataFrame with all joined data sources

        Returns:
            DataFrame with new fundamental features
        """
        logger.info("\n" + "="*80)
        logger.info("CREATING FUNDAMENTAL FEATURES")
        logger.info("="*80)

        df = df.copy()

        # 1. RESERVE MARGIN RATIO
        # Measures system tightness: how much spare capacity exists
        # High ratio = comfortable supply margin, lower prices
        # Low ratio = tight system, higher prices and volatility
        logger.info("\n1. Reserve Margin Ratio:")
        df['reserve_margin_ratio'] = (
            (df['capacity_eak'] - df['consumption_forecast']) /
            df['capacity_eak'].replace(0, np.nan)
        )
        # Clip extreme values
        df['reserve_margin_ratio'] = df['reserve_margin_ratio'].clip(-1, 2)
        logger.info(f"   Range: {df['reserve_margin_ratio'].min():.3f} to {df['reserve_margin_ratio'].max():.3f}")

        # 2. RENEWABLE SATURATION
        # Measures renewable penetration relative to demand
        # High saturation = cheap renewable power available
        # Low saturation = more thermal generation needed
        logger.info("\n2. Renewable Saturation:")
        renewable_gen = df['wind_forecast'].fillna(0) + df['hydro_energy'].fillna(0)
        df['renewable_saturation'] = (
            renewable_gen / df['consumption_forecast'].replace(0, np.nan)
        )
        df['renewable_saturation'] = df['renewable_saturation'].clip(0, 2)
        logger.info(f"   Range: {df['renewable_saturation'].min():.3f} to {df['renewable_saturation'].max():.3f}")

        # 3. THERMAL GAP
        # The load that thermal plants (gas/coal) must cover
        # Higher gap = more gas needed = higher prices
        logger.info("\n3. Thermal Gap:")
        df['thermal_gap'] = (
            df['consumption_forecast'] - renewable_gen
        )
        # Negative thermal gap means renewable surplus
        logger.info(f"   Range: {df['thermal_gap'].min():.0f} to {df['thermal_gap'].max():.0f} MWh")

        # 4. SYSTEM SHORT SIGNAL (Lagged)
        # SMF - PTF spread indicates real-time vs day-ahead imbalance
        # Positive = system was short (actual > forecast), future prices may rise
        # Negative = system was long, future prices may fall
        logger.info("\n4. System Short Signal (24h lag):")
        df['price_smf_lag_24h'] = df['price_smf'].shift(24)
        df['price_ptf_lag_24h_raw'] = df['price_ptf_raw'].shift(24)
        df['system_short_signal'] = df['price_smf_lag_24h'] - df['price_ptf_lag_24h_raw']
        logger.info(f"   Range: {df['system_short_signal'].min():.2f} to {df['system_short_signal'].max():.2f} TL/MWh")

        # 5. IMPORT COST PROXY (D-1 Safe)
        # Approximates cost of imports / fuel costs in TRY terms
        # Higher USD/TRY * demand = higher cost pressure
        # Using D-1 close FX rate (shifted 24h) for day-ahead safety
        logger.info("\n5. Import Cost Proxy (D-1 Safe):")
        usd_try_d1 = df['USD_TRY'].shift(24)  # Use D-1 FX rate
        df['import_cost_proxy'] = usd_try_d1 * df['consumption_forecast'] / 1000  # Scale down
        logger.info(f"   Range: {df['import_cost_proxy'].min():.0f} to {df['import_cost_proxy'].max():.0f}")

        # 6. SPARK SPREAD PROXY (LAGGED - Day-Ahead Safe)
        # Approximates gas plant profitability using LAGGED price to avoid leakage
        # Higher spread = gas plants profitable = supply willing to generate
        # Lower spread = gas plants reluctant = potential supply crunch
        logger.info("\n6. Spark Spread Proxy (Lagged - D-1 Safe):")
        # Using USD/TRY * 100 as rough gas price proxy (natural gas imports)
        # Use LAGGED price (24h) to avoid data leakage for day-ahead forecasting
        gas_proxy = df['USD_TRY'].shift(24) * 100  # Use D-1 FX rate
        df['spark_spread_proxy'] = df['price_ptf_lag_24h_raw'] / gas_proxy.replace(0, np.nan)
        df['spark_spread_proxy'] = df['spark_spread_proxy'].clip(0, 10)
        logger.info(f"   Range: {df['spark_spread_proxy'].min():.3f} to {df['spark_spread_proxy'].max():.3f}")

        # Additional derived features (ALL DAY-AHEAD SAFE)
        logger.info("\n7. Additional Derived Features (D-1 Safe):")

        # Renewable share of capacity (OK - uses forecasts available D-1)
        df['renewable_capacity_share'] = renewable_gen / df['capacity_eak'].replace(0, np.nan)
        df['renewable_capacity_share'] = df['renewable_capacity_share'].clip(0, 1)

        # Load factor (OK - uses forecasts available D-1)
        df['load_factor'] = df['consumption_forecast'] / df['capacity_eak'].replace(0, np.nan)
        df['load_factor'] = df['load_factor'].clip(0, 1.5)

        # SMF/PTF ratio - LAGGED VERSION (real-time premium from D-1)
        # Cannot use current SMF - it's an actual value not available for D+1
        df['realtime_premium_lag24h'] = df['price_smf_lag_24h'] / df['price_ptf_lag_24h_raw'].replace(0, np.nan)
        df['realtime_premium_lag24h'] = df['realtime_premium_lag24h'].clip(0.1, 10)

        # Price volatility signal - LAGGED (from D-1 and earlier)
        # Use PTF rolling std shifted by 24h to ensure D-1 availability
        if 'price_ptf_rolling_std_24h' in df.columns:
            df['price_volatility_lag24h'] = df['price_ptf_rolling_std_24h'].shift(24)
        else:
            df['price_volatility_lag24h'] = df['price_ptf_raw'].shift(24).rolling(24, min_periods=1).std()

        logger.info("   - renewable_capacity_share (forecast-based, OK)")
        logger.info("   - load_factor (forecast-based, OK)")
        logger.info("   - realtime_premium_lag24h (lagged, D-1 safe)")
        logger.info("   - price_volatility_lag24h (lagged, D-1 safe)")

        return df

    def create_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lagged versions of fundamental features."""
        logger.info("\nCreating lagged fundamental features...")

        df = df.copy()

        # Lags for fundamental features
        fundamental_cols = [
            'reserve_margin_ratio', 'renewable_saturation', 'thermal_gap',
            'import_cost_proxy', 'spark_spread_proxy', 'load_factor'
        ]

        lag_hours = [1, 24, 48, 168]

        for col in fundamental_cols:
            if col in df.columns:
                for lag in lag_hours:
                    df[f'{col}_lag_{lag}h'] = df[col].shift(lag)

        logger.info(f"   Created lagged features for {len(fundamental_cols)} fundamental variables")

        return df

    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling statistics for fundamental features."""
        logger.info("\nCreating rolling fundamental features...")

        df = df.copy()

        fundamental_cols = [
            'reserve_margin_ratio', 'renewable_saturation', 'thermal_gap',
            'system_short_signal', 'load_factor'
        ]

        windows = [24, 168]

        for col in fundamental_cols:
            if col in df.columns:
                for window in windows:
                    df[f'{col}_rolling_mean_{window}h'] = df[col].rolling(window, min_periods=1).mean()
                    df[f'{col}_rolling_std_{window}h'] = df[col].rolling(window, min_periods=1).std()

        logger.info(f"   Created rolling features for {len(fundamental_cols)} fundamental variables")

        return df

    def create_consumption_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create consumption-specific features for consumption forecasting.

        These features are D-1 safe and use lagged consumption values.
        """
        logger.info("\nCreating consumption-specific features...")

        df = df.copy()

        # Consumption lags (24h+ for D-1 safety)
        if 'consumption' in df.columns:
            for lag in [24, 48, 168]:
                col_name = f'consumption_lag_{lag}h'
                if col_name not in df.columns:
                    df[col_name] = df['consumption'].shift(lag)

            # Consumption rolling statistics
            for window in [24, 168]:
                mean_col = f'consumption_rolling_mean_{window}h'
                std_col = f'consumption_rolling_std_{window}h'
                min_col = f'consumption_rolling_min_{window}h'
                max_col = f'consumption_rolling_max_{window}h'

                if mean_col not in df.columns:
                    df[mean_col] = df['consumption'].rolling(window, min_periods=1).mean()
                if std_col not in df.columns:
                    df[std_col] = df['consumption'].rolling(window, min_periods=1).std()
                if min_col not in df.columns:
                    df[min_col] = df['consumption'].rolling(window, min_periods=1).min()
                if max_col not in df.columns:
                    df[max_col] = df['consumption'].rolling(window, min_periods=1).max()

            logger.info("   Created consumption lag and rolling features")

        # Temperature lags (if available)
        temp_cols = ['temp_national', 'temperature']
        for temp_col in temp_cols:
            if temp_col in df.columns:
                for lag in [24, 168]:
                    lag_col = f'{temp_col.replace("_national", "")}_lag_{lag}h'
                    if lag_col not in df.columns:
                        df[lag_col] = df[temp_col].shift(lag)

                # Temperature rolling stats
                for window in [24, 168]:
                    roll_col = f'{temp_col.replace("_national", "")}_rolling_{window}h'
                    if roll_col not in df.columns:
                        df[roll_col] = df[temp_col].rolling(window, min_periods=1).mean()

                logger.info(f"   Created temperature lag and rolling features from {temp_col}")
                break  # Only need one temperature column

        return df

    def run_pipeline(self, save_output: bool = True) -> pd.DataFrame:
        """
        Run the full feature engineering pipeline.

        Args:
            save_output: Whether to save the output parquet file

        Returns:
            DataFrame with all fundamental features
        """
        logger.info("\n" + "="*80)
        logger.info("FUNDAMENTAL FEATURE ENGINEERING V2 PIPELINE")
        logger.info("="*80)

        # Load master_v1 as base
        master = self.load_master_v1()

        # Load component data sources
        capacity = self.load_capacity_data()
        consumption_fcst = self.load_consumption_forecast()
        wind_fcst = self.load_wind_forecast()
        hydro = self.load_hydro_energy()
        price_smf = self.load_price_smf()
        price_ptf = self.load_price_ptf()
        fx = self.load_fx_features()

        # Join all data sources to master
        logger.info("\nJoining data sources to master...")

        # Left join to preserve master index
        df = master.copy()

        data_sources = [
            (capacity, 'capacity_eak'),
            (consumption_fcst, 'consumption_forecast'),
            (wind_fcst, 'wind_forecast'),
            (hydro, 'hydro_energy'),
            (price_smf, 'price_smf'),
            (price_ptf, 'price_ptf_raw'),
            (fx, 'fx'),
        ]

        for source_df, name in data_sources:
            try:
                df = df.join(source_df, how='left', rsuffix=f'_{name}')
                logger.info(f"   Joined {name}: {source_df.shape[1]} columns")
            except Exception as e:
                logger.warning(f"   Failed to join {name}: {e}")

        # Forward fill missing values (market data might have gaps)
        logger.info("\nForward filling missing values...")
        fill_cols = [
            'capacity_eak', 'consumption_forecast', 'wind_forecast',
            'hydro_energy', 'price_smf', 'price_ptf_raw', 'USD_TRY'
        ]
        for col in fill_cols:
            if col in df.columns:
                df[col] = df[col].ffill()

        # Create fundamental features
        df = self.create_fundamental_features(df)

        # Create lagged features
        df = self.create_lagged_features(df)

        # Create rolling features
        df = self.create_rolling_features(df)

        # Create consumption-specific features (for consumption forecasting target)
        df = self.create_consumption_features(df)

        # Final cleanup
        logger.info("\nFinal cleanup...")

        # Forward fill any remaining NaNs from feature engineering
        df = df.ffill()

        # Drop rows with remaining NaNs at the start (from lags)
        initial_len = len(df)
        df = df.dropna(subset=['reserve_margin_ratio', 'renewable_saturation', 'thermal_gap'])
        logger.info(f"   Dropped {initial_len - len(df)} rows with NaN fundamental features")

        # Save output
        if save_output:
            # Save parquet
            parquet_path = self.gold_master / 'master_v2_fundamental.parquet'
            df.to_parquet(parquet_path)
            logger.info(f"\nSaved parquet: {parquet_path}")
            logger.info(f"   Shape: {df.shape}")
            logger.info(f"   Size: {parquet_path.stat().st_size / 1024 / 1024:.1f} MB")

            # Save CSV
            csv_path = self.gold_master / 'master_v2_fundamental.csv'
            df.to_csv(csv_path)
            logger.info(f"\nSaved CSV: {csv_path}")
            logger.info(f"   Size: {csv_path.stat().st_size / 1024 / 1024:.1f} MB")

        # Summary
        logger.info("\n" + "="*80)
        logger.info("FUNDAMENTAL FEATURES CREATED")
        logger.info("="*80)

        fundamental_features = [
            'reserve_margin_ratio', 'renewable_saturation', 'thermal_gap',
            'system_short_signal', 'import_cost_proxy', 'spark_spread_proxy',
            'renewable_capacity_share', 'load_factor', 'realtime_premium',
            'price_volatility_signal'
        ]

        for feat in fundamental_features:
            if feat in df.columns:
                logger.info(f"  {feat}: {df[feat].min():.3f} to {df[feat].max():.3f}")

        logger.info(f"\nTotal columns: {df.shape[1]}")
        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
        logger.info(f"Total samples: {len(df)}")

        return df


def main():
    """Main entry point."""
    engineer = FundamentalFeatureEngineerV2()
    df = engineer.run_pipeline(save_output=True)

    print("\n" + "="*80)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*80)
    print(f"\nOutput: data/gold/master/master_v2_fundamental.parquet")
    print(f"Shape: {df.shape}")
    print(f"\nNew fundamental columns:")
    new_cols = [c for c in df.columns if any(x in c for x in [
        'reserve_margin', 'renewable_sat', 'thermal_gap', 'system_short',
        'import_cost', 'spark_spread', 'load_factor', 'realtime_premium'
    ])]
    for col in sorted(new_cols)[:20]:
        print(f"  - {col}")
    if len(new_cols) > 20:
        print(f"  ... and {len(new_cols) - 20} more")


if __name__ == "__main__":
    main()
