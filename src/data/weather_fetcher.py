"""
Demand-Side Weather Data Pipeline
==================================
Fetches weather data at major population centers for electricity demand forecasting.

Author: ForeWatt Team
Date: October 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import openmeteo_requests
import requests_cache
from retry_requests import retry
from pathlib import Path
import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DemandWeatherFetcher:
    """
    Fetches and processes weather data at population centers for demand forecasting.
    
    Features:
    - Population-weighted city selection
    - Automatic caching and retry logic
    - Europe/Istanbul timezone handling
    - Demand-relevant feature engineering
    """
    
    # Top 10 Turkish cities by official 2024 population (TÜİK data)
    # Total Turkey population: 85,664,944
    DEMAND_CITIES = {
        'Istanbul': {
            'lat': 41.0082,
            'lon': 28.9784,
            'pop_weight': 0.1833,  # 15,701,602
            'population': 15_701_602,
            'region': 'Marmara'
        },
        'Ankara': {
            'lat': 39.9334,
            'lon': 32.8597,
            'pop_weight': 0.0685,  # 5,864,049
            'population': 5_864_049,
            'region': 'Central Anatolia'
        },
        'Izmir': {
            'lat': 38.4237,
            'lon': 27.1428,
            'pop_weight': 0.0524,  # 4,493,242
            'population': 4_493_242,
            'region': 'Aegean'
        },
        'Bursa': {
            'lat': 40.1826,
            'lon': 29.0665,
            'pop_weight': 0.0378,  # 3,238,618
            'population': 3_238_618,
            'region': 'Marmara'
        },
        'Antalya': {
            'lat': 36.8969,
            'lon': 30.7133,
            'pop_weight': 0.0318,  # 2,722,103
            'population': 2_722_103,
            'region': 'Mediterranean'
        },
        'Konya': {
            'lat': 37.8746,
            'lon': 32.4932,
            'pop_weight': 0.0272,  # 2,330,024
            'population': 2_330_024,
            'region': 'Central Anatolia'
        },
        'Adana': {
            'lat': 37.0000,
            'lon': 35.3213,
            'pop_weight': 0.0266,  # 2,280,484
            'population': 2_280_484,
            'region': 'Mediterranean'
        },
        'Sanliurfa': {
            'lat': 37.1591,
            'lon': 38.7969,
            'pop_weight': 0.0261,  # 2,237,745
            'population': 2_237_745,
            'region': 'Southeastern Anatolia'
        },
        'Gaziantep': {
            'lat': 37.0662,
            'lon': 37.3833,
            'pop_weight': 0.0256,  # 2,193,363
            'population': 2_193_363,
            'region': 'Southeastern Anatolia'
        },
        'Kocaeli': {
            'lat': 40.8533,
            'lon': 29.8815,
            'pop_weight': 0.0249,  # 2,130,006
            'population': 2_130_006,
            'region': 'Marmara'
        },
    }
    # Total weight: 0.5042 (represents 50.42% of Turkey's population)
    # Total population covered: 43,191,236 / 85,664,944 (official 2024 TÜİK data)
    # Top 10 cities: 42,191,236 / 85,664,944 = 49.25%
    
    # Weather variables for demand forecasting
    HOURLY_VARIABLES = [
        'temperature_2m',              # Core demand driver
        'relative_humidity_2m',        # For heat index calculation
        'precipitation',               # Affects behavior patterns
        'rain',                        # Liquid precipitation
        'cloud_cover',                 # Natural lighting (industrial/commercial)
        'wind_speed_10m',              # Wind chill factor
        'surface_pressure',            # Weather pattern indicator
        'apparent_temperature',        # Feels-like temperature
    ]
    
    def __init__(self, cache_dir: str = '.cache'):
        """
        Initialize the weather fetcher with caching.
        
        Args:
            cache_dir: Directory for request caching
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Setup Open-Meteo client with caching and retry
        cache_session = requests_cache.CachedSession(
            str(self.cache_dir / 'openmeteo_cache'), 
            expire_after=-1  # Cache forever (historical data doesn't change)
        )
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        self.client = openmeteo_requests.Client(session=retry_session)
        
        logger.info(f"Initialized DemandWeatherFetcher with {len(self.DEMAND_CITIES)} cities (49.25% population coverage)")

    def _split_date_range(self, start_date: str, end_date: str, max_days: int = 365) -> List[Tuple[str, str]]:
        """Split date range into chunks to respect API limits."""
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')

        chunks = []
        current = start

        while current <= end:
            chunk_end = min(current + timedelta(days=max_days - 1), end)
            chunks.append((
                current.strftime('%Y-%m-%d'),
                chunk_end.strftime('%Y-%m-%d')
            ))
            current = chunk_end + timedelta(days=1)

        return chunks

    def fetch_city_weather(
        self,
        city_name: str,
        start_date: str,
        end_date: str,
        use_forecast_api: bool = False,
        retry_attempts: int = 3,
        retry_delay: float = 5.0
    ) -> pd.DataFrame:
        """Fetch weather data for a single city with retry logic and chunking."""
        if city_name not in self.DEMAND_CITIES:
            raise ValueError(f"City {city_name} not found in DEMAND_CITIES")

        city_info = self.DEMAND_CITIES[city_name]
        date_chunks = self._split_date_range(start_date, end_date, max_days=365)

        logger.info(f"Fetching weather for {city_name} ({start_date} to {end_date})...")
        if len(date_chunks) > 1:
            logger.info(f"  Splitting into {len(date_chunks)} yearly chunks")

        all_data = []

        for chunk_idx, (chunk_start, chunk_end) in enumerate(date_chunks, 1):
            if len(date_chunks) > 1 and chunk_idx % 5 == 1:
                progress = (chunk_idx / len(date_chunks)) * 100
                logger.info(f"  Progress: {progress:.0f}% (chunk {chunk_idx}/{len(date_chunks)})")

            for attempt in range(1, retry_attempts + 1):
                try:
                    if use_forecast_api:
                        url = "https://api.open-meteo.com/v1/forecast"
                        past_days = (datetime.now().date() - datetime.strptime(chunk_start, '%Y-%m-%d').date()).days
                        params = {
                            "latitude": city_info['lat'],
                            "longitude": city_info['lon'],
                            "hourly": self.HOURLY_VARIABLES,
                            "timezone": "Europe/Istanbul",
                            "past_days": min(past_days, 92),
                        }
                    else:
                        url = "https://archive-api.open-meteo.com/v1/archive"
                        params = {
                            "latitude": city_info['lat'],
                            "longitude": city_info['lon'],
                            "start_date": chunk_start,
                            "end_date": chunk_end,
                            "hourly": self.HOURLY_VARIABLES,
                            "timezone": "Europe/Istanbul",
                        }

                    responses = self.client.weather_api(url, params=params)
                    response = responses[0]
                    hourly = response.Hourly()

                    time_range = pd.date_range(
                        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                        freq=pd.Timedelta(seconds=hourly.Interval()),
                        inclusive="left"
                    )
                    time_range = time_range.tz_convert('Europe/Istanbul')

                    data = {"datetime": time_range}
                    for i, var in enumerate(self.HOURLY_VARIABLES):
                        values = hourly.Variables(i).ValuesAsNumpy()
                        data[var] = values

                    df = pd.DataFrame(data)
                    df.set_index('datetime', inplace=True)
                    df['city'] = city_name
                    df['region'] = city_info['region']
                    df['pop_weight'] = city_info['pop_weight']

                    all_data.append(df)
                    break

                except Exception as e:
                    error_msg = str(e)[:200]
                    if attempt == retry_attempts:
                        logger.error(f"  Failed after {retry_attempts} attempts: {error_msg}")
                        raise

                    if attempt < retry_attempts:
                        wait_time = retry_delay * (2 ** (attempt - 1))
                        time.sleep(wait_time)

            if chunk_idx < len(date_chunks):
                time.sleep(2)

        if not all_data:
            logger.warning(f"No data collected for {city_name}")
            return pd.DataFrame()

        combined_df = pd.concat(all_data, ignore_index=False)
        logger.info(f"  ✓ Total fetched: {len(combined_df)} records for {city_name}")
        return combined_df
    
    def fetch_all_cities(
        self,
        start_date: str,
        end_date: str,
        cities: Optional[List[str]] = None,
        delay_between_requests: float = 7.0
    ) -> Dict[str, pd.DataFrame]:
        """Fetch weather data for all (or specified) cities."""
        if cities is None:
            cities = list(self.DEMAND_CITIES.keys())

        city_data = {}
        failed_cities = []

        days_ago = (datetime.now().date() - datetime.strptime(end_date, '%Y-%m-%d').date()).days
        use_forecast_api = days_ago < 5

        if use_forecast_api:
            logger.warning(f"End date is within 5-day delay. Using forecast API.")

        logger.info(f"Fetching {len(cities)} cities with {delay_between_requests}s delay between requests")

        for idx, city in enumerate(cities, 1):
            try:
                df = self.fetch_city_weather(city, start_date, end_date, use_forecast_api)
                city_data[city] = df

                if idx < len(cities):
                    time.sleep(delay_between_requests)

            except Exception as e:
                error_msg = str(e)[:200]
                logger.error(f"Skipping {city}: {error_msg}")
                failed_cities.append(city)

                if "rate limit" in str(e).lower() or "minutely" in str(e).lower():
                    logger.warning(f"Rate limit detected. Waiting 65s...")
                    time.sleep(65)

        if failed_cities:
            logger.warning(f"Failed cities: {', '.join(failed_cities)}")

        logger.info(f"✓ Fetched {len(city_data)}/{len(cities)} cities successfully")
        return city_data
    
    def create_national_features(self, city_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Aggregate city-level weather into national population-weighted features.
        
        Args:
            city_data: Dictionary of city weather DataFrames
            
        Returns:
            DataFrame with national-level weather features
        """
        logger.info("Creating national population-weighted features...")
        
        # Get time index from first city (all should be aligned)
        first_city = list(city_data.values())[0]
        time_index = first_city.index
        
        # Initialize national features
        national = pd.DataFrame(index=time_index)
        
        # Calculate population-weighted averages
        national['temp_national'] = sum(
            city_data[city]['temperature_2m'] * self.DEMAND_CITIES[city]['pop_weight']
            for city in city_data.keys()
        )
        
        national['humidity_national'] = sum(
            city_data[city]['relative_humidity_2m'] * self.DEMAND_CITIES[city]['pop_weight']
            for city in city_data.keys()
        )
        
        national['wind_speed_national'] = sum(
            city_data[city]['wind_speed_10m'] * self.DEMAND_CITIES[city]['pop_weight']
            for city in city_data.keys()
        )
        
        national['apparent_temp_national'] = sum(
            city_data[city]['apparent_temperature'] * self.DEMAND_CITIES[city]['pop_weight']
            for city in city_data.keys()
        )
        
        # Precipitation (sum makes sense - any city having rain affects demand)
        national['precipitation_national'] = sum(
            city_data[city]['precipitation'] * self.DEMAND_CITIES[city]['pop_weight']
            for city in city_data.keys()
        )
        
        # Cloud cover (affects lighting, industrial/commercial electricity use)
        national['cloud_cover_national'] = sum(
            city_data[city]['cloud_cover'] * self.DEMAND_CITIES[city]['pop_weight']
            for city in city_data.keys()
        )
        
        # Regional temperature spread (measure of geographic diversity)
        temps = pd.DataFrame({
            city: city_data[city]['temperature_2m'] 
            for city in city_data.keys()
        })
        national['temp_std'] = temps.std(axis=1)  # Standard deviation across cities
        
        logger.info(f"✓ Created {len(national.columns)} national weather features")
        
        return national
    
    def create_demand_features(self, national_weather: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer demand-relevant features from national weather.
        
        Features include:
        - Heating/Cooling Degree Days (HDD/CDD)
        - Heat Index (apparent temperature)
        - Temperature lags and rolling means
        - Extreme temperature flags
        - Weather momentum (temperature changes)
        
        Args:
            national_weather: DataFrame with national weather aggregates
            
        Returns:
            DataFrame with engineered demand features
        """
        logger.info("Engineering demand-side features...")
        
        df = national_weather.copy()
        
        # ═══════════════════════════════════════
        # DEGREE DAYS (Primary demand drivers)
        # ═══════════════════════════════════════
        
        # Heating Degree Days (base 18°C - Turkish building codes)
        df['HDD'] = np.maximum(18 - df['temp_national'], 0)
        
        # Cooling Degree Days (base 18°C)
        df['CDD'] = np.maximum(df['temp_national'] - 18, 0)
        
        # Alternative base temperatures (for sensitivity analysis)
        df['HDD_15'] = np.maximum(15 - df['temp_national'], 0)
        df['CDD_21'] = np.maximum(df['temp_national'] - 21, 0)
        
        # ═══════════════════════════════════════
        # HEAT INDEX & COMFORT
        # ═══════════════════════════════════════
        
        # Heat index (feels-like temperature) - important for AC demand
        df['heat_index'] = self._calculate_heat_index(
            df['temp_national'], 
            df['humidity_national']
        )
        
        # Wind chill index (feels-like in cold) - important for heating demand
        df['wind_chill'] = self._calculate_wind_chill(
            df['temp_national'],
            df['wind_speed_national']
        )
        
        # ═══════════════════════════════════════
        # EXTREME TEMPERATURE FLAGS
        # ═══════════════════════════════════════
        
        df['is_hot'] = (df['temp_national'] > 30).astype(int)  # AC surge threshold
        df['is_very_hot'] = (df['temp_national'] > 35).astype(int)  # Extreme AC
        df['is_cold'] = (df['temp_national'] < 5).astype(int)  # Heating surge
        df['is_very_cold'] = (df['temp_national'] < 0).astype(int)  # Extreme heating
        
        # ═══════════════════════════════════════
        # TEMPERATURE LAGS
        # ═══════════════════════════════════════
        
        df['temp_lag_1h'] = df['temp_national'].shift(1)
        df['temp_lag_2h'] = df['temp_national'].shift(2)
        df['temp_lag_3h'] = df['temp_national'].shift(3)
        df['temp_lag_24h'] = df['temp_national'].shift(24)
        df['temp_lag_168h'] = df['temp_national'].shift(168)  # Weekly
        
        # ═══════════════════════════════════════
        # ROLLING STATISTICS
        # ═══════════════════════════════════════
        
        # 24-hour rolling mean (daily average)
        df['temp_rolling_24h'] = df['temp_national'].rolling(window=24, min_periods=12).mean()
        
        # 7-day rolling mean (weekly trend)
        df['temp_rolling_7d'] = df['temp_national'].rolling(window=168, min_periods=84).mean()
        
        # 24-hour rolling standard deviation (daily volatility)
        df['temp_std_24h'] = df['temp_national'].rolling(window=24, min_periods=12).std()
        
        # ═══════════════════════════════════════
        # TEMPERATURE MOMENTUM (Changes)
        # ═══════════════════════════════════════
        
        df['temp_change_1h'] = df['temp_national'].diff(1)
        df['temp_change_3h'] = df['temp_national'].diff(3)
        df['temp_change_24h'] = df['temp_national'].diff(24)
        
        # Rapid temperature drops/rises (shock to demand)
        df['temp_shock'] = (np.abs(df['temp_change_3h']) > 5).astype(int)
        
        # ═══════════════════════════════════════
        # CALENDAR FEATURES
        # ═══════════════════════════════════════
        
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_year'] = df.index.dayofyear
        df['month'] = df.index.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Cyclical encoding for hour (important for neural networks)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Cyclical encoding for day of week
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # ═══════════════════════════════════════
        # WEATHER PATTERN FEATURES
        # ═══════════════════════════════════════
        
        # Precipitation flag (affects indoor activity)
        df['is_raining'] = (df['precipitation_national'] > 0.1).astype(int)
        df['is_heavy_rain'] = (df['precipitation_national'] > 5.0).astype(int)
        
        # Cloud cover (affects natural lighting → industrial/commercial load)
        df['is_cloudy'] = (df['cloud_cover_national'] > 70).astype(int)
        
        logger.info(f"✓ Created {len(df.columns)} total features")
        
        return df
    
    @staticmethod
    def _calculate_heat_index(temp_c: pd.Series, humidity: pd.Series) -> pd.Series:
        """
        Calculate heat index (feels-like temperature in hot weather).
        Based on Steadman formula adapted for Celsius.
        
        Only applies when temp > 27°C, otherwise returns actual temperature.
        """
        # Simplified Rothfusz regression (converted to Celsius)
        c1 = -8.78469475556
        c2 = 1.61139411
        c3 = 2.33854883889
        c4 = -0.14611605
        c5 = -0.012308094
        c6 = -0.0164248277778
        c7 = 0.002211732
        c8 = 0.00072546
        c9 = -0.000003582
        
        T = temp_c
        R = humidity
        
        HI = (c1 + c2*T + c3*R + c4*T*R + c5*T**2 + 
              c6*R**2 + c7*T**2*R + c8*T*R**2 + c9*T**2*R**2)
        
        # Only apply when hot (temp > 27°C)
        heat_index = np.where(temp_c > 27, HI, temp_c)
        
        return pd.Series(heat_index, index=temp_c.index)
    
    @staticmethod
    def _calculate_wind_chill(temp_c: pd.Series, wind_kmh: pd.Series) -> pd.Series:
        """
        Calculate wind chill index (feels-like temperature in cold weather).
        Based on North American wind chill formula.
        
        Only applies when temp < 10°C and wind > 4.8 km/h.
        """
        # Convert wind to km/h if needed (input should already be km/h from Open-Meteo)
        V = wind_kmh
        T = temp_c
        
        # Wind chill formula (Environment Canada)
        WC = (13.12 + 0.6215*T - 11.37*V**0.16 + 0.3965*T*V**0.16)
        
        # Only apply when cold and windy
        wind_chill = np.where((temp_c < 10) & (wind_kmh > 4.8), WC, temp_c)
        
        return pd.Series(wind_chill, index=temp_c.index)
    
    def save_to_parquet(
        self, 
        df: pd.DataFrame, 
        filepath: str,
        layer: str = 'silver'
    ):
        """
        Save DataFrame to parquet file (medallion architecture).
        
        Args:
            df: DataFrame to save
            filepath: Output file path
            layer: Data layer ('bronze', 'silver', or 'gold')
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_parquet(filepath, engine='pyarrow', compression='snappy')
        logger.info(f"✓ Saved {layer} layer data to {filepath} ({len(df)} rows)")
    
    def run_pipeline(
        self,
        start_date: str,
        end_date: str,
        output_dir: str = './data',
        delay_between_requests: float = 7.0,
        retry_failed: bool = True
    ) -> pd.DataFrame:
        """
        Run the complete demand weather pipeline: fetch → aggregate → engineer features.

        Args:
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
            output_dir: Output directory for parquet files
            delay_between_requests: Seconds to wait between API calls (default 7s)
            retry_failed: If True, retry failed cities once after 65s delay

        Returns:
            DataFrame with engineered demand features
        """
        output_path = Path(output_dir)

        logger.info("="*60)
        logger.info("DEMAND WEATHER PIPELINE - Starting")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info("="*60)

        # Step 1: Fetch raw city data (BRONZE layer)
        city_data = self.fetch_all_cities(start_date, end_date, delay_between_requests=delay_between_requests)

        # Step 1.5: Retry failed cities if enabled
        if retry_failed:
            all_cities = set(self.DEMAND_CITIES.keys())
            fetched_cities = set(city_data.keys())
            failed_cities = all_cities - fetched_cities

            if failed_cities:
                logger.info(f"Retrying {len(failed_cities)} failed cities after 65s delay...")
                time.sleep(65)  # Wait for rate limit to reset

                retry_data = self.fetch_all_cities(
                    start_date,
                    end_date,
                    cities=list(failed_cities),
                    delay_between_requests=delay_between_requests
                )

                # Merge retry results into main dataset
                city_data.update(retry_data)
                logger.info(f"After retry: {len(city_data)}/{len(all_cities)} cities fetched successfully")

        # Save bronze layer
        bronze_dir = output_path / 'bronze' / 'demand_weather'
        for city_name, df in city_data.items():
            self.save_to_parquet(
                df,
                bronze_dir / f"{city_name.lower()}_{start_date}_{end_date}.parquet",
                layer='bronze'
            )
        
        # Step 2: Aggregate to national level (SILVER layer)
        national_weather = self.create_national_features(city_data)
        
        # Save silver layer
        silver_path = output_path / 'silver' / 'demand_weather' / f'national_{start_date}_{end_date}.parquet'
        self.save_to_parquet(national_weather, silver_path, layer='silver')
        
        # Step 3: Engineer demand features (GOLD layer)
        demand_features = self.create_demand_features(national_weather)
        
        # Save gold layer
        gold_path = output_path / 'gold' / 'demand_features' / f'demand_features_{start_date}_{end_date}.parquet'
        self.save_to_parquet(demand_features, gold_path, layer='gold')
        
        logger.info("="*60)
        logger.info("DEMAND WEATHER PIPELINE - Completed Successfully")
        logger.info(f"Output: {gold_path}")
        logger.info("="*60)
        
        return demand_features


# ═══════════════════════════════════════════════════════════
# EXAMPLE USAGE & TESTING
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Example usage of the Demand Weather Pipeline
    """
    
    # Initialize fetcher
    fetcher = DemandWeatherFetcher(cache_dir='.cache')
    
    # Example 1: Fetch recent 2 weeks for testing
    print("\n" + "="*60)
    print("EXAMPLE 1: Fetching Recent Data (2 weeks)")
    print("="*60)
    
    recent_features = fetcher.run_pipeline(
        start_date='2024-10-01',
        end_date='2024-10-14',
        output_dir='./data'
    )
    
    print("\nSample of engineered features:")
    print(recent_features.tail(24))  # Last 24 hours
    
    print("\nFeature summary:")
    print(recent_features.describe())
    
    # Example 2: Fetch historical training data (2020-2024)
    print("\n" + "="*60)
    print("EXAMPLE 2: Fetching Historical Training Data")
    print("="*60)
    
    historical_features = fetcher.run_pipeline(
        start_date='2020-01-01',
        end_date='2024-09-30',
        output_dir='./data'
    )
    
    print(f"\n✓ Historical data: {len(historical_features)} hourly records")
    print(f"  Date range: {historical_features.index.min()} to {historical_features.index.max()}")
    print(f"  Features: {len(historical_features.columns)} columns")
    
    # Example 3: Check for missing values
    print("\n" + "="*60)
    print("DATA QUALITY CHECK")
    print("="*60)
    
    missing = historical_features.isnull().sum()
    missing_pct = (missing / len(historical_features) * 100).round(2)
    
    print("\nMissing values by feature:")
    for col in missing[missing > 0].index:
        print(f"  {col}: {missing[col]} ({missing_pct[col]}%)")
    
    if missing.sum() == 0:
        print("  ✓ No missing values!")
    
    # Example 4: Visualize key features (if matplotlib available)
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # Plot last 7 days
        recent_week = historical_features.last('7D')
        
        # Temperature and degree days
        axes[0].plot(recent_week.index, recent_week['temp_national'], label='Temperature', linewidth=2)
        axes[0].plot(recent_week.index, recent_week['heat_index'], label='Heat Index', linestyle='--')
        axes[0].set_ylabel('Temperature (°C)')
        axes[0].legend()
        axes[0].set_title('Temperature Metrics (Last 7 Days)')
        axes[0].grid(True, alpha=0.3)
        
        # HDD and CDD
        axes[1].fill_between(recent_week.index, 0, recent_week['HDD'], label='HDD', alpha=0.5)
        axes[1].fill_between(recent_week.index, 0, recent_week['CDD'], label='CDD', alpha=0.5)
        axes[1].set_ylabel('Degree Days')
        axes[1].legend()
        axes[1].set_title('Heating/Cooling Degree Days')
        axes[1].grid(True, alpha=0.3)
        
        # Weather patterns
        axes[2].plot(recent_week.index, recent_week['humidity_national'], label='Humidity (%)', linewidth=2)
        axes[2].plot(recent_week.index, recent_week['cloud_cover_national'], label='Cloud Cover (%)', linewidth=2)
        axes[2].set_ylabel('Percentage (%)')
        axes[2].set_xlabel('Date')
        axes[2].legend()
        axes[2].set_title('Weather Patterns')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('./demand_weather_visualization.png', dpi=150)
        print("\n✓ Visualization saved to: demand_weather_visualization.png")
        
    except ImportError:
        print("\nNote: Install matplotlib for visualizations: pip install matplotlib")
