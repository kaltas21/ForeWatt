"""
ForeWatt API - Cloud Run Entry Point
=====================================
FastAPI application for forecast generation and retrieval.

Endpoints:
- POST /forecast - Trigger hourly forecast
- GET /forecast/price - Get price forecasts
- GET /forecast/consumption - Get consumption forecasts
- GET /forecast/latest - Get latest 24h forecasts
- GET /api/realtime/{model} - Real-time data from Firestore
- GET /api/historical/{model} - Historical data from Parquet
- GET /api/anomaly/{model} - Anomaly detection
- GET /api/alerts - Alert management
- GET /health - Health check

Designed for Cloud Run with Cloud Scheduler triggering /forecast every hour.
"""

import os
import logging
import functools
import hashlib
import time
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel

# Google Cloud Firestore
try:
    from google.cloud import firestore
    FIRESTORE_AVAILABLE = True
except ImportError:
    FIRESTORE_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="ForeWatt Forecast API",
    description="Electricity price and consumption forecasting API",
    version="1.0.0"
)

# CORS for dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GZIP compression for faster response delivery (80-90% size reduction)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Lazy load pipeline to avoid cold start issues
_pipeline = None
_firestore_client = None


def get_pipeline():
    """Lazy load the forecast pipeline."""
    global _pipeline
    if _pipeline is None:
        from src.pipeline import ForecastPipeline
        _pipeline = ForecastPipeline()
    return _pipeline


def get_firestore():
    """Lazy load Firestore client."""
    global _firestore_client
    if _firestore_client is None and FIRESTORE_AVAILABLE:
        _firestore_client = firestore.Client(project="forewatt-483109")
    return _firestore_client


def save_to_firestore(forecast_type: str, forecasts: list, forecast_time: datetime):
    """Save forecasts to Firestore for real-time access."""
    db = get_firestore()
    if not db:
        logger.warning("Firestore not available, skipping save")
        return

    try:
        # Save to forecasts/latest/{type}
        doc_ref = db.collection("forecasts").document("latest").collection(forecast_type).document("current")
        doc_ref.set({
            "forecast_time": forecast_time.isoformat(),
            "updated_at": datetime.now().isoformat(),
            "data": forecasts
        })

        # Also save to history for the specific hour
        hour_key = forecast_time.strftime("%Y-%m-%d_%H")
        history_ref = db.collection("forecast_history").document(forecast_type).collection(hour_key).document("forecast")
        history_ref.set({
            "forecast_time": forecast_time.isoformat(),
            "data": forecasts
        })

        logger.info(f"Saved {len(forecasts)} {forecast_type} forecasts to Firestore")
    except Exception as e:
        logger.error(f"Failed to save to Firestore: {e}")


# =============================================================================
# Request/Response Models
# =============================================================================

class ForecastResponse(BaseModel):
    forecast_time: str
    runtime_seconds: float
    price: dict
    consumption: dict


class ForecastData(BaseModel):
    forecast_time: str
    target_time: str
    forecast_value: float
    model_version: str


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )


@app.get("/debug/epias")
async def debug_epias():
    """Debug endpoint to check EPIAS connectivity and credentials."""
    import os
    result = {
        "timestamp": datetime.now().isoformat(),
        "epias_username_set": bool(os.getenv('EPTR_USERNAME')),
        "epias_password_set": bool(os.getenv('EPTR_PASSWORD')),
        "fetcher_available": False,
        "test_fetch": None,
        "error": None
    }

    try:
        fetcher = get_epias_fetcher()
        result["fetcher_available"] = fetcher is not None

        if fetcher:
            # Try to fetch yesterday's price data as a test
            yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            today = datetime.now().strftime('%Y-%m-%d')

            df = fetcher.fetch_dataset('price_ptf', yesterday, today)
            if df is not None and not df.empty:
                result["test_fetch"] = {
                    "status": "success",
                    "rows": len(df),
                    "columns": list(df.columns),
                    "sample_row": df.iloc[0].to_dict() if len(df) > 0 else None
                }
            else:
                result["test_fetch"] = {"status": "empty", "rows": 0}
    except Exception as e:
        result["error"] = str(e)

    return result


@app.post("/forecast", response_model=ForecastResponse)
async def run_forecast(forecast_time: Optional[str] = None):
    """
    Trigger forecast generation.

    Called by Cloud Scheduler every hour.

    Args:
        forecast_time: Optional ISO format time (defaults to now)
    """
    try:
        pipeline = get_pipeline()

        if forecast_time:
            ft = datetime.fromisoformat(forecast_time)
        else:
            ft = None

        results = pipeline.run(ft)

        # Save to Firestore for real-time access
        ft_parsed = datetime.fromisoformat(results['forecast_time'])

        # Get the forecast data from storage
        price_df = pipeline.storage.get_latest_forecast('price')
        consumption_df = pipeline.storage.get_latest_forecast('consumption')

        if not price_df.empty:
            price_data = price_df.to_dict(orient='records')
            # Convert timestamps to strings for Firestore
            for row in price_data:
                row['forecast_time'] = str(row['forecast_time'])
                row['target_time'] = str(row['target_time'])
            save_to_firestore('price', price_data, ft_parsed)

        if not consumption_df.empty:
            consumption_data = consumption_df.to_dict(orient='records')
            for row in consumption_data:
                row['forecast_time'] = str(row['forecast_time'])
                row['target_time'] = str(row['target_time'])
            save_to_firestore('consumption', consumption_data, ft_parsed)

        return ForecastResponse(**results)

    except Exception as e:
        logger.error(f"Forecast failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/forecast/latest")
async def get_latest_forecasts():
    """Get the latest 24-hour forecasts for both price and consumption."""
    try:
        pipeline = get_pipeline()

        price_df = pipeline.storage.get_latest_forecast('price')
        consumption_df = pipeline.storage.get_latest_forecast('consumption')

        result = {
            'price': [],
            'consumption': [],
            'forecast_time': None,
        }

        if not price_df.empty:
            result['forecast_time'] = price_df['forecast_time'].iloc[0].isoformat()
            result['price'] = price_df.to_dict(orient='records')

        if not consumption_df.empty:
            result['consumption'] = consumption_df.to_dict(orient='records')

        return result

    except Exception as e:
        logger.error(f"Failed to get latest forecasts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/forecast/price")
async def get_price_forecasts(
    start_date: Optional[str] = Query(None, description="Start date (ISO format)"),
    end_date: Optional[str] = Query(None, description="End date (ISO format)"),
    limit: int = Query(1000, description="Max rows to return")
):
    """
    Get price forecasts.

    Args:
        start_date: Filter from this date
        end_date: Filter until this date
        limit: Maximum rows
    """
    try:
        pipeline = get_pipeline()

        # Parse dates and make timezone-naive for comparison
        start = None
        end = None
        if start_date:
            start = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            if start.tzinfo:
                start = start.replace(tzinfo=None)
        if end_date:
            end = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            if end.tzinfo:
                end = end.replace(tzinfo=None)

        df = pipeline.storage.load_forecasts('price', start, end)

        if df.empty:
            return {'data': [], 'count': 0}

        # Apply limit
        df = df.head(limit)

        # Convert to serializable format
        df['forecast_time'] = df['forecast_time'].astype(str)
        df['target_time'] = df['target_time'].astype(str)

        return {
            'data': df.to_dict(orient='records'),
            'count': len(df)
        }

    except Exception as e:
        logger.error(f"Failed to get price forecasts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/forecast/consumption")
async def get_consumption_forecasts(
    start_date: Optional[str] = Query(None, description="Start date (ISO format)"),
    end_date: Optional[str] = Query(None, description="End date (ISO format)"),
    limit: int = Query(1000, description="Max rows to return")
):
    """
    Get consumption forecasts.

    Args:
        start_date: Filter from this date
        end_date: Filter until this date
        limit: Maximum rows
    """
    try:
        pipeline = get_pipeline()

        # Parse dates and make timezone-naive for comparison
        start = None
        end = None
        if start_date:
            start = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            if start.tzinfo:
                start = start.replace(tzinfo=None)
        if end_date:
            end = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            if end.tzinfo:
                end = end.replace(tzinfo=None)

        df = pipeline.storage.load_forecasts('consumption', start, end)

        if df.empty:
            return {'data': [], 'count': 0}

        # Apply limit
        df = df.head(limit)

        # Convert to serializable format
        df['forecast_time'] = df['forecast_time'].astype(str)
        df['target_time'] = df['target_time'].astype(str)

        return {
            'data': df.to_dict(orient='records'),
            'count': len(df)
        }

    except Exception as e:
        logger.error(f"Failed to get consumption forecasts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/forecast/history/{target_hour}")
async def get_forecast_history(
    target_hour: str,
    forecast_type: str = Query("price", description="'price' or 'consumption'")
):
    """
    Get all forecasts that predicted a specific target hour.

    Useful for analyzing forecast accuracy.

    Args:
        target_hour: The target hour to look up (ISO format)
        forecast_type: 'price' or 'consumption'
    """
    try:
        pipeline = get_pipeline()

        target = datetime.fromisoformat(target_hour)
        df = pipeline.storage.get_forecast_history(forecast_type, target)

        if df.empty:
            return {'data': [], 'count': 0}

        df['forecast_time'] = df['forecast_time'].astype(str)
        df['target_time'] = df['target_time'].astype(str)

        return {
            'data': df.to_dict(orient='records'),
            'count': len(df)
        }

    except Exception as e:
        logger.error(f"Failed to get forecast history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/storage/stats")
async def get_storage_stats():
    """Get storage statistics."""
    try:
        pipeline = get_pipeline()
        return pipeline.storage.get_storage_stats()
    except Exception as e:
        logger.error(f"Failed to get storage stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Historical Actuals Endpoints (EPIAS Data)
# =============================================================================

import pandas as pd
from pathlib import Path

# Cache for historical data
_historical_data = None
_epias_fetcher = None
_epias_cache = {}  # Cache for EPIAS API data

# Query result cache with TTL (30 minutes)
_query_cache: Dict[str, Dict[str, Any]] = {}
_CACHE_TTL_SECONDS = 1800  # 30 minutes


def get_cache_key(prefix: str, **kwargs) -> str:
    """Generate a cache key from prefix and kwargs."""
    key_str = f"{prefix}:" + ":".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
    return hashlib.md5(key_str.encode()).hexdigest()


def get_cached_result(cache_key: str) -> Optional[Any]:
    """Get cached result if still valid."""
    if cache_key in _query_cache:
        entry = _query_cache[cache_key]
        if time.time() - entry['timestamp'] < _CACHE_TTL_SECONDS:
            logger.info(f"Cache hit for {cache_key[:16]}...")
            return entry['data']
        else:
            del _query_cache[cache_key]
    return None


def set_cached_result(cache_key: str, data: Any):
    """Cache a result with timestamp."""
    _query_cache[cache_key] = {
        'timestamp': time.time(),
        'data': data
    }
    # Limit cache size to 100 entries
    if len(_query_cache) > 100:
        oldest_key = min(_query_cache.keys(), key=lambda k: _query_cache[k]['timestamp'])
        del _query_cache[oldest_key]

# Lazy-loaded forecasters for model inference
_price_forecaster = None
_consumption_forecaster = None


def get_price_forecaster():
    """Lazy load price forecaster model."""
    global _price_forecaster
    if _price_forecaster is None:
        try:
            from src.pipeline.price_inference import PriceForecaster
            _price_forecaster = PriceForecaster()
            logger.info("Price forecaster loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load price forecaster: {e}")
            _price_forecaster = False  # Mark as unavailable
    return _price_forecaster if _price_forecaster else None


def get_consumption_forecaster():
    """Lazy load consumption forecaster model."""
    global _consumption_forecaster
    if _consumption_forecaster is None:
        try:
            from src.pipeline.consumption_inference import ConsumptionForecaster
            _consumption_forecaster = ConsumptionForecaster()
            logger.info("Consumption forecaster loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load consumption forecaster: {e}")
            _consumption_forecaster = False  # Mark as unavailable
    return _consumption_forecaster if _consumption_forecaster else None


def generate_simple_forecasts(forecast_type: str, timestamps: pd.Series, actual_data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate simple forecasts for timestamps using hourly patterns from actual data.

    This is used when full model inference isn't possible (e.g., recent dates beyond parquet).
    Uses hourly averages from actual data to create pattern-based forecasts.

    Args:
        forecast_type: 'price' or 'consumption'
        timestamps: Series of timestamps to forecast
        actual_data: DataFrame with 'timestamp' and value column (price/consumption)

    Returns:
        DataFrame with 'timestamp' and 'forecast' columns
    """
    if actual_data.empty:
        return pd.DataFrame()

    value_col = 'price' if forecast_type == 'price' else 'consumption'

    # Ensure we have the value column
    if value_col not in actual_data.columns:
        logger.warning(f"Column {value_col} not found in actual data")
        return pd.DataFrame()

    # Calculate hourly patterns from actual data
    actual_data = actual_data.copy()
    actual_data['hour'] = pd.to_datetime(actual_data['timestamp']).dt.hour

    hourly_avg = actual_data.groupby('hour')[value_col].mean().to_dict()
    overall_avg = actual_data[value_col].mean()

    # Fill missing hours with overall average
    for h in range(24):
        if h not in hourly_avg:
            hourly_avg[h] = overall_avg

    # Generate forecasts based on hourly patterns
    forecasts = []
    timestamps_dt = pd.to_datetime(timestamps)

    for ts in timestamps_dt:
        hour = ts.hour
        base_value = hourly_avg.get(hour, overall_avg)

        # Add slight variation based on day of week
        dow = ts.weekday()
        if dow >= 5:  # Weekend
            if forecast_type == 'price':
                base_value *= 0.95  # Prices slightly lower on weekends
            else:
                base_value *= 0.92  # Consumption lower on weekends

        forecasts.append({
            'timestamp': ts,
            'forecast': float(base_value)
        })

    result = pd.DataFrame(forecasts)
    logger.info(f"Generated {len(result)} simple {forecast_type} forecasts from hourly patterns")
    return result


def generate_model_forecasts(forecast_type: str, timestamps: pd.Series) -> pd.DataFrame:
    """
    Generate model forecasts for given timestamps using the trained models.

    Uses the full feature set from the master parquet file.

    Args:
        forecast_type: 'price' or 'consumption'
        timestamps: Series of timestamps to forecast

    Returns:
        DataFrame with 'timestamp' and 'forecast' columns
    """
    # Load full feature data from parquet
    master_df = get_historical_data()
    if master_df.empty:
        logger.warning("No master data available for forecasting")
        return pd.DataFrame()

    # Ensure timestamps are timezone-naive for comparison
    if master_df['timestamp'].dt.tz is not None:
        master_df = master_df.copy()
        master_df['timestamp'] = master_df['timestamp'].dt.tz_localize(None)

    # Filter to requested timestamps
    timestamps_naive = pd.to_datetime(timestamps)
    if hasattr(timestamps_naive, 'dt') and timestamps_naive.dt.tz is not None:
        timestamps_naive = timestamps_naive.dt.tz_localize(None)

    # Find matching rows in master data
    forecast_df = master_df[master_df['timestamp'].isin(timestamps_naive)].copy()

    if forecast_df.empty:
        logger.info(f"No matching timestamps in master data for forecasting")
        return pd.DataFrame()

    logger.info(f"Generating {forecast_type} forecasts for {len(forecast_df)} timestamps")

    try:
        if forecast_type == 'price':
            forecaster = get_price_forecaster()
            if forecaster is None:
                return pd.DataFrame()

            # Set index for forecaster (it expects DatetimeIndex)
            forecast_df = forecast_df.set_index('timestamp')

            # Generate predictions (without error correction for historical)
            predictions, _ = forecaster.predict(forecast_df, apply_correction=False)

            result = pd.DataFrame({
                'timestamp': forecast_df.index,
                'forecast': predictions
            }).reset_index(drop=True)

        else:  # consumption
            forecaster = get_consumption_forecaster()
            if forecaster is None:
                return pd.DataFrame()

            # Generate predictions
            predictions = forecaster.predict(forecast_df)

            result = pd.DataFrame({
                'timestamp': forecast_df['timestamp'],
                'forecast': predictions
            })

        logger.info(f"Generated {len(result)} {forecast_type} forecasts")
        return result

    except Exception as e:
        logger.error(f"Failed to generate {forecast_type} forecasts: {e}", exc_info=True)
        return pd.DataFrame()

def make_timestamps_naive(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure timestamp column is timezone-naive."""
    if df.empty or 'timestamp' not in df.columns:
        return df
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    if df['timestamp'].dt.tz is not None:
        # Remove timezone info (convert to naive)
        df['timestamp'] = df['timestamp'].dt.tz_localize(None)
    return df


def get_historical_data(force_refresh: bool = False):
    """Load historical EPIAS data from GCS (with local fallback)."""
    global _historical_data

    if _historical_data is None or force_refresh:
        try:
            # Try GCS first
            from src.pipeline.gcs_storage import load_master_from_gcs
            _historical_data = load_master_from_gcs(force_refresh=force_refresh)
            if not _historical_data.empty:
                _historical_data = make_timestamps_naive(_historical_data)
                logger.info(f"Loaded {len(_historical_data)} records from GCS")
                return _historical_data
        except Exception as e:
            logger.warning(f"GCS load failed: {e}, falling back to local file")

        # Fallback to local bundled file
        data_path = Path(__file__).parent / 'data' / 'gold' / 'master' / 'master_v2_fundamental.parquet'
        if data_path.exists():
            _historical_data = pd.read_parquet(data_path)
            _historical_data = make_timestamps_naive(_historical_data)
            logger.info(f"Loaded {len(_historical_data)} records from local file")
        else:
            logger.warning(f"Historical data not found at {data_path}")
            _historical_data = pd.DataFrame()

    return _historical_data


def get_epias_fetcher():
    """Lazy load EPIAS fetcher (uses credentials from environment/Secret Manager)."""
    global _epias_fetcher
    if _epias_fetcher is None:
        try:
            from src.data.epias_fetcher import EpiasDataFetcher
            _epias_fetcher = EpiasDataFetcher()
            logger.info("EPIAS fetcher initialized successfully")
        except Exception as e:
            logger.warning(f"EPIAS fetcher not available: {e}")
            _epias_fetcher = False  # Mark as unavailable
    return _epias_fetcher if _epias_fetcher else None


def fetch_epias_live_data(data_type: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch live data from EPIAS API.

    Args:
        data_type: 'price' or 'consumption'
        start_date: Start date 'YYYY-MM-DD'
        end_date: End date 'YYYY-MM-DD'

    Returns:
        DataFrame with timestamp, price/consumption columns
    """
    global _epias_cache

    cache_key = f"{data_type}_{start_date}_{end_date}"

    # Check cache first (valid for 5 minutes)
    if cache_key in _epias_cache:
        cached_time, cached_data = _epias_cache[cache_key]
        if (datetime.now() - cached_time).seconds < 300:
            logger.info(f"Using cached EPIAS data for {cache_key}")
            return cached_data

    fetcher = get_epias_fetcher()
    if not fetcher:
        logger.warning("EPIAS fetcher not available - credentials may be missing")
        return pd.DataFrame()

    try:
        # Map data_type to EPIAS dataset
        if data_type == 'price':
            dataset = 'price_ptf'
            value_col = 'price'
        else:
            dataset = 'consumption_actual'
            value_col = 'consumption'

        logger.info(f"Fetching {dataset} from EPIAS: {start_date} to {end_date}")
        df = fetcher.fetch_dataset(dataset, start_date, end_date)

        if df is None or df.empty:
            logger.warning(f"No data returned from EPIAS for {dataset}")
            return pd.DataFrame()

        logger.info(f"EPIAS returned {len(df)} rows with columns: {list(df.columns)}")

        # Standardize columns based on EPIAS response
        df_result = pd.DataFrame()

        # Find timestamp column - EPIAS uses various names
        ts_cols = [c for c in df.columns if any(kw in c.lower() for kw in ['date', 'time', 'tarih', 'saat'])]
        if ts_cols:
            logger.info(f"Found timestamp column: {ts_cols[0]}")
            df_result['timestamp'] = pd.to_datetime(df[ts_cols[0]])
        else:
            logger.warning(f"No timestamp column found in: {list(df.columns)}")
            return pd.DataFrame()

        # Find value column based on data type
        if data_type == 'price':
            # For price_ptf (MCP), eptr2 returns columns like 'marketClearingPrice' or 'price'
            val_cols = [c for c in df.columns if any(kw in c.lower() for kw in ['price', 'fiyat', 'ptf', 'mcp', 'clearing'])]
        else:
            # For consumption_actual (rt-cons), eptr2 returns 'consumption' or similar
            val_cols = [c for c in df.columns if any(kw in c.lower() for kw in ['consumption', 'tüketim', 'load', 'demand', 'value'])]

        if val_cols:
            logger.info(f"Found value column: {val_cols[0]}")
            df_result[value_col] = pd.to_numeric(df[val_cols[0]], errors='coerce')
        else:
            logger.warning(f"No value column found for {data_type} in: {list(df.columns)}")
            return pd.DataFrame()

        if df_result.empty or 'timestamp' not in df_result.columns:
            logger.warning("Could not parse EPIAS response - result is empty")
            return pd.DataFrame()

        # Remove timezone for consistency (convert to naive datetime)
        if df_result['timestamp'].dt.tz is not None:
            df_result['timestamp'] = df_result['timestamp'].dt.tz_convert('Europe/Istanbul').dt.tz_localize(None)

        # Drop any rows with NaN values
        df_result = df_result.dropna()

        # Sort by timestamp
        df_result = df_result.sort_values('timestamp').reset_index(drop=True)

        # Cache the result
        _epias_cache[cache_key] = (datetime.now(), df_result)

        logger.info(f"Successfully fetched {len(df_result)} records from EPIAS for {data_type}")
        return df_result

    except Exception as e:
        logger.error(f"Failed to fetch from EPIAS: {e}", exc_info=True)
        return pd.DataFrame()


def ensure_tz_naive(dt) -> datetime:
    """Ensure datetime is timezone-naive."""
    if dt is None:
        return dt
    if hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
        # Convert to UTC then remove timezone
        if hasattr(dt, 'tz_convert'):
            return dt.tz_convert('UTC').replace(tzinfo=None)
        return dt.replace(tzinfo=None)
    return dt


def get_combined_actual_data(data_type: str, start: datetime, end: datetime) -> pd.DataFrame:
    """
    Get actual data combining parquet (historical) and EPIAS API (recent).

    For dates within parquet: use parquet data
    For dates beyond parquet: fetch from EPIAS API
    """
    value_col = 'price_real' if data_type == 'price' else 'consumption'
    output_col = 'price' if data_type == 'price' else 'consumption'

    # Ensure input datetimes are timezone-naive
    start = ensure_tz_naive(start)
    end = ensure_tz_naive(end)

    # Load parquet data
    parquet_df = get_historical_data()

    if parquet_df.empty:
        # No parquet data, try EPIAS only
        epias_df = fetch_epias_live_data(
            data_type,
            start.strftime('%Y-%m-%d'),
            end.strftime('%Y-%m-%d')
        )
        if not epias_df.empty:
            # Ensure timezone-naive
            if epias_df['timestamp'].dt.tz is not None:
                epias_df['timestamp'] = epias_df['timestamp'].dt.tz_localize(None)
            # Filter by exact time range (EPIAS returns full days)
            filtered = epias_df[
                (epias_df['timestamp'] >= start) &
                (epias_df['timestamp'] <= end)
            ][['timestamp', output_col]].copy()
            return filtered
        return pd.DataFrame()

    # Make a copy to avoid modifying cached data
    parquet_df = parquet_df.copy()

    # Ensure timestamps are timezone-naive for comparison
    if parquet_df['timestamp'].dt.tz is not None:
        parquet_df['timestamp'] = parquet_df['timestamp'].dt.tz_localize(None)

    # Find the last timestamp in parquet
    parquet_max_time = parquet_df['timestamp'].max()
    logger.info(f"Parquet data ends at: {parquet_max_time}")

    result_parts = []

    # Part 1: Data from parquet (if start is within parquet range)
    if start <= parquet_max_time:
        parquet_end = min(end, parquet_max_time)
        parquet_result = parquet_df[
            (parquet_df['timestamp'] >= start) &
            (parquet_df['timestamp'] <= parquet_end)
        ][['timestamp', value_col]].copy()
        parquet_result = parquet_result.rename(columns={value_col: output_col})
        result_parts.append(parquet_result)
        logger.info(f"Got {len(parquet_result)} records from parquet")

    # Part 2: Data from EPIAS API (if end is beyond parquet)
    if end > parquet_max_time:
        epias_start = max(start, parquet_max_time + timedelta(hours=1))
        epias_df = fetch_epias_live_data(
            data_type,
            epias_start.strftime('%Y-%m-%d'),
            end.strftime('%Y-%m-%d')
        )
        if not epias_df.empty:
            # Ensure timezone-naive
            if epias_df['timestamp'].dt.tz is not None:
                epias_df['timestamp'] = epias_df['timestamp'].dt.tz_localize(None)
            # Filter to exact range
            epias_df = epias_df[
                (epias_df['timestamp'] >= epias_start) &
                (epias_df['timestamp'] <= end)
            ]
            result_parts.append(epias_df[['timestamp', output_col]])
            logger.info(f"Got {len(epias_df)} records from EPIAS API")

    if not result_parts:
        return pd.DataFrame()

    # Combine and sort - ensure all parts are tz-naive
    for i, part in enumerate(result_parts):
        if part['timestamp'].dt.tz is not None:
            result_parts[i] = part.copy()
            result_parts[i]['timestamp'] = result_parts[i]['timestamp'].dt.tz_localize(None)

    combined = pd.concat(result_parts, ignore_index=True)
    combined = combined.drop_duplicates(subset=['timestamp']).sort_values('timestamp')

    return combined


def get_forecast_data(forecast_type: str, start: datetime = None, end: datetime = None) -> pd.DataFrame:
    """
    Load forecast data from parquet files.
    Returns most recent forecast for each target_time.
    """
    forecast_dir = Path(__file__).parent / 'data' / 'forecasts' / forecast_type
    if not forecast_dir.exists():
        return pd.DataFrame()

    # Load relevant parquet files based on date range
    all_forecasts = []
    for parquet_file in sorted(forecast_dir.glob('*.parquet')):
        try:
            df = pd.read_parquet(parquet_file)
            all_forecasts.append(df)
        except Exception as e:
            logger.warning(f"Failed to read {parquet_file}: {e}")

    if not all_forecasts:
        return pd.DataFrame()

    forecasts = pd.concat(all_forecasts, ignore_index=True)

    # Convert times
    forecasts['target_time'] = pd.to_datetime(forecasts['target_time'])
    forecasts['forecast_time'] = pd.to_datetime(forecasts['forecast_time'])

    # Remove timezone for consistency
    if forecasts['target_time'].dt.tz is not None:
        forecasts['target_time'] = forecasts['target_time'].dt.tz_localize(None)
    if forecasts['forecast_time'].dt.tz is not None:
        forecasts['forecast_time'] = forecasts['forecast_time'].dt.tz_localize(None)

    # Filter by date range if specified
    if start is not None:
        forecasts = forecasts[forecasts['target_time'] >= start]
    if end is not None:
        forecasts = forecasts[forecasts['target_time'] <= end]

    # Keep only the most recent forecast for each target_time
    forecasts = forecasts.sort_values('forecast_time', ascending=False)
    forecasts = forecasts.drop_duplicates(subset=['target_time'], keep='first')
    forecasts = forecasts.sort_values('target_time')

    return forecasts


@app.get("/history/price")
async def get_historical_prices(
    start_date: Optional[str] = Query(None, description="Start date (ISO format)"),
    end_date: Optional[str] = Query(None, description="End date (ISO format)"),
    limit: int = Query(5000, description="Max rows to return")
):
    """
    Get historical EPIAS price data with our forecasts.

    - Actual data: From EPIAS (parquet for historical, live API for recent)
    - Forecast data: From our model predictions
    """
    try:
        # Parse dates
        if start_date:
            start = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            if start.tzinfo:
                start = start.replace(tzinfo=None)
        else:
            start = datetime.now() - timedelta(days=7)

        if end_date:
            end = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            if end.tzinfo:
                end = end.replace(tzinfo=None)
        else:
            end = datetime.now()

        # Get ACTUAL data from EPIAS (parquet + live API)
        actual_df = get_combined_actual_data('price', start, end)

        if actual_df.empty:
            return {'data': [], 'count': 0, 'has_forecasts': False, 'source': 'none'}

        # Prepare result
        result = actual_df.head(limit).copy()

        # Try to get saved forecasts first
        saved_forecasts = get_forecast_data('price', start, end)
        has_forecasts = False

        if not saved_forecasts.empty:
            # Use saved forecasts from parquet
            forecast_df = saved_forecasts[['target_time', 'forecast_value']].copy()
            forecast_df = forecast_df.rename(columns={'target_time': 'timestamp', 'forecast_value': 'forecast'})
            result = result.merge(forecast_df, on='timestamp', how='left')
            has_forecasts = bool(result['forecast'].notna().any())
            logger.info(f"Using {result['forecast'].notna().sum()} saved forecasts from parquet")

        # If no saved forecasts, try model forecasts (requires master parquet features)
        if not has_forecasts:
            model_forecasts = generate_model_forecasts('price', result['timestamp'])
            if not model_forecasts.empty:
                result = result.merge(model_forecasts, on='timestamp', how='left')
                has_forecasts = bool(result['forecast'].notna().any())
                logger.info(f"Generated {result['forecast'].notna().sum()} model forecasts")

        # If still no forecasts, use simple pattern-based forecasts from actual data
        if not has_forecasts:
            simple_forecasts = generate_simple_forecasts('price', result['timestamp'], actual_df)
            if not simple_forecasts.empty:
                result = result.merge(simple_forecasts, on='timestamp', how='left')
                has_forecasts = bool(result['forecast'].notna().any())
                logger.info(f"Generated {result['forecast'].notna().sum()} simple pattern forecasts")

        result['timestamp'] = result['timestamp'].astype(str)

        # Convert numpy types to native Python for JSON serialization
        data_records = result.to_dict(orient='records')
        for record in data_records:
            for key, value in record.items():
                if hasattr(value, 'item'):  # numpy types have .item() method
                    record[key] = value.item()

        return {
            'data': data_records,
            'count': len(result),
            'has_forecasts': has_forecasts,
            'source': 'epias'
        }

    except Exception as e:
        logger.error(f"Failed to get historical prices: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history/consumption")
async def get_historical_consumption(
    start_date: Optional[str] = Query(None, description="Start date (ISO format)"),
    end_date: Optional[str] = Query(None, description="End date (ISO format)"),
    limit: int = Query(5000, description="Max rows to return")
):
    """
    Get historical EPIAS consumption data with our forecasts.

    - Actual data: From EPIAS (parquet for historical, live API for recent)
    - Forecast data: From our model predictions
    """
    try:
        # Parse dates
        if start_date:
            start = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            if start.tzinfo:
                start = start.replace(tzinfo=None)
        else:
            start = datetime.now() - timedelta(days=7)

        if end_date:
            end = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            if end.tzinfo:
                end = end.replace(tzinfo=None)
        else:
            end = datetime.now()

        # Get ACTUAL data from EPIAS (parquet + live API)
        actual_df = get_combined_actual_data('consumption', start, end)

        if actual_df.empty:
            return {'data': [], 'count': 0, 'has_forecasts': False, 'source': 'none'}

        # Prepare result
        result = actual_df.head(limit).copy()

        # Try to get saved forecasts first
        saved_forecasts = get_forecast_data('consumption', start, end)
        has_forecasts = False

        if not saved_forecasts.empty:
            # Use saved forecasts from parquet
            forecast_df = saved_forecasts[['target_time', 'forecast_value']].copy()
            forecast_df = forecast_df.rename(columns={'target_time': 'timestamp', 'forecast_value': 'forecast'})
            result = result.merge(forecast_df, on='timestamp', how='left')
            has_forecasts = bool(result['forecast'].notna().any())
            logger.info(f"Using {result['forecast'].notna().sum()} saved consumption forecasts from parquet")

        # If no saved forecasts, try model forecasts (requires master parquet features)
        if not has_forecasts:
            model_forecasts = generate_model_forecasts('consumption', result['timestamp'])
            if not model_forecasts.empty:
                result = result.merge(model_forecasts, on='timestamp', how='left')
                has_forecasts = bool(result['forecast'].notna().any())
                logger.info(f"Generated {result['forecast'].notna().sum()} consumption model forecasts")

        # If still no forecasts, use simple pattern-based forecasts from actual data
        if not has_forecasts:
            simple_forecasts = generate_simple_forecasts('consumption', result['timestamp'], actual_df)
            if not simple_forecasts.empty:
                result = result.merge(simple_forecasts, on='timestamp', how='left')
                has_forecasts = bool(result['forecast'].notna().any())
                logger.info(f"Generated {result['forecast'].notna().sum()} simple consumption forecasts")

        result['timestamp'] = result['timestamp'].astype(str)

        # Convert numpy types to native Python for JSON serialization
        data_records = result.to_dict(orient='records')
        for record in data_records:
            for key, value in record.items():
                if hasattr(value, 'item'):  # numpy types have .item() method
                    record[key] = value.item()

        return {
            'data': data_records,
            'count': len(result),
            'has_forecasts': has_forecasts,
            'source': 'epias'
        }

    except Exception as e:
        logger.error(f"Failed to get historical consumption: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Pre-Aggregated Endpoints (Fast Dashboard Loading)
# =============================================================================

@app.get("/api/aggregates/day-type/{model}")
async def get_day_type_comparison(
    model: str,
    days: int = Query(30, description="Number of days to analyze")
):
    """
    Get pre-aggregated weekday vs weekend comparison.

    This is a fast endpoint that returns hourly patterns grouped by day type.
    Used by the Compare page for Day Type Comparison chart.
    """
    if model not in ['price', 'consumption']:
        raise HTTPException(status_code=400, detail="Model must be 'price' or 'consumption'")

    # Check cache first
    cache_key = get_cache_key('day_type', model=model, days=days)
    cached = get_cached_result(cache_key)
    if cached:
        return cached

    try:
        # Load historical data
        df = get_historical_data()
        if df.empty:
            return {'weekday': [], 'weekend': [], 'diffPercent': 0, 'error': 'No data available'}

        # Get the value column
        value_col = 'price_ptf' if model == 'price' else 'consumption'
        if value_col not in df.columns:
            return {'weekday': [], 'weekend': [], 'diffPercent': 0, 'error': f'Column {value_col} not found'}

        # Filter to requested date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
        filtered_df = df.loc[mask].copy()

        if filtered_df.empty:
            return {'weekday': [], 'weekend': [], 'diffPercent': 0, 'error': 'No data in range'}

        # Add day of week
        filtered_df['hour'] = filtered_df['timestamp'].dt.hour
        filtered_df['day_of_week'] = filtered_df['timestamp'].dt.dayofweek
        filtered_df['is_weekend'] = filtered_df['day_of_week'].isin([5, 6])

        # Calculate hourly averages for weekday and weekend
        weekday_hourly = filtered_df[~filtered_df['is_weekend']].groupby('hour')[value_col].mean()
        weekend_hourly = filtered_df[filtered_df['is_weekend']].groupby('hour')[value_col].mean()

        # Build response
        weekday_data = [
            {'label': f'{h:02d}:00', 'value': float(weekday_hourly.get(h, 0))}
            for h in range(24)
        ]
        weekend_data = [
            {'label': f'{h:02d}:00', 'value': float(weekend_hourly.get(h, 0))}
            for h in range(24)
        ]

        # Calculate percentage difference
        weekday_avg = weekday_hourly.mean() if len(weekday_hourly) > 0 else 0
        weekend_avg = weekend_hourly.mean() if len(weekend_hourly) > 0 else 0
        diff_percent = ((weekday_avg - weekend_avg) / weekday_avg * 100) if weekday_avg > 0 else 0

        result = {
            'weekday': weekday_data,
            'weekend': weekend_data,
            'diffPercent': round(float(diff_percent), 1),
            'dataPoints': len(filtered_df),
            'dateRange': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            }
        }

        # Cache the result
        set_cached_result(cache_key, result)

        return result

    except Exception as e:
        logger.error(f"Failed to get day type comparison: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/aggregates/hourly/{model}")
async def get_hourly_aggregates(
    model: str,
    days: int = Query(30, description="Number of days to analyze")
):
    """
    Get pre-aggregated hourly statistics.

    Returns min, max, mean, std for each hour of the day.
    Used for anomaly detection and pattern analysis.
    """
    if model not in ['price', 'consumption']:
        raise HTTPException(status_code=400, detail="Model must be 'price' or 'consumption'")

    # Check cache
    cache_key = get_cache_key('hourly_agg', model=model, days=days)
    cached = get_cached_result(cache_key)
    if cached:
        return cached

    try:
        df = get_historical_data()
        if df.empty:
            return {'data': [], 'error': 'No data available'}

        value_col = 'price_ptf' if model == 'price' else 'consumption'
        if value_col not in df.columns:
            return {'data': [], 'error': f'Column {value_col} not found'}

        # Filter to date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
        filtered_df = df.loc[mask].copy()

        if filtered_df.empty:
            return {'data': [], 'error': 'No data in range'}

        filtered_df['hour'] = filtered_df['timestamp'].dt.hour

        # Aggregate by hour
        hourly_stats = filtered_df.groupby('hour')[value_col].agg(['min', 'max', 'mean', 'std', 'count'])

        result = {
            'data': [
                {
                    'hour': h,
                    'label': f'{h:02d}:00',
                    'min': float(hourly_stats.loc[h, 'min']) if h in hourly_stats.index else 0,
                    'max': float(hourly_stats.loc[h, 'max']) if h in hourly_stats.index else 0,
                    'mean': float(hourly_stats.loc[h, 'mean']) if h in hourly_stats.index else 0,
                    'std': float(hourly_stats.loc[h, 'std']) if h in hourly_stats.index else 0,
                    'count': int(hourly_stats.loc[h, 'count']) if h in hourly_stats.index else 0
                }
                for h in range(24)
            ],
            'totalDataPoints': len(filtered_df),
            'dateRange': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            }
        }

        set_cached_result(cache_key, result)
        return result

    except Exception as e:
        logger.error(f"Failed to get hourly aggregates: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/aggregates/daily/{model}")
async def get_daily_aggregates(
    model: str,
    days: int = Query(30, description="Number of days to return")
):
    """
    Get pre-aggregated daily statistics.

    Returns daily min, max, mean for fast historical overview.
    Used by Historical page for quick loading.
    """
    if model not in ['price', 'consumption']:
        raise HTTPException(status_code=400, detail="Model must be 'price' or 'consumption'")

    # Check cache
    cache_key = get_cache_key('daily_agg', model=model, days=days)
    cached = get_cached_result(cache_key)
    if cached:
        return cached

    try:
        df = get_historical_data()
        if df.empty:
            return {'data': [], 'error': 'No data available'}

        value_col = 'price_ptf' if model == 'price' else 'consumption'
        if value_col not in df.columns:
            return {'data': [], 'error': f'Column {value_col} not found'}

        # Filter to date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
        filtered_df = df.loc[mask].copy()

        if filtered_df.empty:
            return {'data': [], 'error': 'No data in range'}

        filtered_df['date'] = filtered_df['timestamp'].dt.date

        # Aggregate by day
        daily_stats = filtered_df.groupby('date')[value_col].agg(['min', 'max', 'mean', 'std', 'count'])
        daily_stats = daily_stats.reset_index()

        result = {
            'data': [
                {
                    'date': str(row['date']),
                    'min': float(row['min']),
                    'max': float(row['max']),
                    'mean': float(row['mean']),
                    'std': float(row['std']) if not pd.isna(row['std']) else 0,
                    'count': int(row['count'])
                }
                for _, row in daily_stats.iterrows()
            ],
            'totalDays': len(daily_stats),
            'dateRange': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            }
        }

        set_cached_result(cache_key, result)
        return result

    except Exception as e:
        logger.error(f"Failed to get daily aggregates: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/aggregates/statistics/{model}")
async def get_statistics(
    model: str,
    days: int = Query(30, description="Number of days to analyze")
):
    """
    Get overall statistics for a time period.

    Returns mean, median, std, min, max, percentiles.
    Used for summary cards and anomaly thresholds.
    """
    if model not in ['price', 'consumption']:
        raise HTTPException(status_code=400, detail="Model must be 'price' or 'consumption'")

    # Check cache
    cache_key = get_cache_key('statistics', model=model, days=days)
    cached = get_cached_result(cache_key)
    if cached:
        return cached

    try:
        df = get_historical_data()
        if df.empty:
            return {'error': 'No data available'}

        value_col = 'price_ptf' if model == 'price' else 'consumption'
        if value_col not in df.columns:
            return {'error': f'Column {value_col} not found'}

        # Filter to date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
        values = df.loc[mask, value_col].dropna()

        if len(values) == 0:
            return {'error': 'No data in range'}

        result = {
            'mean': float(values.mean()),
            'median': float(values.median()),
            'std': float(values.std()),
            'min': float(values.min()),
            'max': float(values.max()),
            'p25': float(values.quantile(0.25)),
            'p75': float(values.quantile(0.75)),
            'p95': float(values.quantile(0.95)),
            'count': int(len(values)),
            'dateRange': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            }
        }

        set_cached_result(cache_key, result)
        return result

    except Exception as e:
        logger.error(f"Failed to get statistics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Real-time API Endpoints (Firestore-backed)
# =============================================================================

class RealTimeDataPoint(BaseModel):
    timestamp: str
    value: float


class RealTimeSummary(BaseModel):
    avgActual: float
    avgForecast: float
    peakActual: dict
    peakForecast: dict


class RealTimeResponse(BaseModel):
    modelType: str
    unit: str
    timezone: str
    lastUpdated: str
    actual: List[dict]
    pivotTime: str
    forecast: List[dict]
    summary: dict


@app.get("/api/realtime/{model}")
async def get_realtime_data(model: str):
    """
    Get real-time data for dashboard.
    - Pivot: floor(current_time) - 2 hours (EPIAS delay)
    - Actuals: 12 hours BEFORE pivot (ending at pivot, not including pivot hour)
    - Forecasts: 12 hours AFTER pivot (starting from pivot hour)
    """
    if model not in ['price', 'consumption']:
        raise HTTPException(status_code=400, detail="Model must be 'price' or 'consumption'")

    try:
        unit = 'TL/MWh' if model == 'price' else 'MWh'
        now = datetime.now()

        # Pivot = floor(current_time) - 2 hours (EPIAS has 2-hour delay)
        pivot_time = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=2)

        # =====================================================
        # 1. Get REAL EPIAS actuals (12 hours BEFORE pivot)
        # Actual data ends at pivot (exclusive), starts 12 hours before
        # =====================================================
        actuals = []
        value_col = 'price' if model == 'price' else 'consumption'

        actual_start = pivot_time - timedelta(hours=12)
        actual_end = pivot_time - timedelta(hours=1)  # End 1 hour before pivot (last actual hour)
        actual_df = get_combined_actual_data(model, actual_start, actual_end)

        if not actual_df.empty:
            actual_df = actual_df.sort_values('timestamp')
            for _, row in actual_df.iterrows():
                if pd.notna(row[value_col]):
                    actuals.append({
                        'timestamp': str(row['timestamp']),
                        'value': float(row[value_col])
                    })

        logger.info(f"RealTime {model}: Got {len(actuals)} actuals from EPIAS")

        # =====================================================
        # 2. Generate forecasts for next 12 hours
        # =====================================================
        forecast_data = []

        # Try to get from pipeline storage first (if scheduler ran)
        try:
            pipeline = get_pipeline()
            forecast_df = pipeline.storage.get_latest_forecast(model)
            if not forecast_df.empty:
                for _, row in forecast_df.iterrows():
                    try:
                        target = pd.to_datetime(row['target_time'])
                        if target.tzinfo:
                            target = target.tz_localize(None)

                        if target >= pivot_time:
                            val = float(row['forecast_value'])
                            uncertainty = val * 0.1 if model == 'price' else val * 0.05
                            forecast_data.append({
                                'timestamp': str(row['target_time']),
                                'value': val,
                                'lower': val - uncertainty,
                                'upper': val + uncertainty
                            })
                    except Exception as e:
                        continue
                logger.info(f"RealTime {model}: Got {len(forecast_data)} forecasts from storage")
        except Exception as e:
            logger.warning(f"Pipeline storage read failed: {e}")

        # If no stored forecasts, generate on-the-fly using model
        if not forecast_data:
            logger.info(f"No stored forecasts, generating on-the-fly for {model}")
            try:
                # Generate simple forecasts based on recent average
                if actuals:
                    avg_recent = sum(a['value'] for a in actuals[-6:]) / min(len(actuals), 6)
                else:
                    avg_recent = 2500 if model == 'price' else 35000  # Reasonable defaults

                # Generate 12 hours of forecasts starting FROM pivot_time
                for h in range(12):
                    target_time = pivot_time + timedelta(hours=h)
                    hour = target_time.hour

                    if model == 'price':
                        # Price typically higher during day (8-20)
                        if 8 <= hour <= 20:
                            multiplier = 1.1 + 0.05 * (1 if 17 <= hour <= 20 else 0)
                        else:
                            multiplier = 0.9
                        uncertainty_pct = 0.1
                    else:  # consumption
                        # Consumption higher during day
                        if 8 <= hour <= 22:
                            multiplier = 1.05 + 0.1 * (1 if 18 <= hour <= 21 else 0)
                        else:
                            multiplier = 0.85
                        uncertainty_pct = 0.05

                    val = avg_recent * multiplier
                    uncertainty = val * uncertainty_pct
                    forecast_data.append({
                        'timestamp': target_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'value': val,
                        'lower': val - uncertainty,
                        'upper': val + uncertainty
                    })

                logger.info(f"Generated {len(forecast_data)} forecasts on-the-fly for {model}")
            except Exception as e:
                logger.error(f"Failed to generate forecasts: {e}")

        # Keep first 12 forecasts
        forecast_data = forecast_data[:12]

        # Calculate summary
        avg_actual = sum(a['value'] for a in actuals) / len(actuals) if actuals else 0
        avg_forecast = sum(f['value'] for f in forecast_data) / len(forecast_data) if forecast_data else 0

        peak_actual = max(actuals, key=lambda x: x['value']) if actuals else {'value': 0, 'timestamp': ''}
        peak_forecast = max(forecast_data, key=lambda x: x['value']) if forecast_data else {'value': 0, 'timestamp': ''}

        def format_time(ts):
            try:
                dt = datetime.fromisoformat(str(ts).replace(' ', 'T'))
                return dt.strftime('%H:%M')
            except:
                return ''

        return {
            'modelType': model,
            'unit': unit,
            'timezone': 'Europe/Istanbul',
            'lastUpdated': now.isoformat(),
            'actual': actuals,
            'pivotTime': pivot_time.isoformat(),
            'forecast': forecast_data,
            'summary': {
                'avgActual': float(avg_actual),
                'avgForecast': float(avg_forecast),
                'peakActual': {'value': float(peak_actual['value']), 'time': format_time(peak_actual['timestamp'])},
                'peakForecast': {'value': float(peak_forecast['value']), 'time': format_time(peak_forecast['timestamp'])}
            }
        }

    except Exception as e:
        logger.error(f"Failed to get realtime data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/anomaly/{model}")
async def get_anomaly_data(model: str, days: int = Query(7, description="Number of days to analyze")):
    """
    Get anomaly detection data.
    Calculates anomalies based on statistical deviation from mean.
    Uses EPIAS live data when parquet isn't available.
    """
    if model not in ['price', 'consumption']:
        raise HTTPException(status_code=400, detail="Model must be 'price' or 'consumption'")

    try:
        # Get data for the specified period
        now = datetime.now()
        start = now - timedelta(days=days)

        value_col = 'price' if model == 'price' else 'consumption'

        # Use combined data (parquet + EPIAS fallback)
        result = get_combined_actual_data(model, start, now)

        if result.empty:
            logger.warning(f"No data available for anomaly detection ({model})")
            return {'summary': {}, 'anomalies': [], 'scoreDistribution': {}, 'error': 'No data available'}

        values = result[value_col].values

        # Calculate statistics
        mean_val = values.mean()
        std_val = values.std()

        # Simple anomaly detection: z-score based
        anomalies = []
        scores = []

        # Convert numpy values to Python native types
        mean_val_py = float(mean_val)
        std_val_py = float(std_val)

        for _, row in result.iterrows():
            actual = row[value_col]
            if pd.isna(actual):
                continue
            actual_py = float(actual)
            # Use mean as "expected" for simple anomaly detection
            residual = abs(actual_py - mean_val_py)
            z_score = residual / std_val_py if std_val_py > 0 else 0
            anomaly_score = min(z_score / 3, 1.0)  # Normalize to 0-1
            is_anomaly = bool(anomaly_score > 0.8)

            scores.append(float(anomaly_score))
            anomalies.append({
                'timestamp': str(row['timestamp']),
                'actual': actual_py,
                'forecast': mean_val_py,
                'residual': float(residual),
                'anomalyScore': float(anomaly_score),
                'isAnomaly': is_anomaly
            })

        anomaly_count = sum(1 for a in anomalies if a['isAnomaly'])
        residuals = [a['residual'] for a in anomalies]

        return {
            'summary': {
                'totalRows': int(len(anomalies)),
                'anomalyCount': int(anomaly_count),
                'anomalyRate': float((anomaly_count / len(anomalies) * 100) if anomalies else 0),
                'maxScore': float(max(scores)) if scores else 0.0,
                'maxResidual': float(max(residuals)) if residuals else 0.0,
                'meanResidual': float(sum(residuals) / len(residuals)) if residuals else 0.0
            },
            'anomalies': anomalies,
            'scoreDistribution': {
                'count': int(len(scores)),
                'mean': float(sum(scores) / len(scores)) if scores else 0.0,
                'std': 0.2,
                'min': float(min(scores)) if scores else 0.0,
                'max': float(max(scores)) if scores else 0.0
            }
        }

    except Exception as e:
        logger.error(f"Failed to get anomaly data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class AlertCreate(BaseModel):
    type: str
    severity: str
    title: str
    message: str


@app.get("/api/alerts")
async def get_alerts():
    """Get all alerts from Firestore."""
    db = get_firestore()
    alerts = []

    if db:
        try:
            docs = db.collection("alerts").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(50).stream()
            for doc in docs:
                alert = doc.to_dict()
                alert['id'] = doc.id
                alerts.append(alert)
        except Exception as e:
            logger.warning(f"Failed to get alerts from Firestore: {e}")

    # Add default system alert if no alerts
    if not alerts:
        alerts = [{
            'id': '1',
            'type': 'SYSTEM',
            'severity': 'info',
            'title': 'API Connected',
            'message': 'Dashboard is connected to the ForeWatt API.',
            'timestamp': datetime.now().isoformat(),
            'read': False
        }]

    return alerts


@app.post("/api/alerts")
async def create_alert(alert: AlertCreate):
    """Create a new alert in Firestore."""
    db = get_firestore()

    alert_data = {
        'type': alert.type,
        'severity': alert.severity,
        'title': alert.title,
        'message': alert.message,
        'timestamp': datetime.now().isoformat(),
        'read': False
    }

    if db:
        try:
            doc_ref = db.collection("alerts").add(alert_data)
            alert_data['id'] = doc_ref[1].id
        except Exception as e:
            logger.warning(f"Failed to save alert to Firestore: {e}")
            alert_data['id'] = 'local-' + datetime.now().strftime('%Y%m%d%H%M%S')
    else:
        alert_data['id'] = 'local-' + datetime.now().strftime('%Y%m%d%H%M%S')

    return alert_data


# =============================================================================
# Data Update Endpoint (for Cloud Scheduler)
# =============================================================================

@app.post("/api/update-data")
async def update_data(
    days_back: int = Query(7, description="Number of days to look back for new data"),
    force: bool = Query(False, description="Force update even if data is recent")
):
    """
    Update the master parquet file with new EPIAS and weather data.

    This endpoint is designed to be called by Cloud Scheduler daily.
    It fetches new data from EPIAS and Open-Meteo, generates features,
    and saves the updated parquet to Google Cloud Storage.

    Args:
        days_back: Number of days to look back for new data
        force: Force update even if data is recent (less than 24h old)

    Returns:
        Dictionary with update statistics
    """
    try:
        from src.pipeline.data_updater import IncrementalDataUpdater
        from src.pipeline.gcs_storage import get_gcs_storage, save_master_to_gcs

        logger.info(f"Starting data update (days_back={days_back}, force={force})")

        # Use GCS-backed storage
        gcs_storage = get_gcs_storage()

        # Create updater with GCS master path
        updater = IncrementalDataUpdater()

        # Override the master path to use a temp file
        import tempfile
        temp_master = tempfile.NamedTemporaryFile(suffix='.parquet', delete=False)
        temp_path = temp_master.name
        temp_master.close()

        # Download current master from GCS
        current_df = gcs_storage.load_master_df(force_refresh=True)
        if not current_df.empty:
            current_df.to_parquet(temp_path, index=False)
            updater.master_path = Path(temp_path)
        else:
            logger.warning("No existing master data found in GCS")
            return {
                'success': False,
                'records_added': 0,
                'error': 'No existing master data in GCS'
            }

        result = updater.update_master_parquet(days_back=days_back, force=force)

        # If update was successful, save back to GCS
        if result['success']:
            updated_df = pd.read_parquet(temp_path)
            if save_master_to_gcs(updated_df):
                logger.info(f"Saved updated master to GCS ({len(updated_df)} records)")
                result['gcs_saved'] = True
            else:
                logger.error("Failed to save to GCS")
                result['gcs_saved'] = False

        # Clean up temp file
        import os
        try:
            os.unlink(temp_path)
        except:
            pass

        # Clear the cached historical data so next request gets fresh data
        global _historical_data
        _historical_data = None
        gcs_storage.clear_cache()

        # Log result
        if result['success']:
            logger.info(f"Data update successful: {result['records_added']} records added")

            # Create an alert for successful update
            db = get_firestore()
            if db:
                try:
                    db.collection("alerts").add({
                        'type': 'DATA_UPDATE',
                        'severity': 'success',
                        'title': 'Data Updated',
                        'message': f"Added {result['records_added']} new records. Last timestamp: {result['last_timestamp_after']}",
                        'timestamp': datetime.now().isoformat(),
                        'read': False
                    })
                except Exception as e:
                    logger.warning(f"Failed to create update alert: {e}")
        else:
            logger.warning(f"Data update failed: {result['error']}")

        return result

    except Exception as e:
        logger.error(f"Data update failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data-status")
async def get_data_status():
    """
    Get the current status of available data.

    Returns information about:
    - Parquet data (historical with forecasts)
    - EPIAS live data (recent actuals)
    - Forecast parquet files
    """
    try:
        result = {
            'status': 'ok',
            'parquet': {'available': False},
            'epias_live': {'available': False},
            'forecasts': {'available': False}
        }

        # Check parquet master data
        df = get_historical_data()
        if not df.empty:
            result['parquet'] = {
                'available': True,
                'first_timestamp': str(df['timestamp'].min()),
                'last_timestamp': str(df['timestamp'].max()),
                'record_count': len(df)
            }
            # Use parquet end date as main reference
            result['last_timestamp'] = str(df['timestamp'].max())
            result['first_timestamp'] = str(df['timestamp'].min())

        # Check forecast parquet files
        forecast_dir = Path(__file__).parent / 'data' / 'forecasts' / 'price'
        if forecast_dir.exists():
            forecast_files = sorted(forecast_dir.glob('*.parquet'))
            if forecast_files:
                # Get date range from last forecast file
                last_file = forecast_files[-1]
                try:
                    forecast_df = pd.read_parquet(last_file)
                    forecast_df['target_time'] = pd.to_datetime(forecast_df['target_time'])
                    if forecast_df['target_time'].dt.tz is not None:
                        forecast_df['target_time'] = forecast_df['target_time'].dt.tz_localize(None)

                    result['forecasts'] = {
                        'available': True,
                        'file_count': len(forecast_files),
                        'last_file': last_file.name,
                        'first_target': str(pd.read_parquet(forecast_files[0])['target_time'].min()),
                        'last_target': str(forecast_df['target_time'].max())
                    }
                    # Update last_timestamp to include forecast range
                    if 'last_timestamp' not in result or result['last_timestamp'] is None:
                        result['last_timestamp'] = str(forecast_df['target_time'].max())
                        result['first_timestamp'] = str(pd.read_parquet(forecast_files[0])['target_time'].min())
                except Exception as e:
                    logger.warning(f"Failed to read forecast files: {e}")

        # Check EPIAS live availability (try a small request)
        try:
            yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            today = datetime.now().strftime('%Y-%m-%d')
            epias_df = fetch_epias_live_data('price', yesterday, today)
            if not epias_df.empty:
                result['epias_live'] = {
                    'available': True,
                    'latest_record': str(epias_df['timestamp'].max())
                }
                # If no parquet, use EPIAS dates
                if result['last_timestamp'] is None or result.get('parquet', {}).get('available') is False:
                    result['last_timestamp'] = str(epias_df['timestamp'].max())
                    result['first_timestamp'] = str(epias_df['timestamp'].min())
        except Exception as e:
            logger.warning(f"EPIAS check failed: {e}")

        # Determine best data source recommendation
        if result['forecasts']['available']:
            result['recommended_source'] = 'parquet'
            result['has_forecasts'] = True
        elif result['epias_live'].get('available'):
            result['recommended_source'] = 'epias'
            result['has_forecasts'] = False
        else:
            result['recommended_source'] = 'none'
            result['has_forecasts'] = False

        return result

    except Exception as e:
        logger.error(f"Failed to get data status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )
