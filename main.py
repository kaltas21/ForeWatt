"""
ForeWatt API - Cloud Run Entry Point
=====================================
FastAPI application for forecast generation and retrieval.

Endpoints:
- POST /forecast - Trigger hourly forecast
- GET /forecast/price - Get price forecasts
- GET /forecast/consumption - Get consumption forecasts
- GET /forecast/latest - Get latest 24h forecasts
- GET /health - Health check

Designed for Cloud Run with Cloud Scheduler triggering /forecast every hour.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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

# Lazy load pipeline to avoid cold start issues
_pipeline = None


def get_pipeline():
    """Lazy load the forecast pipeline."""
    global _pipeline
    if _pipeline is None:
        from src.pipeline import ForecastPipeline
        _pipeline = ForecastPipeline()
    return _pipeline


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

        start = datetime.fromisoformat(start_date) if start_date else None
        end = datetime.fromisoformat(end_date) if end_date else None

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

        start = datetime.fromisoformat(start_date) if start_date else None
        end = datetime.fromisoformat(end_date) if end_date else None

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
