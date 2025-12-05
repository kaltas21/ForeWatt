import os
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from influxdb_client import InfluxDBClient, Point, WritePrecision
import pandas as pd

app = FastAPI(title="ForeWatt API", version="1.0.0")

# CORS middleware for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# InfluxDB client
client = InfluxDBClient(
    url=os.getenv("INFLUXDB_URL", "http://influxdb:8086"),
    token=os.getenv("INFLUXDB_TOKEN"),
    org=os.getenv("INFLUXDB_ORG"),
)
write_api = client.write_api()
query_api = client.query_api()
bucket = os.getenv("INFLUXDB_BUCKET")

# Predictions directory
PREDICTIONS_DIR = Path(__file__).parent.parent / "data" / "predictions"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/forewatt/window")
def get_consumption_window(
    hours: int = Query(default=24, ge=1, le=168, description="Number of hours to return"),
    include_forecast: bool = Query(default=True, description="Include forecast data")
):
    """
    Get consumption data window for real-time visualization.

    Returns:
    - historical: Last N hours of actual consumption
    - forecast: Next 12 hours of predictions (if available)
    - metadata: Generation time, model info, etc.
    """
    result = {
        "timestamp": datetime.now().isoformat(),
        "historical": [],
        "forecast": [],
        "metadata": {}
    }

    # Load latest forecast file (contains both historical and forecast data)
    latest_file = PREDICTIONS_DIR / "latest_forecast.json"

    if latest_file.exists():
        try:
            with open(latest_file) as f:
                forecast_data = json.load(f)

            # Historical data
            if "historical_timestamps" in forecast_data and "historical_values" in forecast_data:
                for ts, val in zip(forecast_data["historical_timestamps"], forecast_data["historical_values"]):
                    result["historical"].append({
                        "timestamp": ts,
                        "value": val,
                        "type": "actual"
                    })

            # Forecast data
            if include_forecast and "timestamps" in forecast_data and "predictions" in forecast_data:
                lower = forecast_data.get("lower_bound", [None] * len(forecast_data["predictions"]))
                upper = forecast_data.get("upper_bound", [None] * len(forecast_data["predictions"]))

                for i, (ts, val) in enumerate(zip(forecast_data["timestamps"], forecast_data["predictions"])):
                    result["forecast"].append({
                        "timestamp": ts,
                        "value": val,
                        "lower": lower[i] if lower else None,
                        "upper": upper[i] if upper else None,
                        "type": "forecast"
                    })

            result["metadata"] = {
                "generated_at": forecast_data.get("generated_at"),
                "model": forecast_data.get("model"),
                "horizon": forecast_data.get("horizon"),
                "base_time": forecast_data.get("base_time")
            }

        except Exception as e:
            result["error"] = str(e)
    else:
        result["error"] = "No forecast data available. Run: python services/scheduler.py --once"

    return result


@app.get("/api/forewatt/latest")
def get_latest_forecast():
    """Get the full latest forecast JSON."""
    latest_file = PREDICTIONS_DIR / "latest_forecast.json"

    if latest_file.exists():
        with open(latest_file) as f:
            return json.load(f)

    return {"error": "No forecast data available"}

@app.get("/write_test")
def write_test():
    p = Point("load").tag("region","TR").field("value", 12345).time(None, WritePrecision.S)
    write_api.write(bucket=bucket, record=p)
    return {"ok": True}

@app.get("/read_test")
def read_test():
    q = f'from(bucket:"{bucket}") |> range(start: -1h) |> limit(n:5)'
    tables = query_api.query(q)
    count = sum(1 for _t in tables for _r in _t.records)
    return {"rows": count}
