"""
ForeWatt API Module
===================
Backend API clients and endpoints for the ForeWatt dashboard.

Data Sources:
- EPIAS: Real-time consumption and price data
- Open-Meteo: Weather forecasts for 4 Turkish cities
- Firestore: Real-time data store

Endpoints:
- /api/realtime/{model} - Real-time forecasts + actuals
- /api/historical/{model} - Historical data from Parquet
- /api/weather - Current and forecast weather
- /api/chat - Gemini 3 Flash AI assistant
- /api/alerts - Alert management
"""

from .epias import EPIASClient
from .weather import WeatherClient
from .firebase import FirestoreClient

__all__ = [
    'EPIASClient',
    'WeatherClient',
    'FirestoreClient',
]
