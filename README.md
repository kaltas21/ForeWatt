# ForeWatt

**Electricity Market Forecasting Platform for the Turkish Market**

ForeWatt is an end-to-end electricity market forecasting platform designed for the Turkish electricity market operated by EPIAS. The system provides 24-hour ahead forecasts for both energy consumption and day-ahead market prices using gradient boosting ensemble methods.

## Key Features

- **Consumption Forecasting**: CatBoost model achieving 1.95% sMAPE with 23 engineered features
- **Price Forecasting**: Hybrid CatBoost + LightGBM ensemble with adaptive error correction achieving 11.71% sMAPE
- **Real-time Dashboard**: React-based visualization with interactive charts, anomaly detection, and AI chatbot
- **Production Deployment**: Google Cloud Run with hourly automated forecasts and Firestore real-time storage
- **Medallion Data Architecture**: Bronze → Silver → Gold data pipeline for robust data management

## Model Performance

| Model | Configuration | Test sMAPE | Test MAE | R² |
|-------|--------------|------------|----------|-----|
| Consumption | CatBoost V2, lr=0.1 | 1.95% | 808.5 MWh | 0.969 |
| Price | CHybrid V14, 4yr data | 11.71% | 48.2 TL/MWh | 0.871 |

## Architecture Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Data Sources  │     │   ML Pipeline   │     │    Frontend     │
│  ─────────────  │     │  ─────────────  │     │  ─────────────  │
│  • EPIAS API    │────▶│  • Feature Eng  │────▶│  • React 19     │
│  • Open-Meteo   │     │  • CatBoost     │     │  • ECharts      │
│  • EVDS (TCMB)  │     │  • LightGBM     │     │  • Gemini AI    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                      │                       │
         ▼                      ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Google Cloud Platform                         │
│  Cloud Run • Cloud Scheduler • Firestore • Cloud Storage        │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker (optional, for containerized deployment)

### Backend Setup

```bash
# Clone the repository
git clone https://github.com/kaltas21/ForeWatt.git
cd ForeWatt

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your credentials (EPIAS, etc.)
```

### Frontend Setup

```bash
cd forewatt-dashboard

# Install dependencies
npm install

# Start development server
npm run dev
```

### Docker Setup (Local Development)

```bash
# Start all services (API, scheduler, dashboard, MLflow, InfluxDB)
docker-compose up -d
```

## Configuration

Create a `.env` file with the following variables:

```env
# EPIAS Credentials
EPTR_USERNAME=your_email@example.com
EPTR_PASSWORD=your_password

# Optional: InfluxDB (for local time-series storage)
INFLUXDB_TOKEN=your_token
INFLUXDB_BUCKET=epias

# Optional: Google Cloud
GOOGLE_CLOUD_PROJECT=your-project-id
```

## Usage

### Running the Forecast Pipeline

```bash
# Single forecast run
python services/scheduler.py --once

# Continuous mode (hourly forecasts)
python services/scheduler.py

# Custom interval (every 5 minutes for testing)
python services/scheduler.py --interval 5
```

### Starting the API Server

```bash
# Development
uvicorn main:app --reload --port 8000

# Production
uvicorn main:app --host 0.0.0.0 --port 8080
```

### Training Models

```bash
# Train consumption model
python src/models/consumption_train.py

# Train price model (CHybrid V14)
python src/models/price_train.py
```

### Running Experiments

```bash
# Run all experiments
python experiments/run_all_experiments.py

# Run consumption experiments only
python experiments/run_consumption_experiments.py

# Run price experiments only
python experiments/run_price_experiments.py
```

## Project Structure

```
ForeWatt/
├── main.py                      # Cloud Run FastAPI entry point
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Cloud Run container
├── docker-compose.yml           # Local development stack
│
├── src/                         # Core Python package
│   ├── data/                    # Data fetching & preprocessing
│   │   ├── epias_fetcher.py     # EPIAS API integration
│   │   ├── weather_fetcher.py   # Open-Meteo weather API
│   │   └── evds_fetcher.py      # Central Bank macro data
│   ├── features/                # Feature engineering
│   │   ├── lag_features.py      # Temporal lag features
│   │   ├── rolling_features.py  # Rolling window statistics
│   │   └── calendar_features.py # Calendar encodings
│   ├── models/                  # Model training
│   │   ├── consumption_train.py # CatBoost consumption model
│   │   └── price_train.py       # CHybrid V14 price model
│   ├── pipeline/                # Inference pipeline
│   │   ├── forecast_pipeline.py # Main forecast orchestration
│   │   └── storage.py           # Parquet storage manager
│   └── anomaly/                 # Anomaly detection
│       └── isolationForest.py
│
├── data/                        # Medallion data architecture
│   ├── bronze/                  # Raw data from APIs
│   ├── silver/                  # Cleaned & validated
│   ├── gold/                    # Feature-engineered
│   │   └── master/              # Master dataset
│   └── forecasts/               # Monthly forecast archives
│
├── models/                      # Trained ML models
│   ├── consumption/             # CatBoost model
│   └── price/                   # CHybrid V14 ensemble
│
├── forewatt-dashboard/          # React frontend
│   ├── views/                   # Dashboard pages
│   ├── components/              # Reusable components
│   └── contexts/                # React contexts
│
├── experiments/                 # ML experiments
│   ├── results/                 # Experiment outputs
│   └── report_plots/            # Generated figures
│
└── services/                    # Background services
    └── scheduler.py             # Hourly pipeline orchestration
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/forecast` | POST | Trigger hourly forecast generation |
| `/forecast/latest` | GET | Get latest 24-hour forecasts |
| `/api/realtime/{model}` | GET | Real-time data from Firestore |
| `/api/historical/{model}` | GET | Historical data from Parquet |
| `/api/anomaly/{model}` | GET | Anomaly detection results |
| `/api/aggregates/hourly/{model}` | GET | Hourly aggregated patterns |

## Dashboard Views

- **Home**: Model selection with performance metrics
- **Real-Time**: Live 24-hour forecast visualization with confidence intervals
- **Historical**: Custom date range analysis with statistics
- **Anomaly**: Statistical anomaly detection with configurable sensitivity
- **Compare**: Period-over-period analysis (day/week/month)
- **Alerts**: Threshold-based alert configuration and history

## Data Sources

| Source | Data | Update Frequency |
|--------|------|------------------|
| EPIAS | Consumption, prices, generation, capacity | Hourly (2h delay) |
| Open-Meteo | Temperature, humidity, wind, cloud cover | Hourly |
| EVDS (TCMB) | FX rates, inflation indices | Daily |

## Technologies

**Backend**: Python 3.11, FastAPI, CatBoost, LightGBM, Pandas, scikit-learn

**Frontend**: React 19, TypeScript, Vite, ECharts, Tailwind CSS

**Cloud**: Google Cloud Run, Cloud Scheduler, Firestore, Cloud Storage

**Data**: Apache Parquet, InfluxDB (optional), MLflow

## License

This project is developed as part of COMP 491 - Computer Engineering Design Project at Koç University.

## Authors

- Zeynep Öykü Aslan
- Kaan Altaş
- Zeliha Paycı

**Project Advisor**: Asst. Prof. Gözde Gül Şahin

## Acknowledgments

- EPIAS for providing access to the Transparency Platform API
- Open-Meteo for free weather data API
- Central Bank of Turkey (TCMB) for EVDS macroeconomic data
