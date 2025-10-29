# ForeWatt — Hourly Electricity Demand Forecasting, Uncertainty, and Anomaly Diagnostics

> **ForeWatt** is a fully reproducible, open-source platform for **1–24h** electricity demand forecasting with **calibrated prediction intervals**, **actionable anomaly diagnostics**, and an optional **EV load-shifting optimizer**.
> Stack: **FastAPI**, **MLflow**, **InfluxDB**, **Streamlit**, **Docker Compose**.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#license)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](#requirements)
[![Reproducible](https://img.shields.io/badge/reproducible-mlflow%20%7C%20docker-informational)](#mlops--reproducibility)

---

## Table of Contents
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Data Pipeline](#data-pipeline)
- [Models & Forecasting](#models--forecasting)
- [Development](#development)
- [License](#license)

---

## Features

### Core Capabilities
- **Multi-horizon forecasting**: 1–24 hour ahead predictions at hourly resolution
- **Calibrated uncertainty**: Split conformal prediction intervals with 90% coverage
- **Anomaly detection**: IsolationForest with level-shift, drift, and feature attribution diagnostics
- **Medallion data architecture**: Bronze → Silver → Gold layers with data quality gates
- **Population-weighted weather**: Open-Meteo integration for Turkey's top 10 cities (49% population coverage)
- **EPİAŞ integration**: Turkish electricity market data via eptr2 library
- **MLOps best practices**: MLflow experiment tracking, model registry, champion/challenger workflow

### Forecasting Models
- **Baselines**: Prophet, CatBoost, XGBoost
- **Deep Learning**: N-HiTS (Neural Hierarchical Interpolation for Time Series)
- **Ensemble**: Weighted median aggregation with failover
- **Optional**: PatchTST, Temporal Fusion Transformer, foundation models (TimesFM, Moirai)

### Optimization (Stretch Goal)
- **EV load shifting**: Linear programming with PuLP + CBC solver
- **Cost minimization**: EPİAŞ MCP wholesale price signals
- **Constraint satisfaction**: Power limits, energy-by-deadline requirements

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       DATA INGESTION                             │
│  EPİAŞ (eptr2) + Open-Meteo → APScheduler → Bronze (Parquet)   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    DATA QUALITY & NORMALIZATION                  │
│  Schema/Range/Monotonicity Checks → Silver (Normalized)         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     FEATURE ENGINEERING                          │
│  Lags, Rolls, Fourier, Calendar, Weather → Gold (ML-Ready)      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL TRAINING & ENSEMBLE                     │
│  N-HiTS + Prophet + CatBoost → MLflow Registry                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   UNCERTAINTY QUANTIFICATION                     │
│  Split Conformal Prediction (28-day rolling window)             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      SERVING & MONITORING                        │
│  FastAPI (/forecast, /intervals, /anomalies) + Streamlit UI     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
ForeWatt/
├── src/                           # Core source code (Python 3.11+)
│   ├── __init__.py
│   ├── data/                      # Data ingestion & processing
│   │   ├── __init__.py
│   │   └── weather_fetcher.py     # Open-Meteo population-weighted weather
│   ├── features/                  # Feature engineering (TODO)
│   ├── models/                    # Forecasting models (TODO)
│   ├── uncertainty/               # Conformal prediction (TODO)
│   ├── anomaly/                   # IsolationForest + diagnostics (TODO)
│   ├── optimization/              # EV load shifting (TODO)
│   └── evaluation/                # Metrics & validation (TODO)
│
├── data/                          # Medallion architecture
│   ├── bronze/                    # Raw API data (EPİAŞ, PJM, weather)
│   │   ├── epias/
│   │   ├── pjm/
│   │   └── demand_weather/
│   ├── silver/                    # Normalized, schema-validated
│   │   ├── epias/
│   │   ├── pjm/
│   │   └── demand_weather/
│   ├── gold/                      # Feature-engineered, ML-ready
│   │   └── demand_features/
│   ├── unused/                    # Archived data
│   │   └── RES_GES_Data.csv      # Renewable energy plants (future use)
│   ├── influx/                    # InfluxDB volume mount
│   └── mlflow/                    # MLflow volume mount
│
├── api/                           # FastAPI REST service
│   ├── Dockerfile
│   ├── main.py                    # API endpoints
│   └── requirements.txt
│
├── dashboard/                     # Streamlit UI
│   ├── Dockerfile
│   ├── app.py                     # Dashboard
│   └── requirements.txt
│
├── notebooks/                     # Jupyter notebooks (TODO)
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_models.ipynb
│   └── 04_model_comparison.ipynb
│
├── tests/                         # Unit & integration tests (TODO)
│
├── configs/                       # Model configurations (TODO)
│   ├── nhits.yaml
│   ├── catboost.yaml
│   └── ensemble.yaml
│
├── docs/                          # Documentation
│   └── Forewatt_COMP491_Proposal.txt
│
├── .env.example                   # Environment variable template
├── .gitignore                     # Git ignore rules
├── docker-compose.yml             # Multi-service orchestration
├── requirements.txt               # Root Python dependencies
├── LICENSE                        # MIT License
└── README.md                      # This file
```

### Medallion Architecture

**Bronze Layer** (Raw)
- EPİAŞ hourly load/price (via eptr2)
- PJM benchmarking data
- Open-Meteo weather for 10 Turkish cities
- Stored as timestamped Parquet files

**Silver Layer** (Normalized)
- Schema validation (dtypes, column presence)
- Range checks (physically plausible bounds)
- Monotonicity checks (timestamp ordering)
- Duplicate removal
- Timezone standardization (Europe/Istanbul)
- Conservative gap handling:
  - Load: ≤2h forward fill, else drop
  - Weather: ≤6h linear interpolation, else drop

**Gold Layer** (Feature-Engineered)
- Lag features: 1, 2, 3, 6, 12, 24, 168 hours
- Rolling statistics: 3, 6, 12, 24, 168-hour windows
- Fourier terms: 24h, 168h periodicities
- Calendar features: hour, day, week, month, holidays, Ramadan
- Weather features: HDD/CDD, heat index, wind chill, temperature momentum
- Cyclical encodings: sin/cos for hour and day-of-week
- Versioned transformers: deterministic, serialized with models

---

## Getting Started

### Prerequisites
- **Docker & Docker Compose** (recommended for production deployment)
- **Python 3.11+** (for local development)
- **EPİAŞ account**: Register at [EPİAŞ Transparency Platform](https://www.epias.com.tr/en/transparency-platform/)

### Quick Start (Docker)

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ForeWatt.git
   cd ForeWatt
   ```

2. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your EPİAŞ credentials
   nano .env
   ```

3. **Launch services**
   ```bash
   docker-compose up -d
   ```

4. **Access services**
   - API: http://localhost:8000/docs (OpenAPI/Swagger)
   - Dashboard: http://localhost:8501
   - MLflow: http://localhost:5000
   - InfluxDB: http://localhost:8086

### Local Development

1. **Create virtual environment**
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate  # macOS/Linux
   # .venv\Scripts\activate    # Windows
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   # Add your EPİAŞ credentials
   ```

4. **Run weather data pipeline**
   ```bash
   python -m src.data.weather_fetcher
   ```

---

## Data Pipeline

### Weather Data (Open-Meteo)

The weather pipeline (`src/data/weather_fetcher.py`) fetches hourly data for Turkey's top 10 cities by population, covering 49.25% of the national population.

**Cities & Coverage**:
- Istanbul (18.3%), Ankara (6.9%), Izmir (5.2%), Bursa (3.8%), Antalya (3.2%)
- Konya (2.7%), Adana (2.7%), Şanlıurfa (2.6%), Gaziantep (2.6%), Kocaeli (2.5%)

**Features**:
- Temperature (2m), apparent temperature, humidity, wind speed, precipitation
- Cloud cover, surface pressure, rain
- Population-weighted national aggregation
- Heat index, wind chill, heating/cooling degree days (HDD/CDD)
- 60+ engineered demand features

**Usage**:
```python
from src.data.weather_fetcher import DemandWeatherFetcher

fetcher = DemandWeatherFetcher(cache_dir='.cache')
features = fetcher.run_pipeline(
    start_date='2020-01-01',
    end_date='2024-12-31',
    output_dir='./data'
)
```

### EPİAŞ Data (TODO)
- Hourly electricity consumption (national + 21 DSO regions)
- Day-ahead market clearing prices (MCP)
- Balancing market data
- Via eptr2 library with credentials from .env

---

## Models & Forecasting

### Training Protocol
- **Temporal cross-validation**: 4-fold expanding window
- **Test set**: 12 months held-out (strictly future data)
- **Hyperparameter optimization**: Bayesian search (Optuna), 30-50 trials
- **MLflow logging**: Parameters, metrics, artifacts, model versions

### Evaluation Metrics
- **Point forecasts**: sMAPE, MASE, MAE, RMSE
- **Probabilistic**: Pinball loss, CRPS, Winkler score
- **Uncertainty**: Coverage rate, interval width, sharpness
- **Anomaly detection**: Precision, recall, F1 at ≤5% FPR

### Performance Targets
- **Day-ahead (24h)**: 4-6% sMAPE, MASE < 1.0
- **Short-term (1-6h)**: 2-3% sMAPE, MASE < 0.5
- **Conformal intervals**: 90% coverage (±5% tolerance)
- **API latency**: p95 < 300ms (single horizon, CPU)

---

## Development

### Roadmap (Gantt Chart in Proposal)

**Phase 1: Foundation** (Weeks 1-2) ✅
- ✅ GitHub, Docker, MLflow setup
- ✅ EPİAŞ/Open-Meteo API access
- ✅ Project structure & medallion architecture

**Phase 2: Data Pipeline** (Weeks 1-6) 🚧
- ✅ Weather data fetcher (Open-Meteo)
- 🚧 EPİAŞ ingestion (eptr2)
- 🚧 Data quality checks
- 🚧 Feature engineering

**Phase 3: Baseline Models** (Weeks 3-6)
- Prophet, CatBoost, XGBoost
- Temporal cross-validation
- MLflow experiment tracking

**Phase 4: Advanced Models** (Weeks 5-9)
- N-HiTS (NeuralForecast)
- Ensemble aggregation
- Conformal calibration

**Phase 5: Anomaly & Dashboard** (Weeks 8-10)
- IsolationForest + diagnostics
- Streamlit UI rebuild
- API endpoints

**Phase 6: Optimization & Polish** (Weeks 9-12)
- EV load shifting (stretch goal)
- Final documentation
- Poster & report

### Contributing
This is an academic project (Koç University COMP 491). External contributions are welcome after January 2026.

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Advisor**: Prof. Dr. Gözde Gül Şahin
- **Team**: Zeynep Öykü Aslan, Kaan Altaş, Zeliha Paycı
- **Institution**: Koç University, Department of Computer Engineering
- **Course**: COMP 491 - Computer Engineering Design (Fall 2025)