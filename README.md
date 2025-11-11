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
- [Data Sources & Coverage](#data-sources--coverage)
- [Complete Data Pipeline](#complete-data-pipeline)
- [Key Design Decisions](#key-design-decisions)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Data Pipeline](#data-pipeline)
- [Models & Forecasting](#models--forecasting)
- [Validation & Testing](#validation--testing)
- [Development](#development)
- [Data Summary](#data-summary)
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

## Data Sources & Coverage

### 1. EPİAŞ (Turkish Electricity Market)
**Source**: EPİAŞ Transparency Platform via `eptr2` library
**Status**: ✅ Full 2020-2024 data fetched (~43,824 hourly records)
**Datasets** (12):
- **Consumption**: Actual load (target variable), day-ahead forecast
- **Prices**: PTF (day-ahead), SMF (balancing), IDM (intraday), WAP (weighted average)
- **Generation**: Real-time generation by source, available capacity (EAK), generation plans (KGÜP)
- **Forecasts**: Wind forecasts, hydro reservoir data

### 2. Weather Data (Open-Meteo)
**Status**: ✅ Full 2020-2024 data fetched
**Coverage**: 10 Turkish cities (49.25% population)
**Cities**: Istanbul, Ankara, Izmir, Bursa, Antalya, Konya, Adana, Sanliurfa, Gaziantep, Kocaeli
**Variables** (8): Temperature, humidity, precipitation, rain, cloud cover, wind speed, pressure, apparent temperature
**Processing**: Population-weighted national aggregates, 60+ engineered features

### 3. Macroeconomic Indicators (EVDS)
**Source**: Turkish Central Bank API
**Status**: ✅ Bronze data available
**Indicators**: TÜFE (CPI), ÜFE (PPI), M2 (money supply), TL_FAIZ (interest rates)
**Purpose**: Deflation pipeline for TL normalization

### 4. External Features (FX & Gold)
**Source**: EVDS API
**Status**: ✅ Scripts complete
**Variables**: USD/TRY, EUR/TRY, XAU/TRY, FX basket (0.5×USD + 0.5×EUR)
**Features**: Daily rates, 7/30-day momentum, 30-day volatility

---

## Complete Data Pipeline

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────────┐
│ BRONZE LAYER (Raw Ingestion)                                │
│ • EPİAŞ: 12 datasets (hourly, 2020-2024)                    │
│ • Weather: 10 cities × 8 variables (hourly)                 │
│ • Macro: TÜFE/ÜFE/M2/TL_FAIZ (monthly)                      │
│ • FX/Gold: USD/EUR/XAU vs TRY (daily)                       │
│ Format: Parquet (primary) + CSV (secondary)                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ SILVER LAYER (Normalized & Validated)                       │
│ • Quality gates: Schema, range, monotonicity, duplicates    │
│ • Timezone standardization (Europe/Istanbul)                │
│ • Conservative gap handling (forward fill/interpolation)    │
│ • Population-weighted weather aggregates                    │
│ • Deflator indices (DID_index, base=100 at 2022-01)        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ GOLD LAYER (ML-Ready Features)                              │
│ • Deflated prices (real TL/MWh, inflation-adjusted)         │
│ • Weather demand features (60+ engineered)                   │
│ • External features (FX/Gold with momentum/volatility)      │
│ • Calendar features (holidays, temporal, cyclical)          │
│ • Lag features (1h, 2h, 3h, 24h, 168h)                      │
│ • Rolling stats (24h, 7d means & std)                       │
└─────────────────────────────────────────────────────────────┘
```

### Pipeline Execution Order

#### Phase 1: Data Ingestion (Bronze & Silver Layers)
```bash
# Step 1: Fetch raw data (Bronze layer)
python src/data/epias_fetcher.py          # EPİAŞ electricity data (12 datasets)
python src/data/weather_fetcher.py        # Weather data (10 cities)
python src/data/evds_fetcher.py           # Macroeconomic indicators

# Step 2: Build deflation pipeline
python src/data/deflator_builder.py       # Create deflator indices (Factor Analysis + DFM)
python src/data/deflate_prices.py         # Deflate electricity prices

# Step 3: Build external features (optional)
python src/data/external_features_fetcher.py  # FX/Gold features with momentum/volatility

# Step 4: Build calendar features (optional)
python src/data/calendar_builder.py       # Turkish holiday calendars
python src/analysis/make_gold_calendar_features.py  # Calendar features

# Step 5: Validate
python src/data/validate_deflation_pipeline.py  # 7 automated tests
```

#### Phase 2: Feature Engineering (Gold Layer)
```bash
# Complete pipeline (recommended - runs all steps)
python src/features/run_feature_pipeline.py

# Or run individual modules:
python src/features/lag_features.py       # Lag features (1h-168h for consumption, temp, price)
python src/features/rolling_features.py   # Rolling stats (24h, 168h windows)
python src/features/merge_features.py     # Merge all gold layers into master dataset
```

#### Phase 3: Model Training (TODO)
```bash
# python src/models/train.py
```

#### Phase 4: Serve Predictions (TODO)
```bash
# docker-compose up -d
```

## Feature Engineering

### Modular Feature Pipeline

ForeWatt uses a **hybrid approach** to feature engineering:
- **Modular scripts** for key feature types (lag, rolling)
- **Unified merge script** to combine all gold layers
- **Versioned output** with date + feature hash

### Feature Types

#### 1. Lag Features (`lag_features.py`)
**Target variable (consumption)**:
- Lags: 1h, 2h, 3h, 6h, 12h, 24h, 48h, 168h

**Temperature (key weather driver)**:
- Lags: 1h, 2h, 3h, 24h, 168h

**Electricity price (economic signal)**:
- Lags: 1h, 24h, 168h

**Total**: 16 lag features

#### 2. Rolling Features (`rolling_features.py`)
**For consumption, temperature, price_ptf**:
- Windows: 24h, 168h (1 day, 1 week)
- Statistics: mean, std, min, max
- **Total**: 24 rolling features

**Derived volatility features**:
- `consumption_range_24h`: max - min (daily volatility)
- `consumption_cv_24h`: std / mean (coefficient of variation)
- `temp_range_24h`: diurnal temperature range
- **Total**: 3 derived features

**Total rolling features**: 27

#### 3. Weather Demand Features (existing)
Already engineered in `weather_fetcher.py`:
- Heating/Cooling Degree Days (HDD/CDD)
- Heat index & wind chill
- Extreme temperature flags
- Temperature momentum & shocks
- **Total**: 60+ features

#### 4. Master Dataset (`merge_features.py`)
Combines all gold layers:
- Target: `consumption` (Silver EPİAŞ)
- Weather: 60+ demand features (Gold)
- Prices: Deflated PTF (Gold)
- External: FX/Gold (Gold - optional)
- Calendar: Holidays + temporal (Gold - optional)
- Lags: 16 lag features (Gold)
- Rolling: 27 rolling features (Gold)

**Output**: `data/gold/master/master_v1_{date}_{hash}.parquet`
**Metadata**: `master_v1_{date}_{hash}_metadata.json`

### Running Feature Engineering

#### Quick Start
```bash
# Setup environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run complete pipeline
python src/features/run_feature_pipeline.py
```

#### Custom Options
```bash
# Custom date range
python src/features/run_feature_pipeline.py --start 2020-01-01 --end 2023-12-31

# Custom version
python src/features/run_feature_pipeline.py --version v2
```

#### Individual Modules
```bash
# Run specific feature modules
python src/features/lag_features.py
python src/features/rolling_features.py
python src/features/merge_features.py
```

### Output Files

**Lag features**:
- `data/gold/lag_features/lag_features_2020-01-01_2024-12-31.parquet`
- `data/gold/lag_features/lag_features_2020-01-01_2024-12-31.csv`

**Rolling features**:
- `data/gold/rolling_features/rolling_features_2020-01-01_2024-12-31.parquet`
- `data/gold/rolling_features/rolling_features_2020-01-01_2024-12-31.csv`

**Master dataset**:
- `data/gold/master/master_v1_2025-11-11_{hash}.parquet`
- `data/gold/master/master_v1_2025-11-11_{hash}.csv`
- `data/gold/master/master_v1_2025-11-11_{hash}_metadata.json`

---

### Key Scripts

| Script | Purpose | Status |
|--------|---------|--------|
| **Data Ingestion** |
| `epias_fetcher.py` | Fetch 12 EPİAŞ electricity datasets | ✅ Complete |
| `weather_fetcher.py` | Fetch & engineer 60+ weather features | ✅ Complete |
| `evds_fetcher.py` | Fetch Turkish macro indicators | ✅ Complete |
| `deflator_builder.py` | Build inflation deflator indices (DFM + Factor Analysis) | ✅ Complete |
| `deflate_prices.py` | Deflate electricity prices to real values | ✅ Complete |
| `external_features_fetcher.py` | Fetch FX/Gold features | ✅ Complete |
| `calendar_builder.py` | Build Turkish holiday calendars | ✅ Complete |
| `validate_deflation_pipeline.py` | Validate TL normalization (7 tests) | ✅ Complete |
| **Feature Engineering** |
| `lag_features.py` | Generate lag features (16 features) | ✅ Complete |
| `rolling_features.py` | Generate rolling window features (27 features) | ✅ Complete |
| `merge_features.py` | Merge all gold layers into master dataset | ✅ Complete |
| `run_feature_pipeline.py` | Orchestrate complete feature pipeline | ✅ Complete |

---

## Key Design Decisions

### Why Deflation is Separate from External Features?
- **Deflator (DID)**: Removes domestic inflation (TÜFE, ÜFE, M2, TL_FAIZ) from electricity prices
- **External Features (FX, Gold)**: Preserved as predictive signals
- **Reason**: FX/import shocks ARE predictive (e.g., gas price spike → electricity price spike). Removing them destroys valuable information.

### Why DFM over Simple CPI?
- TÜFE alone includes import price effects (overfits to commodity shocks)
- Dynamic Factor Model (DFM) extracts "pure" domestic inflation, filters FX/commodity noise
- Kalman smoothing reduces noise

### Why Linear Interpolation (Monthly → Hourly)?
- Avoids step changes at month boundaries
- Reflects gradual inflation accumulation
- Monthly → Daily (linear) → Hourly (forward fill)

### Why Population-Weighted Weather?
- Turkey's population is highly concentrated (top 10 cities = 49.25%)
- National aggregate reflects where people (and electricity demand) actually are
- Regional spread (temp_std) captures geographic diversity

### Why Dual Format Storage (Parquet + CSV)?
- **Parquet**: Fast, compressed, columnar (primary for ML pipelines)
- **CSV**: Human-readable, Excel-compatible (secondary for inspection)
- Best of both worlds: performance + accessibility

---

## Project Structure

```
ForeWatt/
├── src/                           # Core source code (Python 3.11+)
│   ├── __init__.py
│   ├── data/                      # Data ingestion & processing
│   │   ├── __init__.py
│   │   ├── epias_fetcher.py       # EPİAŞ electricity data
│   │   ├── weather_fetcher.py     # Open-Meteo population-weighted weather
│   │   ├── evds_fetcher.py        # Turkish Central Bank macro data
│   │   ├── deflator_builder.py    # Inflation deflator (DFM + Factor Analysis)
│   │   ├── deflate_prices.py      # Price deflation
│   │   └── external_features_fetcher.py  # FX/Gold features
│   ├── features/                  # Feature engineering (modular)
│   │   ├── __init__.py
│   │   ├── lag_features.py        # Lag features (16 features)
│   │   ├── rolling_features.py    # Rolling window features (27 features)
│   │   ├── merge_features.py      # Master dataset merger
│   │   └── run_feature_pipeline.py  # Pipeline orchestrator
│   ├── analysis/                  # Exploratory analysis & validation
│   ├── models/                    # Forecasting models (TODO)
│   ├── uncertainty/               # Conformal prediction (TODO)
│   ├── anomaly/                   # IsolationForest + diagnostics (TODO)
│   ├── optimization/              # EV load shifting (TODO)
│   └── evaluation/                # Metrics & validation (TODO)
│
├── data/                          # Medallion architecture
│   ├── bronze/                    # Raw API data (timestamped Parquet + CSV)
│   │   ├── epias/                 # 12 EPİAŞ datasets (2020-2024)
│   │   ├── demand_weather/        # 10 city weather (hourly)
│   │   ├── evds/                  # Macro indicators (monthly)
│   │   └── external/              # FX/Gold rates (daily)
│   ├── silver/                    # Normalized, validated
│   │   ├── epias/                 # Deduplicated, timezone-standardized
│   │   ├── demand_weather/        # Population-weighted national aggregates
│   │   ├── evds/                  # Deflator indices (DID_index)
│   │   └── calendar/              # Holiday tables
│   ├── gold/                      # Feature-engineered, ML-ready
│   │   ├── demand_features/       # 60+ weather demand features
│   │   ├── deflated_prices/       # Real TL/MWh (inflation-adjusted)
│   │   ├── external_features/     # FX/Gold with momentum/volatility (optional)
│   │   ├── calendar_features/     # Holiday & temporal features (optional)
│   │   ├── lag_features/          # Lag features (16 features)
│   │   ├── rolling_features/      # Rolling window features (27 features)
│   │   └── master/                # Unified ML-ready dataset (v1_{date}_{hash})
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
   # Edit .env with your API credentials
   nano .env
   ```

   **Required variables**:
   ```env
   # EPİAŞ (Turkish Electricity Market)
   EPTR_USERNAME=your_email@example.com
   EPTR_PASSWORD=your_password

   # EVDS (Turkish Central Bank - for macro data & FX)
   EVDS_API_KEY=your_evds_api_key

   # InfluxDB
   INFLUXDB_TOKEN=your_token
   INFLUXDB_ORG=forewatt
   INFLUXDB_BUCKET=epias

   # MLflow
   MLFLOW_TRACKING_URI=http://localhost:5050

   # Service Ports
   API_PORT=8000
   DASHBOARD_PORT=8501
   MLFLOW_PORT=5050
   INFLUXDB_PORT=8086

   # Timezone
   TIMEZONE=Europe/Istanbul
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

## Validation & Testing

### Deflation Pipeline Validation
**Script**: `src/data/validate_deflation_pipeline.py`

**7 Automated Tests**:
1. **Prerequisites**: Dependencies, API keys, directory structure
2. **Synthetic Data**: Generate test inflation/price data
3. **EVDS Fetcher**: Validate macro data ingestion
4. **Deflator Builder**: Test Factor Analysis + DFM
5. **Interpolation**: Monthly → Daily → Hourly
6. **Price Deflation**: Apply deflation formula
7. **Output Quality**: Stationarity (ADF test), variance reduction (20-40%)

**Usage**:
```bash
# Dry-run mode (no API calls, synthetic data)
python src/data/validate_deflation_pipeline.py --dry-run

# Full validation (requires EVDS API key)
python src/data/validate_deflation_pipeline.py
```

### Data Quality Checks (Silver Layer)
- **Schema validation**: dtypes, column presence, required fields
- **Range checks**: Physically plausible bounds (e.g., load > 0, temp in [-50, 60])
- **Monotonicity checks**: Timestamp ordering, no duplicates
- **Gap handling**: Conservative forward fill (≤2h load, ≤6h weather)
- **Outlier detection**: >5 std from mean flagged for review

### Calendar Tests
**Scripts**: `tests/test_calendar*.py`
- Unit tests for calendar features
- End-to-end tests for holiday handling
- Half-day PM validation (e.g., Oct 28, 2025)

---

## Development

### Roadmap (Gantt Chart in Proposal)

**Phase 1: Foundation** (Weeks 1-2) ✅
- ✅ GitHub, Docker, MLflow setup
- ✅ EPİAŞ/Open-Meteo API access
- ✅ Project structure & medallion architecture

**Phase 2: Data Pipeline** (Weeks 1-6) ✅
- ✅ Weather data fetcher (Open-Meteo)
- ✅ EPİAŞ ingestion (eptr2) - 12 datasets, 2020-2024
- ✅ Data quality checks (schema, range, monotonicity)
- ✅ Feature engineering (60+ weather demand features)
- ✅ Deflation pipeline (TL normalization via DFM)
- ✅ External features (FX/Gold from EVDS)
- ✅ Calendar features (Turkish holidays)

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

## Data Summary

**Coverage**: 2020-01-01 to 2024-12-31 (5 years)
**Total Records**: ~43,824 hourly observations

**Features**:
- **Target**: consumption (Silver EPİAŞ)
- **EPİAŞ**: 12 datasets (consumption, prices, generation, forecasts)
- **Weather**: 60+ engineered demand features (HDD, CDD, heat index, momentum)
- **Macro**: Deflation indices (Factor Analysis + DFM)
- **External**: FX/Gold with momentum/volatility (optional)
- **Calendar**: Turkish holidays + temporal features (optional)
- **Lag features**: 16 features (consumption, temperature, price)
- **Rolling features**: 27 features (24h/168h windows + derived volatility)
- **Total features**: 100+ in master dataset

**Storage**:
- Dual format: Parquet (primary) + CSV (secondary)
- Bronze: Raw API data
- Silver: Normalized, validated
- Gold: ML-ready features + master dataset
- Versioning: `master_v{version}_{date}_{hash}.parquet`

---

## Acknowledgments

- **Advisor**: Prof. Dr. Gözde Gül Şahin
- **Team**: Zeynep Öykü Aslan, Kaan Altaş, Zeliha Paycı
- **Institution**: Koç University, Department of Computer Engineering
- **Course**: COMP 491 - Computer Engineering Design (Fall 2025)

---

**Last Updated**: 2025-11-11
**Pipeline Status**: Phase 2 Complete (Data ingestion & feature engineering ✅)
**Next Phase**: Model development (baseline models, N-HiTS, ensemble)

---

## Quick Start Guide

### 1. Environment Setup
```bash
# Clone repository
git clone https://github.com/yourusername/ForeWatt.git
cd ForeWatt

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
nano .env  # Add EPTR_USERNAME, EPTR_PASSWORD, EVDS_API_KEY
```

### 2. Run Data Pipeline
```bash
# Fetch data (Bronze & Silver layers)
python src/data/epias_fetcher.py
python src/data/weather_fetcher.py
python src/data/evds_fetcher.py

# Build deflation pipeline
python src/data/deflator_builder.py
python src/data/deflate_prices.py

# Validate
python src/data/validate_deflation_pipeline.py
```

### 3. Run Feature Engineering
```bash
# Complete pipeline (creates master dataset)
python src/features/run_feature_pipeline.py

# Output: data/gold/master/master_v1_{date}_{hash}.parquet
```

### 4. Next Steps (TODO)
```bash
# Train models
# python src/models/train.py

# Start services
# docker-compose up -d
```