# ForeWatt — Hourly Electricity Demand Forecasting for Turkey

> **ForeWatt** is a fully reproducible, open-source platform for **1–24h** electricity demand forecasting with **calibrated prediction intervals**, **actionable anomaly diagnostics**, and an optional **EV load-shifting optimizer**.
> Stack: **FastAPI**, **MLflow**, **InfluxDB**, **Streamlit**, **Docker Compose**.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#license)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](#requirements)
[![Reproducible](https://img.shields.io/badge/reproducible-mlflow%20%7C%20docker-informational)](#mlops--reproducibility)

---

## Quick Links

- **[Complete Documentation](COMPLETE_DOCUMENTATION.md)** - Full codebase documentation with all technical details
- **[Feature Engineering Guide](docs/FEATURE_ENGINEERING.md)** - Detailed feature engineering decisions and rationale
- **[Project Proposal](docs/Forewatt_COMP491_Proposal.txt)** - Academic project proposal with Gantt charts

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Data Pipeline](#data-pipeline)
- [Feature Engineering](#feature-engineering)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Development Status](#development-status)
- [License](#license)

---

## Features

### Core Capabilities

- **Multi-horizon forecasting**: 1-24 hour ahead predictions at hourly resolution
- **Calibrated uncertainty**: Split conformal prediction intervals with 90% coverage
- **Anomaly detection**: IsolationForest with level-shift, drift, and feature attribution diagnostics
- **Medallion data architecture**: Bronze → Silver → Gold layers with data quality gates
- **Population-weighted weather**: Open-Meteo integration for Turkey's top 10 cities (49% population coverage)
- **EPİAŞ integration**: Turkish electricity market data via eptr2 library
- **MLOps best practices**: MLflow experiment tracking, model registry, champion/challenger workflow

### Forecasting Models

- **Baselines**: Prophet, CatBoost, XGBoost, SARIMAX
- **Deep Learning**: N-HiTS (Neural Hierarchical Interpolation for Time Series) *(planned)*
- **Ensemble**: Weighted median aggregation with failover *(planned)*
- **Optional**: PatchTST, Temporal Fusion Transformer, foundation models *(planned)*

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
│  Lags, Rolls, Calendar, Weather → Gold (ML-Ready)               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL TRAINING & ENSEMBLE                     │
│  CatBoost + XGBoost + Prophet → MLflow Registry                 │
└─────────────────────────────────────────────────────────────────┘
```

**See [Complete Documentation](COMPLETE_DOCUMENTATION.md) for full architecture details.**

---

## Quick Start

### Prerequisites

- **Python 3.11+** (for local development)
- **Docker & Docker Compose** (for production deployment)
- **EPİAŞ account**: [Register here](https://www.epias.com.tr/en/transparency-platform/)
- **EVDS API key**: [Get from Turkish Central Bank](https://evds2.tcmb.gov.tr/)

### Local Development

```bash
# Clone repository
git clone https://github.com/yourusername/ForeWatt.git
cd ForeWatt

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API credentials (EPİAŞ, EVDS)
```

### Running the Pipeline

```bash
# Phase 1: Data Ingestion (Bronze & Silver)
python src/data/epias_fetcher.py          # EPİAŞ electricity data
python src/data/weather_fetcher.py        # Weather data (10 cities)
python src/data/evds_fetcher.py           # Macroeconomic indicators
python src/data/deflator_builder.py       # Inflation deflator
python src/data/deflate_prices.py         # Price deflation

# Phase 2: Feature Engineering (Gold)
python src/features/run_feature_pipeline.py

# Phase 3: Model Training
python src/models/run_baselines.py --all
```

### Docker Deployment (Production)

```bash
# Launch services
docker-compose up -d

# Access services:
# - API:       http://localhost:8000/docs
# - Dashboard: http://localhost:8501
# - MLflow:    http://localhost:5050
# - InfluxDB:  http://localhost:8086
```

---

## Data Pipeline

### Data Sources

| Source | Coverage | Status | Purpose |
|--------|----------|--------|---------|
| **EPİAŞ** | 2020-2024 (hourly) | ✅ Complete | Consumption, prices, generation |
| **Open-Meteo** | 10 cities (49.25% pop) | ✅ Complete | Population-weighted weather |
| **EVDS** | 2020-2024 (monthly) | ✅ Complete | Macroeconomic indicators |
| **Turkish Holidays** | 2020-2025 | ✅ Complete | 50 holiday days |

### Medallion Architecture

- **Bronze Layer**: Raw API data (Parquet + CSV dual format)
- **Silver Layer**: Normalized, validated, timezone-standardized
- **Gold Layer**: ML-ready features with versioned master dataset

**Total Data Volume**: ~397 MB (Bronze: 153 MB, Silver: 95 MB, Gold: 149 MB)

---

## Feature Engineering

### Master Dataset

- **Rows**: 43,848 (5 years hourly: 2020-01-01 to 2024-12-31)
- **Features**: 106 + 1 timestamp
- **Missing**: 0.03% (structural from lag features)
- **Version**: v1, Hash: `a567fe49`

### Feature Categories

| Category | Count | Examples |
|----------|-------|----------|
| **Lag Features** | 16 | consumption_lag_24h, temp_lag_168h, price_lag_1h |
| **Rolling Features** | 27 | consumption_rolling_mean_24h, temp_rolling_std_168h |
| **Calendar Features** | 12 | is_holiday_hour, dow_sin, month_cos |
| **Weather Features** | 35+ | temp_national, HDD, CDD, heat_index, wind_chill |
| **Price Features** | 17+ | price_real, priceEur_real (inflation-adjusted) |

**See [Feature Engineering Guide](docs/FEATURE_ENGINEERING.md) for complete details and design decisions.**

---

## Model Performance

### Baseline Model Comparison

| Model | MAE | RMSE | MAPE | sMAPE | MASE | Training Time |
|-------|-----|------|------|-------|------|---------------|
| **CatBoost** | **519.9** | 845.5 | 1.24 | 1.25 | **0.27** | 2.8s |
| **XGBoost** | 527.6 | 826.8 | 1.27 | 1.28 | 0.28 | 4.8s |
| Prophet | 2644 | 3284 | 7.15 | 6.81 | 1.37 | 20.9s |
| SARIMAX | 54608 | 62994 | 138 | 155 | 29.2 | 9.5s |

**Winner**: CatBoost (MASE < 1.0 = better than naive seasonal forecast)

### Performance Targets

- **Day-ahead (24h)**: 4-6% sMAPE, MASE < 1.0
- **Short-term (1-6h)**: 2-3% sMAPE, MASE < 0.5
- **API latency**: p95 < 300ms (single horizon, CPU)

---

## Project Structure

```
ForeWatt/
├── src/                           # Core source code (27 modules, 7,347 lines)
│   ├── data/                      # Data ingestion (11 modules, 3,766 lines)
│   ├── features/                  # Feature engineering (5 modules, 1,419 lines)
│   ├── models/                    # Model training (6 modules, 2,162 lines)
│   └── analysis/                  # Visualization & validation
│
├── data/                          # Medallion architecture (397 MB)
│   ├── bronze/                    # Raw API data (153 MB)
│   ├── silver/                    # Normalized data (95 MB)
│   └── gold/                      # ML-ready features (149 MB)
│       └── master/                # Master dataset (106 features)
│
├── api/                           # FastAPI REST service
├── dashboard/                     # Streamlit UI
├── tests/                         # Unit & integration tests
├── reports/                       # Model comparison results
├── docs/                          # Documentation
│   ├── FEATURE_ENGINEERING.md     # Feature engineering details
│   └── Forewatt_COMP491_Proposal.txt
│
├── COMPLETE_DOCUMENTATION.md      # Full codebase documentation
├── README.md                      # This file
├── requirements.txt               # Python dependencies (50+ packages)
├── docker-compose.yml             # Multi-service orchestration
└── .env.example                   # Environment variable template
```

---

## Documentation

### Available Documentation

1. **[COMPLETE_DOCUMENTATION.md](COMPLETE_DOCUMENTATION.md)** - Comprehensive codebase documentation
   - Complete architecture overview
   - All source code module documentation
   - Data pipeline details
   - Feature engineering explanations
   - Model training protocols
   - Configuration and setup guides

2. **[docs/FEATURE_ENGINEERING.md](docs/FEATURE_ENGINEERING.md)** - Feature engineering guide
   - Design philosophy and rationale
   - Every feature explained
   - Validation results
   - Missing value strategy
   - Versioning approach

3. **[docs/Forewatt_COMP491_Proposal.txt](docs/Forewatt_COMP491_Proposal.txt)** - Project proposal
   - Academic project background
   - Gantt charts and milestones
   - Research objectives

### Key Scripts Documentation

| Script | Purpose | Status |
|--------|---------|--------|
| `src/data/epias_fetcher.py` | Fetch 12 EPİAŞ datasets | ✅ Complete |
| `src/data/weather_fetcher.py` | Fetch & engineer 60+ weather features | ✅ Complete |
| `src/data/evds_fetcher.py` | Fetch Turkish macro indicators | ✅ Complete |
| `src/data/deflator_builder.py` | Build inflation deflator (DFM) | ✅ Complete |
| `src/data/deflate_prices.py` | Deflate electricity prices | ✅ Complete |
| `src/features/run_feature_pipeline.py` | Orchestrate feature pipeline | ✅ Complete |
| `src/models/run_baselines.py` | Train baseline models | ✅ Complete |

**See [COMPLETE_DOCUMENTATION.md](COMPLETE_DOCUMENTATION.md#source-code-documentation) for detailed module documentation.**

---

## Development Status

### Phase 1: Foundation ✅ (Complete)
- GitHub, Docker, MLflow setup
- EPİAŞ/Open-Meteo API access
- Project structure & medallion architecture

### Phase 2: Data Pipeline ✅ (Complete)
- Weather data fetcher (60+ engineered features)
- EPİAŞ ingestion (12 datasets, 5 years)
- Deflation pipeline (TL normalization)
- External features (FX/Gold)
- Calendar features (Turkish holidays)
- Lag features (16 features)
- Rolling features (27 features)
- Master dataset (106 features)

### Phase 3: Baseline Models ✅ (Complete)
- Prophet, CatBoost, XGBoost, SARIMAX
- MLflow experiment tracking
- Model comparison report

### Phase 4: Advanced Models (In Progress)
- N-HiTS (Neural Hierarchical Interpolation)
- PatchTST, Temporal Fusion Transformer
- Ensemble aggregation
- Conformal prediction intervals

### Phase 5: Production (Planned)
- Anomaly detection (IsolationForest + diagnostics)
- Streamlit dashboard rebuild
- FastAPI endpoints
- EV load shifting optimizer (stretch goal)

---

## Team

- **Advisor**: Prof. Dr. Gözde Gül Şahin
- **Team**: Zeynep Öykü Aslan, Kaan Altaş, Zeliha Paycı
- **Institution**: Koç University, Department of Computer Engineering
- **Course**: COMP 491 - Computer Engineering Design (Fall 2025)

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Contributing

This is an academic project (Koç University COMP 491). External contributions are welcome after January 2026.

---

## Acknowledgments

- **EPİAŞ** for electricity market data
- **Open-Meteo** for weather API
- **Turkish Central Bank (TCMB)** for EVDS API
- **Koç University** for academic support

---

**Last Updated**: November 18, 2025
**Version**: 1.0
**Status**: Phase 2 Complete (Data Pipeline & Feature Engineering ✅)
**Next**: Advanced Models (N-HiTS, Ensemble, Conformal Prediction)

---

## Quick Reference

### Environment Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Edit with your credentials
```

### Run Complete Pipeline
```bash
python src/data/epias_fetcher.py
python src/data/weather_fetcher.py
python src/data/evds_fetcher.py
python src/data/deflator_builder.py
python src/data/deflate_prices.py
python src/features/run_feature_pipeline.py
python src/models/run_baselines.py --all
```

### Docker Services
```bash
docker-compose up -d
# API: http://localhost:8000/docs
# Dashboard: http://localhost:8501
# MLflow: http://localhost:5050
```

**For detailed instructions, see [COMPLETE_DOCUMENTATION.md](COMPLETE_DOCUMENTATION.md)**
