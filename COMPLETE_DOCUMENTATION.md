# ForeWatt - Complete Codebase Documentation

**Version:** 1.0
**Last Updated:** November 18, 2025
**Project:** Hourly Electricity Demand Forecasting for Turkey
**Team:** Koç University COMP 491 Fall 2025
**License:** MIT

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture Overview](#architecture-overview)
3. [Data Sources & Coverage](#data-sources--coverage)
4. [Complete Data Pipeline](#complete-data-pipeline)
5. [Source Code Documentation](#source-code-documentation)
6. [Feature Engineering](#feature-engineering)
7. [Model Development](#model-development)
8. [Project Structure](#project-structure)
9. [Configuration & Setup](#configuration--setup)
10. [Validation & Testing](#validation--testing)
11. [Development Roadmap](#development-roadmap)
12. [Appendices](#appendices)

---

## Project Overview

### What is ForeWatt?

ForeWatt is a **fully reproducible, open-source platform** for **1-24 hour ahead electricity demand forecasting** with **calibrated prediction intervals**, **actionable anomaly diagnostics**, and an optional **EV load-shifting optimizer**.

### Technology Stack

- **Backend:** FastAPI
- **Experiment Tracking:** MLflow
- **Time Series Database:** InfluxDB
- **Dashboard:** Streamlit
- **Orchestration:** Docker Compose
- **ML Libraries:** Darts, Prophet, NeuralForecast, CatBoost, XGBoost
- **Data Processing:** Pandas, NumPy, PyArrow

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
- **Deep Learning**: N-HiTS (Neural Hierarchical Interpolation for Time Series)
- **Ensemble**: Weighted median aggregation with failover
- **Optional**: PatchTST, Temporal Fusion Transformer, foundation models (TimesFM, Moirai)

### Performance Targets

- **Day-ahead (24h)**: 4-6% sMAPE, MASE < 1.0
- **Short-term (1-6h)**: 2-3% sMAPE, MASE < 0.5
- **Conformal intervals**: 90% coverage (±5% tolerance)
- **API latency**: p95 < 300ms (single horizon, CPU)

---

## Architecture Overview

### System Architecture

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

### Medallion Architecture

**Bronze Layer (Raw Data)**
- EPİAŞ hourly load/price (via eptr2)
- Open-Meteo weather for 10 Turkish cities
- EVDS macroeconomic indicators
- Stored as timestamped Parquet + CSV files

**Silver Layer (Normalized & Validated)**
- Schema validation (dtypes, column presence)
- Range checks (physically plausible bounds)
- Monotonicity checks (timestamp ordering)
- Duplicate removal
- Timezone standardization (Europe/Istanbul)
- Conservative gap handling

**Gold Layer (Feature-Engineered)**
- Lag features: 16 features (consumption, temperature, price)
- Rolling statistics: 27 features (24h/168h windows)
- Calendar features: 12 features (holidays, temporal, cyclical)
- Weather features: 60+ engineered demand features
- Versioned transformers and master dataset

---

## Data Sources & Coverage

### 1. EPİAŞ (Turkish Electricity Market)

**Source**: EPİAŞ Transparency Platform via `eptr2` library
**Status**: ✅ Full 2020-2024 data fetched (~43,824 hourly records)
**Coverage**: 5 years of hourly data (2020-01-01 to 2024-12-31)

**Datasets (12)**:

1. **consumption_actual** - Real-time consumption (target variable)
2. **consumption_forecast** - Day-ahead consumption forecast
3. **price_ptf** - Day-Ahead Market Price (PTF/MCP)
4. **price_smf** - Balancing Power Market Price
5. **price_idm** - Intraday Market Quantity
6. **price_wap** - Weighted Average Price
7. **generation_realtime** - Generation by source/plant
8. **capacity_eak** - Available Capacity (EAK)
9. **plan_kgup** - Day-Ahead Generation Plan
10. **plan_kudup** - Bilateral Contract Plan
11. **wind_forecast** - Wind Generation Forecast
12. **hydro_reservoir_volume** - Hydropower Reservoir Volume

### 2. Weather Data (Open-Meteo)

**Status**: ✅ Full 2020-2024 data fetched
**Coverage**: 10 Turkish cities (49.25% population)
**Resolution**: Hourly

**Cities Covered**:
- Istanbul (18.3%), Ankara (6.9%), Izmir (5.2%), Bursa (3.8%), Antalya (3.2%)
- Konya (2.7%), Adana (2.7%), Şanlıurfa (2.6%), Gaziantep (2.6%), Kocaeli (2.5%)

**Variables (8)**:
- temperature_2m, relative_humidity_2m
- precipitation, rain
- cloud_cover
- wind_speed_10m
- surface_pressure
- apparent_temperature

**Processing**: Population-weighted national aggregates, 60+ engineered features

### 3. Macroeconomic Indicators (EVDS)

**Source**: Turkish Central Bank (EVDS) API
**Status**: ✅ Bronze data available
**Frequency**: Monthly

**Indicators**:
- **TÜFE**: Turkish CPI (Consumer Price Index, 2003=100)
- **ÜFE**: Turkish PPI (Producer Price Index, 2003=100)
- **M2**: Money supply M2 (Million TL)
- **TL_FAIZ**: TL deposit interest rate (%)

**Purpose**: Deflation pipeline for TL normalization

### 4. External Features (FX & Gold)

**Source**: EVDS API
**Status**: ✅ Scripts complete
**Frequency**: Daily → Hourly (forward fill)

**Variables**:
- USD/TRY, EUR/TRY, XAU/TRY
- FX basket: 0.5×USD + 0.5×EUR

**Engineered Features**:
- 7-day and 30-day momentum
- 30-day volatility

### 5. Turkish Holiday Calendar

**Source**: Static JSON file
**Status**: ✅ Complete (2020-2025)
**Coverage**: 50 holiday days

**Holiday Types**:
- Official holidays (7 types): New Year, Republic Day, etc.
- Religious holidays (2 types): Ramazan Bayramı, Kurban Bayramı
- Half-day holidays: October 28 PM only

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

#### Phase 3: Model Training

```bash
# Run all baseline models
python src/models/run_baselines.py --all

# Or run individual models
python src/models/run_baselines.py --models prophet catboost xgboost
```

---

## Source Code Documentation

### Data Modules (`src/data/`)

#### 1. EPİAŞ Data Fetcher (`epias_fetcher.py`)

**Purpose**: Fetches electricity market data from Turkey's EPİAŞ Transparency Platform.

**Main Class**: `EpiasDataFetcher`

**Key Features**:
- Fetches 12 different datasets (consumption, prices, generation, capacity, forecasts)
- Medallion architecture (Bronze → Silver layers)
- Dual format export (Parquet + CSV)
- Automatic retry logic and rate limiting
- Europe/Istanbul timezone handling
- Date range chunking (respects 1-year API limit)

**Main Methods**:
```python
fetch_dataset(dataset_name, start_date, end_date) -> pd.DataFrame
run_pipeline(start_date, end_date, output_dir) -> Dict[str, pd.DataFrame]
validate_silver_layer(df, dataset_name) -> Tuple[pd.DataFrame, Dict]
```

**Dependencies**: `eptr2`, `pandas`, `numpy`, `python-dotenv`

**Input**: API credentials (EPTR_USERNAME, EPTR_PASSWORD), date range
**Output**: Bronze (raw) and Silver (validated) Parquet/CSV files

**File Location**: `/home/user/ForeWatt/src/data/epias_fetcher.py`

---

#### 2. EVDS Macro Fetcher (`evds_fetcher.py`)

**Purpose**: Fetches Turkish macroeconomic indicators from Central Bank (EVDS).

**Main Function**: `fetch_evds_data(start_date, end_date) -> pd.DataFrame`

**Fetched Series**:
- TÜFE: Turkish CPI (2003=100)
- ÜFE: Turkish PPI (2003=100)
- M2: Money supply M2 (Million TL)
- TL_FAIZ: TL deposit interest rate (%)

**Features**:
- Monthly frequency data
- Date normalization (YYYY-MM format)
- Rebase index to custom base date
- Dual format output (CSV + Parquet)

**Output Files**:
- `data/bronze/macro/macro_evds_raw.csv`
- `data/bronze/macro/macro_evds_YYYY-MM-DD_YYYY-MM-DD.parquet`

**Dependencies**: `evds` (evdspy), `pandas`, `python-dotenv`

**File Location**: `/home/user/ForeWatt/src/data/evds_fetcher.py`

---

#### 3. Weather Fetcher (`weather_fetcher.py`)

**Purpose**: Fetches population-weighted weather data for demand forecasting.

**Main Class**: `DemandWeatherFetcher`

**Coverage**:
- Top 10 Turkish cities by population (49.25% national coverage)
- Population-weighted aggregation to national features

**Weather Variables (8)**:
- temperature_2m
- relative_humidity_2m
- precipitation, rain
- cloud_cover
- wind_speed_10m
- surface_pressure
- apparent_temperature

**Engineered Features (60+)**:
- Heating/Cooling Degree Days (HDD/CDD)
- Heat index, wind chill
- Temperature lags (1h, 2h, 3h, 24h, 168h)
- Rolling statistics (24h, 7d means/std)
- Extreme temperature flags
- Cyclical encodings (hour, day of week)

**Main Methods**:
```python
fetch_city_weather(city, start_date, end_date) -> pd.DataFrame
fetch_all_cities(start_date, end_date) -> Dict[str, pd.DataFrame]
create_national_features(city_data) -> pd.DataFrame
create_demand_features(national_weather) -> pd.DataFrame
run_pipeline(start_date, end_date) -> pd.DataFrame
```

**Output Layers**:
- Bronze: Individual city weather (10 cities)
- Silver: National aggregated weather
- Gold: Engineered demand features (60+ columns)

**Dependencies**: `openmeteo_requests`, `requests_cache`, `retry_requests`, `pandas`, `numpy`

**File Location**: `/home/user/ForeWatt/src/data/weather_fetcher.py`

---

#### 4. External Features Fetcher (`external_features_fetcher.py`)

**Purpose**: Fetches FX rates and gold prices as exogenous features.

**Main Class**: `ExternalFeaturesFetcher`

**Rationale**:
- Turkey imports ~70% of energy (natural gas, oil)
- Electricity prices correlate with FX rates (import cost pass-through)
- Gold reflects capital flight and inflation expectations

**Fetched Series (EVDS)**:
- USD/TRY: US Dollar exchange rate
- EUR/TRY: Euro exchange rate
- XAU/TRY: Gold price in Turkish Lira

**Derived Features**:
- FX_basket: Weighted average (0.5 × USD + 0.5 × EUR)
- USD_TRY_mom7d, EUR_TRY_mom7d: 7-day momentum
- USD_TRY_mom30d, EUR_TRY_mom30d: 30-day momentum
- FX_volatility: 30-day rolling std

**Methodology**:
- Daily data → Hourly via forward fill
- Preserves import/FX shock signals

**Main Methods**:
```python
fetch_daily_fx(start_date, end_date) -> pd.DataFrame
convert_to_hourly(daily_df) -> pd.DataFrame
save_features(daily_df, hourly_df, start_date, end_date)
```

**Output**: `data/gold/external/fx_features_{daily,hourly}_YYYY-MM-DD_YYYY-MM-DD.{parquet,csv}`

**Dependencies**: `evds`, `pandas`, `numpy`

**File Location**: `/home/user/ForeWatt/src/data/external_features_fetcher.py`

---

#### 5. Calendar Builder (`calendar_builder.py`)

**Purpose**: Builds Turkish holiday calendar from static JSON.

**Input**: `src/data/static/tr_holidays_2020_2025.json`

**Holiday Types**:
- Turkish official holidays (New Year, Republic Day, etc.)
- Religious holidays (Ramadan, Kurban Bayramı) - multi-day
- Half-day holidays (e.g., Oct 28 PM only)

**Main Function**: `build_calendar_tables()`

**Output Layers**:
- Bronze: `calendar_raw.csv` - Raw holiday spans
- Silver: `calendar_days.csv` - Exploded daily holidays
- Silver: `calendar_full_days.csv` - Full daily calendar with flags

**Output Columns**:
- date_only, dow, month
- is_weekend, is_holiday_day
- is_holiday_weekend, is_holiday_weekday
- holiday_name, half_day

**Dependencies**: `pandas`, `json`

**File Location**: `/home/user/ForeWatt/src/data/calendar_builder.py`

---

#### 6. Deflator Builder (`deflator_builder.py`)

**Purpose**: Builds deflation indices to convert nominal TL prices to real values.

**Methods**:
1. **Baseline DID**: Factor Analysis on TÜFE, ÜFE, M2, TL_FAIZ
2. **DFM DID**: Dynamic Factor Model with Kalman smoothing

**Main Functions**:
```python
build_did_baseline() -> None
build_did_dfm() -> None
```

**Methodology**:
1. Load EVDS macro data (TÜFE, ÜFE, M2, TL_FAIZ)
2. Compute growth rates (MoM, YoY)
3. Extract inflation factor via Factor Analysis
4. Calibrate to TÜFE using OLS
5. Build cumulative deflator index (base=100 at 2022-01)

**Output Files**:
- `data/silver/macro/deflator_did_baseline.{parquet,csv}`
- `data/silver/macro/deflator_did_dfm.{parquet,csv}`

**Output Columns**:
- DATE (YYYY-MM format)
- DID_monthly, DID_index (base=100 at 2022-01)
- pi_hat_monthly (inflation estimate)

**Dependencies**: `sklearn`, `statsmodels`, `pandas`, `numpy`

**File Location**: `/home/user/ForeWatt/src/data/deflator_builder.py`

---

#### 7. Price Deflator (`deflate_prices.py`)

**Purpose**: Applies deflation to EPİAŞ price datasets.

**Main Class**: `PriceDeflator`

**Datasets Requiring Deflation**:
- price_ptf, price_smf, price_idm, price_wap (all in TL/MWh)

**Methodology**:
1. Load monthly deflator index
2. Interpolate monthly → daily (linear) → hourly (forward fill)
3. Join hourly prices with hourly deflator
4. Apply formula: `real_price = nominal_price / (DID_index / 100)`

**Main Methods**:
```python
deflate_dataset(dataset_name, start_date, end_date) -> pd.DataFrame
deflate_all_price_datasets(start_date, end_date) -> Dict
```

**Output**: `data/gold/epias/{dataset}_deflated_YYYY-MM-DD_YYYY-MM-DD.{parquet,csv}`

**Output Columns**: Original columns + `*_real` columns

**Dependencies**: `pandas`, `numpy`, `pathlib`

**File Location**: `/home/user/ForeWatt/src/data/deflate_prices.py`

---

#### 8. Validation Pipeline (`validate_deflation_pipeline.py`)

**Purpose**: Tests complete TL normalization pipeline.

**Main Class**: `DeflationPipelineValidator`

**Validation Steps**:
1. Check prerequisites (API keys, dependencies)
2. Create synthetic test data
3. Test EVDS fetcher
4. Test deflator builder (baseline + DFM)
5. Test interpolation (monthly → daily → hourly)
6. Test price deflation
7. Validate output quality (stationarity, variance reduction)

**Usage**:
```bash
python src/data/validate_deflation_pipeline.py --dry-run  # Synthetic data
python src/data/validate_deflation_pipeline.py --full     # Real data
```

**Dependencies**: `pandas`, `numpy`, `statsmodels`

**File Location**: `/home/user/ForeWatt/src/data/validate_deflation_pipeline.py`

---

### Features Modules (`src/features/`)

#### 1. Lag Features Generator (`lag_features.py`)

**Purpose**: Creates lagged features for target and exogenous variables.

**Main Class**: `LagFeaturesGenerator`

**Lag Configurations**:
- **Consumption lags**: 1h, 2h, 3h, 6h, 12h, 24h, 48h, 168h (1 week)
- **Temperature lags**: 1h, 2h, 3h, 24h, 168h
- **Price lags**: 1h, 24h, 168h

**Total Features**: 16 lag features

**Rationale**:
- Short lags (1-3h): Capture inertia and autoregressive patterns
- Daily lag (24h): Capture day-over-day changes
- Weekly lag (168h): Capture week-over-week patterns

**Main Methods**:
```python
load_consumption(start_date, end_date) -> pd.DataFrame
load_temperature(start_date, end_date) -> pd.DataFrame
load_price(start_date, end_date) -> pd.DataFrame
create_lags(df, column, lags) -> pd.DataFrame
generate_all_lags(start_date, end_date) -> pd.DataFrame
run_pipeline(start_date, end_date) -> pd.DataFrame
```

**Output**: `data/gold/lag_features/lag_features_YYYY-MM-DD_YYYY-MM-DD.{parquet,csv}`

**Dependencies**: `pandas`, `numpy`

**File Location**: `/home/user/ForeWatt/src/features/lag_features.py`

---

#### 2. Rolling Features Generator (`rolling_features.py`)

**Purpose**: Creates rolling window statistics for time series.

**Main Class**: `RollingFeaturesGenerator`

**Windows**: 24h (daily), 168h (weekly)

**Statistics**: mean, std, min, max

**Total Features**: 27 rolling features

**Features Created**:
- `{variable}_rolling_{stat}_{window}h` (e.g., `consumption_rolling_mean_24h`)
- Derived features:
  - `consumption_range_24h`: Daily volatility (max - min)
  - `consumption_cv_24h`: Coefficient of variation (std / mean)
  - `temp_range_24h`: Diurnal temperature range

**Main Methods**:
```python
create_rolling_stats(df, column, windows, stats) -> pd.DataFrame
create_derived_features(df) -> pd.DataFrame
generate_all_rolling(start_date, end_date) -> pd.DataFrame
run_pipeline(start_date, end_date) -> pd.DataFrame
```

**Output**: `data/gold/rolling_features/rolling_features_YYYY-MM-DD_YYYY-MM-DD.{parquet,csv}`

**Dependencies**: `pandas`, `numpy`

**File Location**: `/home/user/ForeWatt/src/features/rolling_features.py`

---

#### 3. Calendar Features Generator (`calendar_features.py`)

**Purpose**: Creates calendar and holiday features for forecasting.

**Main Class**: `CalendarFeaturesGenerator`

**Total Features**: 12 calendar features

**Features Created**:
- **Temporal**: dow, dom, month, weekofyear, is_weekend
- **Holiday-related**: is_holiday_day, is_holiday_hour, holiday_name
- **Cyclical encodings**: dow_sin, dow_cos, month_sin, month_cos

**Main Methods**:
```python
load_calendar_days() -> pd.DataFrame
create_hourly_calendar(start_date, end_date) -> pd.DataFrame
run_pipeline(start_date, end_date) -> pd.DataFrame
```

**Output**: `data/gold/calendar_features/calendar_features_YYYY-MM-DD_YYYY-MM-DD.{parquet,csv}`

**Dependencies**: `pandas`, `numpy`

**File Location**: `/home/user/ForeWatt/src/features/calendar_features.py`

---

#### 4. Master Feature Merger (`merge_features.py`)

**Purpose**: Merges all Gold layer features into unified ML-ready dataset.

**Main Class**: `MasterFeatureMerger`

**Gold Layers Merged**:
1. Target variable: consumption_actual (Silver EPİAŞ)
2. Demand weather features: 60+ engineered features (Gold)
3. Deflated prices: Real TL/MWh (Gold)
4. External features: FX/Gold with momentum/volatility (Gold - optional)
5. Calendar features: Holidays + temporal (Gold - optional)
6. Lag features: Target + temp + price lags (Gold)
7. Rolling features: 24h/168h windows (Gold)

**Versioning Strategy**:
- Filename: `master_v{version}_{date}_{hash}.parquet`
- Hash: MD5 of sorted feature names (first 8 chars)
- Metadata JSON includes version info, feature list, statistics

**Main Methods**:
```python
load_target(start_date, end_date) -> pd.DataFrame
load_weather_features(start_date, end_date) -> pd.DataFrame
load_deflated_prices(start_date, end_date) -> pd.DataFrame
merge_all_features(start_date, end_date) -> pd.DataFrame
save_master_dataset(df, start_date, end_date)
run_pipeline(start_date, end_date) -> pd.DataFrame
```

**Output**:
- `data/gold/master/master_v{version}_{date}_{hash}.{parquet,csv}`
- `data/gold/master/master_v{version}_{date}_{hash}_metadata.json`

**Dependencies**: `pandas`, `hashlib`, `json`

**File Location**: `/home/user/ForeWatt/src/features/merge_features.py`

---

#### 5. Feature Pipeline Runner (`run_feature_pipeline.py`)

**Purpose**: Orchestrates complete feature engineering pipeline.

**Pipeline Steps**:
1. Generate lag features
2. Generate rolling features
3. Generate calendar features
4. Merge all features into master dataset

**Main Function**: `run_complete_pipeline(start_date, end_date, version)`

**Usage**:
```bash
python src/features/run_feature_pipeline.py --start 2020-01-01 --end 2024-12-31
```

**Dependencies**: All feature modules

**File Location**: `/home/user/ForeWatt/src/features/run_feature_pipeline.py`

---

### Models Modules (`src/models/`)

#### 1. Evaluation Metrics (`evaluate.py`)

**Purpose**: Unified evaluation metrics for time series forecasting.

**Metrics Implemented**:
1. **MAE** (Mean Absolute Error)
2. **RMSE** (Root Mean Squared Error)
3. **MAPE** (Mean Absolute Percentage Error)
4. **sMAPE** (Symmetric MAPE)
5. **MASE** (Mean Absolute Scaled Error) - time series specific

**Main Function**: `evaluate_forecast(y_true, y_pred, y_train, seasonality, model_name) -> Dict[str, float]`

**MASE Interpretation**:
- MASE < 1: Better than naive seasonal forecast
- MASE = 1: Same as naive seasonal forecast
- MASE > 1: Worse than naive seasonal forecast

**Additional Functions**:
```python
calculate_residuals(y_true, y_pred) -> Dict[str, np.ndarray]
forecast_bias(y_true, y_pred) -> float
```

**Dependencies**: `numpy`, `pandas`

**File Location**: `/home/user/ForeWatt/src/models/evaluate.py`

---

#### 2. Prophet Baseline (`train_prophet.py`)

**Purpose**: Facebook Prophet with Turkish holidays and weather regressors.

**Main Class**: `ProphetForecaster`

**Features**:
- Turkish national holidays (2020-2025)
- Weather regressors: temp_national, humidity, wind_speed, HDD, CDD, heat_index, wind_chill
- Daily/weekly/yearly seasonality
- MLflow tracking

**Hyperparameters**:
- seasonality_mode: 'multiplicative'
- changepoint_prior_scale: 0.05
- seasonality_prior_scale: 10.0
- holidays_prior_scale: 10.0

**Main Function**: `run_prophet_baseline(experiment_name, run_name, test_size, mlflow_uri) -> Dict[str, float]`

**Dependencies**: `prophet`, `mlflow`, `pandas`, `numpy`

**File Location**: `/home/user/ForeWatt/src/models/train_prophet.py`

---

#### 3. SARIMAX Baseline (`train_sarimax.py`)

**Purpose**: Seasonal ARIMA with exogenous variables.

**Main Class**: `SARIMAXForecaster`

**Configuration**:
- Order: (1, 1, 1) - ARIMA parameters
- Seasonal order: (1, 1, 1, 24) - Seasonal parameters with 24h period
- Exogenous variables: temp_national, HDD, CDD, is_holiday_hour, hour_sin, hour_cos, dow_sin, dow_cos

**Note**: Computationally expensive, uses subset of data (last 4 months) for training.

**Main Function**: `run_sarimax_baseline(experiment_name, run_name, test_size, order, seasonal_order, mlflow_uri) -> Dict[str, float]`

**Dependencies**: `statsmodels`, `mlflow`, `pandas`, `numpy`

**File Location**: `/home/user/ForeWatt/src/models/train_sarimax.py`

---

#### 4. XGBoost Baseline (`train_xgboost.py`)

**Purpose**: Extreme Gradient Boosting for time series regression.

**Main Class**: `XGBoostForecaster`

**Hyperparameters**:
- n_estimators: 1000
- learning_rate: 0.1
- max_depth: 6
- subsample: 0.8
- colsample_bytree: 0.8
- reg_alpha: 0.0 (L1)
- reg_lambda: 1.0 (L2)

**Features**:
- Uses all numeric features from master dataset
- Excludes object columns (text features)
- Feature importance logging

**Main Function**: `run_xgboost_baseline(experiment_name, run_name, val_size, test_size, hyperparams, mlflow_uri) -> Dict[str, float]`

**Dependencies**: `xgboost`, `mlflow`, `pandas`, `numpy`

**File Location**: `/home/user/ForeWatt/src/models/train_xgboost.py`

---

#### 5. CatBoost Baseline (`train_catboost.py`)

**Purpose**: Gradient boosting with categorical features support.

**Main Class**: `CatBoostForecaster`

**Hyperparameters**:
- iterations: 1000
- learning_rate: 0.1
- depth: 6
- l2_leaf_reg: 3.0
- early_stopping_rounds: 50

**Features**:
- Automatic categorical feature handling
- Supports holiday_name as categorical
- Binary flags (0/1) treated as categorical
- Feature importance logging

**Main Function**: `run_catboost_baseline(experiment_name, run_name, val_size, test_size, hyperparams, mlflow_uri) -> Dict[str, float]`

**Dependencies**: `catboost`, `mlflow`, `pandas`, `numpy`

**File Location**: `/home/user/ForeWatt/src/models/train_catboost.py`

---

#### 6. Baseline Orchestrator (`run_baselines.py`)

**Purpose**: Runs all baseline models and generates comparison report.

**Main Class**: `BaselineOrchestrator`

**Models**:
- Prophet (with holidays & weather)
- CatBoost (with categorical features)
- XGBoost (gradient boosting)
- SARIMAX (seasonal ARIMA)

**Main Method**: `run_all(models) -> Dict[str, Dict[str, float]]`

**Usage**:
```bash
python src/models/run_baselines.py --all
python src/models/run_baselines.py --models prophet catboost
```

**Output**: `reports/baseline_comparison.csv`

**Dependencies**: All model training modules

**File Location**: `/home/user/ForeWatt/src/models/run_baselines.py`

---

### Analysis Modules (`src/analysis/`)

#### 1. Gold Calendar Features Generator (`make_gold_calendar_features.py`)

**Purpose**: Applies calendar features to silver weather data.

**Input**: `data/silver/demand_weather/*.{parquet,csv}`
**Output**: `data/gold/demand_features/*_w_calendar.{parquet,csv}`

**Main Function**: `main()`

**Dependencies**: `src.data.calendar_features`

**File Location**: `/home/user/ForeWatt/src/analysis/make_gold_calendar_features.py`

---

#### 2. Deflator Plotter (`plot_deflator.py`)

**Purpose**: Visualizes deflator indices for comparison.

**Plots**:
- DFM/Kalman Deflator
- Baseline (TÜFE-based) Deflator
- TÜFE (rebased to 2022=100)

**Usage**:
```bash
python src/analysis/plot_deflator.py
```

**Output**: `deflator_visualization.png` (1000×600)

**Dependencies**: `pandas`, `matplotlib`

**File Location**: `/home/user/ForeWatt/src/analysis/plot_deflator.py`

---

## Feature Engineering

### Design Philosophy

#### 1. Hybrid Modular Approach

**Decision**: Create separate, independent modules for each feature type (lag, rolling, calendar) with a unified merger.

**Why**:
- **Maintainability**: Each module can be updated independently
- **Flexibility**: Can run individual pipelines or complete pipeline
- **Debugging**: Easier to isolate issues to specific feature types
- **Scalability**: New feature types can be added without modifying existing code
- **Reusability**: Modules can be reused for different date ranges or versions

#### 2. Target + Key Exogenous Focus

**Decision**: Create lag features ONLY for target variable (consumption) and key exogenous variables (temperature, price).

**Why**:
- **Curse of Dimensionality**: Creating lags for all 60+ raw features would result in 600+ features
- **Diminishing Returns**: Most features (e.g., cloud_cover, humidity) have minimal predictive power when lagged
- **Model Efficiency**: Smaller feature sets train faster and generalize better
- **Interpretability**: Easier to understand which historical patterns drive predictions

### Feature Types

#### 1. Lag Features (16 features)

**Lag Periods Selected**:
- **1h, 2h, 3h**: Immediate short-term autoregressive patterns
- **6h**: Quarter-day patterns
- **12h**: Half-day patterns (morning/evening peaks)
- **24h**: Daily seasonality (same hour yesterday) - **MOST IMPORTANT**
- **48h**: Two-day patterns
- **168h**: Weekly seasonality (same hour last week) - **SECOND MOST IMPORTANT**

**Variable-Specific Lag Coverage**:

| Variable | Lags Created | Why |
|----------|--------------|-----|
| **Consumption** | 1h, 2h, 3h, 6h, 12h, 24h, 48h, 168h | Full coverage - target variable requires all patterns |
| **Temperature** | 1h, 2h, 3h, 24h, 168h | Focus on immediate and daily/weekly patterns |
| **Price (PTF)** | 1h, 24h, 168h | Focus on immediate and daily/weekly patterns |

#### 2. Rolling Window Features (27 features)

**Window Sizes Selected**:
- **24h (Daily)**: Short-term trends and daily volatility
- **168h (Weekly)**: Medium-term trends and weekly patterns

**Statistics Computed**:

| Statistic | Why |
|-----------|-----|
| **Mean** | Central tendency; smooth trend indicator |
| **Std** | Volatility measure; demand stability indicator |
| **Min** | Lower bound; identifies baseline demand |
| **Max** | Upper bound; identifies peak demand |
| **Range** | Demand variability (max - min) |
| **CV** | Normalized volatility (std / mean) |

**Total Rolling Features**:
- Consumption: 6 stats × 2 windows = 12 features
- Temperature: 6 stats × 2 windows = 12 features
- Price: 6 stats × 2 windows = 12 features
- **Plus derived**: `consumption_range_24h`, `consumption_cv_24h`, `temp_range_24h`
- **Total**: 27 rolling features

#### 3. Calendar Features (12 features)

| Feature | Type | Values | Why |
|---------|------|--------|-----|
| **dow** | Categorical | 0-6 (Mon-Sun) | Day of week; captures weekly patterns |
| **dom** | Categorical | 1-31 | Day of month; captures billing cycles |
| **month** | Categorical | 1-12 | Month; captures seasonal patterns |
| **weekofyear** | Categorical | 1-53 | ISO week number |
| **is_weekend** | Binary | 0/1 | Weekend flag; critical demand differentiator |
| **is_holiday_day** | Binary | 0/1 | Day-level holiday flag |
| **is_holiday_hour** | Binary | 0/1 | Hour-level holiday flag (handles half-days) |
| **holiday_name** | Categorical | String | Holiday type |
| **dow_sin** | Continuous | [-1, 1] | Day of week cyclical encoding |
| **dow_cos** | Continuous | [-1, 1] | Day of week cyclical encoding |
| **month_sin** | Continuous | [-1, 1] | Month cyclical encoding |
| **month_cos** | Continuous | [-1, 1] | Month cyclical encoding |

**Key Design Decisions**:

1. **Hourly Resolution**: Features generated at hourly resolution to match target variable
2. **Half-Day Holiday Handling**: Special logic for holidays like October 28 PM only
3. **Cyclical Encodings**: Sin/cos encodings preserve circular nature of time variables
4. **Two-Level Holiday Flags**: Both day-level and hour-level flags for flexibility

#### 4. Weather Demand Features (60+ features)

Already engineered in `weather_fetcher.py`:
- Heating/Cooling Degree Days (HDD/CDD)
- Heat index & wind chill
- Extreme temperature flags
- Temperature momentum & shocks
- Population-weighted aggregates

#### 5. Master Dataset

**Combines all gold layers**:
- Target: `consumption` (Silver EPİAŞ)
- Weather: 60+ demand features (Gold)
- Prices: Deflated PTF (Gold)
- External: FX/Gold (Gold - optional)
- Calendar: Holidays + temporal (Gold - optional)
- Lags: 16 lag features (Gold)
- Rolling: 27 rolling features (Gold)

**Output**: `data/gold/master/master_v1_{date}_{hash}.parquet`
**Metadata**: `master_v1_{date}_{hash}_metadata.json`

### Master Dataset Statistics

- **Total Features**: 106 (excluding timestamp)
- **Rows**: 43,848 (5 years hourly)
- **Date Range**: 2020-01-01 to 2024-12-31
- **Timezone**: Europe/Istanbul (UTC+3)
- **Missing**: 0.03% (structural from lag features)
- **Feature Hash**: `a567fe49`

### Feature Selection Rationale

**Expected Feature Importance**:

**Tier 1 (Critical)**:
1. `consumption_lag_24h` - Same hour yesterday
2. `consumption_lag_168h` - Same hour last week
3. `is_holiday_day` / `is_holiday_hour` - Holiday effects
4. `hour` - Time of day
5. `dow` - Day of week
6. `temp_national` - Current temperature

**Tier 2 (Important)**:
7. `consumption_rolling_mean_24h` - Daily trend
8. `consumption_lag_1h` - Immediate history
9. `month` - Seasonal effects
10. `temperature_lag_24h` - Yesterday's weather

**Tier 3 (Helpful)**:
- Other consumption lags (2h, 3h, 6h, 12h, 48h)
- Rolling statistics (std, min, max)
- Weather features (humidity, wind, precipitation)
- Price features

### Missing Value Strategy

**Philosophy**: Preserve Structural Missingness

**Decision**: Do NOT impute missing values caused by lag/rolling window operations.

**Why**:
- **Structural missingness is informative**: Missing 168h lag indicates first week of dataset
- **Imputation introduces bias**: Filling with mean/median assumes average behavior
- **Modern ML models handle missing values**: XGBoost, LightGBM, CatBoost have native support

**Missing Value Breakdown**:
- Lag features: 1-168 missing values (depending on lag period)
- Rolling features: 23-167 missing values (depending on window)
- All other features: 0 missing values
- **Total dataset missingness: 0.03%**

### Versioning Strategy

**Feature Hash Versioning**:

**Hash Computation**:
```python
feature_cols = sorted([c for c in df.columns if c != 'timestamp'])
hash_input = '|'.join(feature_cols)
hash_full = hashlib.md5(hash_input.encode()).hexdigest()
feature_hash = hash_full[:8]  # First 8 characters
```

**Filename Format**:
```
master_v{version}_{date}_{hash}.parquet
master_v1_2025-11-12_a567fe49.parquet
```

**Version History**:

| Hash | Date | Features | Description |
|------|------|----------|-------------|
| `0403682c` | 2025-11-11 | 95 | Initial version WITHOUT calendar features |
| `a567fe49` | 2025-11-12 | 106 | Current version WITH calendar features |

---

## Model Development

### Baseline Model Results

From `reports/baseline_comparison.csv`:

| Model | MAE | RMSE | MAPE | sMAPE | MASE | Training Time |
|-------|-----|------|------|-------|------|---------------|
| **CatBoost** | 519.9 | 845.5 | 1.24 | 1.25 | 0.27 | 2.8s |
| **XGBoost** | 527.6 | 826.8 | 1.27 | 1.28 | 0.28 | 4.8s |
| Prophet | 2644 | 3284 | 7.15 | 6.81 | 1.37 | 20.9s |
| SARIMAX | 54608 | 62994 | 138 | 155 | 29.2 | 9.5s |

**Winner**: CatBoost (MASE < 1.0 = better than naive forecast)

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
│   │   ├── external_features_fetcher.py  # FX/Gold features
│   │   ├── calendar_builder.py    # Turkish holiday calendar
│   │   └── validate_deflation_pipeline.py  # Pipeline validation
│   ├── features/                  # Feature engineering (modular)
│   │   ├── __init__.py
│   │   ├── lag_features.py        # Lag features (16 features)
│   │   ├── rolling_features.py    # Rolling window features (27 features)
│   │   ├── calendar_features.py   # Calendar features (12 features)
│   │   ├── merge_features.py      # Master dataset merger
│   │   └── run_feature_pipeline.py  # Pipeline orchestrator
│   ├── analysis/                  # Exploratory analysis & validation
│   │   ├── make_gold_calendar_features.py
│   │   └── plot_deflator.py
│   ├── models/                    # Forecasting models
│   │   ├── evaluate.py            # Evaluation metrics
│   │   ├── train_prophet.py       # Prophet baseline
│   │   ├── train_catboost.py      # CatBoost baseline
│   │   ├── train_xgboost.py       # XGBoost baseline
│   │   ├── train_sarimax.py       # SARIMAX baseline
│   │   └── run_baselines.py       # Baseline orchestrator
│   ├── uncertainty/               # Conformal prediction (TODO)
│   ├── anomaly/                   # IsolationForest + diagnostics (TODO)
│   ├── optimization/              # EV load shifting (TODO)
│   └── evaluation/                # Metrics & validation (TODO)
│
├── data/                          # Medallion architecture
│   ├── bronze/                    # Raw API data (timestamped Parquet + CSV)
│   │   ├── epias/                 # 12 EPİAŞ datasets (2020-2024)
│   │   ├── demand_weather/        # 10 city weather (hourly)
│   │   ├── macro/                 # Macro indicators (monthly)
│   │   └── calendar/              # Holiday definitions
│   ├── silver/                    # Normalized, validated
│   │   ├── epias/                 # Deduplicated, timezone-standardized
│   │   ├── demand_weather/        # Population-weighted national aggregates
│   │   ├── macro/                 # Deflator indices (DID_index)
│   │   └── calendar/              # Holiday tables
│   ├── gold/                      # Feature-engineered, ML-ready
│   │   ├── demand_features/       # 60+ weather demand features
│   │   ├── epias/                 # Deflated prices (real TL/MWh)
│   │   ├── external/              # FX/Gold with momentum/volatility
│   │   ├── calendar_features/     # Holiday & temporal features
│   │   ├── lag_features/          # Lag features (16 features)
│   │   ├── rolling_features/      # Rolling window features (27 features)
│   │   └── master/                # Unified ML-ready dataset
│   ├── unused/                    # Archived data
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
├── tests/                         # Unit & integration tests
│   ├── test_calendar.py
│   ├── test_calendar_end_to_end.py
│   └── test_calendar_features_unit.py
│
├── reports/                       # Model comparison results
│   └── baseline_comparison.csv
│
├── docs/                          # Documentation
│   ├── FEATURE_ENGINEERING.md
│   └── Forewatt_COMP491_Proposal.txt
│
├── .env.example                   # Environment variable template
├── .gitignore                     # Git ignore rules
├── docker-compose.yml             # Multi-service orchestration
├── requirements.txt               # Root Python dependencies
├── pytest.ini                     # Test configuration
├── LICENSE                        # MIT License
└── README.md                      # Main documentation
```

---

## Configuration & Setup

### Prerequisites

- **Docker & Docker Compose** (recommended for production deployment)
- **Python 3.11+** (for local development)
- **EPİAŞ account**: Register at [EPİAŞ Transparency Platform](https://www.epias.com.tr/en/transparency-platform/)

### Environment Variables

**Required variables** (`.env` file):

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

### Quick Start (Docker)

```bash
# Clone repository
git clone https://github.com/yourusername/ForeWatt.git
cd ForeWatt

# Configure environment variables
cp .env.example .env
# Edit .env with your API credentials

# Launch services
docker-compose up -d
```

**Access services**:
- API: http://localhost:8000/docs (OpenAPI/Swagger)
- Dashboard: http://localhost:8501
- MLflow: http://localhost:5050
- InfluxDB: http://localhost:8086

### Local Development

```bash
# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Add your EPİAŞ and EVDS credentials
```

### Running the Complete Pipeline

```bash
# Phase 1: Data Ingestion
python src/data/epias_fetcher.py
python src/data/weather_fetcher.py
python src/data/evds_fetcher.py
python src/data/deflator_builder.py
python src/data/deflate_prices.py

# Phase 2: Feature Engineering
python src/features/run_feature_pipeline.py

# Phase 3: Model Training
python src/models/run_baselines.py --all
```

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
- **Range checks**: Physically plausible bounds
- **Monotonicity checks**: Timestamp ordering, no duplicates
- **Gap handling**: Conservative forward fill
- **Outlier detection**: >5 std from mean flagged for review

### Calendar Tests

**Scripts**: `tests/test_calendar*.py`

- Unit tests for calendar features
- End-to-end tests for holiday handling
- Half-day PM validation (e.g., Oct 28, 2025)

### Feature Engineering Validation

**All validation tests PASSED** ✅

1. **Lag Features**: 43,848 rows × 20 columns, lag correctness verified
2. **Rolling Features**: 43,848 rows × 31 columns, rolling calculations verified
3. **Calendar Features**: 43,848 rows × 13 columns, 840 holiday hours detected
4. **Master Dataset**: 43,848 rows × 107 columns, 0.03% missing (structural only)

---

## Development Roadmap

### Phase 1: Foundation ✅ (Weeks 1-2)

- ✅ GitHub, Docker, MLflow setup
- ✅ EPİAŞ/Open-Meteo API access
- ✅ Project structure & medallion architecture

### Phase 2: Data Pipeline ✅ (Weeks 1-6)

- ✅ Weather data fetcher (Open-Meteo)
- ✅ EPİAŞ ingestion (eptr2) - 12 datasets, 2020-2024
- ✅ Data quality checks (schema, range, monotonicity)
- ✅ Feature engineering (60+ weather demand features)
- ✅ Deflation pipeline (TL normalization via DFM)
- ✅ External features (FX/Gold from EVDS)
- ✅ Calendar features (Turkish holidays)
- ✅ Lag features (16 features)
- ✅ Rolling features (27 features)
- ✅ Master dataset creation (106 features)

### Phase 3: Baseline Models (Weeks 3-6)

- ✅ Prophet baseline
- ✅ CatBoost baseline (Best: MASE 0.27)
- ✅ XGBoost baseline (MASE 0.28)
- ✅ SARIMAX baseline
- ✅ Temporal cross-validation
- ✅ MLflow experiment tracking

### Phase 4: Advanced Models (Weeks 5-9)

- N-HiTS (NeuralForecast)
- PatchTST, Temporal Fusion Transformer
- Ensemble aggregation
- Conformal calibration

### Phase 5: Anomaly & Dashboard (Weeks 8-10)

- IsolationForest + diagnostics
- Streamlit UI rebuild
- API endpoints

### Phase 6: Optimization & Polish (Weeks 9-12)

- EV load shifting (stretch goal)
- Final documentation
- Poster & report

---

## Appendices

### A. Complete Feature Inventory (106 features)

#### Lag Features (16)
- `consumption_lag_1h`, `consumption_lag_2h`, `consumption_lag_3h`
- `consumption_lag_6h`, `consumption_lag_12h`, `consumption_lag_24h`
- `consumption_lag_48h`, `consumption_lag_168h`
- `temperature_lag_1h`, `temperature_lag_2h`, `temperature_lag_3h`
- `temperature_lag_24h`, `temperature_lag_168h`
- `price_ptf_lag_1h`, `price_ptf_lag_24h`, `price_ptf_lag_168h`

#### Rolling Features (27)
- `consumption_rolling_mean_24h`, `consumption_rolling_std_24h`
- `consumption_rolling_min_24h`, `consumption_rolling_max_24h`
- `consumption_rolling_mean_168h`, `consumption_rolling_std_168h`
- `consumption_rolling_min_168h`, `consumption_rolling_max_168h`
- `temperature_rolling_mean_24h`, `temperature_rolling_std_24h`
- `temperature_rolling_min_24h`, `temperature_rolling_max_24h`
- `temperature_rolling_mean_168h`, `temperature_rolling_std_168h`
- `temperature_rolling_min_168h`, `temperature_rolling_max_168h`
- `price_ptf_rolling_mean_24h`, `price_ptf_rolling_std_24h`
- `price_ptf_rolling_min_24h`, `price_ptf_rolling_max_24h`
- `price_ptf_rolling_mean_168h`, `price_ptf_rolling_std_168h`
- `price_ptf_rolling_min_168h`, `price_ptf_rolling_max_168h`
- `consumption_range_24h`, `consumption_cv_24h`, `temp_range_24h`

#### Calendar Features (12)
- `dow`, `dom`, `month`, `weekofyear`
- `is_weekend`, `is_holiday_day`, `is_holiday_hour`, `holiday_name`
- `dow_sin`, `dow_cos`, `month_sin`, `month_cos`

#### Weather Features (35+)
- Temperature: `temp_national`, `temp_std`, various lags/rolling
- Humidity: `humidity_national`
- Wind: `wind_speed_national`, `wind_chill`
- Precipitation: `precipitation_national`, rain flags
- Cloud: `cloud_cover_national`, cloud flags
- Derived: `apparent_temp_national`, `heat_index`, `HDD`, `CDD`

#### Price Features (17+)
- Nominal: `price`, `priceEur`, `priceUsd`
- Real: `price_real`, `priceEur_real`, `priceUsd_real`
- Lags and rolling statistics

---

### B. Key Design Decisions

#### Why Deflation is Separate from External Features?

- **Deflator (DID)**: Removes domestic inflation from electricity prices
- **External Features (FX, Gold)**: Preserved as predictive signals
- **Reason**: FX/import shocks ARE predictive. Removing them destroys valuable information.

#### Why DFM over Simple CPI?

- TÜFE alone includes import price effects
- Dynamic Factor Model extracts "pure" domestic inflation
- Kalman smoothing reduces noise

#### Why Linear Interpolation (Monthly → Hourly)?

- Avoids step changes at month boundaries
- Reflects gradual inflation accumulation
- Monthly → Daily (linear) → Hourly (forward fill)

#### Why Population-Weighted Weather?

- Turkey's population is highly concentrated (top 10 cities = 49.25%)
- National aggregate reflects where electricity demand actually is
- Regional spread captures geographic diversity

#### Why Dual Format Storage (Parquet + CSV)?

- **Parquet**: Fast, compressed, columnar (primary for ML pipelines)
- **CSV**: Human-readable, Excel-compatible (secondary for inspection)
- Best of both worlds: performance + accessibility

---

### C. Dependencies Summary

**Data Acquisition**:
- `eptr2` - EPİAŞ API client
- `evds` (evdspy) - EVDS API client
- `openmeteo_requests` - Open-Meteo weather API
- `requests_cache`, `retry_requests` - API caching and retry

**Data Processing**:
- `pandas`, `numpy` - Core data manipulation
- `pyarrow` - Parquet file I/O
- `python-dotenv` - Environment variable management

**Feature Engineering**:
- `sklearn` - Factor Analysis, StandardScaler
- `statsmodels` - DFM, SARIMAX

**Model Training**:
- `prophet` - Facebook Prophet
- `xgboost` - XGBoost
- `catboost` - CatBoost
- `statsmodels` - SARIMAX
- `mlflow` - Experiment tracking

**Visualization**:
- `matplotlib` - Plotting

---

### D. Acknowledgments

- **Advisor**: Prof. Dr. Gözde Gül Şahin
- **Team**: Zeynep Öykü Aslan, Kaan Altaş, Zeliha Paycı
- **Institution**: Koç University, Department of Computer Engineering
- **Course**: COMP 491 - Computer Engineering Design (Fall 2025)

---

### E. Current Status

**Last Updated**: November 18, 2025
**Pipeline Status**: Phase 2 Complete (Data ingestion & feature engineering ✅)
**Master Dataset**: v1, 106 features, 43,848 rows, hash: a567fe49
**Best Baseline Model**: CatBoost (MAE: 519.9, MASE: 0.27)
**Next Phase**: Advanced models (N-HiTS, ensemble, conformal prediction)

---

## License

MIT License - see LICENSE file for details.

---

**End of Complete Documentation**
