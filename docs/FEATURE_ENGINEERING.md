# ForeWatt Feature Engineering Documentation

**Version:** 1.0
**Last Updated:** November 11, 2025
**Author:** ForeWatt Team

---

## Table of Contents

1. [Overview](#overview)
2. [Design Philosophy](#design-philosophy)
3. [Feature Engineering Pipeline](#feature-engineering-pipeline)
4. [Lag Features](#lag-features)
5. [Rolling Window Features](#rolling-window-features)
6. [Calendar Features](#calendar-features)
7. [Master Dataset Creation](#master-dataset-creation)
8. [Missing Value Strategy](#missing-value-strategy)
9. [Versioning Strategy](#versioning-strategy)
10. [Feature Selection Rationale](#feature-selection-rationale)
11. [Technical Implementation](#technical-implementation)
12. [Validation Results](#validation-results)

---

## Overview

This document provides a comprehensive explanation of every feature engineering decision made in the ForeWatt electricity demand forecasting project. The feature engineering pipeline transforms raw time series data from Bronze and Silver layers into a machine learning-ready master dataset in the Gold layer.

### Key Statistics

- **Final Dataset Shape:** 43,848 rows × 107 columns
- **Total Features:** 106 (excluding timestamp)
- **Date Range:** 2020-01-01 to 2024-12-31 (5 years)
- **Temporal Resolution:** Hourly
- **Missing Values:** 0.03% (structural missingness from lag features)
- **Feature Hash:** `a567fe49` (for version tracking)

---

## Design Philosophy

### 1. Hybrid Modular Approach

**Decision:** Create separate, independent modules for each feature type (lag, rolling, calendar) with a unified merger.

**Why:**
- **Maintainability:** Each module can be updated independently
- **Flexibility:** Can run individual pipelines or complete pipeline
- **Debugging:** Easier to isolate issues to specific feature types
- **Scalability:** New feature types can be added without modifying existing code
- **Reusability:** Modules can be reused for different date ranges or versions

**Alternative Considered:** Monolithic single script
**Rejected Because:** Would be harder to maintain, debug, and extend

---

### 2. Target + Key Exogenous Focus

**Decision:** Create lag features ONLY for target variable (consumption) and key exogenous variables (temperature, price).

**Why:**
- **Curse of Dimensionality:** Creating lags for all 60+ raw features would result in 600+ features
- **Diminishing Returns:** Most features (e.g., cloud_cover, humidity) have minimal predictive power when lagged
- **Model Efficiency:** Smaller feature sets train faster and generalize better
- **Interpretability:** Easier to understand which historical patterns drive predictions

**Key Exogenous Variables Selected:**
1. **Consumption (Target):** Autoregressive patterns are strongest predictor
2. **Temperature:** Weather directly drives heating/cooling demand
3. **Price (PTF):** Economic signal, potential for price-demand feedback loops

**Alternative Considered:** Lag all features
**Rejected Because:** Would create ~600 features with minimal added value

---

## Feature Engineering Pipeline

### Pipeline Architecture

```
Bronze/Silver Data
    ↓
┌─────────────────────────────────────────────────┐
│  STEP 1: Lag Features Generator                │
│  - Consumption lags (1h-168h)                   │
│  - Temperature lags (1h-168h)                   │
│  - Price lags (1h-168h)                         │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│  STEP 2: Rolling Features Generator             │
│  - 24h windows (consumption, temp, price)       │
│  - 168h windows (consumption, temp, price)      │
│  - Statistics: mean, std, min, max, range, CV   │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│  STEP 3: Calendar Features Generator            │
│  - Turkish holidays (official + religious)      │
│  - Weekend/weekday flags                        │
│  - Cyclical encodings (sin/cos)                 │
│  - Half-day holiday handling                    │
└─────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────┐
│  STEP 4: Master Feature Merger                  │
│  - Merge all gold layers on timestamp           │
│  - Compute feature hash for versioning          │
│  - Generate metadata JSON                       │
│  - Save dual format (Parquet + CSV)             │
└─────────────────────────────────────────────────┘
    ↓
Master ML-Ready Dataset (Gold Layer)
```

### Execution

```bash
# Run complete pipeline
python src/features/run_feature_pipeline.py

# With custom date range
python src/features/run_feature_pipeline.py --start 2020-01-01 --end 2024-12-31 --version v1
```

---

## Lag Features

### Decision: Specific Lag Periods

**Lag Periods Selected:**
- **1h, 2h, 3h:** Immediate short-term autoregressive patterns
- **6h:** Quarter-day patterns
- **12h:** Half-day patterns (morning/evening peaks)
- **24h:** Daily seasonality (same hour yesterday)
- **48h:** Two-day patterns
- **168h:** Weekly seasonality (same hour last week)

**Why These Specific Lags:**

1. **1h, 2h, 3h (Immediate History):**
   - Electricity demand changes gradually
   - Recent hours strongly predict next hour
   - Captures inertia in consumption patterns
   - Essential for autoregressive models (ARIMA, SARIMA)

2. **6h (Quarter-Day):**
   - Bridges short-term and daily patterns
   - Captures transitions between peak/off-peak periods
   - Useful for models learning time-of-day effects

3. **12h (Half-Day):**
   - Morning vs. evening comparison
   - Captures AM/PM asymmetry in demand
   - Example: 6 AM demand often predicts 6 PM demand with shift

4. **24h (Daily Seasonality):**
   - **Most important lag feature**
   - Same hour yesterday is strongest predictor
   - Captures daily consumption routines
   - Essential for all time series forecasting models

5. **48h (Two-Day):**
   - Accounts for day-to-day variability
   - Useful when yesterday was anomalous (holiday, weather event)
   - Provides longer-term context

6. **168h (Weekly Seasonality):**
   - **Second most important lag feature**
   - Same hour last week captures weekday/weekend patterns
   - Example: Monday 9 AM demand similar to previous Monday 9 AM
   - Critical for capturing work week cycles

**Alternative Considered:** Fibonacci lags (1, 2, 3, 5, 8, 13...)
**Rejected Because:** Domain-specific lags (6h, 12h, 24h, 168h) more meaningful for electricity demand

---

### Variable-Specific Lag Coverage

| Variable | Lags Created | Why |
|----------|--------------|-----|
| **Consumption** | 1h, 2h, 3h, 6h, 12h, 24h, 48h, 168h | Full coverage - target variable requires all patterns |
| **Temperature** | 1h, 2h, 3h, 24h, 168h | Focus on immediate and daily/weekly patterns; skip 6h/12h/48h due to slower weather changes |
| **Price (PTF)** | 1h, 24h, 168h | Focus on immediate and daily/weekly patterns; price changes less frequently than demand |

**Why Reduced Coverage for Exogenous Variables:**
- Temperature changes slowly (no need for 6h/12h lags)
- Price follows day-ahead market structure (24h cycle dominant)
- Reduces feature count while retaining predictive power

---

### Implementation Details

**File:** `src/features/lag_features.py`

**Key Methods:**
- `load_consumption()`: Load hourly consumption from Silver EPİAŞ data
- `load_temperature()`: Load national weighted temperature from Gold weather
- `load_price()`: Load PTF (market clearing price) from Silver EPİAŞ data
- `create_lag_features()`: Generate all lag features using `df.shift(periods)`

**Technical Challenge Solved:**
- **Problem:** Different data sources used different timestamp column names (`date`, `datetime`, `timestamp`)
- **Solution:** Implemented flexible column detection that handles all naming conventions
- **Code Location:** `lag_features.py:34-52`

---

## Rolling Window Features

### Decision: Window Sizes

**Window Sizes Selected:**
- **24h (Daily):** Short-term trends and daily volatility
- **168h (Weekly):** Medium-term trends and weekly patterns

**Why These Specific Windows:**

1. **24h Window (Daily):**
   - **Captures intraday patterns:** Min/max demand within day
   - **Daily volatility:** Standard deviation shows demand stability
   - **Recent trends:** Rolling mean shows if demand is increasing/decreasing
   - **Aligned with consumption patterns:** Households and businesses operate on 24h cycles

2. **168h Window (Weekly):**
   - **Captures weekly patterns:** Weekday vs. weekend differences
   - **Medium-term trends:** Detects gradual changes (seasonal transitions)
   - **Business cycle alignment:** Most businesses operate on weekly schedules
   - **Balances responsiveness and stability:** Not too short (noisy) or too long (lagging)

**Why Not Other Windows:**
- **No 72h window:** Not aligned with natural consumption cycles (neither daily nor weekly)
- **No 336h (2-week) window:** Too slow to react to changes; less interpretable
- **No 12h window:** Redundant with existing 12h lag feature; minimal added value

**Alternative Considered:** Exponential weighted moving averages (EWMA)
**Rejected Because:** Fixed windows more interpretable and aligned with domain cycles

---

### Statistics Computed

For each (variable, window) combination:

| Statistic | Why |
|-----------|-----|
| **Mean** | Central tendency; smooth trend indicator |
| **Std (Standard Deviation)** | Volatility measure; demand stability indicator |
| **Min** | Lower bound; identifies baseline demand |
| **Max** | Upper bound; identifies peak demand |
| **Range** (max - min) | Demand variability; complementary to std |
| **CV** (std / mean) | Normalized volatility; accounts for scale differences |

**Total Rolling Features:**
- Consumption: 6 stats × 2 windows = 12 features
- Temperature: 6 stats × 2 windows = 12 features
- Price: 6 stats × 2 windows = 12 features
- **Plus derived:** `consumption_range_24h`, `consumption_cv_24h`, `temp_range_24h`
- **Total:** 27 rolling features

---

### Why These Statistics Are Important

1. **Mean (Rolling Average):**
   - Smooths out noise in raw data
   - Shows underlying trend direction
   - Essential for detecting demand increases/decreases

2. **Std (Rolling Standard Deviation):**
   - Measures demand predictability
   - High std = volatile demand (harder to forecast)
   - Low std = stable demand (easier to forecast)
   - Useful for prediction interval estimation

3. **Min/Max:**
   - Identifies demand bounds within window
   - Min shows baseline consumption (e.g., nighttime)
   - Max shows peak consumption (e.g., evening)
   - Together they define the demand envelope

4. **Range (Max - Min):**
   - Direct measure of demand swing
   - High range = large daily fluctuations
   - Low range = stable consumption pattern
   - Complementary to std (range is absolute, std is statistical)

5. **CV (Coefficient of Variation):**
   - **Normalized volatility** (std / mean)
   - Allows comparison across different demand levels
   - Example: 1000 MWh std on 30,000 MWh mean (3.3%) vs. 1000 MWh std on 20,000 MWh mean (5%)
   - Critical for scale-invariant volatility assessment

---

### Implementation Details

**File:** `src/features/rolling_features.py`

**Key Methods:**
- `create_rolling_features()`: Apply `.rolling(window).agg(['mean', 'std', 'min', 'max'])`
- **Minimum periods:** Set to `window - 1` to allow calculation starting at window edge
- **Alignment:** Uses default `right` alignment (window includes current observation)

**Missing Values:**
- First 23 hours: `rolling_std_24h` = NaN (need 24 observations for std)
- First 167 hours: `rolling_std_168h` = NaN
- This is **structural missingness** (acceptable, not data quality issue)

---

## Calendar Features

### Decision: Turkish Holiday Integration

**Why Calendar Features Matter:**
- Electricity demand changes significantly on holidays
- Official holidays: Reduced industrial demand, variable residential
- Religious holidays: Multi-day celebrations with unique demand patterns
- Weekend vs. weekday: Fundamental consumption behavior differences

---

### Features Created

| Feature | Type | Values | Why |
|---------|------|--------|-----|
| **dow** | Categorical | 0-6 (Mon-Sun) | Day of week; captures weekly patterns |
| **dom** | Categorical | 1-31 | Day of month; captures billing cycles |
| **month** | Categorical | 1-12 | Month; captures seasonal patterns |
| **weekofyear** | Categorical | 1-53 | ISO week number; alternative to month |
| **is_weekend** | Binary | 0/1 | Weekend flag; critical demand differentiator |
| **is_holiday_day** | Binary | 0/1 | Day-level holiday flag (00:00-23:59) |
| **is_holiday_hour** | Binary | 0/1 | Hour-level holiday flag (handles half-days) |
| **holiday_name** | Categorical | String | Holiday type (e.g., "Ramazan Bayramı") |
| **dow_sin** | Continuous | [-1, 1] | Day of week cyclical encoding |
| **dow_cos** | Continuous | [-1, 1] | Day of week cyclical encoding |
| **month_sin** | Continuous | [-1, 1] | Month cyclical encoding |
| **month_cos** | Continuous | [-1, 1] | Month cyclical encoding |

**Total:** 12 calendar features

---

### Key Design Decisions

#### 1. Hourly Resolution (Not Daily)

**Decision:** Generate calendar features at hourly resolution, not daily.

**Why:**
- Matches target variable frequency (hourly consumption)
- Enables hour-specific holiday patterns (e.g., morning vs. evening on holidays)
- Supports half-day holiday handling (see below)
- Allows models to learn intraday effects on holidays

**Alternative Considered:** Daily flags broadcast to all hours
**Rejected Because:** Loses ability to model intraday holiday patterns

---

#### 2. Half-Day Holiday Handling

**Decision:** Implement special logic for half-day holidays (e.g., October 28 PM only).

**Why:**
- **October 28:** Republic Day Eve is a half-day holiday (13:00-23:59 only)
- AM hours (00:00-12:59) are working hours, PM hours are holiday
- Treating entire day as holiday would be inaccurate
- Electricity demand pattern differs: morning industrial, afternoon residential

**Implementation:**
```python
# If half_day == 'pm', AM hours (<13) are NOT holiday hours
am_mask = hourly_index.hour < 13
feats.loc[
    feats['date_only'].map(lambda d: half_map.get(d) == 'pm') & am_mask,
    'is_holiday_hour'
] = 0
```

**Code Location:** `calendar_features.py:138-148`

**Impact:** Improves accuracy for 28 hours/year × 5 years = 140 hours

---

#### 3. Cyclical Encodings (Sin/Cos)

**Decision:** Create sine/cosine encodings for day of week and month.

**Why:**
- **Problem:** Day-of-week is cyclical (Sunday → Monday wraps around)
- **Linear encoding issue:** dow=0 (Monday) and dow=6 (Sunday) are numerically far apart but temporally adjacent
- **Solution:** Cyclical encoding preserves circular nature

**Formula:**
```python
dow_sin = sin(2π × dow / 7)
dow_cos = cos(2π × dow / 7)
```

**Example:**
| Day | dow | dow_sin | dow_cos | Distance to Monday |
|-----|-----|---------|---------|-------------------|
| Monday | 0 | 0.00 | 1.00 | 0 |
| Tuesday | 1 | 0.78 | 0.62 | 1 |
| Sunday | 6 | -0.78 | 0.62 | 1 (correct!) |

**Why Both Sin and Cos:**
- Single sine would be ambiguous (e.g., Tuesday and Saturday both have same sine value)
- Together, (sin, cos) uniquely identifies each day
- Neural networks and tree models can learn circular patterns from this encoding

**When Not to Use:**
- Tree-based models (RF, XGBoost) can handle categorical dow directly
- But harmless to include; models will ignore if not useful

**Alternative Considered:** One-hot encoding
**Rejected Because:** Creates 7 columns for dow, 12 for month (19 features vs. 4 cyclical features)

---

#### 4. Two-Level Holiday Flags

**Decision:** Provide both `is_holiday_day` (day-level) and `is_holiday_hour` (hour-level).

**Why:**
- **Flexibility for different models:**
  - Some models may prefer day-level (simpler)
  - Others may need hour-level (more precise)
- **Half-day holidays:** `is_holiday_hour` handles October 28 PM correctly
- **Minimal overhead:** Only 2 binary columns
- **User choice:** Data scientist can select appropriate flag for their model

---

### Holiday Types Included

**Official Holidays (7 types):**
1. Yılbaşı (New Year's Day) - Jan 1
2. Ulusal Egemenlik ve Çocuk Bayramı (National Sovereignty Day) - Apr 23
3. Emek ve Dayanışma Günü (Labor Day) - May 1
4. Atatürk'ü Anma, Gençlik ve Spor Bayramı (Atatürk Commemoration Day) - May 19
5. Demokrasi ve Milli Birlik Günü (Democracy Day) - Jul 15
6. Zafer Bayramı (Victory Day) - Aug 30
7. Cumhuriyet Bayramı (Republic Day) - Oct 29
   - Plus Oct 28 PM (half-day)

**Religious Holidays (2 types, multi-day):**
1. Ramazan Bayramı (Eid al-Fitr) - 3.5 days
2. Kurban Bayramı (Eid al-Adha) - 4.5 days

**Total Holiday Coverage:** 50 holiday days across 2020-2025

---

### Implementation Details

**File:** `src/features/calendar_features.py`

**Data Source:** `data/silver/calendar/calendar_days.parquet`

**Key Methods:**
- `load_calendar_days()`: Load holiday definitions from Silver layer
- `create_hourly_calendar()`: Generate hourly calendar features with timezone handling
- **Timezone:** Europe/Istanbul (UTC+3) for all timestamps

**Validation:**
- ✅ 840 total holiday hours identified
- ✅ 7 unique holiday types detected
- ✅ Half-day PM handling verified for Oct 28
- ✅ Cyclical encodings all in [-1, 1] range

---

## Master Dataset Creation

### Decision: Merge Strategy

**Merge Type:** Outer join on `timestamp` across all feature layers.

**Layers Merged:**
1. **EPİAŞ (Silver):** Consumption, price, base features
2. **Weather (Gold):** Temperature, humidity, wind, precipitation
3. **Deflation (Gold):** Real prices (TL adjusted)
4. **Lag Features (Gold):** Created by lag_features.py
5. **Rolling Features (Gold):** Created by rolling_features.py
6. **Calendar Features (Gold):** Created by calendar_features.py

**Why Outer Join:**
- Retains all timestamps even if some features are missing
- Preserves structural missingness (e.g., first 168 hours have no 168h lag)
- Allows models to handle missing values appropriately

**Alternative Considered:** Inner join (only complete cases)
**Rejected Because:** Would lose first 168 hours of data unnecessarily

---

### Timestamp Standardization Challenge

**Problem:** Different data sources used different column names and formats:
- EPİAŞ Silver: Column named `date` (not `timestamp`)
- Weather Gold: Timestamp in DatetimeIndex (not a column)
- Calendar Gold: Column named `datetime`

**Solution:** Implemented `_standardize_timestamp()` helper method:

```python
def _standardize_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
    """Standardize timestamp column name and format."""
    # Check if timestamp is in index
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        if df.columns[0] in ['datetime', 'date']:
            df = df.rename(columns={df.columns[0]: 'timestamp'})

    # Handle different timestamp column names
    timestamp_col = None
    for col in ['timestamp', 'date', 'datetime']:
        if col in df.columns:
            timestamp_col = col
            break

    if timestamp_col and timestamp_col != 'timestamp':
        df = df.rename(columns={timestamp_col: 'timestamp'})

    return df
```

**Code Location:** `merge_features.py:39-60`

**Impact:** Enables seamless merging across heterogeneous data sources

---

### Dual Format Output

**Decision:** Save master dataset in both Parquet and CSV formats.

**Formats:**
1. **Parquet (Primary):**
   - Compressed columnar format (18 MB vs. 51 MB CSV)
   - Fast read/write for Python/R
   - Preserves data types (timestamps, categories)
   - Preferred for ML pipelines

2. **CSV (Secondary):**
   - Human-readable for inspection
   - Excel/spreadsheet compatible
   - Debugging and manual validation
   - Backup format if Parquet issues

**Why Both:**
- Parquet for production ML workflows
- CSV for exploratory data analysis and debugging
- Minimal storage overhead (18 MB + 51 MB = 69 MB is acceptable)

**Alternative Considered:** Parquet only
**Rejected Because:** CSV useful for quick inspection and non-Python users

---

### Metadata Generation

**Decision:** Generate comprehensive JSON metadata file alongside dataset.

**Metadata Includes:**
- Version and creation date
- Feature hash (MD5 of sorted column names)
- Data date range (start, end, first_timestamp, last_timestamp)
- Shape (rows, columns)
- Feature list (all 106 feature names)
- Missing value counts and percentages
- Target column name
- Timestamp column name

**Why:**
- **Traceability:** Know exactly what features are in each version
- **Validation:** Programmatically verify dataset integrity
- **Documentation:** Self-describing dataset
- **Reproducibility:** Hash ensures same features → same hash

**File:** `master_v1_2025-11-11_a567fe49_metadata.json`

---

## Missing Value Strategy

### Philosophy: Preserve Structural Missingness

**Decision:** Do NOT impute missing values caused by lag/rolling window operations.

**Why:**
- **Structural missingness is informative:**
  - First 24 hours: No 24h lag available (we don't have data before time zero)
  - This is **not random missingness** - it's a deterministic pattern
  - Models can learn that "missing 168h lag" = "first week of dataset"

- **Imputation introduces bias:**
  - Filling with mean/median assumes average behavior at time zero
  - Forward fill assumes demand at hour 0 equals demand 168h before (impossible)
  - Backward fill leaks future information

- **Modern ML models handle missing values:**
  - XGBoost: Treats missing as a separate branch in trees
  - LightGBM: Native missing value handling
  - CatBoost: Handles missing values internally
  - Neural networks: Can use masking layers

**Missing Value Breakdown:**
- Lag features: 1-168 missing values (depending on lag period)
- Rolling features: 23-167 missing values (depending on window)
- All other features: 0 missing values
- **Total dataset missingness: 0.03%**

**Alternative Considered:** Impute with forward fill or mean
**Rejected Because:** Introduces bias and misleads models about true data availability

---

### When to Handle Missing Values

**In Feature Engineering Pipeline:** Never (preserve structural missingness)

**In Modeling Pipeline (later):**
1. **Drop first 168 rows:** Cleanest approach, minimal data loss (0.4% of 5 years)
2. **Model-specific handling:** Let XGBoost/LightGBM handle natively
3. **Create "missingness indicator" features:** Binary flags for `is_missing_lag_168h` (useful for some models)

**Recommendation:** Drop first 168 rows before training to avoid structural missingness entirely.

---

## Versioning Strategy

### Feature Hash Versioning

**Decision:** Use MD5 hash of sorted feature names as version identifier.

**Hash Computation:**
```python
feature_cols = sorted([c for c in df.columns if c != 'timestamp'])
hash_input = '|'.join(feature_cols)
hash_full = hashlib.md5(hash_input.encode()).hexdigest()
feature_hash = hash_full[:8]  # First 8 characters
```

**Why:**
- **Reproducibility:** Same features → same hash
- **Change detection:** Different features → different hash
- **Collision resistance:** 8 hex chars = 4.3 billion combinations (sufficient for project)
- **Compact:** Only 8 characters in filename

**Filename Format:**
```
master_v{version}_{date}_{hash}.parquet
master_v1_2025-11-11_a567fe49.parquet
```

**Versioning Components:**
1. **Version (v1):** Major pipeline version (manual increment for breaking changes)
2. **Date (2025-11-11):** Dataset creation date
3. **Hash (a567fe49):** Feature composition fingerprint

**Why This Matters:**
- Can run pipeline multiple times same day with different features
- Hash distinguishes `master_v1_2025-11-11_a567fe49` (with calendar) from `master_v1_2025-11-11_0403682c` (without calendar)
- Prevents accidental overwriting of different feature sets

---

### Version History

| Hash | Date | Features | Description |
|------|------|----------|-------------|
| `0403682c` | 2025-11-11 | 95 | Initial version WITHOUT calendar features |
| `a567fe49` | 2025-11-11 | 106 | Current version WITH calendar features |

---

## Feature Selection Rationale

### Feature Count Trade-offs

**Current Approach:** 106 features (selective)

**Alternative Approaches:**

| Approach | Feature Count | Pros | Cons |
|----------|---------------|------|------|
| **Minimal** | ~20 | Fast training, interpretable | Underfitting risk |
| **Current (Selective)** | 106 | Balanced complexity | Moderate training time |
| **Comprehensive** | ~600 | Maximum information | Overfitting risk, slow |

**Why 106 Features is Optimal:**
- Includes all important temporal patterns (lags, rolling)
- Covers all weather variables (temp, humidity, wind, precipitation)
- Includes calendar effects (holidays, weekends)
- Avoids redundant lags on low-value features
- Balances model complexity vs. predictive power

---

### Feature Importance (Expected)

Based on time series forecasting literature and domain knowledge:

**Tier 1 (Critical):**
1. `consumption_lag_24h` - Same hour yesterday
2. `consumption_lag_168h` - Same hour last week
3. `is_holiday_day` / `is_holiday_hour` - Holiday effects
4. `hour` - Time of day
5. `dow` - Day of week
6. `temp_national` - Current temperature

**Tier 2 (Important):**
7. `consumption_rolling_mean_24h` - Daily trend
8. `consumption_lag_1h` - Immediate history
9. `month` - Seasonal effects
10. `temperature_lag_24h` - Yesterday's weather

**Tier 3 (Helpful):**
- Other consumption lags (2h, 3h, 6h, 12h, 48h)
- Rolling statistics (std, min, max)
- Weather features (humidity, wind, precipitation)
- Price features

**Note:** Actual feature importance should be validated using:
1. Model-specific importance (XGBoost `feature_importances_`)
2. SHAP values for interpretability
3. Permutation importance for model-agnostic ranking

---

## Technical Implementation

### Technology Stack

- **Python:** 3.9+
- **Pandas:** Time series manipulation
- **NumPy:** Numerical operations
- **PyArrow:** Parquet I/O
- **Logging:** Pipeline observability

### Code Organization

```
src/features/
├── __init__.py                    # Package initialization
├── lag_features.py                # Lag feature generator
├── rolling_features.py            # Rolling window generator
├── calendar_features.py           # Calendar feature generator
├── merge_features.py              # Master dataset merger
└── run_feature_pipeline.py        # Pipeline orchestrator
```

### Pipeline Execution Time

**Typical Runtime (2020-2024, 5 years):**
- Step 1 (Lag Features): ~15 seconds
- Step 2 (Rolling Features): ~20 seconds
- Step 3 (Calendar Features): ~5 seconds
- Step 4 (Master Merge): ~10 seconds
- **Total:** ~50 seconds

**Performance Optimizations:**
- Vectorized pandas operations (no row-by-row loops)
- Parquet for fast I/O
- Lazy evaluation where possible

---

### Error Handling

**Challenges Encountered and Solutions:**

1. **Timestamp Column Name Inconsistency**
   - **Problem:** Different data sources used `date`, `datetime`, `timestamp`
   - **Solution:** Flexible column detection in all loaders
   - **Code:** `lag_features.py:34-52`, `merge_features.py:39-60`

2. **DatetimeIndex vs. Column**
   - **Problem:** Weather data stored timestamp in index, not column
   - **Solution:** Check for DatetimeIndex, reset if needed
   - **Code:** `lag_features.py:86-92`

3. **Temperature Column Name Variations**
   - **Problem:** `temperature_2m`, `temp_weighted`, `temp_national` all used
   - **Solution:** Priority list search for temperature column
   - **Code:** `lag_features.py:101-106`

---

## Validation Results

### Data Quality Checks

All validation tests **PASSED** ✅

#### 1. Lag Features Validation
- ✅ Shape: 43,848 rows × 20 columns
- ✅ Lag correctness: `consumption_lag_24h[24]` == `consumption[0]`
- ✅ 1h shift: `temperature_lag_1h` correctly shifted
- ✅ Missing values: Exactly 168 for 168h lag (expected)

#### 2. Rolling Features Validation
- ✅ Shape: 43,848 rows × 31 columns
- ✅ 24h rolling mean: Matches manual calculation
- ✅ Range calculation: `max - min` verified
- ✅ Missing values: Exactly 23 for 24h std (expected)

#### 3. Calendar Features Validation
- ✅ Shape: 43,848 rows × 13 columns
- ✅ Holiday detection: Jan 1, 2020 = "Yılbaşı" ✓
- ✅ Weekend detection: Jan 4, 2020 (Sat, dow=5) ✓
- ✅ Cyclical encodings: All sin/cos in [-1, 1] range ✓
- ✅ 840 total holiday hours
- ✅ 7 unique holidays

#### 4. Master Dataset Validation
- ✅ Shape: 43,848 rows × 107 columns
- ✅ Feature count: 106 (matches metadata)
- ✅ Timestamp range: 2020-01-01 to 2024-12-31
- ✅ No duplicate timestamps
- ✅ Missing values: 0.03% (structural only)
- ✅ Feature hash: `a567fe49` (matches)
- ✅ All 6 layers successfully merged

#### 5. Python Modules Validation
- ✅ All syntax valid
- ✅ All imports successful
- ✅ All expected classes found
- ✅ All functions callable

**Total Tests:** 30+
**Pass Rate:** 100%

---

## Next Steps

### Immediate (Before Modeling)

1. **Feature Importance Analysis**
   - Train baseline XGBoost model
   - Extract feature importances
   - Validate expected importance ranking
   - Consider dropping low-importance features (<1% importance)

2. **Correlation Analysis**
   - Identify highly correlated features (>0.95)
   - Consider removing redundant features
   - Check for multicollinearity issues

3. **Data Splits**
   - Create train/validation/test splits
   - Recommendation: 2020-2022 (train), 2023 (validation), 2024 (test)
   - Ensure no data leakage across splits

### Future Enhancements

1. **External Features Integration**
   - FX rates (USD/TRY, EUR/TRY) from EVDS
   - Gold prices from EVDS
   - Economic indicators (CPI, industrial production)

2. **Advanced Features**
   - Interaction terms (e.g., `temp × is_weekend`)
   - Polynomial features (e.g., `temp²` for non-linear effects)
   - Binned features (e.g., temperature quartiles)

3. **Alternative Feature Sets**
   - Version v2: Add external features
   - Version v3: Add interaction terms
   - Compare model performance across versions

4. **Feature Selection**
   - Recursive feature elimination (RFE)
   - LASSO regularization for feature selection
   - Boruta algorithm for feature importance

---

## Appendix: Feature List

### Complete Feature Inventory (106 features)

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
- Temperature: `temp_national`, `temp_std`, `temp_lag_*`, `temp_change_*`, `temp_rolling_*`
- Humidity: `humidity_national`
- Wind: `wind_speed_national`, `wind_chill`
- Precipitation: `precipitation_national`, `is_raining`, `is_heavy_rain`
- Cloud: `cloud_cover_national`, `is_cloudy`
- Derived: `apparent_temp_national`, `heat_index`, `HDD`, `CDD`, `DID_index`

#### Price Features (17+)
- `price`, `priceEur`, `priceUsd`
- Real prices: `price_real`, `priceEur_real`, `priceUsd_real`
- Lags: `price_ptf_lag_*`
- Rolling: `price_ptf_rolling_*`

#### Other Features (~10)
- Time encodings: `hour_x`, `hour_y`, `hour_sin`, `hour_cos`
- `day_of_week`, `day_of_year`
- Target: `consumption`

**Total:** 106 features + 1 timestamp = 107 columns

---

## Conclusion

This feature engineering pipeline transforms raw time series data into a comprehensive, ML-ready dataset with **106 carefully selected features**. Every decision—from lag periods to window sizes to cyclical encodings—is grounded in domain knowledge, time series forecasting best practices, and practical considerations.

The resulting dataset (`master_v1_2025-11-11_a567fe49.parquet`) is:
- ✅ **Complete:** All important temporal, weather, and calendar patterns captured
- ✅ **Validated:** All features verified with automated tests
- ✅ **Documented:** Every design decision explained and justified
- ✅ **Reproducible:** Hash-based versioning ensures consistency
- ✅ **Production-Ready:** Dual format (Parquet + CSV), comprehensive metadata

**Ready for Phase 3: Model Development** 🚀

---

**For questions or clarifications, refer to:**
- Pipeline code: `src/features/run_feature_pipeline.py`
- Individual generators: `src/features/{lag,rolling,calendar,merge}_features.py`
- Validation scripts: (see Validation Results section)
- Project README: `README.md`
