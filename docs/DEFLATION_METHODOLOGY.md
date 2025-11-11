# TL Normalization Methodology for ForeWatt

## Overview

This document describes the complete methodology for handling Turkish Lira (TL) normalization in electricity price forecasting, balancing the removal of domestic inflation while preserving import/FX shock signals.

**Key Principle:** Separate domestic inflation (remove via deflator) from external shocks (preserve as features).

---

## 1. Domestic Inflation Deflator (DID)

### Purpose
Remove domestic/monetary inflation from electricity prices WITHOUT removing FX/import shocks.

**Why this matters:**
- Turkey experienced 40-80% annual inflation (2021-2024)
- Electricity prices contain:
  - **Domestic inflation component** (TL devaluation against basket of goods) → Remove this
  - **Import shock component** (gas/oil prices, FX rates) → Keep this as signal

### Method: Dynamic Factor Model (DFM) with Kalman Smoothing

**Algorithm:**
1. Extract single latent "domestic inflation" factor from multiple domestic indicators
2. Use Kalman filter to smooth estimates
3. Calibrate factor to actual inflation rates

**Why DFM over single index:**
- Using only TÜFE → Includes import price effects (Turkey imports 70% of energy)
- Using only ÜFE → Too volatile, sensitive to commodity shocks
- **DFM approach:** Combines multiple domestic indicators → Extracts common "purely domestic" inflation signal → FX/import effects are treated as noise and excluded

### Inputs (Monthly Frequency)

| Indicator | Source | Role |
|-----------|--------|------|
| **TÜFE** | EVDS | Consumer price inflation (mandatory) |
| **ÜFE** | EVDS | Domestic producer inflation |
| **M2** | EVDS | Money supply annual growth (monetary policy stance) |
| **TL_FAIZ** | EVDS | TL deposit interest rate (credit conditions) |

**Not used (by design):**
- ❌ USD/TRY → External shock, not domestic inflation
- ❌ Oil/Gas prices → External shock, not domestic inflation
- ❌ EUR/TRY → External shock, not domestic inflation
- ❌ Gold → Capital flight indicator, not domestic inflation

### Temporal Interpolation

**Monthly → Hourly conversion:**

```
Monthly DID_index (60 values, 2020-2024)
    ↓ Linear interpolation
Daily DID_index (1,826 values)
    ↓ Forward fill
Hourly DID_index (43,824 values)
```

**Why linear interpolation (not step function):**
- Avoids artificial volatility at month boundaries
- Reflects gradual inflation accumulation
- Prevents model from learning spurious month-end patterns

**Implementation:** `src/data/deflate_prices.py::_interpolate_monthly_to_hourly()`

### Deflation Formula

```python
PTF_real = PTF_nominal / (DID_index / 100)
SMF_real = SMF_nominal / (DID_index / 100)
GIP_real = GIP_nominal / (DID_index / 100)
```

**Base period:** 2022-01 = 100
- Chosen as middle of dataset (2020-2024)
- Avoids extrapolation issues at edges

**Example:**
```
Date: 2024-01-15 14:00
PTF_nominal: 1250.50 TL/MWh
DID_index: 145.8 (domestic inflation 45.8% since 2022-01)

PTF_real = 1250.50 / 1.458 = 857.92 TL/MWh (in 2022-01 TL)
```

**Interpretation:** The Jan 2024 price has the same purchasing power as 858 TL/MWh in Jan 2022.

---

## 2. External Shock Features (NOT Deflators)

### Purpose
Preserve FX/import shock signals as explicit features for the forecasting model.

**Why separate from deflator:**
- Import/FX shocks ARE predictive signal (e.g., gas price spike → electricity price spike)
- We want model to learn this relationship
- Removing these from prices would destroy valuable information

### Features (Daily → Hourly Forward Fill)

| Feature | Source | Update Frequency | Role |
|---------|--------|------------------|------|
| **USD/TRY** | TCMB EVDS | Daily | Primary FX anchor (US dollar) |
| **EUR/TRY** | TCMB EVDS | Daily | Secondary FX anchor (Euro, ~40% of trade) |
| **FX_basket** | Derived | Daily | Weighted: 0.5×USD/TRY + 0.5×EUR/TRY |
| **XAU/TRY** | TCMB EVDS | Daily | Gold in TL (capital flight / expectations indicator) |

**Not included (per spec):**
- ❌ Natural gas prices → Not explicitly fetched (can be added later if needed)
- ❌ Oil prices → Not explicitly fetched

**Temporal resolution:**
- Raw data: Daily (EVDS updates EOD)
- Convert to hourly: Forward fill (all 24 hours in day get same value)

### Usage in Modeling

These features enter N-HiTS (and other models) as **exogenous covariates**:

```python
# Model input structure
X = [
    load_lags,         # Lagged consumption values
    PTF_real_lags,     # Lagged real prices (deflated)
    calendar_features, # Hour, day, week, holidays
    weather_features,  # Temperature, wind, etc.
    fx_features,       # USD/TRY, EUR/TRY, XAU/TRY ← External shocks
]
```

**Why this works:**
- PTF_real contains "baseline" price dynamics (supply/demand, seasonal patterns)
- FX features capture "deviation" from baseline due to external shocks
- Model learns: "When USD/TRY spikes, PTF_real often rises 2-3 hours later"

---

## 3. Complete Data Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│ STEP 1: FETCH DOMESTIC MACRO DATA (EVDS)                    │
│ src/data/evds_fetcher.py                                     │
│ Input:  API key, date range (2020-01-01 to 2024-12-31)      │
│ Output: data/bronze/macro/macro_evds_2020-01-01_*.parquet   │
│         Columns: DATE, TUFE, UFE, M2, TL_FAIZ (monthly)     │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ STEP 2: BUILD DID DEFLATOR (DFM/KALMAN)                     │
│ src/data/deflator_builder.py                                 │
│ Method: build_did_dfm()                                      │
│ Output: data/silver/macro/deflator_did_dfm.parquet          │
│         Columns: DATE, DID_index (monthly, base=100)        │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ STEP 3: FETCH EPİAŞ ELECTRICITY DATA                        │
│ src/data/epias_fetcher.py                                    │
│ Output: data/silver/epias/price_ptf_normalized_*.parquet    │
│         Columns: datetime, Price (hourly, nominal TL/MWh)   │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ STEP 4: DEFLATE PRICES (INTERPOLATE + JOIN)                 │
│ src/data/deflate_prices.py                                   │
│ Process:                                                     │
│   a) Load monthly DID_index                                  │
│   b) Interpolate: Monthly → Daily (linear) → Hourly (ffill) │
│   c) Join hourly prices with hourly deflator                 │
│   d) Apply: Price_real = Price_nominal / (DID_index / 100)  │
│ Output: data/gold/epias/price_ptf_deflated_*.parquet        │
│         Columns: datetime, Price (nominal), Price_real       │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ STEP 5: FETCH EXTERNAL FEATURES (FX/GOLD)                   │
│ TODO: src/data/external_features_fetcher.py                 │
│ Inputs: USD/TRY, EUR/TRY, XAU/TRY (daily)                   │
│ Process: Forward fill to hourly                              │
│ Output: data/gold/external/fx_features_*.parquet            │
│         Columns: datetime, USD_TRY, EUR_TRY, XAU_TRY        │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ STEP 6: FEATURE ENGINEERING (GOLD LAYER)                    │
│ src/features/build_features.py                               │
│ Inputs:                                                      │
│   - consumption_*.parquet (MW, no deflation)                 │
│   - price_ptf_deflated_*.parquet (use Price_real column)    │
│   - fx_features_*.parquet (USD/TRY, EUR/TRY, XAU/TRY)      │
│   - calendar features, weather features                      │
│ Output: data/gold/features/demand_features_*.parquet        │
│         Ready for model training                             │
└──────────────────────────────────────────────────────────────┘
```

---

## 4. Key Design Decisions & Rationale

### Why NOT use ENAG?
- **Original plan:** Use ENAG (alternative CPI from private sector)
- **Implementation:** Use EVDS TÜFE instead
- **Reason:** Data availability and official source reliability

### Why DFM over simple CPI deflation?
| Approach | Pros | Cons |
|----------|------|------|
| **Use TÜFE alone** | Simple | Includes import price effects (overfits to commodity shocks) |
| **Use ÜFE alone** | Producer-focused | Too volatile, not consumer-relevant |
| **DFM (multi-indicator)** | ✅ Extracts "pure" domestic inflation<br>✅ Filters out FX/commodity noise<br>✅ Statistically robust (Kalman smoothing) | More complex |

### Why linear interpolation (not forward fill)?
```python
# Option 1: Forward fill (step function)
DID_2024-01: 145.8 → All hours in Jan use 145.8
DID_2024-02: 150.2 → Sudden jump on Feb 1st (artificial volatility)

# Option 2: Linear interpolation (our choice)
DID_2024-01-01: 145.8
DID_2024-01-15: 147.9 (interpolated)
DID_2024-01-31: 150.0 (interpolated)
DID_2024-02-01: 150.2 (smooth transition)
```

**Result:** No artificial month-boundary effects.

### Why keep FX features separate (not in deflator)?
```python
# BAD: Deflate using comprehensive index including FX
comprehensive_index = f(TÜFE, ÜFE, USD/TRY, Oil, Gas)
Price_real = Price_nominal / comprehensive_index
# Problem: Removes ALL external shock signal → Model can't learn FX → price relationship

# GOOD: Deflate using domestic-only index, keep FX as feature
DID = f(TÜFE, ÜFE, M2, TL_FAIZ)  # No FX
Price_real = Price_nominal / DID
X = [Price_real_lags, USD_TRY, EUR_TRY, XAU_TRY]  # FX as explicit feature
# Result: Preserves external shock information while removing domestic inflation
```

---

## 5. Implementation Checklist

### ✅ Completed
- [x] EVDS fetcher (TÜFE, ÜFE, M2, TL_FAIZ)
- [x] Deflator builder (DFM/Kalman method)
- [x] Monthly → Daily → Hourly interpolation
- [x] Price deflation utility (PTF, SMF, GIP, WAP)
- [x] Dual format saving (Parquet + CSV)
- [x] Documentation (this file + DEFLATION_GUIDE.md)

### ❌ TODO (Next Steps)
- [ ] **External features fetcher** (`src/data/external_features_fetcher.py`)
  - Fetch USD/TRY, EUR/TRY, XAU/TRY from EVDS
  - Daily → Hourly forward fill
  - Save to `data/gold/external/fx_features_*.parquet`

- [ ] **Feature engineering integration** (`src/features/build_features.py`)
  - Load deflated prices (use `*_real` columns)
  - Load FX features
  - Create lags, rolling stats on REAL prices
  - Merge FX features as exogenous covariates

- [ ] **Model training updates**
  - N-HiTS: Add FX features as `future_covariates`
  - CatBoost: Add FX features as input columns
  - Validate: Compare model with/without FX features (ablation study)

---

## 6. Validation & Sanity Checks

### Check 1: Deflation reduces variance
```python
# Before deflation
std(PTF_nominal) = 250 TL/MWh  # High variance (includes inflation)

# After deflation
std(PTF_real) = 180 TL/MWh     # Lower variance (inflation removed)

# Expected: 20-40% variance reduction
```

### Check 2: Real prices should be stationary (or near-stationary)
```python
from statsmodels.tsa.stattools import adfuller

# Nominal prices: non-stationary (p > 0.05)
adf_nominal = adfuller(PTF_nominal)  # p=0.42 → non-stationary

# Real prices: stationary (p < 0.05)
adf_real = adfuller(PTF_real)        # p=0.01 → stationary ✅
```

### Check 3: FX features correlate with price residuals
```python
# After removing domestic inflation, remaining variation should correlate with FX
residuals = PTF_real - PTF_real.rolling(24).mean()
corr(residuals, USD_TRY) > 0.3  # Expect positive correlation ✅
```

### Check 4: Interpolation is smooth (no discontinuities)
```python
# Check daily rate of change
did_daily_change = DID_hourly.resample('D').first().pct_change()
assert did_daily_change.abs().max() < 0.05  # Max 5% daily change
```

---

## 7. Troubleshooting

### Issue: Real prices are negative
**Cause:** DID_index values are corrupted or incorrectly scaled
**Fix:** Check that DID_index base=100 at 2022-01, rebuild deflator

### Issue: High correlation between DID and USD/TRY
**Cause:** DFM is capturing FX effects (should be filtered out)
**Fix:** Review DFM inputs, ensure no FX-related variables included

### Issue: Model performance worse with deflation
**Cause:** Removed too much signal (over-deflation)
**Fix:**
- Use baseline deflator (simpler than DFM)
- Check if FX features are included as covariates
- Validate deflation on simple model (CatBoost) first

### Issue: Interpolation creates artifacts
**Cause:** Monthly values have errors/outliers
**Fix:**
- Smooth monthly DID before interpolation (3-month MA)
- Use cubic spline instead of linear interpolation

---

## 8. References

### Academic Papers
- **DFM for inflation:** Stock & Watson (1989), "New Indexes of Coincident and Leading Economic Indicators"
- **Conformal prediction:** Vovk et al. (2005), "Algorithmic Learning in a Random World"
- **Energy forecasting:** Lago et al. (2021), "Forecasting Electricity Prices"

### Data Sources
- **EVDS API:** https://evds2.tcmb.gov.tr/
- **EPİAŞ Transparency:** https://seffaflik.epias.com.tr/
- **TCMB FX Rates:** https://www.tcmb.gov.tr/kurlar/kurlar_tr.html

### Code Files
- `src/data/evds_fetcher.py` - EVDS data fetching
- `src/data/deflator_builder.py` - DID index construction
- `src/data/deflate_prices.py` - Price deflation with interpolation
- `docs/DEFLATION_GUIDE.md` - User-facing guide

---

**Last Updated:** November 2025
**Status:** Core deflation pipeline complete, external features fetcher pending
**Authors:** ForeWatt Team - Koç University COMP 491
