# Turkish Lira Deflation Guide

## Overview

This guide explains how to normalize EPİAŞ electricity price data from nominal Turkish Lira to real values using EVDS-based deflator indices.

**Why deflation is necessary:**
- Turkey has experienced high inflation (40-80% annually in 2021-2024)
- Nominal price trends reflect both electricity market dynamics AND currency devaluation
- Machine learning models need real (inflation-adjusted) prices to learn true market patterns
- Without deflation, models would confuse inflation with genuine price changes

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. FETCH MACROECONOMIC DATA (EVDS)                              │
│    src/data/evds_fetcher.py                                     │
│    → Fetches TÜFE (CPI), ÜFE (PPI), M2, TL_FAIZ                │
│    → Output: data/bronze/macro/macro_evds_2020-01-01_2024-12-31│
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. BUILD DEFLATOR INDICES                                       │
│    src/data/deflator_builder.py                                 │
│    → Method 1: Factor Analysis (baseline)                       │
│    → Method 2: Dynamic Factor Model (DFM/Kalman)                │
│    → Output: data/silver/macro/deflator_did_baseline.parquet    │
│              data/silver/macro/deflator_did_dfm.parquet          │
│    → DID_index: Deflator index (base=100 at 2022-01)            │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. FETCH EPİAŞ ELECTRICITY DATA                                 │
│    src/data/epias_fetcher.py                                    │
│    → Fetches prices, consumption, generation, etc.              │
│    → Output: data/silver/epias/price_*_normalized_*.parquet     │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. DEFLATE PRICE DATA                                           │
│    src/data/deflate_prices.py                                   │
│    → Joins hourly prices with monthly deflator index            │
│    → Formula: real_price = nominal_price / (DID_index / 100)    │
│    → Output: data/gold/epias/price_*_deflated_*.parquet         │
│              Contains both nominal AND real (*_real) columns    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Which Datasets Need Deflation?

### ✅ REQUIRE Deflation (Price Data in TL/MWh)

| Dataset | Description | Files |
|---------|-------------|-------|
| `price_ptf` | Day-Ahead Market Clearing Price (Piyasa Takas Fiyatı) | `data/silver/epias/price_ptf_normalized_*.parquet` |
| `price_smf` | Balancing Power Market Price (Sistem Marjinal Fiyatı) | `data/silver/epias/price_smf_normalized_*.parquet` |
| `price_idm` | Intraday Market Quantity/Price (Gün İçi Piyasa) | `data/silver/epias/price_idm_normalized_*.parquet` |
| `price_wap` | Weighted Average Price | `data/silver/epias/price_wap_normalized_*.parquet` |

### ❌ DO NOT Deflate (Physical Units)

| Dataset | Description | Unit | Reason |
|---------|-------------|------|--------|
| `consumption_*` | Electricity consumption | MW | Physical quantity |
| `generation_*` | Generation by source | MW | Physical quantity |
| `capacity_*` | Available capacity | MW | Physical quantity |
| `wind_forecast` | Wind generation forecast | MW | Physical quantity |
| `hydro_*` | Hydropower data | Volume/Energy | Physical quantity |

---

## Step-by-Step Usage

### Step 1: Fetch Macroeconomic Data (EVDS)

```bash
python src/data/evds_fetcher.py
```

**Requirements:**
- Set `EVDS_API_KEY` in `.env` file
- Get key from: https://evds2.tcmb.gov.tr/

**Output:**
- `data/bronze/macro/macro_evds_2020-01-01_2024-12-31.parquet` (60 months, 2020-2024)
- Columns: `DATE`, `TUFE`, `UFE`, `M2`, `TL_FAIZ`, `TUFE_rebased`, `UFE_rebased`

### Step 2: Build Deflator Indices

```bash
python src/data/deflator_builder.py
```

**Output:**
- `data/silver/macro/deflator_did_baseline.parquet` (Factor Analysis)
- `data/silver/macro/deflator_did_dfm.parquet` (Dynamic Factor Model)
- Key column: `DID_index` (base=100 at 2022-01)

**Example deflator values:**
```
DATE       DID_index
2020-01    65.2      (prices were 35% lower in real terms)
2022-01    100.0     (base month)
2024-12    145.8     (prices 46% higher in real terms)
```

### Step 3: Fetch EPİAŞ Electricity Data

```bash
python src/data/epias_fetcher.py
```

**Requirements:**
- Set `EPTR_USERNAME` and `EPTR_PASSWORD` in `.env` file
- Register at: https://seffaflik.epias.com.tr/

**Output:**
- `data/silver/epias/price_ptf_normalized_2020-01-01_2024-12-31.parquet`
- `data/silver/epias/price_smf_normalized_2020-01-01_2024-12-31.parquet`
- ... and other datasets

### Step 4: Deflate Price Data

```bash
python src/data/deflate_prices.py
```

**Output:**
- `data/gold/epias/price_ptf_deflated_2020-01-01_2024-12-31.parquet`
- Contains both:
  - Original nominal columns (e.g., `Price`)
  - Deflated real columns (e.g., `Price_real`)

**Example:**
```python
# Before deflation (nominal TL/MWh)
Date                Price
2024-01-15 14:00    1250.50

# After deflation (real TL/MWh, base=2022-01)
Date                Price    Price_real  DID_index
2024-01-15 14:00    1250.50  857.92      145.8
```

---

## Advanced Usage (Python API)

### Deflate a Single Dataset

```python
from src.data.deflate_prices import PriceDeflator

# Initialize deflator (baseline or dfm)
deflator = PriceDeflator(deflator_method='baseline')

# Deflate price_ptf dataset
df = deflator.deflate_dataset(
    dataset_name='price_ptf',
    start_date='2020-01-01',
    end_date='2024-12-31',
    layer='silver',  # Use normalized silver layer
    output_layer='gold'
)

# Access real prices
print(df[['Date', 'Price', 'Price_real']].head())
```

### Deflate All Price Datasets

```python
from src.data.deflate_prices import PriceDeflator

# Use DFM deflator (more sophisticated)
deflator = PriceDeflator(deflator_method='dfm')

# Deflate all price datasets at once
results = deflator.deflate_all_price_datasets(
    start_date='2020-01-01',
    end_date='2024-12-31',
    layer='silver'
)

# Check results
for dataset_name, df in results.items():
    print(f"{dataset_name}: {len(df)} records")
```

---

## Understanding the Deflation Formula

### Formula

```
real_price = nominal_price / (DID_index / 100)
```

### Intuition

- **DID_index = 100** (base month, 2022-01): No adjustment
- **DID_index > 100** (inflation): Deflate downward
  - Example: DID=145.8 → Divide by 1.458 → Real price is 68.6% of nominal
- **DID_index < 100** (deflation): Inflate upward
  - Example: DID=65.2 → Divide by 0.652 → Real price is 153% of nominal

### Example Calculation

**Nominal price in Jan 2024:** 1250 TL/MWh
**DID_index in Jan 2024:** 145.8

**Real price (base=2022-01):**
```
real_price = 1250 / (145.8 / 100)
          = 1250 / 1.458
          = 857.9 TL/MWh
```

**Interpretation:** The Jan 2024 price of 1250 TL/MWh has the same purchasing power as 857.9 TL/MWh in Jan 2022.

---

## File Structure After Pipeline

```
data/
├── bronze/
│   ├── macro/
│   │   ├── macro_evds_2020-01-01_2024-12-31.parquet (✅ monthly EVDS data)
│   │   └── macro_evds_raw.csv                        (legacy format)
│   └── epias/
│       └── price_ptf_2020-01-01_2024-12-31.parquet   (raw EPİAŞ data)
│
├── silver/
│   ├── macro/
│   │   ├── deflator_did_baseline.parquet             (✅ deflator index, baseline)
│   │   └── deflator_did_dfm.parquet                  (✅ deflator index, DFM)
│   └── epias/
│       └── price_ptf_normalized_2020-01-01_2024-12-31.parquet (normalized prices)
│
└── gold/
    └── epias/
        └── price_ptf_deflated_2020-01-01_2024-12-31.parquet   (✅ deflated prices)
```

---

## Key Changes from Original Code

### 1. **Date Format Alignment** (evds_fetcher.py)
   - **Before:** `start_date="01-01-2021"`, `end_date="01-01-2025"` (DD-MM-YYYY)
   - **After:** `start_date="2020-01-01"`, `end_date="2024-12-31"` (YYYY-MM-DD)
   - **Reason:** Match EPİAŞ pipeline format, cover full training period

### 2. **Dual Format Saving** (evds_fetcher.py, deflator_builder.py)
   - **Before:** CSV only
   - **After:** Both Parquet + CSV
   - **Reason:** Parquet is faster and smaller, aligns with EPİAŞ pipeline

### 3. **Backward Compatibility** (deflator_builder.py)
   - Added `_load_bronze_data()` function
   - Prefers Parquet, falls back to CSV
   - No breaking changes for existing workflows

### 4. **Documentation** (All files)
   - Added comprehensive docstrings
   - Clarified date format expectations (YYYY-MM-DD for input, YYYY-MM for monthly aggregation)
   - Explained deflation methodology

---

## Frequently Asked Questions

### Q: Which deflator method should I use (baseline vs DFM)?
**A:** Start with **baseline** (Factor Analysis). It's simpler and equally effective. Use **DFM** only if you want Kalman-smoothed estimates (reduces noise but more complex).

### Q: What if EPİAŞ data has hours outside the deflator date range?
**A:** The deflator will forward-fill (for future dates) or back-fill (for past dates). A warning will be logged. Ideally, fetch EVDS data covering your full EPİAŞ date range.

### Q: Can I change the base month (2022-01)?
**A:** Yes! Edit `BASE_MONTH` in `deflator_builder.py`. Choose a month near the middle of your dataset.

### Q: Do I need to re-run deflation if I fetch new EPİAŞ data?
**A:** Only if the new data falls outside the deflator's date range. Otherwise, reuse existing deflator indices.

### Q: What about forecasting future prices?
**A:** For future horizons (e.g., 2025), you'll need to:
   1. Extrapolate the deflator index (use recent trend or assume constant inflation)
   2. OR: Forecast in real terms, then convert back to nominal using assumed inflation

---

## Troubleshooting

### Error: "Bronze EVDS data not found"
**Solution:** Run `python src/data/evds_fetcher.py` first

### Error: "Deflator file not found"
**Solution:** Run `python src/data/deflator_builder.py` first

### Error: "Price data not found"
**Solution:** Run `python src/data/epias_fetcher.py` first

### Warning: "Missing deflator values"
**Solution:** Extend EVDS date range to cover all EPİAŞ data. Forward/back-fill is only a stopgap.

---

## Next Steps

After deflation:
1. ✅ Use `*_real` columns for modeling (NOT nominal prices)
2. ✅ Feature engineering: Lags, rolling stats on **real** prices
3. ✅ Model training: N-HiTS, CatBoost, etc. on **real** prices
4. ✅ Forecasting: Generate predictions in **real** terms
5. ⚠️ For reporting: Convert real forecasts back to nominal (multiply by current DID_index)

---

**Last Updated:** November 2025
**Authors:** ForeWatt Team
