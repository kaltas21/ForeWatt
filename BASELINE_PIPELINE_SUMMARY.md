# Baseline Pipeline with Intelligent Feature Selection

## 🎯 Overview

A comprehensive baseline modeling pipeline for ForeWatt with **intelligent feature selection** that automatically optimizes features for different model types and prediction targets.

### Two Prediction Targets
1. **Demand Forecasting** (`consumption`) - Electricity consumption in MWh
2. **Price Forecasting** (`price_real`) - Today's Turkish Lira (deflated/inflation-adjusted)

### Five Baseline Models
1. **CatBoost** - Gradient boosting with categorical features
2. **XGBoost** - Extreme gradient boosting
3. **LightGBM** - Fast gradient boosting
4. **Prophet** - Facebook's time series forecaster
5. **SARIMAX** - Statistical ARIMA with exogenous variables

---

## 🧠 Intelligent Feature Selection

### The Problem
Different model types require different features:
- **Statistical models** (Prophet, SARIMAX) handle autocorrelation internally → don't need lag features
- **Boosting models** (CatBoost, XGBoost, LightGBM) have no time series structure → need lag features

Using the same features for all models leads to:
- ❌ Redundancy in statistical models (overfitting)
- ❌ Poor performance in boosting models (missing patterns)

### The Solution
Automatic feature selection based on model type:

| Feature Type | Statistical Models | Boosting Models |
|--------------|-------------------|-----------------|
| Temporal/Calendar | ✅ | ✅ |
| Weather (core + derived) | ✅ | ✅ |
| Temperature dynamics | ✅ | ✅ |
| Lag features (1h-168h) | ❌ | ✅ |
| Rolling statistics | ❌ | ✅ |
| **Total Features** | **~50-60** | **~100+** |

---

## 📁 Project Structure

```
src/models/baseline/
├── __init__.py              # Package exports
├── feature_selector.py      # Intelligent feature selection logic
├── model_trainer.py         # Unified model trainer (all 5 models)
├── data_loader.py           # Data loading and temporal splitting
├── pipeline_runner.py       # Main orchestrator with MLflow
├── examples.py              # 6 usage examples
├── test_pipeline.py         # Validation tests
├── run_pipeline.sh          # Quick start bash script
└── README.md                # Complete documentation
```

---

## 🚀 Quick Start

### 1. Run Complete Pipeline (Both Targets, All Models)
```bash
cd /home/user/ForeWatt
python src/models/baseline/pipeline_runner.py
```

### 2. Run for Specific Target
```bash
# Demand forecasting only
python src/models/baseline/pipeline_runner.py --targets consumption

# Price forecasting only
python src/models/baseline/pipeline_runner.py --targets price_real
```

### 3. Run Specific Models
```bash
# Boosting models only
python src/models/baseline/pipeline_runner.py --models catboost xgboost lightgbm

# Statistical models only
python src/models/baseline/pipeline_runner.py --models prophet sarimax

# Single model
python src/models/baseline/pipeline_runner.py --models catboost
```

### 4. Quick Test (Fastest)
```bash
# CatBoost + Demand only
bash src/models/baseline/run_pipeline.sh quick
```

---

## 📊 Feature Groups

### Core Features (All Models)

**1. Temporal Core** (17 features)
- Hour: `hour_x`, `hour_y`, `hour_sin`, `hour_cos`
- Day: `dow`, `dom`, `day_of_week`, `day_of_year`
- Month: `month_x`, `month_y`, `weekofyear`
- Cyclical: `dow_sin_x`, `dow_cos_x`, `dow_sin_y`, `dow_cos_y`, `month_sin`, `month_cos`

**2. Calendar** (5 features)
- `is_weekend_x`, `is_weekend_y`
- `is_holiday_day`, `is_holiday_hour`
- `holiday_name` (categorical, boosting only)

**3. Weather Core** (7 features)
- `temp_national`, `humidity_national`, `wind_speed_national`
- `apparent_temp_national`, `precipitation_national`
- `cloud_cover_national`, `temp_std`

**4. Weather Derived** (14 features)
- Degree days: `HDD`, `CDD`, `HDD_15`, `CDD_21`
- Comfort: `heat_index`, `wind_chill`
- Binary flags: `is_hot`, `is_very_hot`, `is_cold`, `is_very_cold`, `is_raining`, `is_heavy_rain`, `is_cloudy`

**5. Temperature Dynamics** (5 features)
- `temp_change_1h`, `temp_change_3h`, `temp_change_24h`
- `temp_shock`, `temp_range_24h`

### Boosting-Only Features

**6. Consumption Lags** (8 features)
- `consumption_lag_1h`, `consumption_lag_2h`, `consumption_lag_3h`
- `consumption_lag_6h`, `consumption_lag_12h`, `consumption_lag_24h`
- `consumption_lag_48h`, `consumption_lag_168h`

**7. Temperature Lags** (10 features)
- Short-term: `temp_lag_1h`, `temp_lag_2h`, `temp_lag_3h`
- Daily/weekly: `temp_lag_24h`, `temp_lag_168h`
- Also: `temperature_lag_*` variants

**8. Price Lags** (3 features)
- `price_ptf_lag_1h`, `price_ptf_lag_24h`, `price_ptf_lag_168h`

**9. Rolling Statistics** (30+ features)
- Consumption: 24h/168h windows × (mean, std, min, max)
- Temperature: 24h/168h windows × (mean, std, min, max)
- Price: 24h/168h windows × (mean, std, min, max)
- Derived: `consumption_range_24h`, `consumption_cv_24h`, `temp_range_24h`

---

## 🎯 Why This Design Works

### For Statistical Models (Prophet, SARIMAX)

**What they do internally:**
- Model trends, seasonality, holidays
- Handle autocorrelation with AR/MA terms
- Capture weekly/yearly patterns

**What they need from us:**
- External regressors (weather, calendar)
- Domain knowledge features (HDD/CDD, holidays)

**What hurts them:**
- Lag features → redundant with internal AR terms
- Rolling stats → redundant with MA terms
- Result: Overfitting, poor generalization

### For Boosting Models (CatBoost, XGBoost, LightGBM)

**What they do internally:**
- Tree-based feature interactions
- Automatic feature selection
- Handle non-linearities

**What they DON'T do:**
- No time series structure
- No autocorrelation modeling

**What they need from us:**
- Lag features → capture recent history
- Rolling stats → capture trends
- All available features → let model select

---

## 📈 Expected Performance

### Demand Forecasting (`consumption`)
- **Target**: 2-3% sMAPE (1-6h), 4-6% sMAPE (24h)
- **Best models**: CatBoost, XGBoost, LightGBM
- **Top features**:
  - `consumption_lag_24h` (same hour yesterday)
  - `consumption_lag_168h` (same hour last week)
  - `temp_national`
  - `hour_x`

### Price Forecasting (`price_real`)
- **Target**: 5-10% sMAPE (more volatile)
- **Best models**: CatBoost, XGBoost
- **Top features**:
  - `price_ptf_lag_24h`
  - `consumption` (demand drives price)
  - `temp_national`
  - `hour_x`

---

## 📊 Outputs

### 1. MLflow Tracking
All experiments logged to MLflow:
- **Experiments**:
  - `ForeWatt-Baseline-Consumption`
  - `ForeWatt-Baseline-Price-Real`
- **Metrics**: MAE, RMSE, MAPE, sMAPE, MASE, training time
- **Artifacts**: Feature importance CSVs, predictions

View with:
```bash
mlflow ui --backend-store-uri /home/user/ForeWatt/mlruns
```

### 2. Comparison Reports
Saved to `reports/baseline/`:
- `baseline_comparison_consumption.csv` - Model rankings for demand
- `baseline_comparison_price_real.csv` - Model rankings for price
- `baseline_results_*.json` - Full results

### 3. Feature Importance
Saved to `reports/baseline/`:
- `feature_importance_summary_consumption.csv` - Top features for demand
- `feature_importance_summary_price_real.csv` - Top features for price
- Per-model CSVs in MLflow artifacts

---

## 🧪 Testing

### Validation Tests
```bash
python src/models/baseline/test_pipeline.py
```

Tests:
- ✅ Imports
- ✅ Data loading (master dataset)
- ✅ Feature selector (boosting vs statistical)
- ✅ Data split (temporal)
- ✅ Model trainer initialization

### Example Scripts
```bash
# List examples
python src/models/baseline/examples.py

# Run specific example
python src/models/baseline/examples.py --example 1
```

Examples:
1. Full pipeline (both targets, all models)
2. Demand forecasting (boosting only)
3. Price forecasting (custom hyperparameters)
4. Feature analysis (compare selections)
5. Feature importance (single model deep dive)
6. Quick test (sanity check)

---

## 📐 Evaluation Metrics

- **MAE** (Mean Absolute Error): Average error in MWh or TL
- **RMSE** (Root Mean Squared Error): Penalizes large errors
- **MAPE** (Mean Absolute Percentage Error): Percentage error
- **sMAPE** (Symmetric MAPE): Bounded [0, 200], addresses MAPE asymmetry
- **MASE** (Mean Absolute Scaled Error): Scale-independent
  - MASE < 1: Better than naive seasonal forecast
  - MASE = 1: Same as naive
  - MASE > 1: Worse than naive

**Primary metric**: MASE (used for model ranking)

---

## 🔄 Data Split

- **Train**: 70% (oldest data)
- **Validation**: 10% (for early stopping)
- **Test**: 20% (newest data)

Split is **temporal** (no shuffling) to respect time series structure.

---

## 💡 Key Design Decisions

### 1. Why Separate Feature Selection?
- Statistical models: Built-in autocorrelation → simpler features
- Boosting models: No time structure → need all features
- One-size-fits-all hurts both types

### 2. Why Not Use All Features for Statistical Models?
- Prophet/SARIMAX have AR/MA components
- Adding lags creates redundancy
- Result: Overfitting, slower training, worse generalization

### 3. Why Include Lag Features for Boosting?
- Trees don't know about time
- Lag features explicitly provide recent history
- Critical for time series forecasting

### 4. Why Two Targets?
- Demand forecasting: Grid stability (primary goal)
- Price forecasting: Economic planning (secondary goal)
- Different patterns, different optimal features

### 5. Why Temporal Split?
- Time series data has temporal dependencies
- Shuffling breaks causality
- Test on future data (realistic scenario)

---

## 🚧 Current Limitations & Future Work

### Current Limitations
1. **Single horizon**: Currently predicts 1-step ahead
2. **No ensemble**: Models run independently
3. **No uncertainty**: Point forecasts only (no intervals)
4. **No hyperparameter tuning**: Uses defaults

### Future Enhancements
1. **Multi-horizon forecasting**: 1-24h ahead
2. **Ensemble methods**: Weighted average of best models
3. **Conformal prediction**: Calibrated uncertainty intervals
4. **Hyperparameter optimization**: Optuna integration
5. **Online learning**: Incremental model updates
6. **Cross-validation**: Time series CV for robust evaluation

---

## 📚 Usage Examples

### Example 1: Quick Test
```bash
python src/models/baseline/pipeline_runner.py --targets consumption --models catboost
```

### Example 2: Boosting Models Comparison
```bash
python src/models/baseline/pipeline_runner.py --models catboost xgboost lightgbm
```

### Example 3: Custom Hyperparameters
```python
from src.models.baseline import BaselinePipeline, data_loader

# Load data
df = data_loader.load_master_data()
train_df, val_df, test_df = data_loader.train_val_test_split(df)

# Initialize pipeline
pipeline = BaselinePipeline(target='consumption')

# Custom hyperparameters
hyperparams = {
    'iterations': 2000,
    'learning_rate': 0.05,
    'depth': 8
}

# Train
results = pipeline.run_model('catboost', train_df, val_df, test_df, hyperparams)
```

---

## ✅ What's Done

- ✅ Intelligent feature selector (statistical vs boosting)
- ✅ Unified model trainer (5 models)
- ✅ Data loader with temporal splitting
- ✅ Pipeline orchestrator with MLflow
- ✅ Dual target support (demand + price)
- ✅ Feature importance analysis
- ✅ Comparison reports (CSV + JSON)
- ✅ Validation tests
- ✅ Example scripts
- ✅ Comprehensive documentation

---

## 📞 Contact & Contributing

**Team**: ForeWatt (Koç University COMP 491)
**License**: MIT
**Status**: Ready for use

For issues or questions, see project README.

---

## 🎓 References

- **Feature Engineering**: `src/features/`
- **Data Pipeline**: `src/data/`
- **Evaluation**: `src/models/evaluate.py`
- **Master Dataset**: `data/gold/master/`
- **Documentation**: `docs/`

---

**Ready to run! Start with:**
```bash
python src/models/baseline/pipeline_runner.py --models catboost --targets consumption
```
