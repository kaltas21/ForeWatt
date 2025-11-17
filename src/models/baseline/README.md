# Baseline Models with Intelligent Feature Selection

Optimized baseline models for ForeWatt with automatic feature selection for:
- **Demand forecasting** (`consumption`)
- **Price forecasting** (`price_real` - today's Turkish Lira)

## Key Features

### 🎯 Dual Target Support
- **Consumption**: Electricity demand in MWh
- **Price (Real)**: Today's Turkish Lira (deflated/inflation-adjusted)

### 🤖 Five Baseline Models
1. **CatBoost** - Gradient boosting with categorical features
2. **XGBoost** - Extreme gradient boosting
3. **LightGBM** - Fast gradient boosting
4. **Prophet** - Facebook's time series forecaster
5. **SARIMAX** - Statistical ARIMA with exogenous variables

### 🧠 Intelligent Feature Selection

The pipeline automatically selects optimal features based on model type:

#### Statistical Models (Prophet, SARIMAX)
- ✅ Temporal/Calendar features
- ✅ Weather features (core + derived)
- ✅ Temperature dynamics
- ❌ Lag features (models handle autocorrelation internally)
- ❌ Rolling statistics (redundant with internal modeling)
- **~50-60 features**

#### Boosting Models (CatBoost, XGBoost, LightGBM)
- ✅ ALL features including:
  - Temporal/Calendar
  - Weather (core + derived)
  - Lag features (1h, 2h, 3h, 6h, 12h, 24h, 48h, 168h)
  - Rolling statistics (24h, 168h windows)
  - Temperature dynamics
  - Price features (for demand) / Consumption features (for price)
- **~100+ features**

## Project Structure

```
baseline/
├── __init__.py                # Package initialization
├── feature_selector.py        # Intelligent feature selection logic
├── model_trainer.py           # Unified model trainer
├── data_loader.py             # Data loading and splitting
├── pipeline_runner.py         # Main orchestrator
└── README.md                  # This file
```

## Quick Start

### Run Complete Pipeline (Both Targets, All Models)

```bash
python src/models/baseline/pipeline_runner.py
```

### Run for Specific Target

```bash
# Demand forecasting only
python src/models/baseline/pipeline_runner.py --targets consumption

# Price forecasting only
python src/models/baseline/pipeline_runner.py --targets price_real
```

### Run Specific Models

```bash
# Run only boosting models
python src/models/baseline/pipeline_runner.py --models catboost xgboost lightgbm

# Run only statistical models
python src/models/baseline/pipeline_runner.py --models prophet sarimax

# Run single model
python src/models/baseline/pipeline_runner.py --models catboost
```

### Combine Options

```bash
# Price prediction with boosting models only
python src/models/baseline/pipeline_runner.py \
    --targets price_real \
    --models catboost xgboost lightgbm
```

## Usage in Code

```python
from src.models.baseline import run_baseline_pipeline

# Run complete pipeline
results = run_baseline_pipeline(
    targets=['consumption', 'price_real'],
    models=['catboost', 'xgboost', 'lightgbm', 'prophet', 'sarimax'],
    val_size=0.1,
    test_size=0.2
)

# Results structure:
# {
#   'consumption': {
#     'catboost': {'MAE': 500.2, 'RMSE': 650.3, 'sMAPE': 2.1, 'MASE': 0.45, ...},
#     'xgboost': {...},
#     ...
#   },
#   'price_real': {
#     'catboost': {...},
#     ...
#   }
# }
```

## Feature Selection Details

### Why Different Features for Different Models?

**Statistical Models (Prophet, SARIMAX)**:
- These models have built-in autocorrelation handling (AR/MA terms, seasonality components)
- Adding lag features and rolling stats creates redundancy and overfitting
- They work best with **external regressors** that explain the signal (weather, calendar)
- Simpler feature set = faster training + better generalization

**Boosting Models (CatBoost, XGBoost, LightGBM)**:
- No built-in time series structure
- **Need lag features** to understand recent history
- **Need rolling stats** to capture trends and patterns
- Can handle many features and automatically select important ones
- More features = better capture of complex patterns

### Feature Groups

1. **Temporal Core** (all models)
   - `hour_x`, `hour_y`, `hour_sin`, `hour_cos`
   - `dow`, `dom`, `day_of_week`, `day_of_year`
   - `month_x`, `month_y`, `weekofyear`
   - Cyclical encodings

2. **Calendar** (all models)
   - `is_weekend_x`, `is_weekend_y`
   - `is_holiday_day`, `is_holiday_hour`
   - `holiday_name` (boosting only)

3. **Weather Core** (all models)
   - `temp_national`, `humidity_national`, `wind_speed_national`
   - `apparent_temp_national`, `precipitation_national`
   - `cloud_cover_national`, `temp_std`

4. **Weather Derived** (all models)
   - `HDD`, `CDD`, `heat_index`, `wind_chill`
   - Binary flags: `is_hot`, `is_cold`, `is_raining`, etc.

5. **Temperature Dynamics** (all models)
   - `temp_change_1h`, `temp_change_3h`, `temp_change_24h`
   - `temp_shock`, `temp_range_24h`

6. **Lag Features** (boosting only)
   - Consumption lags: 1h, 2h, 3h, 6h, 12h, 24h, 48h, 168h
   - Temperature lags: 1h, 2h, 3h, 24h, 168h
   - Price lags: 1h, 24h, 168h

7. **Rolling Statistics** (boosting only)
   - 24h and 168h windows
   - Mean, std, min, max for consumption/temperature/price
   - Range, CV (coefficient of variation)

## Outputs

### 1. MLflow Tracking
All experiments logged to MLflow:
- **Experiment**: `ForeWatt-Baseline-Consumption` or `ForeWatt-Baseline-Price-Real`
- **Metrics**: MAE, RMSE, MAPE, sMAPE, MASE, training time
- **Artifacts**: Feature importance, predictions

### 2. Comparison Reports
Saved to `reports/baseline/`:
- `baseline_comparison_consumption.csv` - Model comparison for demand
- `baseline_comparison_price_real.csv` - Model comparison for price
- `baseline_results_consumption.json` - Full results JSON
- `baseline_results_price_real.json` - Full results JSON

### 3. Feature Importance
Saved to `reports/baseline/`:
- `feature_importance_summary_consumption.csv` - Aggregated importance for demand
- `feature_importance_summary_price_real.csv` - Aggregated importance for price
- Per-model importance CSVs in MLflow artifacts

## Evaluation Metrics

- **MAE** (Mean Absolute Error): Average error in MWh or TL
- **RMSE** (Root Mean Squared Error): Penalizes large errors
- **MAPE** (Mean Absolute Percentage Error): Percentage error
- **sMAPE** (Symmetric MAPE): Bounded [0, 200], addresses MAPE asymmetry
- **MASE** (Mean Absolute Scaled Error): Scale-independent
  - MASE < 1: Better than naive seasonal forecast
  - MASE = 1: Same as naive
  - MASE > 1: Worse than naive

**Primary metric**: MASE (used for ranking models)

## Data Split

- **Train**: 70% (oldest data)
- **Validation**: 10% (for early stopping)
- **Test**: 20% (newest data)

Split is **temporal** (no shuffling) to respect time series structure.

## Advanced Usage

### Custom Hyperparameters

```python
from src.models.baseline import BaselinePipeline

pipeline = BaselinePipeline(target='consumption')

# Load and split data
from src.models.baseline.data_loader import load_master_data, train_val_test_split
df = load_master_data()
train_df, val_df, test_df = train_val_test_split(df)

# Run with custom hyperparameters
custom_hyperparams = {
    'iterations': 2000,
    'learning_rate': 0.05,
    'depth': 8
}

results = pipeline.run_model(
    model_type='catboost',
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    hyperparams=custom_hyperparams
)
```

### Feature Importance Analysis

```python
from src.models.baseline.feature_selector import FeatureSelector

selector = FeatureSelector(target='consumption')

# Get features for specific model
features = selector.get_features_for_model_type('catboost', 'consumption')
print(f"CatBoost features: {len(features)}")

# Print summary
selector.print_feature_summary('catboost', 'consumption')
selector.print_feature_summary('prophet', 'consumption')
```

## Expected Performance

### Demand Forecasting (consumption)
- **Target**: 2-3% sMAPE (short-term), 4-6% sMAPE (day-ahead)
- **Best models**: CatBoost, XGBoost, LightGBM
- **Top features**: `consumption_lag_24h`, `consumption_lag_168h`, `temp_national`, `hour_x`

### Price Forecasting (price_real)
- **Target**: 5-10% sMAPE (more volatile than demand)
- **Best models**: CatBoost, XGBoost
- **Top features**: `price_ptf_lag_24h`, `consumption`, `temp_national`, `hour_x`

## Troubleshooting

### ModuleNotFoundError
```bash
# Ensure project root is in PYTHONPATH
export PYTHONPATH=/home/user/ForeWatt:$PYTHONPATH
```

### Missing Data
```bash
# Check master dataset exists
ls -lh data/gold/master/master_v*.parquet
```

### SARIMAX Slow
SARIMAX is computationally expensive. The pipeline automatically:
- Uses only top 10 most correlated features
- Limits iterations to 100
- Consider skipping SARIMAX for large datasets

### Prophet Warnings
Prophet may show warnings about missing holidays - this is expected and doesn't affect performance.

## Next Steps

1. **Hyperparameter Tuning**: Use Optuna to optimize model hyperparameters
2. **Ensemble**: Combine best models (e.g., weighted average of CatBoost + XGBoost)
3. **Conformal Prediction**: Add uncertainty quantification with MAPIE
4. **Multi-horizon**: Extend to 1-24h ahead forecasts
5. **Online Learning**: Implement incremental updates as new data arrives

## References

- ForeWatt Documentation: `docs/`
- Feature Engineering: `src/features/`
- Evaluation Metrics: `src/models/evaluate.py`
- Master Dataset: `data/gold/master/`

---

**Author**: ForeWatt Team
**Date**: November 2025
**License**: MIT
