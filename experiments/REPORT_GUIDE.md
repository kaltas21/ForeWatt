# ForeWatt Experiment Results - Report Guide

## Results Summary

### Consumption Model
- **Best Model:** Higher Learning Rate (lr=0.1)
- **Test sMAPE:** 1.95%
- **Test MAE:** 808.5 MWh
- **Test Period:** June 2024 - October 2025 (12,429 hours)

### Price Model
- **Best Model:** Full Data (4 years) + Hybrid Correction
- **Test sMAPE:** 11.71%
- **Test MAE:** 48.2 TL/MWh
- **Test Period:** June 2024 - October 2025 (12,429 hours)

---

## Recommended Tables for Report

### Table 1: Consumption Model - All Metrics Summary

| Metric | Best Model (Higher LR) | Default Model | Unit |
|--------|----------------------|---------------|------|
| **MAE** | 808.5 | 854.2 | MWh |
| **RMSE** | 1194.8 | 1266.8 | MWh |
| **MAPE** | 1.94% | 2.06% | % |
| **sMAPE** | 1.95% | 2.07% | % |
| **MASE** | 0.76 | 0.80 | - |
| **MBE** | -341.6 | -311.9 | MWh |
| **R²** | 0.969 | 0.965 | - |
| **Max Error** | 7,779 | 8,338 | MWh |
| **Median AE** | 536.5 | 561.2 | MWh |
| **P90 Error** | 1,825.7 | 1,956.5 | MWh |
| **P95 Error** | 2,635.9 | 2,765.2 | MWh |
| **P99 Error** | 4,321.1 | 4,569.9 | MWh |

### Table 2: Price Model - All Metrics Summary

| Metric | Best Model (Hybrid) | Raw Ensemble | Unit |
|--------|---------------------|--------------|------|
| **MAE** | 48.2 | 52.4 | TL/MWh |
| **RMSE** | 69.0 | 75.5 | TL/MWh |
| **MAPE** | 44.0% | 61.0% | % |
| **sMAPE** | 11.71% | 12.58% | % |
| **MASE** | 0.36 | 0.39 | - |
| **MBE** | +2.5 | +6.6 | TL/MWh |
| **R²** | 0.871 | 0.845 | - |
| **Max Error** | 463.6 | 480.6 | TL/MWh |
| **Median AE** | 32.2 | 34.9 | TL/MWh |
| **P90 Error** | 113.0 | 122.0 | TL/MWh |
| **P95 Error** | 149.0 | 165.7 | TL/MWh |
| **P99 Error** | 224.2 | 253.0 | TL/MWh |

### Table 3: Consumption - Data Size Impact

| Training Data | Samples | sMAPE | MAE | R² |
|--------------|---------|-------|-----|-----|
| 1 year (2023) | 8,759 | 2.27% | 934.1 | 0.959 |
| 2 years (2022-23) | 17,518 | 2.16% | 896.2 | 0.963 |
| 3 years (2021-23) | 26,277 | 2.08% | 856.7 | 0.965 |
| **4 years (2020-23)** | **32,765** | **2.07%** | **854.2** | **0.965** |

### Table 4: Price - Data Size Impact

| Training Data | Samples | sMAPE | MAE | R² |
|--------------|---------|-------|-----|-----|
| 1 year | 8,784 | 12.34% | 51.3 | 0.858 |
| 2 years | 17,544 | 12.43% | 52.4 | 0.856 |
| 3 years | 26,304 | 11.82% | 48.9 | 0.866 |
| **4 years (full)** | **34,155** | **11.71%** | **48.2** | **0.871** |

### Table 5: Consumption - Feature Ablation

| Feature Set | Features | sMAPE | MAE | R² |
|------------|----------|-------|-----|-----|
| Lag only | 5 | 3.03% | 1,205.8 | 0.929 |
| Weather only | 8 | 12.95% | 5,174.0 | 0.142 |
| Calendar only | 9 | 10.61% | 4,167.0 | 0.493 |
| Lag + Weather | 13 | 2.97% | 1,183.7 | 0.935 |
| **Lag + Calendar** | **14** | **1.96%** | **799.2** | **0.970** |
| All features | 23 | 2.07% | 854.2 | 0.965 |

### Table 6: Price - Feature Ablation

| Feature Set | Features | sMAPE | MAE | R² |
|------------|----------|-------|-----|-----|
| Price lags only | 14 | 13.51% | 57.8 | 0.817 |
| Market signals only | 15 | 12.84% | 54.7 | 0.839 |
| Calendar only | 11 | 13.00% | 55.7 | 0.820 |
| Price + Market | 23 | 12.13% | 50.6 | 0.859 |
| **All features** | **28** | **11.71%** | **48.2** | **0.871** |

### Table 7: Price - Model Architecture Comparison

| Model | sMAPE | MAE | R² |
|-------|-------|-----|-----|
| CatBoost only | 11.94% | 49.4 | 0.865 |
| LightGBM only | 12.46% | 52.5 | 0.854 |
| Ensemble (61.4%/38.6%) | 11.71% | 48.2 | 0.871 |
| Ensemble (50%/50%) | 11.74% | 48.5 | 0.870 |

### Table 8: Price - Error Correction Impact

| Correction Method | sMAPE | Improvement |
|------------------|-------|-------------|
| Raw (no correction) | 12.58% | - |
| Simple AEC | 11.95% | -0.63% |
| KNN-EC | 11.94% | -0.64% |
| **Hybrid (50/50)** | **11.71%** | **-0.87%** |

### Table 9: Baseline Comparison

| Model | Consumption sMAPE | Price sMAPE |
|-------|-------------------|-------------|
| Naive (Lag-24h) | 5.69% | 21.96% |
| Seasonal Naive (Lag-168h) | 5.48% | 21.60% |
| Rolling Mean (24h) | 15.22% | 25.38% |
| Hourly Mean | 13.37% | 42.16% |
| DOW-Hourly Mean | 12.37% | 43.56% |
| **Our ML Model** | **1.95%** | **11.71%** |
| **Improvement vs Best Baseline** | **2.8x better** | **1.8x better** |

---

## Recommended Plots for Report

### Essential Plots (Must Include)

1. **Data Size Impact** (2 plots)
   - `experiments/results/consumption/*/data_size_comparison.png`
   - `experiments/results/price/*/data_size_comparison.png`

2. **Feature Ablation** (2 plots)
   - `experiments/results/consumption/*/feature_ablation_comparison.png`
   - `experiments/results/price/*/feature_ablation_comparison.png`

3. **Baseline Comparison** (2 plots)
   - `experiments/results/consumption/*/baseline_comparison.png`
   - `experiments/results/price/*/baseline_comparison.png`

4. **Best Model Predictions vs Actual** (2 plots)
   - `experiments/results/consumption/*/hyperparam_higher_lr/predictions_vs_actual.png`
   - `experiments/results/price/*/data_size_4_years/predictions_vs_actual.png`

5. **Error Distribution** (2 plots)
   - `experiments/results/consumption/*/hyperparam_higher_lr/error_distribution.png`
   - `experiments/results/price/*/data_size_4_years/error_distribution.png`

6. **Hourly Error Analysis** (2 plots)
   - `experiments/results/consumption/*/hyperparam_higher_lr/hourly_errors.png`
   - `experiments/results/price/*/data_size_4_years/hourly_errors.png`

7. **Feature Importance** (2 plots)
   - `experiments/results/consumption/*/hyperparam_higher_lr/feature_importance.png`
   - `experiments/results/price/*/data_size_4_years/feature_importance.png`

8. **Scatter: Actual vs Predicted** (2 plots)
   - `experiments/results/consumption/*/hyperparam_higher_lr/scatter_actual_vs_pred.png`
   - `experiments/results/price/*/data_size_4_years/scatter_actual_vs_pred.png`

### Optional Plots (For Detailed Analysis)

9. **Summary Dashboards** (comprehensive single-page views)
   - `experiments/results/*/hyperparam_higher_lr/summary_dashboard.png`
   - `experiments/results/*/data_size_4_years/summary_dashboard.png`

10. **Price Model Comparison**
    - `experiments/results/price/*/model_comparison.png`

11. **Error Correction Impact**
    - `experiments/results/price/*/error_correction_comparison.png`

12. **Hyperparameter Comparison**
    - `experiments/results/consumption/*/hyperparam_comparison.png`

---

## Key Findings Summary

### Consumption Model Insights
1. **More recent data is not always better** - 1 year performs similarly to 4 years
2. **Lag + Calendar features are critical** - Achieve 1.96% sMAPE (best feature subset)
3. **Weather features alone are insufficient** - Only 12.95% sMAPE
4. **Higher learning rate (0.1) improves performance** - 1.95% vs 2.07% baseline
5. **Model beats all baselines by 2.8x** (vs Seasonal Naive 5.48%)

### Price Model Insights
1. **More data helps** - 4 years achieves best results (11.71%)
2. **Ensemble outperforms individual models** - CatBoost (11.94%) + LightGBM (12.46%) → Ensemble (11.71%)
3. **Error correction is essential** - Reduces error from 12.58% to 11.71%
4. **Hybrid correction > Simple AEC > KNN-EC alone**
5. **Model beats all baselines by 1.8x** (vs Seasonal Naive 21.60%)

---

## Files Location

All experiment results are stored in:
```
experiments/results/
├── consumption/20260118_053325/   # Consumption experiments
│   ├── summary.json               # All metrics
│   ├── data_size_*/               # Data size experiments
│   ├── features_*/                # Feature ablation
│   ├── hyperparam_*/              # Hyperparameter tuning
│   └── baselines/                 # Baseline results
└── price/20260118_053425/         # Price experiments
    ├── summary.json               # All metrics
    ├── data_size_*/               # Data size experiments
    ├── features_*/                # Feature ablation
    ├── model_*/                   # Model comparison
    ├── correction_*/              # Error correction
    └── baselines/                 # Baseline results
```
