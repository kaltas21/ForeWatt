# Deep Learning Models for ForeWatt

State-of-the-art deep learning pipeline for multi-horizon electricity forecasting with uncertainty quantification.

## 🎯 Overview

This package implements an advanced deep learning pipeline featuring:

- **3 State-of-the-Art Models**: N-HiTS, TFT, PatchTST
- **Bayesian Hyperparameter Optimization** with Optuna
- **Expanding-Window Temporal Cross-Validation**
- **Multi-Horizon Forecasting** (1-24 hours ahead)
- **Split Conformal Prediction** for calibrated uncertainty intervals
- **Horizon-Wise Evaluation** with sMAPE and MASE

## 📊 Models

### N-HiTS (Neural Hierarchical Interpolation for Time Series)
- **Paper**: "N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting" (AAAI 2023)
- **Strengths**: Multi-rate sampling, hierarchical interpolation, excellent for multiple seasonalities
- **Best for**: Complex seasonal patterns (daily + weekly + yearly)
- **Architecture**: Stacked blocks with identity, trend, and seasonality components

### TFT (Temporal Fusion Transformer)
- **Paper**: "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting" (2021)
- **Strengths**: Attention mechanisms, interpretability, handles static & dynamic features
- **Best for**: Multi-variate forecasting with complex dependencies
- **Architecture**: Variable selection + LSTM + multi-head attention

### PatchTST (Patched Time Series Transformer)
- **Paper**: "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers" (ICLR 2023)
- **Strengths**: Efficient long-sequence modeling via patching, channel independence
- **Best for**: Long-horizon forecasting with many features
- **Architecture**: Patch embedding + transformer encoder

## 🧠 Feature Engineering

### Fixed Feature Set (v2_deep_learning)

**Total**: ~80-100 features across 6 groups

#### 1. Lagged Targets (8 features)
- `{target}_lag_{1,2,3,6,12,24,48,168}h`
- Captures recent history and weekly patterns

#### 2. Rolling Statistics (32 features)
- Windows: 24h, 168h
- Stats: mean, std, min, max
- Variables: consumption/price, temperature

#### 3. Fourier Seasonality (24 features) ⭐ **NEW**
- **Daily** (5 orders): 10 sin/cos features
- **Weekly** (3 orders): 6 sin/cos features
- **Yearly** (4 orders): 8 sin/cos features
- Smooth representation of periodic patterns

#### 4. Calendar Encodings (9 features)
- Cyclical: `hour_sin/cos`, `dow_sin/cos`, `month_sin/cos`
- Binary: `is_weekend`, `is_holiday_day`, `is_holiday_hour`

#### 5. Weather Covariates (15+ features)
- Current: temp, humidity, wind, precipitation
- Lags: `temp_lag_{1,24,168}h`
- Derived: HDD, CDD, heat_index
- Binary: is_hot, is_cold

#### 6. Cross-Domain (3-5 features)
- For **demand**: price features
- For **price**: consumption features

## 🔄 Expanding-Window Cross-Validation

### Strategy

Respects temporal order and tests on increasingly future data:

```
Fold 1: Train[0:50%] → Validate[50%:60%]
Fold 2: Train[0:60%] → Validate[60%:70%]
Fold 3: Train[0:70%] → Validate[70%:80%]
Fold 4: Train[0:80%] → Validate[80%:90%]
Fold 5: Train[0:90%] → Validate[90%:100%]
```

### Benefits
- ✅ No data leakage
- ✅ Mimics production (retrain with all historical data)
- ✅ Tests generalization to future
- ✅ More robust than single split

### Configuration
```python
cv = ExpandingWindowCV(
    n_splits=5,              # Number of folds
    min_train_size=0.5,      # Minimum 50% for first fold
    val_size=0.1,            # 10% validation
    gap=0                    # No gap between train/val
)
```

## 🎛️ Bayesian Hyperparameter Optimization

### Method: Optuna (Tree-structured Parzen Estimator)

Efficiently searches hyperparameter space using Bayesian optimization:

1. **Sample** hyperparameters from search space
2. **Train** model with sampled params on training fold
3. **Evaluate** on validation fold (sMAPE)
4. **Update** Bayesian posterior
5. **Repeat** for N trials
6. **Select** best hyperparameters (min mean sMAPE across CV folds)

### Search Spaces

#### N-HiTS
```python
{
    'stack_types': [['identity', 'trend', 'seasonality']],
    'n_blocks': [1, 2, 3],
    'n_pool_kernel_size': [2, 4, 8],
    'hidden_size': [128, 256, 512],
    'learning_rate': [1e-4, 1e-3, 1e-2],
    'batch_size': [32, 64, 128],
    'max_steps': [500, 1000, 2000]
}
```

#### TFT
```python
{
    'hidden_size': [32, 64, 128, 256],
    'n_heads': [2, 4, 8],
    'dropout': [0.1, 0.2, 0.3],
    'learning_rate': [1e-4, 1e-3, 1e-2],
    'batch_size': [32, 64, 128]
}
```

#### PatchTST
```python
{
    'patch_len': [8, 16, 24],
    'stride': [4, 8, 16],
    'n_layers': [2, 3, 4],
    'hidden_size': [64, 128, 256],
    'n_heads': [4, 8, 16]
}
```

### Early Stopping
- **Metric**: Validation sMAPE
- **Patience**: 50 iterations
- **Min delta**: 0.001

### Pruning
- **Pruner**: MedianPruner
- **Strategy**: Stop bad trials early (below median performance)
- **Benefit**: 2-3x faster optimization

## 📈 Horizon-Wise Evaluation

### Multi-Horizon Forecasting

For each time t, predict h=1,2,...,24 hours ahead:

```
t=100: Predict [101, 102, ..., 124]
t=101: Predict [102, 103, ..., 125]
...
```

### Metrics Per Horizon

**Primary**: sMAPE (Symmetric Mean Absolute Percentage Error)
- Bounded [0, 200]
- Symmetric treatment of over/under-forecasts
- Industry standard

**Secondary**: MASE (Mean Absolute Scaled Error)
- Scale-independent
- Compares to naive seasonal forecast
- MASE < 1 → better than naive

**Additional**: MAE, RMSE

### Aggregation Levels
1. **Per-horizon**: sMAPE for h=1, h=2, ..., h=24
2. **Horizon groups**:
   - Short-term (h=1-6): Tactical decisions
   - Day-ahead (h=24): Market participation
   - Full day (h=1-24): Comprehensive
3. **Overall**: Mean across all horizons

## 🔒 Split Conformal Prediction

### Method

Provides calibrated prediction intervals with **coverage guarantees**:

1. **Split**: Train (60%) / Calibration (20%) / Test (20%)
2. **Train**: Fit model on train set
3. **Calibrate**: Compute residuals on calibration set
4. **Quantiles**: Calculate α-quantile of |residuals|
5. **Intervals**: prediction ± quantile

### Coverage Guarantee

For **90% prediction interval** (α=0.1):
- At least 90% of true values fall within interval
- Guaranteed in expectation (finite sample)
- Post-hoc calibration (no model retraining)

### Horizon-Specific Calibration

Different horizons have different uncertainty:
- **h=1**: Low uncertainty (near future)
- **h=24**: High uncertainty (day ahead)

**Solution**: Separate conformal predictor per horizon
- **Result**: Tighter intervals for short-term, wider for long-term

## 🚀 Usage

### Quick Start

```bash
# Full pipeline (both targets, all models, Bayesian optimization)
python src/models/deep_learning/pipeline_runner.py

# Single target, single model
python src/models/deep_learning/pipeline_runner.py \
    --targets consumption \
    --models nhits \
    --n_trials 20 \
    --n_folds 3

# Custom horizons and coverage
python src/models/deep_learning/pipeline_runner.py \
    --horizons 24 \
    --alpha 0.1 \
    --n_trials 50
```

### Python API

```python
from src.models.deep_learning import (
    DeepLearningFeaturePreparer,
    ExpandingWindowCV,
    SplitConformalPredictor
)

# 1. Prepare features
preparer = DeepLearningFeaturePreparer(target='consumption')
X, y, feature_names = preparer.prepare_features(df)

# 2. Setup CV
cv = ExpandingWindowCV(n_splits=5)

# 3. Run optimization + training
# (See pipeline_runner.py for complete implementation)

# 4. Evaluate with conformal intervals
conformal = SplitConformalPredictor(alpha=0.1)
conformal.fit(model, X_train, y_train, X_calib, y_calib)

y_pred, y_intervals = conformal.predict(X_test)
metrics = conformal.evaluate_coverage(y_test, y_pred, y_intervals)
```

## 📊 Outputs

### 1. MLflow Tracking
- **Experiment**: `ForeWatt-DeepLearning-{Target}`
- **Per trial**: Hyperparameters, CV metrics
- **Best model**: Final metrics, model artifact, feature importance
- **Artifacts**: Predictions, intervals, plots

### 2. Reports (reports/deep_learning/)
- `model_comparison_{target}.csv` - All models ranked
- `best_hyperparameters_{model}_{target}.json` - Optimal params
- `horizon_metrics_{model}_{target}.csv` - Per-horizon sMAPE/MASE
- `conformal_coverage_{model}_{target}.csv` - Interval quality
- `cv_results_{model}_{target}.json` - Cross-validation details

### 3. Plots (reports/deep_learning/plots/)
- Multi-horizon forecast vs actual
- Prediction intervals with coverage
- Per-horizon error breakdown
- Attention weights (TFT)

## 🎯 Expected Performance

### Demand Forecasting (consumption)
| Horizon | Target sMAPE | Expected Best Model |
|---------|--------------|---------------------|
| h=1     | <1.5%        | N-HiTS              |
| h=6     | <2.5%        | N-HiTS / TFT        |
| h=24    | <4.0%        | TFT / PatchTST      |
| **Mean**| **<3.0%**    | **TFT**             |

### Price Forecasting (price_real)
| Horizon | Target sMAPE | Expected Best Model |
|---------|--------------|---------------------|
| h=1     | <3.0%        | N-HiTS              |
| h=6     | <5.0%        | TFT                 |
| h=24    | <8.0%        | TFT / PatchTST      |
| **Mean**| **<6.0%**    | **TFT**             |

### Uncertainty Quantification
- **Coverage**: 90% ± 2% (calibrated)
- **Sharpness**: Minimize interval width while maintaining coverage
- **Winkler Score**: <500 for demand, <1000 for price

## ⚙️ Configuration

### Hardware
- **GPU**: Highly recommended (10-20x speedup)
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 10GB for models + artifacts

### Dependencies
```
neuralforecast>=1.6.0      # N-HiTS, TFT, PatchTST
pytorch>=2.0.0             # Deep learning
pytorch-lightning>=2.0.0   # Training framework
optuna>=3.0.0              # Bayesian optimization
mapie>=0.8.0               # Conformal prediction
mlflow>=2.8.0              # Experiment tracking
```

### Computational Cost
- **Per trial**: 5-30 minutes (depends on model, data size, GPU)
- **Full optimization (50 trials × 3 models)**: 5-25 hours
- **Recommendations**:
  - Start with 20 trials for quick results
  - Use GPU for production runs
  - Run overnight for full optimization

## 📚 Architecture Details

### Data Flow

```
Raw Data (Master Dataset)
    ↓
Feature Preparation (Fourier + engineering)
    ↓
Train/Calib/Test Split (60/20/20)
    ↓
Expanding-Window CV (5 folds on train set)
    ↓
Bayesian Optimization (Optuna)
    ├── Trial 1: Sample params → Train → Validate → Log
    ├── Trial 2: Update posterior → Sample → Train → Validate
    ├── ...
    └── Trial N: Select best params
    ↓
Final Training (best params on full train set)
    ↓
Conformal Calibration (on calibration set)
    ↓
Test Evaluation (multi-horizon + intervals)
    ↓
Outputs (MLflow + Reports + Plots)
```

### Component Structure

```
DeepLearningPipeline
├── FeaturePreparer
│   ├── add_fourier_features()
│   ├── create_fixed_feature_set()
│   └── prepare_sequences()
├── ExpandingWindowCV
│   └── split() → (train_idx, val_idx) for each fold
├── BayesianOptimizer (Optuna)
│   ├── define_search_space()
│   ├── objective_function() → mean_CV_sMAPE
│   └── optimize() → best_hyperparameters
├── ModelTrainer
│   ├── NHiTS / TFT / PatchTST
│   ├── fit()
│   └── predict_multi_horizon()
├── ConformalPredictor
│   ├── fit(model, X_calib, y_calib)
│   ├── predict(X_test) → (y_pred, y_intervals)
│   └── evaluate_coverage()
└── Evaluator
    ├── horizon_wise_metrics()
    ├── aggregate_metrics()
    └── plot_results()
```

## 🔬 Advanced Features (Future)

1. **Transfer Learning**: Pre-train on consumption, fine-tune on price
2. **Ensemble**: Weighted combination of N-HiTS + TFT + PatchTST
3. **Online Learning**: Incremental updates with new data
4. **Explainability**: SHAP values, attention weights analysis
5. **Multi-target**: Joint forecasting of consumption + price
6. **Exogenous Scenarios**: What-if analysis with custom weather/price inputs

## 📖 References

### Papers
- N-HiTS: Challu et al., AAAI 2023
- TFT: Lim et al., International Journal of Forecasting, 2021
- PatchTST: Nie et al., ICLR 2023
- Conformal Prediction: Shafer & Vovk, 2008

### Libraries
- NeuralForecast: https://github.com/Nixtla/neuralforecast
- Optuna: https://optuna.org/
- MAPIE: https://mapie.readthedocs.io/

## ✅ Implementation Status

- ✅ Feature preparer with Fourier seasonality
- ✅ Expanding-window cross-validation
- ✅ Split conformal prediction
- 🚧 Model trainers (N-HiTS, TFT, PatchTST) - **IN PROGRESS**
- 🚧 Bayesian optimization orchestrator - **IN PROGRESS**
- 🚧 Pipeline runner - **IN PROGRESS**
- 📋 Examples and tests - TODO
- 📋 Production deployment - TODO

## 🚦 Next Steps

1. Complete model trainers with Optuna integration
2. Implement pipeline orchestrator
3. Run full optimization for both targets
4. Analyze results and select best models
5. Deploy to production API

---

**Author**: ForeWatt Team
**Date**: November 2025
**License**: MIT
