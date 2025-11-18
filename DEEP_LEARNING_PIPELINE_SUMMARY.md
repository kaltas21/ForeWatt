# Deep Learning Pipeline for ForeWatt

## 🎯 Overview

State-of-the-art deep learning pipeline for multi-horizon electricity demand and price forecasting with:
- **3 Deep Learning Models**: N-HiTS, TFT, PatchTST
- **Bayesian Hyperparameter Optimization** (Optuna)
- **Expanding-Window Temporal Cross-Validation**
- **Horizon-Wise Evaluation** (1-24h ahead)
- **Split Conformal Prediction** for calibrated uncertainty

---

## 🏗️ Architecture

### Models Implemented

#### 1. **N-HiTS** (Neural Hierarchical Interpolation for Time Series)
- **Type**: Deep learning with hierarchical interpolation
- **Strength**: Multi-rate sampling, handles multiple seasonalities
- **Best for**: Complex seasonal patterns
- **Parameters to optimize**:
  - Stack count (2-4)
  - Hidden size (128-512)
  - Number of blocks per stack
  - Max pool size
  - Learning rate

#### 2. **TFT** (Temporal Fusion Transformer)
- **Type**: Attention-based transformer
- **Strength**: Interpretable attention, handles static/dynamic features
- **Best for**: Multi-variate forecasting with attention
- **Parameters to optimize**:
  - Hidden size (32-256)
  - Number of attention heads
  - Dropout rate
  - Learning rate

#### 3. **PatchTST** (Patched Time Series Transformer)
- **Type**: Transformer with patch-based input
- **Strength**: Efficient long-sequence modeling
- **Best for**: Long-horizon forecasting
- **Parameters to optimize**:
  - Patch size
  - Number of layers
  - Hidden dimension
  - Number of heads
  - Learning rate

---

## 📊 Feature Set (Fixed & Versioned)

### Feature Groups (v2_deep_learning)

**1. Lagged Targets** (8 features)
- `consumption_lag_{1,2,3,6,12,24,48,168}h`
- or `price_real_lag_{1,2,3,6,12,24,48,168}h`

**2. Rolling Statistics** (16 features per window)
- Windows: 24h, 168h
- Stats: mean, std, min, max
- For: consumption/price + temperature

**3. Fourier Seasonality** (24 features)
- **Daily** (5 orders): 10 features (sin/cos pairs)
- **Weekly** (3 orders): 6 features
- **Yearly** (4 orders): 8 features
- Captures smooth periodic patterns

**4. Calendar Encodings** (9 features)
- Cyclical: `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `month_sin`, `month_cos`
- Binary: `is_weekend`, `is_holiday_day`, `is_holiday_hour`

**5. Weather Covariates** (15+ features)
- Current: `temp_national`, `humidity_national`, `wind_speed_national`, `precipitation_national`
- Lags: `temp_lag_{1,24,168}h`
- Derived: `HDD`, `CDD`, `heat_index`, `temp_change_24h`
- Binary: `is_hot`, `is_cold`

**6. Cross-Domain** (3-5 features)
- For demand: price features
- For price: consumption features

**Total**: ~80-100 features

---

## 🔄 Expanding-Window Cross-Validation

### Strategy

```
Fold 1: Train[0:10000]  →  Val[10000:11000]
Fold 2: Train[0:11000]  →  Val[11000:12000]
Fold 3: Train[0:12000]  →  Val[12000:13000]
Fold 4: Train[0:13000]  →  Val[13000:14000]
Fold 5: Train[0:14000]  →  Val[14000:15000]
```

### Benefits
- ✅ Respects temporal ordering (no leakage)
- ✅ Tests on increasingly future data
- ✅ Mimics production scenario (retrain with all data)
- ✅ More robust than single train/val split

### Configuration
- **Folds**: 5 (default, configurable)
- **Min train size**: 50% of data or 1 week minimum
- **Val size**: 10% of data or 24h minimum
- **Gap**: 0 (no gap between train and val)

---

## 🎛️ Bayesian Hyperparameter Optimization

### Method: Optuna (Tree-structured Parzen Estimator)

### Search Spaces (Constrained)

#### N-HiTS
```python
{
    'stack_types': [['identity', 'trend', 'seasonality']],  # Fixed
    'n_blocks': [1, 2, 3],  # Per stack
    'n_pool_kernel_size': [2, 4, 8],
    'n_freq_downsample': [[2, 1, 1], [4, 2, 1], [8, 4, 1]],
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
    'batch_size': [32, 64, 128],
    'max_steps': [500, 1000, 2000]
}
```

#### PatchTST
```python
{
    'patch_len': [8, 16, 24],
    'stride': [4, 8, 16],
    'n_layers': [2, 3, 4],
    'hidden_size': [64, 128, 256],
    'n_heads': [4, 8, 16],
    'learning_rate': [1e-4, 1e-3, 1e-2],
    'batch_size': [32, 64, 128]
}
```

### Optimization Strategy
- **Trials**: 50 per model (configurable)
- **Pruning**: Median pruner (early stopping bad trials)
- **Objective**: Mean sMAPE across CV folds
- **Secondary**: MASE (scale-independent)
- **Timeout**: 12 hours per model (configurable)

---

## 📈 Horizon-Wise Evaluation

### Multi-Horizon Forecasting (1-24h ahead)

For each time point, predict 1, 2, ..., 24 hours ahead:

```
t=0:  Predict [t+1, t+2, ..., t+24]
t=1:  Predict [t+2, t+3, ..., t+25]
...
```

### Metrics Per Horizon
- **sMAPE**: Primary metric (symmetric, bounded)
- **MASE**: Scale-independent (vs naive forecast)
- **MAE**: Absolute error in MWh or TL
- **RMSE**: Penalizes large errors

### Aggregation
- **Per-horizon**: sMAPE for h=1, h=2, ..., h=24
- **Mean across horizons**: Overall performance
- **By horizon group**:
  - Short-term: h=1-6 (tactical)
  - Day-ahead: h=24 (market)
  - Full day: h=1-24 (comprehensive)

---

## 🔒 Split Conformal Prediction

### Method
1. **Split data**: Train / Calibration / Test (60% / 20% / 20%)
2. **Train model**: On train set
3. **Calibrate**: Compute residuals on calibration set
4. **Quantiles**: Calculate α-quantile of absolute residuals
5. **Intervals**: prediction ± quantile

### Coverage Guarantee
For 90% prediction interval (α=0.1):
- At least 90% of true values fall within interval (in expectation)
- Calibrated post-hoc (no model retraining needed)
- Works with any point forecaster

### Horizon-Specific Calibration
- **Problem**: Different horizons have different uncertainty
  - h=1: Low uncertainty (near future)
  - h=24: High uncertainty (day ahead)
- **Solution**: Separate calibration for each horizon
- **Result**: Tighter intervals for short-term, wider for long-term

---

## 🎯 Complete Workflow

```
1. DATA PREPARATION
   ├── Load master dataset (Gold layer)
   ├── Add Fourier seasonality features
   ├── Create fixed feature set (v2_deep_learning)
   └── Split: Train (60%) / Calib (20%) / Test (20%)

2. EXPANDING-WINDOW CV (on Train set)
   ├── Fold 1: Train[0:50%] → Val[50%:60%]
   ├── Fold 2: Train[0:60%] → Val[60%:70%]
   ├── Fold 3: Train[0:70%] → Val[70%:80%]
   ├── Fold 4: Train[0:80%] → Val[80%:90%]
   └── Fold 5: Train[0:90%] → Val[90%:100%]

3. BAYESIAN OPTIMIZATION (per model, per fold)
   ├── Trial 1: Sample hyperparameters from search space
   ├── Train model with sampled params
   ├── Evaluate on validation fold (sMAPE)
   ├── Update Bayesian posterior
   ├── Trial 2-50: Repeat
   └── Select best hyperparameters (best mean sMAPE across folds)

4. FINAL TRAINING
   ├── Train with best hyperparameters on full train set
   ├── Early stopping on calibration set
   └── Save final model

5. CONFORMAL CALIBRATION
   ├── Predict on calibration set
   ├── Compute residuals per horizon
   ├── Calculate α-quantiles (90% coverage)
   └── Store quantiles for prediction intervals

6. TEST EVALUATION
   ├── Predict on test set (multi-horizon)
   ├── Apply conformal intervals
   ├── Evaluate metrics per horizon
   ├── Aggregate results
   └── Save predictions + intervals

7. OUTPUTS
   ├── MLflow: Hyperparameters, metrics, artifacts
   ├── Reports: Per-horizon metrics, best models
   ├── Plots: Forecast vs actual, prediction intervals
   └── Model: Saved for production deployment
```

---

## 📁 File Structure

```
src/models/deep_learning/
├── __init__.py                    # Package exports
├── feature_preparer.py            # ✅ Fourier + feature engineering
├── cv_strategy.py                 # ✅ Expanding-window CV
├── conformal.py                   # ✅ Split conformal prediction
├── models/
│   ├── __init__.py
│   ├── nhits_trainer.py           # N-HiTS with Optuna
│   ├── tft_trainer.py             # TFT with Optuna
│   └── patchtst_trainer.py        # PatchTST with Optuna
├── hyperopt.py                    # Bayesian optimization orchestrator
├── evaluator.py                   # Horizon-wise evaluation
├── pipeline_runner.py             # Main orchestrator (full workflow)
├── examples.py                    # Usage examples
├── test_pipeline.py               # Validation tests
└── README.md                      # Complete documentation
```

---

## 🚀 Usage

### Quick Start
```bash
# Full pipeline: Both targets, all models, with optimization
python src/models/deep_learning/pipeline_runner.py

# Single target, single model (faster)
python src/models/deep_learning/pipeline_runner.py \
    --targets consumption \
    --models nhits \
    --n_trials 20

# With custom horizons
python src/models/deep_learning/pipeline_runner.py \
    --horizons 24 \
    --n_folds 3
```

### Python API
```python
from src.models.deep_learning import DeepLearningPipeline

# Initialize
pipeline = DeepLearningPipeline(
    target='consumption',
    horizon=24,
    n_folds=5,
    n_trials=50
)

# Run optimization + training
results = pipeline.run(
    models=['nhits', 'tft', 'patchtst']
)

# Best model
best_model = results['best_model']
best_params = results['best_hyperparameters']
test_metrics = results['test_metrics']
```

---

## 📊 Expected Outputs

### 1. MLflow Experiments
- **Per trial**: Hyperparameters, CV metrics
- **Per model**: Best params, final metrics, model artifact
- **Comparison**: All models side-by-side

### 2. Reports (reports/deep_learning/)
- `model_comparison_{target}.csv` - All models ranked
- `best_hyperparameters_{model}_{target}.json` - Optimal params
- `horizon_metrics_{model}_{target}.csv` - Per-horizon performance
- `conformal_coverage_{model}_{target}.csv` - Interval quality

### 3. Plots (reports/deep_learning/plots/)
- Forecast vs actual (per horizon)
- Prediction intervals
- Attention weights (TFT only)
- Feature importance (where applicable)

---

## 🎯 Performance Targets

### Demand Forecasting (`consumption`)
| Horizon | Target sMAPE | Expected Model |
|---------|--------------|----------------|
| h=1     | <1.5%        | N-HiTS         |
| h=6     | <2.5%        | N-HiTS / TFT   |
| h=24    | <4.0%        | TFT / PatchTST |
| Mean    | <3.0%        | TFT            |

### Price Forecasting (`price_real`)
| Horizon | Target sMAPE | Expected Model |
|---------|--------------|----------------|
| h=1     | <3.0%        | N-HiTS         |
| h=6     | <5.0%        | TFT            |
| h=24    | <8.0%        | TFT / PatchTST |
| Mean    | <6.0%        | TFT            |

### Uncertainty (Conformal Intervals)
- **Coverage**: 90% ± 2%
- **Sharpness**: Minimize interval width while maintaining coverage

---

## ⚙️ Configuration

### Hardware Requirements
- **GPU**: Recommended (10x faster training)
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 10GB for models + results

### Software Dependencies
```
neuralforecast>=1.6.0      # N-HiTS, TFT, PatchTST
pytorch>=2.0.0             # Deep learning backend
optuna>=3.0.0              # Bayesian optimization
mapie>=0.8.0               # Conformal prediction
mlflow>=2.8.0              # Experiment tracking
```

---

## 🔬 Advanced Features

### 1. Transfer Learning
- Pre-train on one target, fine-tune on another
- Example: Train on consumption, transfer to price

### 2. Ensemble Methods
- Combine best N-HiTS + TFT + PatchTST
- Weighted by CV performance

### 3. Online Learning
- Incremental updates with new data
- Sliding window retraining

### 4. Explainability
- TFT attention weights
- SHAP values for deep models
- Horizon-wise feature importance

---

## 📈 Next Steps

1. **Run full pipeline** on both targets
2. **Analyze best models** per horizon
3. **Deploy best model** to production API
4. **Set up monitoring** for drift detection
5. **Iterate**: Add new features, try new models

---

**Status**: ✅ Core components implemented, ready for model trainers
**Next**: Implement N-HiTS, TFT, PatchTST trainers with Optuna integration
