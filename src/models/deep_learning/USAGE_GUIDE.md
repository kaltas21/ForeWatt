# Deep Learning Pipeline - Complete Usage Guide

## 🚀 Quick Start

### Prerequisites

```bash
# Install dependencies
pip install neuralforecast>=1.6.0
pip install pytorch>=2.0.0
pip install pytorch-lightning>=2.0.0
pip install optuna>=3.0.0
pip install mapie>=0.8.0
pip install mlflow>=2.8.0
```

### Basic Usage

```python
import sys
sys.path.insert(0, '/home/user/ForeWatt')

from src.models.deep_learning import DeepLearningFeaturePreparer, ExpandingWindowCV
from src.models.deep_learning.models import NHiTSTrainer, optimize_nhits
from src.models.deep_learning.evaluator import HorizonWiseEvaluator
from src.models.deep_learning.conformal import SplitConformalPredictor
from src.models.baseline.data_loader import load_master_data, train_val_test_split

# 1. Load data
df = load_master_data()

# 2. Split data
train_df, val_df, test_df = train_val_test_split(df, val_size=0.2, test_size=0.2)

# 3. Prepare features
preparer = DeepLearningFeaturePreparer(target='consumption', max_lag=168)
X_train, y_train, features = preparer.prepare_features(train_df)
X_val, y_val, _ = preparer.prepare_features(val_df)
X_test, y_test, _ = preparer.prepare_features(test_df)

# 4. Optimize hyperparameters with expanding-window CV
cv = ExpandingWindowCV(n_splits=3, min_train_size=int(len(X_train)*0.5))
cv_folds = list(cv.split(X_train))

best_params, best_score = optimize_nhits(
    X_train, y_train, X_val, y_val,
    target='consumption',
    horizon=24,
    n_trials=20,
    cv_folds=cv_folds
)

# 5. Train final model
trainer = NHiTSTrainer(target='consumption', horizon=24)
model, metrics = trainer.train(X_train, y_train, X_val, y_val, best_params)

# 6. Apply conformal prediction
conformal = SplitConformalPredictor(alpha=0.1)
conformal.fit(model, X_train.values, y_train.values, X_val.values, y_val.values)

# 7. Predict with uncertainty
predictions, intervals = conformal.predict(X_test.values)

# 8. Evaluate horizon-wise
evaluator = HorizonWiseEvaluator(horizon=24)
horizon_metrics = evaluator.evaluate_all_horizons(y_test.values, predictions, y_train.values)
aggregated = evaluator.aggregate_metrics(horizon_metrics)

print(f"Mean sMAPE: {aggregated['sMAPE_mean']:.2f}%")
print(f"Short-term sMAPE: {aggregated['sMAPE_short_term']:.2f}%")
```

---

## 📊 Complete Workflow

### Step 1: Data Preparation

```python
from src.models.deep_learning import DeepLearningFeaturePreparer
from src.models.baseline.data_loader import load_master_data

# Load master dataset
df = load_master_data()
print(f"Loaded {len(df)} samples")

# Initialize feature preparer
preparer = DeepLearningFeaturePreparer(
    target='consumption',  # or 'price_real'
    max_lag=168,  # 1 week
    rolling_windows=[24, 168],  # 24h and 168h windows
    fourier_orders={'daily': 5, 'weekly': 3, 'yearly': 4}
)

# Prepare features (adds Fourier seasonality)
X, y, feature_names = preparer.prepare_features(df, add_fourier=True)

print(f"Features: {len(feature_names)}")
preparer.print_feature_summary()
```

### Step 2: Train/Val/Test Split

```python
from src.models.baseline.data_loader import train_val_test_split, prepare_target_data

# Temporal split (60% / 20% / 20%)
train_df, val_df, test_df = train_val_test_split(df, val_size=0.2, test_size=0.2)

# Remove missing targets
train_df, val_df, test_df = prepare_target_data(
    train_df, val_df, test_df, target='consumption'
)

# Prepare features for each split
X_train, y_train, _ = preparer.prepare_features(train_df)
X_val, y_val, _ = preparer.prepare_features(val_df)
X_test, y_test, _ = preparer.prepare_features(test_df)
```

### Step 3: Expanding-Window Cross-Validation Setup

```python
from src.models.deep_learning import ExpandingWindowCV

# Create CV strategy
cv = ExpandingWindowCV(
    n_splits=5,  # 5 folds
    min_train_size=int(len(X_train) * 0.5),  # Start with 50% of train data
    val_size=int(len(X_train) * 0.1),  # 10% for validation
    gap=0  # No gap between train and validation
)

# Generate folds
cv_folds = list(cv.split(X_train))
print(f"Created {len(cv_folds)} CV folds")
```

### Step 4: Bayesian Hyperparameter Optimization

#### Option A: N-HiTS

```python
from src.models.deep_learning.models import optimize_nhits

best_params, best_score = optimize_nhits(
    X_train, y_train,
    X_val, y_val,
    target='consumption',
    horizon=24,
    input_size=168,
    n_trials=50,  # More trials = better optimization
    timeout=3600,  # 1 hour timeout
    cv_folds=cv_folds  # Use CV for robust optimization
)

print(f"Best sMAPE: {best_score:.2f}%")
print(f"Best parameters: {best_params}")
```

#### Option B: TFT

```python
from src.models.deep_learning.models import optimize_tft

best_params_tft, best_score_tft = optimize_tft(
    X_train, y_train,
    X_val, y_val,
    target='consumption',
    horizon=24,
    input_size=168,
    n_trials=50,
    cv_folds=cv_folds
)
```

#### Option C: PatchTST

```python
from src.models.deep_learning.models import optimize_patchtst

best_params_patch, best_score_patch = optimize_patchtst(
    X_train, y_train,
    X_val, y_val,
    target='consumption',
    horizon=24,
    input_size=168,
    n_trials=50,
    cv_folds=cv_folds
)
```

### Step 5: Train Final Model

```python
from src.models.deep_learning.models import NHiTSTrainer

# Initialize trainer with best hyperparameters
trainer = NHiTSTrainer(
    target='consumption',
    horizon=24,
    input_size=168
)

# Train on full train+val data
X_trainval = pd.concat([X_train, X_val])
y_trainval = pd.concat([y_train, y_val])

final_model, final_metrics = trainer.train(
    X_trainval, y_trainval,
    X_test, y_test,  # Use test for early stopping
    hyperparams=best_params
)

print(f"Final model - sMAPE: {final_metrics['sMAPE']:.2f}%")
```

### Step 6: Conformal Prediction (Uncertainty Quantification)

```python
from src.models.deep_learning.conformal import SplitConformalPredictor

# Initialize conformal predictor
conformal = SplitConformalPredictor(
    alpha=0.1,  # 90% coverage
    method='naive'  # or 'plus' for more conservative
)

# Calibrate on validation set
conformal.fit(
    model=trainer,
    X_train=X_train.values,
    y_train=y_train.values,
    X_calib=X_val.values,
    y_calib=y_val.values
)

# Predict with intervals on test set
y_pred, y_intervals = conformal.predict(X_test.values)

# Extract bounds
lower_bounds = y_intervals[:, 0, 0]
upper_bounds = y_intervals[:, 1, 0]

print(f"Prediction interval width: {np.mean(upper_bounds - lower_bounds):.2f}")
```

### Step 7: Horizon-Wise Evaluation

```python
from src.models.deep_learning.evaluator import HorizonWiseEvaluator
from pathlib import Path

# Initialize evaluator
evaluator = HorizonWiseEvaluator(horizon=24)

# Evaluate all horizons
horizon_metrics = evaluator.evaluate_all_horizons(
    y_true=y_test.values,
    y_pred=y_pred,
    y_train=y_train.values
)

# Aggregate metrics
aggregated = evaluator.aggregate_metrics(horizon_metrics)

print(f"\nResults:")
print(f"  Mean sMAPE: {aggregated['sMAPE_mean']:.2f}% ± {aggregated['sMAPE_std']:.2f}%")
print(f"  Short-term (h=1-6): {aggregated['sMAPE_short_term']:.2f}%")
print(f"  Day-ahead (h=24): {aggregated['sMAPE_day_ahead']:.2f}%")

# Save report
report_dir = Path('/home/user/ForeWatt/reports/deep_learning')
report = evaluator.create_evaluation_report(
    horizon_metrics,
    aggregated,
    model_name='NHITS',
    target='consumption',
    save_dir=report_dir
)

# Plot metrics
evaluator.plot_horizon_metrics(
    horizon_metrics,
    save_path=report_dir / 'plots' / 'horizon_metrics_nhits.png'
)
```

### Step 8: Evaluate Conformal Coverage

```python
# Evaluate interval quality
coverage_metrics = conformal.evaluate_coverage(
    y_true=y_test.values,
    y_pred=y_pred,
    y_pis=y_intervals
)

conformal.print_metrics(coverage_metrics)

print(f"Coverage: {coverage_metrics['coverage']*100:.1f}%")
print(f"Target: {coverage_metrics['target_coverage']*100:.1f}%")
print(f"Winkler Score: {coverage_metrics['winkler_score']:.2f}")
```

---

## 🎯 Model Comparison

### Train All Three Models

```python
from src.models.deep_learning.models import NHiTSTrainer, TFTTrainer, PatchTSTTrainer
from src.models.deep_learning.evaluator import compare_models_horizon_wise

models_to_train = {
    'N-HiTS': (NHiTSTrainer, best_params_nhits),
    'TFT': (TFTTrainer, best_params_tft),
    'PatchTST': (PatchTSTTrainer, best_params_patch)
}

results = {}

for model_name, (TrainerClass, params) in models_to_train.items():
    print(f"\n{'='*80}")
    print(f"Training {model_name}")
    print(f"{'='*80}\n")

    trainer = TrainerClass(target='consumption', horizon=24, input_size=168)
    model, metrics = trainer.train(X_train, y_train, X_val, y_val, params)

    # Predict
    predictions = trainer.predict(X_test)

    # Evaluate
    evaluator = HorizonWiseEvaluator(horizon=24)
    horizon_metrics = evaluator.evaluate_all_horizons(y_test.values, predictions, y_train.values)

    results[model_name] = {
        'model': model,
        'metrics': metrics,
        'horizon_metrics': horizon_metrics,
        'predictions': predictions
    }

# Compare all models
model_horizon_metrics = {name: res['horizon_metrics'] for name, res in results.items()}
compare_models_horizon_wise(
    model_horizon_metrics,
    save_path=Path('/home/user/ForeWatt/reports/deep_learning/plots/model_comparison.png')
)

# Print summary
print("\n" + "="*80)
print("MODEL COMPARISON SUMMARY")
print("="*80)
for model_name, res in results.items():
    mean_smape = res['horizon_metrics']['sMAPE'].mean()
    mean_mase = res['horizon_metrics']['MASE'].mean()
    print(f"{model_name:15s}: sMAPE={mean_smape:6.2f}%  MASE={mean_mase:.4f}")
print("="*80)
```

---

## 📈 Advanced Features

### Horizon-Specific Conformal Calibration

```python
from src.models.deep_learning.conformal import horizon_specific_conformal

# Train model to get multi-horizon predictions
# ... (train model as above)

# Calibrate separately for each horizon
conformal_by_horizon = horizon_specific_conformal(
    model=trainer,
    X_train=X_train.values,
    y_train=y_train_multihorizon,  # Shape: (n_samples, 24)
    X_calib=X_val.values,
    y_calib=y_val_multihorizon,  # Shape: (n_samples, 24)
    horizons=24,
    alpha=0.1
)

# Predict with horizon-specific intervals
for h in range(24):
    cp = conformal_by_horizon[h]
    y_pred_h, y_intervals_h = cp.predict(X_test.values)
    print(f"Horizon {h+1}: Interval width = {np.mean(y_intervals_h[:, 1, 0] - y_intervals_h[:, 0, 0]):.2f}")
```

### Custom Search Spaces

```python
import optuna

def custom_optimization(X_train, y_train, X_val, y_val):
    """Custom Optuna optimization with specific constraints."""

    trainer = NHiTSTrainer(target='consumption', horizon=24)

    def objective(trial):
        # Custom hyperparameter suggestions
        params = {
            'stack_types': ['identity', 'trend', 'seasonality'],
            'n_blocks': trial.suggest_int('n_blocks', 1, 4),  # More blocks
            'hidden_size': trial.suggest_categorical('hidden_size', [256, 512, 1024, 2048]),  # Larger networks
            'learning_rate': trial.suggest_loguniform('learning_rate', 5e-5, 5e-3),  # Narrower range
            'batch_size': 64,  # Fixed
            'max_steps': 2000,  # Fixed
            'n_pool_kernel_size': [8, 4, 1],  # Fixed
            'n_freq_downsample': [8, 4, 1],  # Fixed
            'n_mlp_layers': 2,  # Fixed
            'early_stop_patience_steps': 100
        }

        _, metrics = trainer.train(X_train, y_train, X_val, y_val, params)
        return metrics['sMAPE']

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30)

    return study.best_params, study.best_value
```

---

## 💾 Saving and Loading Models

### Save Model

```python
import pickle
from pathlib import Path

# Save trained model
model_path = Path('/home/user/ForeWatt/models/nhits_consumption_v1.pkl')
model_path.parent.mkdir(parents=True, exist_ok=True)

with open(model_path, 'wb') as f:
    pickle.dump({
        'model': trainer.model,
        'hyperparameters': best_params,
        'metrics': final_metrics,
        'feature_names': feature_names,
        'target': 'consumption',
        'horizon': 24
    }, f)

print(f"Model saved to: {model_path}")
```

### Load Model

```python
import pickle

# Load model
with open(model_path, 'rb') as f:
    saved_data = pickle.load(f)

loaded_model = saved_data['model']
loaded_params = saved_data['hyperparameters']
loaded_features = saved_data['feature_names']

# Use for prediction
trainer_loaded = NHiTSTrainer(target='consumption', horizon=24)
trainer_loaded.model = loaded_model

predictions = trainer_loaded.predict(X_new)
```

---

## 🔧 Troubleshooting

### GPU Not Detected

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")

# Force CPU if GPU issues
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
```

### Out of Memory

```python
# Reduce batch size
hyperparams['batch_size'] = 32  # Instead of 64 or 128

# Reduce model size
hyperparams['hidden_size'] = 128  # Instead of 512

# Reduce max_steps
hyperparams['max_steps'] = 500  # Instead of 2000
```

### Slow Training

```python
# Use fewer optimization trials
n_trials = 10  # Instead of 50

# Use fewer CV folds
cv = ExpandingWindowCV(n_splits=3)  # Instead of 5

# Use smaller input_size
input_size = 96  # Instead of 168

# Enable GPU
# Make sure CUDA is installed and available
```

---

## 📚 References

- **N-HiTS Paper**: https://arxiv.org/abs/2201.12886
- **TFT Paper**: https://arxiv.org/abs/1912.09363
- **PatchTST Paper**: https://arxiv.org/abs/2211.14730
- **NeuralForecast**: https://github.com/Nixtla/neuralforecast
- **Optuna**: https://optuna.readthedocs.io/
- **MAPIE**: https://mapie.readthedocs.io/

---

**For more examples, see**: `src/models/deep_learning/examples.py`
