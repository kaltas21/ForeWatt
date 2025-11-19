# ForeWatt Model Storage & Backup Guide

## Overview
This guide explains how models and results are stored in ForeWatt, especially important for RunPod ephemeral instances.

## Storage Locations

### 1. Local Storage (Ephemeral on RunPod)
```
/root/ForeWatt/
├── mlflow/                    # MLflow tracking database
│   ├── mlflow.db             # SQLite database with all experiments
│   └── artifacts/            # Model artifacts logged to MLflow
├── models/                    # Trained models
│   ├── baseline/             # Traditional ML models
│   │   ├── consumption/      # Demand forecasting models
│   │   └── price_real/       # Price forecasting models
│   └── deep_learning/        # Deep learning models
│       ├── consumption/      # Demand forecasting models
│       └── price_real/       # Price forecasting models
└── reports/                   # Results and metrics
    ├── baseline/
    │   └── grid_search/      # CSV files with results
    └── deep_learning/
        ├── logs/             # Training logs
        └── grid_search/      # CSV files with results
```

### 2. Persistent Storage (Survives pod restarts)
```
/workspace/forewatt_backup/
├── mlflow/                    # Backed up MLflow database
├── models/                    # Backed up trained models
├── reports/                   # Backed up results
└── backup_manifest_*.txt     # Backup logs with timestamps
```

## What Gets Saved

### Deep Learning Models
- **Model files**: `.pkl` files containing:
  - Trained neural network (TFT/NHITS/PatchTST)
  - Trainer object with all configuration
  - Hyperparameters
  - Feature selection strategy
  - Validation metrics
- **MLflow logs**: All experiments with:
  - Metrics (sMAPE, MASE per horizon)
  - Parameters
  - Model artifacts
  - Training time
- **Location**: `models/deep_learning/{target}/{model}_{config}_{timestamp}.pkl`

### Baseline Models
- **CSV results**: Grid search results with best configurations
- **MLflow logs**: Experiment tracking
- **Note**: Model objects currently saved in reports, not models/ directory

### MLflow Database
- **File**: `mlflow/mlflow.db` (SQLite)
- **Contains**: All experiment runs, metrics, parameters
- **Artifacts**: Model checkpoints logged via `mlflow.log_artifact()`

## Backup Strategy

### Manual Backup
Run the backup script anytime:
```bash
./backup_to_workspace.sh
```

### Automatic Periodic Backup
Set up a cron-like backup (recommended):
```bash
# Add to your training script or run in background
while true; do
    sleep 3600  # Every hour
    ./backup_to_workspace.sh
done &
```

### Pre-Shutdown Backup
**IMPORTANT**: Before terminating your RunPod instance:
```bash
cd /root/ForeWatt
./backup_to_workspace.sh
```

## Restoring from Backup

If you start a new RunPod instance:
```bash
# Copy backup to working directory
rsync -av /workspace/forewatt_backup/ /root/ForeWatt/

# Start MLflow server with restored database
cd /root/ForeWatt
mlflow server --backend-store-uri sqlite:///mlflow/mlflow.db \
              --default-artifact-root ./mlflow/artifacts \
              --host 0.0.0.0 --port 5050 &
```

## Loading Saved Models

### Deep Learning Models
```python
import joblib
from pathlib import Path

# Load model
model_path = Path("models/deep_learning/consumption/tft_tft_balanced_20251119_120000.pkl")
saved_data = joblib.load(model_path)

model = saved_data['model']
trainer = saved_data['trainer']
config = saved_data['config']
metrics = saved_data['metrics']

# Make predictions
predictions = trainer.predict(X_test)
```

### From MLflow
```python
import mlflow

# Load model from specific run
run_id = "abc123..."
model = mlflow.pytorch.load_model(f"runs:/{run_id}/model")
```

## Storage Space Management

### Check Sizes
```bash
# Check local storage
du -sh /root/ForeWatt/mlflow
du -sh /root/ForeWatt/models
du -sh /root/ForeWatt/reports

# Check backup storage
du -sh /workspace/forewatt_backup
```

### Clean Old Models
```bash
# Keep only best models (manual cleanup)
cd /root/ForeWatt/models/deep_learning/consumption
ls -lt *.pkl | tail -n +11  # List all but top 10
# Review and delete manually
```

## Best Practices

1. **Run backup script after each grid search completes**
2. **Monitor /workspace storage** (limited space)
3. **Keep only best models** - delete underperforming checkpoints
4. **Export MLflow database** periodically to external storage
5. **Download critical models** to your local machine
6. **Use git** to track code changes in /workspace

## External Download

To download to your local machine:
```bash
# From your local terminal (not RunPod)
# Replace PORT with your RunPod SSH port
scp -P PORT root@ssh.runpod.io:/workspace/forewatt_backup/mlflow/mlflow.db ./

# Or use RunPod's file browser to download from /workspace
```

## Troubleshooting

### MLflow shows no runs
- Check if `mlflow.db` exists and has data: `ls -lh mlflow/mlflow.db`
- Restart MLflow server pointing to correct database

### Models not saving
- Check disk space: `df -h`
- Verify models/ directory exists and is writable
- Check grid_search_runner.py has save code

### Backup failed
- Check /workspace is mounted: `ls /workspace`
- Verify rsync is installed: `which rsync`
- Check permissions: `ls -la /workspace`

## Auto-Backup on Training

Add to your training scripts:
```python
import subprocess

def backup_checkpoint():
    subprocess.run(["/root/ForeWatt/backup_to_workspace.sh"])

# Call after each model or every N iterations
backup_checkpoint()
```
