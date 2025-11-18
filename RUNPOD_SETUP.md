# ForeWatt RunPod Training Guide

Complete guide for training ForeWatt models on RunPod with NVIDIA CUDA GPUs.

## Table of Contents
1. [RunPod Setup](#runpod-setup)
2. [Installation](#installation)
3. [Training Models](#training-models)
4. [Monitoring Progress](#monitoring-progress)
5. [Downloading Results](#downloading-results)
6. [Cost Optimization](#cost-optimization)

---

## RunPod Setup

### 1. Choose GPU Instance

**Recommended GPU Options:**
- **RTX 4090** (24GB) - Best price/performance for this project
- **RTX A6000** (48GB) - For ultra-deep models
- **A100** (40GB/80GB) - Maximum performance but expensive

**Minimum Requirements:**
- GPU Memory: 16GB+ (24GB recommended for deep models)
- Storage: 50GB
- CUDA: 12.1 or higher (PyTorch 2.8.0 compatible)

### 2. Select Template

Choose: **PyTorch 2.8** or latest **RunPod PyTorch** template

Alternatively, use: **Ubuntu 22.04 + CUDA 12.1+** and install PyTorch manually

### 3. Storage Configuration

- Volume Size: 50GB minimum
- Persistent Volume: Recommended (to save trained models)

---

## Installation

### Step 1: Connect to RunPod Instance

SSH into your RunPod instance or use the web terminal.

### Step 2: Clone Repository

```bash
# Clone your repository
git clone <your-repository-url> ForeWatt
cd ForeWatt

# Or upload via Jupyter/web interface if available
```

### Step 3: Run Automated Setup

```bash
# Make setup script executable
chmod +x scripts/runpod_setup.sh

# Run setup (installs all dependencies)
./scripts/runpod_setup.sh
```

The setup script will:
- ✓ Install system dependencies
- ✓ Install Python packages
- ✓ Verify CUDA installation
- ✓ Test GPU availability
- ✓ Prepare data directories

### Step 4: Verify Installation

```bash
# Test hardware detection
python src/models/deep_learning/hardware_config.py

# You should see output like:
# ✓ NVIDIA CUDA Available
# GPU 0: NVIDIA GeForce RTX 4090
# Memory: 24.00 GB
```

---

## Training Models

### Quick Start: Train All Models

#### Baseline Models (Traditional ML)
```bash
# Train all baseline models (CatBoost, XGBoost, LightGBM, Prophet)
# Runs all 40+ configurations for consumption and price_real
python src/models/baseline/grid_search_runner.py

# Specific targets only
python src/models/baseline/grid_search_runner.py --targets consumption
python src/models/baseline/grid_search_runner.py --targets price_real

# Specific models only
python src/models/baseline/grid_search_runner.py --models catboost xgboost
```

**Estimated Time:**
- All configurations: 4-6 hours
- Per model: 1-2 hours

#### Deep Learning Models (Neural Networks)
```bash
# Train all deep learning models (N-HiTS, TFT, PatchTST)
# Runs all 30 configurations (10 per model) for both targets
python src/models/deep_learning/grid_search_runner.py

# Specific targets only
python src/models/deep_learning/grid_search_runner.py --targets consumption
python src/models/deep_learning/grid_search_runner.py --targets price_real

# Specific models only
python src/models/deep_learning/grid_search_runner.py --models nhits tft
```

**Estimated Time:**
- All configurations: 12-24 hours (GPU dependent)
- Per model type: 4-8 hours
- Light configs: 5-15 minutes each
- Deep configs: 30-60 minutes each
- Ultra-deep configs: 1-2 hours each

### Advanced: Custom Training

#### Train Specific Configurations

```bash
# Train only recommended configs
python scripts/train_recommended.py

# Train only lightweight configs (for quick testing)
python scripts/train_light_configs.py

# Train only deep configs (maximum performance)
python scripts/train_deep_configs.py
```

#### Bayesian Optimization (Optuna)

```bash
# Run hyperparameter optimization with Optuna
# This will search for best parameters automatically
python src/models/deep_learning/bayesian_optimization.py \
    --model nhits \
    --target consumption \
    --n-trials 100 \
    --timeout 43200  # 12 hours
```

---

## Monitoring Progress

### Real-time Monitoring

#### Option 1: Terminal Monitoring
```bash
# Watch overall progress
watch -n 30 'tail -n 50 reports/baseline/logs/grid_search_run_*.log'
watch -n 30 'tail -n 50 reports/deep_learning/logs/grid_search_run_*.log'

# Watch specific model
tail -f reports/baseline/logs/consumption/catboost/catboost_default_*.log
tail -f reports/deep_learning/logs/consumption/nhits/nhits_balanced_*.log
```

#### Option 2: TensorBoard (Deep Learning)
```bash
# Launch TensorBoard to monitor training
tensorboard --logdir reports/deep_learning/tensorboard --port 6006

# Access via RunPod's port forwarding
# URL: https://<pod-id>-6006.proxy.runpod.net
```

#### Option 3: Progress Script
```bash
# Check training progress summary
python scripts/check_training_progress.py
```

### Log Files

All training logs are saved to:
```
reports/
├── baseline/
│   ├── logs/
│   │   ├── grid_search_run_TIMESTAMP.log      # Overall log
│   │   ├── consumption/
│   │   │   ├── catboost/
│   │   │   │   └── catboost_default_TIMESTAMP.log
│   │   │   ├── xgboost/
│   │   │   └── lightgbm/
│   │   └── price_real/
│   └── grid_search/
│       └── grid_search_results_TIMESTAMP.csv  # Final results
└── deep_learning/
    ├── logs/
    │   ├── grid_search_run_TIMESTAMP.log
    │   ├── consumption/
    │   │   ├── nhits/
    │   │   ├── tft/
    │   │   └── patchtst/
    │   └── price_real/
    └── grid_search/
        └── grid_search_results_TIMESTAMP.csv
```

---

## Downloading Results

### Option 1: Direct Download (Small Files)

```bash
# Download results CSVs
# Use RunPod's file browser or wget from your local machine

# From your local machine:
scp -P <runpod-ssh-port> root@<runpod-ip>:~/ForeWatt/reports/*/grid_search/*.csv ./results/
```

### Option 2: Google Drive Upload

```bash
# Install rclone (if not already installed)
curl https://rclone.org/install.sh | sudo bash

# Configure Google Drive
rclone config

# Upload results
rclone copy reports/ gdrive:ForeWatt_Results/ -P
```

### Option 3: Compressed Archive

```bash
# Create compressed archive of all results
cd /workspace/ForeWatt
tar -czf forewatt_results_$(date +%Y%m%d).tar.gz reports/ models/

# Download via RunPod web interface or SCP
```

### Option 4: Git Push (Code + Small Results)

```bash
# Add and commit results
git add reports/*.csv reports/*.json
git commit -m "Training results from RunPod"
git push origin main
```

---

## Cost Optimization

### Minimize Costs

1. **Use Spot Instances**
   - 50-70% cheaper than on-demand
   - May be interrupted (save checkpoints!)

2. **Right-size GPU**
   - RTX 4090: $0.40-0.60/hr - Best for this project
   - RTX A6000: $0.80-1.20/hr - For ultra-deep models only
   - Don't use A100 unless necessary ($2-4/hr)

3. **Stop Instance When Not Training**
   ```bash
   # After training completes, stop instance immediately
   # Use persistent volume to save data
   ```

4. **Train in Batches**
   ```bash
   # Option 1: Run overnight
   nohup python src/models/baseline/grid_search_runner.py > baseline_training.log 2>&1 &

   # Option 2: Use screen for persistence
   screen -S training
   python src/models/baseline/grid_search_runner.py
   # Detach: Ctrl+A then D
   # Reattach: screen -r training
   ```

5. **Monitor GPU Utilization**
   ```bash
   # Check GPU usage
   nvidia-smi -l 1

   # If GPU usage is low (<50%), reduce model size or batch size
   ```

### Estimated Costs

**Full Training Run (All Models):**
- GPU: RTX 4090 @ $0.50/hr
- Time: ~30 hours total
- **Total Cost: ~$15-20**

**Breakdown:**
- Baseline models: 6 hours × $0.50 = $3
- Deep learning models: 24 hours × $0.50 = $12

**Cost Reduction Tips:**
- Train only recommended configs: ~$5-8
- Train only one target (consumption OR price): ~$7-10
- Use spot instances: Save 50%

---

## Troubleshooting

### CUDA Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solutions:**
1. Reduce batch size in configuration
2. Skip ultra-deep configs on smaller GPUs
3. Use gradient checkpointing (requires code modification)

### Connection Lost

If SSH connection drops:

```bash
# Training continues if using screen/nohup
screen -r training  # Reattach to session

# Check if process still running
ps aux | grep python

# Check logs
tail -f reports/*/logs/*.log
```

### Slow Training

**Check GPU Usage:**
```bash
nvidia-smi

# Look for:
# - GPU Utilization: Should be >80% during training
# - Memory Usage: Should be using significant GPU RAM
# - Processes: Should show python process
```

**If GPU not being used:**
- Check CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
- Verify hardware detection: `python src/models/deep_learning/hardware_config.py`

---

## Quick Reference Commands

```bash
# Setup
./scripts/runpod_setup.sh

# Train everything
python src/models/baseline/grid_search_runner.py
python src/models/deep_learning/grid_search_runner.py

# Monitor
watch -n 10 nvidia-smi
tail -f reports/*/logs/grid_search_run_*.log

# Download results
tar -czf results.tar.gz reports/
# Then download via RunPod web UI

# Clean up
# Stop instance in RunPod dashboard
```

---

## Support

For issues:
1. Check logs in `reports/*/logs/`
2. Verify CUDA: `python src/models/deep_learning/hardware_config.py`
3. Check GPU: `nvidia-smi`
4. Review error messages in grid search logs

---

**Ready to Train!** 🚀

Start with a small test run first:
```bash
python src/models/baseline/grid_search_runner.py --models catboost --targets consumption
```

Then scale up to full training once verified.
