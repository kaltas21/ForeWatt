# RunPod Quick Start Guide

One-page reference for training ForeWatt models on RunPod.

## 🚀 Initial Setup (5 minutes)

```bash
# 1. Clone repository
git clone <your-repo-url> ForeWatt
cd ForeWatt

# 2. Run automated setup
chmod +x scripts/runpod_setup.sh
./scripts/runpod_setup.sh

# 3. Verify installation
python src/models/deep_learning/hardware_config.py
```

## 🧪 Quick Test (2 minutes)

```bash
# Run quick test to verify everything works
chmod +x scripts/quick_test.sh
./scripts/quick_test.sh
```

## 🏃 Start Training

### Baseline Models (4-6 hours)
```bash
# All models, all targets
python src/models/baseline/grid_search_runner.py

# Specific model or target
python src/models/baseline/grid_search_runner.py --models catboost --targets consumption
```

### Deep Learning Models (12-24 hours)
```bash
# All models, all targets
python src/models/deep_learning/grid_search_runner.py

# Specific model or target
python src/models/deep_learning/grid_search_runner.py --models nhits --targets consumption
```

### Background Training (Recommended)
```bash
# Run in background with nohup
nohup python src/models/baseline/grid_search_runner.py > baseline.log 2>&1 &
nohup python src/models/deep_learning/grid_search_runner.py > deep_learning.log 2>&1 &

# Or use screen (can reattach if disconnected)
screen -S training
python src/models/deep_learning/grid_search_runner.py
# Press Ctrl+A then D to detach
# Reattach with: screen -r training
```

## 📊 Monitor Training

### GPU Usage
```bash
# Real-time GPU monitor
chmod +x scripts/monitor_gpu.sh
./scripts/monitor_gpu.sh

# Or use nvidia-smi directly
watch -n 5 nvidia-smi
```

### Training Progress
```bash
# Check progress summary
python scripts/check_training_progress.py

# Tail logs in real-time
tail -f reports/baseline/logs/grid_search_run_*.log
tail -f reports/deep_learning/logs/grid_search_run_*.log

# Watch specific model
tail -f reports/deep_learning/logs/consumption/nhits/*.log
```

## 📥 Get Results

### Check Results
```bash
# List result files
ls -lh reports/baseline/grid_search/
ls -lh reports/deep_learning/grid_search/

# View top results
head -n 20 reports/*/grid_search/grid_search_results_*.csv
```

### Download Results
```bash
# Create archive
tar -czf forewatt_results_$(date +%Y%m%d).tar.gz reports/

# Download via RunPod web interface
# Or use SCP from local machine:
# scp -P <port> root@<ip>:~/ForeWatt/forewatt_results_*.tar.gz ./
```

## 🛑 Stop & Clean Up

```bash
# Check running processes
ps aux | grep python

# Kill training if needed
pkill -f grid_search_runner.py

# Stop RunPod instance via dashboard
# (Don't forget to stop to avoid charges!)
```

## 🔧 Troubleshooting

### Out of Memory
```bash
# Check GPU memory
nvidia-smi

# Error will be logged, training will continue with other configs
# Deep/ultra-deep configs may fail on GPUs < 24GB
```

### Lost Connection
```bash
# If using screen, reattach
screen -r training

# Check if still running
ps aux | grep grid_search_runner

# Check logs
tail -f reports/*/logs/*.log
```

### Slow Training
```bash
# Check GPU utilization (should be >80%)
nvidia-smi

# Ensure CUDA is being used
python -c "import torch; print(torch.cuda.is_available())"
```

## 💰 Cost Optimization

**Recommended GPU: RTX 4090** ($0.40-0.60/hr)
- Full training run: ~30 hours × $0.50 = **$15**
- Baseline only: ~6 hours × $0.50 = **$3**
- Deep learning only: ~24 hours × $0.50 = **$12**

**Use Spot Instances:** Save 50% (but may be interrupted)

**Stop Immediately After Training:**
- Training auto-stops when complete
- Download results and stop instance
- Use persistent volume to save data

## 📁 File Structure

```
ForeWatt/
├── scripts/
│   ├── runpod_setup.sh           # Setup script
│   ├── quick_test.sh             # Quick test
│   ├── monitor_gpu.sh            # GPU monitor
│   └── check_training_progress.py # Progress checker
├── reports/
│   ├── baseline/
│   │   ├── logs/                 # Training logs
│   │   └── grid_search/          # Results CSVs
│   └── deep_learning/
│       ├── logs/                 # Training logs
│       └── grid_search/          # Results CSVs
└── models/
    ├── baseline/                 # Trained baseline models
    └── deep_learning/            # Trained DL models
```

## 🆘 Quick Commands

```bash
# GPU status
nvidia-smi

# Training status
ps aux | grep python

# Check progress
python scripts/check_training_progress.py

# Stop training
pkill -f grid_search_runner

# Tail logs
tail -f reports/*/logs/grid_search_run_*.log

# Create results archive
tar -czf results.tar.gz reports/
```

## 📚 Full Documentation

See `RUNPOD_SETUP.md` for detailed instructions.

---

**Ready? Start here:** `./scripts/runpod_setup.sh`
