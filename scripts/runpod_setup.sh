#!/bin/bash
################################################################################
# ForeWatt RunPod Setup Script
# Automated installation for Linux + CUDA environment
################################################################################

set -e  # Exit on error

echo "████████████████████████████████████████████████████████████████████████████████"
echo "ForeWatt RunPod Setup"
echo "████████████████████████████████████████████████████████████████████████████████"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Project root: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

################################################################################
# Step 1: System Information
################################################################################
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1: System Information"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "OS: $(lsb_release -d | cut -f2)"
echo "Kernel: $(uname -r)"
echo "CPU: $(lscpu | grep 'Model name' | cut -f 2 -d ':' | awk '{$1=$1}1')"
echo "RAM: $(free -h | awk '/^Mem:/ {print $2}')"

if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ NVIDIA driver detected${NC}"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo -e "${RED}✗ NVIDIA driver not found${NC}"
    echo "This setup requires NVIDIA GPU. Please use a GPU instance."
    exit 1
fi

################################################################################
# Step 2: Update System
################################################################################
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2: Update System Packages"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "Updating package lists..."
apt-get update -qq

echo "Installing system dependencies..."
apt-get install -y -qq \
    build-essential \
    git \
    wget \
    curl \
    vim \
    htop \
    tmux \
    screen \
    zip unzip \
    software-properties-common \
    ca-certificates \
    libssl-dev \
    libffi-dev \
    python3-dev

echo -e "${GREEN}✓ System packages updated${NC}"

################################################################################
# Step 3: Python Environment
################################################################################
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 3: Python Environment"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Upgrade pip
echo "Upgrading pip..."
python3 -m pip install --upgrade pip -q

# Install pip tools
echo "Installing pip tools..."
python3 -m pip install --upgrade setuptools wheel -q

echo -e "${GREEN}✓ Python environment ready${NC}"

################################################################################
# Step 4: PyTorch 2.8.0 with CUDA
################################################################################
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 4: PyTorch 2.8.0 with CUDA Support"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check if PyTorch is already installed
if python3 -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
    CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())")

    echo "PyTorch version: $TORCH_VERSION"
    echo "CUDA available: $CUDA_AVAILABLE"

    if [ "$CUDA_AVAILABLE" == "True" ] && [[ "$TORCH_VERSION" == "2.8"* ]]; then
        echo -e "${GREEN}✓ PyTorch 2.8.0 with CUDA already installed${NC}"
    else
        echo -e "${YELLOW}⚠ Installing PyTorch 2.8.0 with CUDA 12.1...${NC}"
        pip3 uninstall -y torch torchvision torchaudio
        pip3 install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    fi
else
    echo "Installing PyTorch 2.8.0 with CUDA 12.1..."
    pip3 install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    echo -e "${GREEN}✓ PyTorch 2.8.0 with CUDA installed${NC}"
fi

################################################################################
# Step 5: Project Dependencies
################################################################################
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 5: Project Dependencies"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "Installing core scientific packages..."
pip3 install -q \
    numpy==1.24.3 \
    pandas==2.0.3 \
    scipy==1.11.1 \
    scikit-learn==1.3.0

echo "Installing visualization packages..."
pip3 install -q \
    matplotlib==3.7.2 \
    seaborn==0.12.2 \
    plotly==5.15.0

echo "Installing ML packages..."
pip3 install -q \
    catboost==1.2 \
    xgboost==1.7.6 \
    lightgbm==4.0.0 \
    prophet==1.1.4 \
    statsmodels==0.14.0

echo "Installing deep learning frameworks..."
pip3 install -q \
    neuralforecast==1.6.4 \
    pytorch-lightning==2.0.6 \
    optuna==3.3.0

echo "Installing utility packages..."
pip3 install -q \
    tqdm==4.65.0 \
    psutil==5.9.5 \
    python-dotenv==1.0.0 \
    pyyaml==6.0.1

echo -e "${GREEN}✓ All dependencies installed${NC}"

################################################################################
# Step 6: Create Directories
################################################################################
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 6: Create Project Directories"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Create directory structure
mkdir -p data/raw
mkdir -p data/processed
mkdir -p models/baseline
mkdir -p models/deep_learning
mkdir -p reports/baseline/logs
mkdir -p reports/baseline/grid_search
mkdir -p reports/deep_learning/logs
mkdir -p reports/deep_learning/grid_search
mkdir -p reports/figures

echo -e "${GREEN}✓ Project directories created${NC}"

################################################################################
# Step 7: Verify Installation
################################################################################
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 7: Verify Installation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "Testing hardware detection..."
if python3 src/models/deep_learning/hardware_config.py; then
    echo -e "${GREEN}✓ Hardware detection successful${NC}"
else
    echo -e "${RED}✗ Hardware detection failed${NC}"
    exit 1
fi

################################################################################
# Step 8: GPU Test
################################################################################
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 8: GPU Functionality Test"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python3 << 'PYTHON_TEST'
import torch
import numpy as np

print("Testing PyTorch CUDA...")
print(f"  PyTorch version: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  GPU count: {torch.cuda.device_count()}")
    print(f"  GPU name: {torch.cuda.get_device_name(0)}")

    # Test tensor operations
    print("\nTesting GPU tensor operations...")
    x = torch.randn(1000, 1000, device='cuda')
    y = torch.matmul(x, x.T)
    print("  ✓ Matrix multiplication on GPU successful")

    # Memory test
    torch.cuda.empty_cache()
    print("  ✓ GPU memory management working")
else:
    print("ERROR: CUDA not available!")
    exit(1)

print("\n✓ All GPU tests passed!")
PYTHON_TEST

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ GPU tests passed${NC}"
else
    echo -e "${RED}✗ GPU tests failed${NC}"
    exit 1
fi

################################################################################
# Complete
################################################################################
echo ""
echo "████████████████████████████████████████████████████████████████████████████████"
echo -e "${GREEN}✓ RunPod Setup Complete!${NC}"
echo "████████████████████████████████████████████████████████████████████████████████"
echo ""
echo "Next steps:"
echo "  1. Ensure data files are in data/ directory"
echo "  2. Run baseline models:"
echo "     python src/models/baseline/grid_search_runner.py"
echo ""
echo "  3. Run deep learning models:"
echo "     python src/models/deep_learning/grid_search_runner.py"
echo ""
echo "  4. Monitor training:"
echo "     tail -f reports/*/logs/grid_search_run_*.log"
echo ""
echo "  5. Check results:"
echo "     ls -lh reports/*/grid_search/"
echo ""
echo "See RUNPOD_SETUP.md for detailed instructions."
echo "████████████████████████████████████████████████████████████████████████████████"
