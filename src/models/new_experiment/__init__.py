"""
New Experiment V2 - Fundamental Price Forecasting
=================================================
Breaking the 10.87% sMAPE floor with supply-demand fundamental features.

Modules:
- data/: Feature engineering for fundamental indicators
- deeplearning/: PatchTST, N-HiTS, TFT (NeuralForecast)
- baseline/: CatBoost, XGBoost, LightGBM, Prophet

All features are D-1 safe (available at day-ahead prediction time).

Target Hardware: NVIDIA RTX 5090 (32GB VRAM)

Author: ForeWatt Team
Date: November 2025
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / 'data'
REPORTS_DIR = PROJECT_ROOT / 'reports' / 'new_experiment'

__version__ = '2.0.0'
__author__ = 'ForeWatt Team'

# Deep Learning module
from .deeplearning import (
    FundamentalFeaturePreparerV2,
    FundamentalGridSearchRunnerV2,
    get_full_grid,
    PatchTSTTrainer,
    NHiTSTrainer,
    TFTTrainer,
)

# Baseline module
from .baseline import (
    BaselineFeaturePreparer,
    BaselineGridSearchRunner,
    get_baseline_grid,
    CatBoostTrainer,
    XGBoostTrainer,
    LightGBMTrainer,
    ProphetTrainer,
)
