"""
Deep Learning Module for New Experiment V2
==========================================
High-performance grid search for price forecasting.

Components:
- FundamentalFeaturePreparerV2: Prepares fundamental features (D-1 safe)
- GridConfigGeneratorV2: Generates 15 configs per model
- FundamentalGridSearchRunnerV2: Runs grid search with logging
- HorizonWiseEvaluator: Per-horizon metrics evaluation
- Trainers: PatchTST, N-HiTS, TFT

Target Hardware: NVIDIA RTX 5090 (32GB VRAM)
"""

from .feature_preparer_v2 import (
    FundamentalFeaturePreparerV2,
    FUNDAMENTAL_V2_FEATURE_STRATEGIES,
    FORBIDDEN_COLUMNS,
    D1_SAFE_COLUMNS,
    load_master_v2
)
from .grid_config_generator_v2 import (
    GridConfigGeneratorV2,
    get_full_grid,
    PATCHTST_CONFIGS,
    NHITS_CONFIGS,
    TFT_CONFIGS,
    TARGETS,
    TARGET_STRATEGIES,
    PRICE_FEATURE_STRATEGIES,
    CONSUMPTION_FEATURE_STRATEGIES
)
from .runner_v2 import FundamentalGridSearchRunnerV2
from .evaluator import HorizonWiseEvaluator, compare_models_horizon_wise
from .hardware_config import HardwareConfig, get_hardware_config

# Trainers
from .models import (
    PatchTSTTrainer,
    NHiTSTrainer,
    TFTTrainer
)

__all__ = [
    # Feature Preparation
    'FundamentalFeaturePreparerV2',
    'FUNDAMENTAL_V2_FEATURE_STRATEGIES',
    'FORBIDDEN_COLUMNS',
    'D1_SAFE_COLUMNS',
    'load_master_v2',
    # Grid Configuration
    'GridConfigGeneratorV2',
    'get_full_grid',
    'PATCHTST_CONFIGS',
    'NHITS_CONFIGS',
    'TFT_CONFIGS',
    'TARGETS',
    'TARGET_STRATEGIES',
    'PRICE_FEATURE_STRATEGIES',
    'CONSUMPTION_FEATURE_STRATEGIES',
    # Runner
    'FundamentalGridSearchRunnerV2',
    # Evaluation
    'HorizonWiseEvaluator',
    'compare_models_horizon_wise',
    # Hardware
    'HardwareConfig',
    'get_hardware_config',
    # Trainers
    'PatchTSTTrainer',
    'NHiTSTrainer',
    'TFTTrainer',
]
