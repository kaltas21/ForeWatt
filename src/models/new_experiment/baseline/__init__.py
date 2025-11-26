"""
Baseline Module for New Experiment V2
=====================================
CatBoost, XGBoost, LightGBM, and Prophet grid search.

Components:
- BaselineFeaturePreparer: D-1 safe feature preparation
- BaselineGridConfigGenerator: 15 configs per model
- BaselineGridSearchRunner: Grid search runner with logging
- Trainers: CatBoost, XGBoost, LightGBM, Prophet

Author: ForeWatt Team - New Experiment V2
Date: November 2025
"""

from .feature_preparer_baseline import (
    BaselineFeaturePreparer,
    BASELINE_FEATURE_STRATEGIES,
    PRICE_FEATURE_TIERS,
    CONSUMPTION_FEATURE_TIERS,
    get_feature_strategy_for_tier
)

from .grid_config_baseline import (
    BaselineGridConfigGenerator,
    get_baseline_grid,
    CATBOOST_CONFIGS,
    XGBOOST_CONFIGS,
    LIGHTGBM_CONFIGS,
    PROPHET_CONFIGS,
    TARGETS,
    TARGET_STRATEGIES
)

from .runner_baseline import BaselineGridSearchRunner

from .models import (
    CatBoostTrainer,
    XGBoostTrainer,
    LightGBMTrainer,
    ProphetTrainer
)

__all__ = [
    # Feature Preparation
    'BaselineFeaturePreparer',
    'BASELINE_FEATURE_STRATEGIES',
    'PRICE_FEATURE_TIERS',
    'CONSUMPTION_FEATURE_TIERS',
    'get_feature_strategy_for_tier',
    # Grid Configuration
    'BaselineGridConfigGenerator',
    'get_baseline_grid',
    'CATBOOST_CONFIGS',
    'XGBOOST_CONFIGS',
    'LIGHTGBM_CONFIGS',
    'PROPHET_CONFIGS',
    'TARGETS',
    'TARGET_STRATEGIES',
    # Runner
    'BaselineGridSearchRunner',
    # Trainers
    'CatBoostTrainer',
    'XGBoostTrainer',
    'LightGBMTrainer',
    'ProphetTrainer',
]
