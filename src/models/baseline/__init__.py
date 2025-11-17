"""
Baseline Models Package for ForeWatt
====================================
Optimized baseline models with intelligent feature selection for:
- Demand forecasting (consumption)
- Price forecasting (price_real - today's Turkish Lira)

Models:
- Statistical: Prophet, SARIMAX
- Gradient Boosting: CatBoost, XGBoost, LightGBM

Author: ForeWatt Team
Date: November 2025
"""

from .feature_selector import FeatureSelector
from .model_trainer import ModelTrainer
from .pipeline_runner import run_baseline_pipeline

__all__ = [
    'FeatureSelector',
    'ModelTrainer',
    'run_baseline_pipeline'
]
