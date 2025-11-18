"""
Deep Learning Models Package for ForeWatt
==========================================
State-of-the-art deep learning models for electricity demand and price forecasting.

Models:
- N-HiTS (Neural Hierarchical Interpolation for Time Series)
- TFT (Temporal Fusion Transformer)
- PatchTST (Patched Time Series Transformer)

Features:
- Expanding-window temporal cross-validation
- Bayesian hyperparameter optimization with Optuna
- Horizon-wise multi-step forecasting (1-24h)
- Split conformal prediction for uncertainty quantification
- Fixed versioned feature set with Fourier seasonality

Author: ForeWatt Team
Date: November 2025
"""

from .feature_preparer import DeepLearningFeaturePreparer
from .cv_strategy import ExpandingWindowCV
# from .conformal import SplitConformalPredictor  # Disabled: mapie import issue

__all__ = [
    'DeepLearningFeaturePreparer',
    'ExpandingWindowCV',
    # 'SplitConformalPredictor'  # Disabled: mapie import issue
]
