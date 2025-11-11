"""
Feature Engineering Modules for ForeWatt
=========================================
Modular feature engineering pipeline following medallion architecture.

Modules:
--------
- lag_features: Create lag features for target and key exogenous variables
- rolling_features: Create rolling window statistics
- merge_features: Merge all gold layers into master ML-ready dataset

Author: ForeWatt Team
Date: November 2025
"""

__version__ = "1.0.0"
