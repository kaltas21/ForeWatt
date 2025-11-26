"""
Data Engineering Module for New Experiment V2
=============================================
Generates fundamental features for price forecasting.

Features:
- reserve_margin_ratio
- renewable_saturation
- thermal_gap
- system_short_signal
- import_cost_proxy
- spark_spread_proxy
"""

from .feature_engineering_v2 import FundamentalFeatureEngineerV2

__all__ = ['FundamentalFeatureEngineerV2']
