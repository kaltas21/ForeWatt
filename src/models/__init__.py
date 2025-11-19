"""
ForeWatt Model Training Module
===============================
Baseline models for electricity demand forecasting.

Models:
-------
- Prophet: Facebook's additive time series model with holidays & weather
- CatBoost: Gradient boosting with categorical feature handling
- XGBoost: Industry-standard gradient boosting
- SARIMAX: Seasonal ARIMA with exogenous features

All models:
- Train on 2020-2023, validate on 2024, test on 2025 (Jan-Oct)
- Log to MLflow (MAE, RMSE, MAPE, MASE)
- Include basic hyperparameter tuning
- Save artifacts (model, predictions, plots)

Author: ForeWatt Team
Date: November 2025
Version: 1.0.0
"""

__version__ = "1.0.0"
