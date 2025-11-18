"""
Baseline Model Hyperparameter Configurations
============================================
Multiple parameter combinations for each baseline model to test different
architectures and feature subsets.

Targets:
- consumption: Electricity demand (MWh)
- price_real: Day-ahead market price (PTF, inflation-adjusted TL/MWh)

GPU Support:
- CatBoost: NVIDIA CUDA only (task_type='GPU')
- XGBoost: NVIDIA CUDA only (tree_method='gpu_hist')
- LightGBM: NVIDIA CUDA only (device='gpu')
- Auto-fallback to CPU if GPU not available

Author: ForeWatt Team
Date: November 2025
"""

from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


def check_gpu_available():
    """
    Check if GPU is available for training.

    Returns:
        tuple: (has_gpu: bool, gpu_type: str)
            gpu_type: 'cuda', 'mps', or 'cpu'
    """
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"✓ NVIDIA CUDA GPU detected: {torch.cuda.get_device_name(0)}")
            return True, 'cuda'
        elif torch.backends.mps.is_available():
            logger.info("✓ Apple Silicon GPU (MPS) detected")
            logger.warning("⚠ Note: Baseline models (CatBoost/XGBoost/LightGBM) don't support Apple Silicon GPU yet")
            logger.info("  These libraries only support NVIDIA CUDA GPUs")
            logger.info("  Falling back to CPU (still very fast on M-series chips)")
            return False, 'mps_unavailable'
    except ImportError:
        pass

    logger.info("⚠ No GPU detected, using CPU")
    return False, 'cpu'


class BaselineHyperparameterConfigs:
    """
    Pre-defined hyperparameter configurations for baseline models.

    Each model has 5 configurations:
    1. Light: Fast training, fewer features
    2. Balanced: Good trade-off (recommended)
    3. Deep: More complex, more features
    4. Price-Optimized: Specialized for price forecasting
    5. Demand-Optimized: Specialized for demand forecasting
    """

    @staticmethod
    def get_catboost_configs() -> Dict[str, Dict[str, Any]]:
        """
        CatBoost gradient boosting configurations.

        Returns:
            Dictionary with 5 configurations
        """
        return {
            'catboost_light': {
                'iterations': 500,
                'learning_rate': 0.1,
                'depth': 4,
                'l2_leaf_reg': 5.0,
                'bagging_temperature': 0.5,
                'random_strength': 0.5,
                'early_stopping_rounds': 30,
                'task_type': 'GPU',  # Use GPU if available (CUDA only)
                'feature_selection': 'top_50',  # Use only top 50 most important features
                'description': 'Fast training with shallow trees'
            },

            'catboost_balanced': {
                'iterations': 1000,
                'learning_rate': 0.1,
                'depth': 6,
                'l2_leaf_reg': 3.0,
                'bagging_temperature': 1.0,
                'random_strength': 1.0,
                'early_stopping_rounds': 50,
                'task_type': 'GPU',  # Use GPU if available (CUDA only)
                'feature_selection': 'all',
                'description': 'Balanced performance and speed (recommended)'
            },

            'catboost_deep': {
                'iterations': 2000,
                'learning_rate': 0.05,
                'depth': 8,
                'l2_leaf_reg': 1.0,
                'bagging_temperature': 1.0,
                'random_strength': 1.0,
                'early_stopping_rounds': 100,
                'task_type': 'GPU',  # Use GPU if available (CUDA only)
                'feature_selection': 'all',
                'description': 'Deep trees for complex patterns'
            },

            'catboost_price_optimized': {
                'iterations': 1500,
                'learning_rate': 0.08,
                'depth': 7,
                'l2_leaf_reg': 2.0,
                'bagging_temperature': 0.8,
                'random_strength': 1.2,
                'early_stopping_rounds': 75,
                'task_type': 'GPU',  # Use GPU if available (CUDA only)
                'feature_selection': 'price_focused',  # Prioritize price-related features
                'description': 'Optimized for price volatility'
            },

            'catboost_demand_optimized': {
                'iterations': 1200,
                'learning_rate': 0.1,
                'depth': 6,
                'l2_leaf_reg': 3.5,
                'bagging_temperature': 0.7,
                'random_strength': 0.8,
                'early_stopping_rounds': 60,
                'task_type': 'GPU',  # Use GPU if available (CUDA only)
                'feature_selection': 'demand_focused',  # Prioritize weather + temporal
                'description': 'Optimized for demand patterns'
            }
        }

    @staticmethod
    def get_xgboost_configs() -> Dict[str, Dict[str, Any]]:
        """
        XGBoost gradient boosting configurations.

        Returns:
            Dictionary with 5 configurations
        """
        return {
            'xgboost_light': {
                'n_estimators': 500,
                'learning_rate': 0.1,
                'max_depth': 4,
                'min_child_weight': 3,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.5,
                'tree_method': 'gpu_hist',  # Use GPU if available (CUDA only)
                'early_stopping_rounds': 30,
                'feature_selection': 'top_50',
                'description': 'Fast, regularized model'
            },

            'xgboost_balanced': {
                'n_estimators': 1000,
                'learning_rate': 0.1,
                'max_depth': 6,
                'min_child_weight': 1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.0,
                'reg_lambda': 1.0,
                'tree_method': 'gpu_hist',  # Use GPU if available (CUDA only)
                'early_stopping_rounds': 50,
                'feature_selection': 'all',
                'description': 'Standard XGBoost (recommended)'
            },

            'xgboost_deep': {
                'n_estimators': 2000,
                'learning_rate': 0.05,
                'max_depth': 8,
                'min_child_weight': 1,
                'subsample': 0.9,
                'colsample_bytree': 0.9,
                'reg_alpha': 0.0,
                'reg_lambda': 0.5,
                'tree_method': 'gpu_hist',  # Use GPU if available (CUDA only)
                'early_stopping_rounds': 100,
                'feature_selection': 'all',
                'description': 'Deep trees with more estimators'
            },

            'xgboost_price_optimized': {
                'n_estimators': 1500,
                'learning_rate': 0.08,
                'max_depth': 7,
                'min_child_weight': 2,
                'subsample': 0.85,
                'colsample_bytree': 0.85,
                'reg_alpha': 0.05,
                'reg_lambda': 1.2,
                'tree_method': 'gpu_hist',  # Use GPU if available (CUDA only)
                'early_stopping_rounds': 75,
                'feature_selection': 'price_focused',
                'description': 'Handles price spikes better'
            },

            'xgboost_demand_optimized': {
                'n_estimators': 1200,
                'learning_rate': 0.1,
                'max_depth': 6,
                'min_child_weight': 1,
                'subsample': 0.8,
                'tree_method': 'gpu_hist',  # Use GPU if available (CUDA only)
                'colsample_bytree': 0.75,
                'reg_alpha': 0.0,
                'reg_lambda': 1.5,
                'early_stopping_rounds': 60,
                'feature_selection': 'demand_focused',
                'description': 'Focus on temporal + weather patterns'
            }
        }

    @staticmethod
    def get_lightgbm_configs() -> Dict[str, Dict[str, Any]]:
        """
        LightGBM gradient boosting configurations.

        Returns:
            Dictionary with 5 configurations
        """
        return {
            'lightgbm_light': {
                'n_estimators': 500,
                'learning_rate': 0.1,
                'max_depth': 4,
                'num_leaves': 15,
                'min_child_samples': 30,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.5,
                'device': 'gpu',  # Use GPU if available (CUDA only)
                'early_stopping_rounds': 30,
                'feature_selection': 'top_50',
                'description': 'Fast LightGBM with leaf-wise growth'
            },

            'lightgbm_balanced': {
                'n_estimators': 1000,
                'learning_rate': 0.1,
                'max_depth': 6,
                'num_leaves': 31,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.0,
                'reg_lambda': 1.0,
                'device': 'gpu',  # Use GPU if available (CUDA only)
                'early_stopping_rounds': 50,
                'feature_selection': 'all',
                'description': 'Balanced LightGBM (recommended)'
            },

            'lightgbm_deep': {
                'n_estimators': 2000,
                'learning_rate': 0.05,
                'max_depth': 8,
                'num_leaves': 63,
                'min_child_samples': 10,
                'subsample': 0.9,
                'colsample_bytree': 0.9,
                'reg_alpha': 0.0,
                'reg_lambda': 0.5,
                'device': 'gpu',  # Use GPU if available (CUDA only)
                'early_stopping_rounds': 100,
                'feature_selection': 'all',
                'description': 'Deep leaf-wise trees'
            },

            'lightgbm_price_optimized': {
                'n_estimators': 1500,
                'learning_rate': 0.08,
                'max_depth': 7,
                'num_leaves': 50,
                'min_child_samples': 15,
                'subsample': 0.85,
                'colsample_bytree': 0.85,
                'reg_alpha': 0.05,
                'reg_lambda': 1.2,
                'device': 'gpu',  # Use GPU if available (CUDA only)
                'early_stopping_rounds': 75,
                'feature_selection': 'price_focused',
                'description': 'Optimized for price prediction'
            },

            'lightgbm_demand_optimized': {
                'n_estimators': 1200,
                'learning_rate': 0.1,
                'max_depth': 6,
                'num_leaves': 40,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.75,
                'reg_alpha': 0.0,
                'reg_lambda': 1.5,
                'device': 'gpu',  # Use GPU if available (CUDA only)
                'early_stopping_rounds': 60,
                'feature_selection': 'demand_focused',
                'description': 'Optimized for demand patterns'
            }
        }

    @staticmethod
    def get_prophet_configs() -> Dict[str, Dict[str, Any]]:
        """
        Prophet (Facebook) time series configurations.

        Returns:
            Dictionary with 5 configurations
        """
        return {
            'prophet_light': {
                'growth': 'linear',
                'changepoint_prior_scale': 0.01,
                'seasonality_prior_scale': 5.0,
                'holidays_prior_scale': 5.0,
                'seasonality_mode': 'additive',
                'yearly_seasonality': True,
                'weekly_seasonality': True,
                'daily_seasonality': False,
                'feature_selection': 'prophet_minimal',  # Only core weather features
                'description': 'Conservative, low-variance Prophet'
            },

            'prophet_balanced': {
                'growth': 'linear',
                'changepoint_prior_scale': 0.05,
                'seasonality_prior_scale': 10.0,
                'holidays_prior_scale': 10.0,
                'seasonality_mode': 'multiplicative',
                'yearly_seasonality': True,
                'weekly_seasonality': True,
                'daily_seasonality': True,
                'feature_selection': 'prophet_standard',
                'description': 'Standard Prophet (recommended)'
            },

            'prophet_flexible': {
                'growth': 'linear',
                'changepoint_prior_scale': 0.5,
                'seasonality_prior_scale': 20.0,
                'holidays_prior_scale': 20.0,
                'seasonality_mode': 'multiplicative',
                'yearly_seasonality': True,
                'weekly_seasonality': True,
                'daily_seasonality': True,
                'feature_selection': 'prophet_extended',
                'description': 'Flexible Prophet with strong seasonality'
            },

            'prophet_price_optimized': {
                'growth': 'linear',
                'changepoint_prior_scale': 0.1,
                'seasonality_prior_scale': 15.0,
                'holidays_prior_scale': 15.0,
                'seasonality_mode': 'multiplicative',
                'yearly_seasonality': True,
                'weekly_seasonality': True,
                'daily_seasonality': True,
                'feature_selection': 'prophet_price',  # Include consumption as regressor
                'description': 'Prophet for price with consumption regressor'
            },

            'prophet_demand_optimized': {
                'growth': 'linear',
                'changepoint_prior_scale': 0.05,
                'seasonality_prior_scale': 12.0,
                'holidays_prior_scale': 15.0,
                'seasonality_mode': 'multiplicative',
                'yearly_seasonality': True,
                'weekly_seasonality': True,
                'daily_seasonality': True,
                'feature_selection': 'prophet_weather',  # Extended weather features
                'description': 'Prophet for demand with weather focus'
            }
        }

    @staticmethod
    def get_all_baseline_configs() -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Get all baseline model configurations.

        Returns:
            Dictionary with all configurations for all models
        """
        return {
            'catboost': BaselineHyperparameterConfigs.get_catboost_configs(),
            'xgboost': BaselineHyperparameterConfigs.get_xgboost_configs(),
            'lightgbm': BaselineHyperparameterConfigs.get_lightgbm_configs(),
            'prophet': BaselineHyperparameterConfigs.get_prophet_configs()
        }

    @staticmethod
    def get_recommended_configs(target: str) -> Dict[str, str]:
        """
        Get recommended configuration for each model based on target.

        Args:
            target: 'consumption' or 'price_real'

        Returns:
            Dictionary mapping model_type to recommended config name
        """
        if target == 'consumption':
            return {
                'catboost': 'catboost_demand_optimized',
                'xgboost': 'xgboost_demand_optimized',
                'lightgbm': 'lightgbm_demand_optimized',
                'prophet': 'prophet_demand_optimized'
            }
        elif target == 'price_real':
            return {
                'catboost': 'catboost_price_optimized',
                'xgboost': 'xgboost_price_optimized',
                'lightgbm': 'lightgbm_price_optimized',
                'prophet': 'prophet_price_optimized'
            }
        else:
            # Default to balanced
            return {
                'catboost': 'catboost_balanced',
                'xgboost': 'xgboost_balanced',
                'lightgbm': 'lightgbm_balanced',
                'prophet': 'prophet_balanced'
            }


# Feature selection strategies
FEATURE_SELECTION_STRATEGIES = {
    # General strategies
    'all': 'Use all available features (~100+)',
    'top_50': 'Use top 50 most important features (based on correlation)',
    'top_30': 'Use top 30 most important features',
    'top_10': 'Use top 10 most important features (SARIMAX only)',

    # Target-specific strategies
    'price_focused': 'Prioritize: price lags, consumption, FX rates, hour encoding',
    'demand_focused': 'Prioritize: consumption lags, weather, temperature, calendar',

    # Model-specific strategies
    'prophet_minimal': 'Temperature, humidity, calendar flags only',
    'prophet_standard': 'Weather core + derived (HDD/CDD) + calendar',
    'prophet_extended': 'Weather + temperature dynamics + calendar',
    'prophet_price': 'Prophet weather + consumption as regressor',
    'prophet_weather': 'Extended weather + temperature lags'
}


def print_config_summary():
    """Print summary of all available configurations."""
    configs = BaselineHyperparameterConfigs.get_all_baseline_configs()

    print("=" * 100)
    print("BASELINE MODEL HYPERPARAMETER CONFIGURATIONS")
    print("=" * 100)
    print(f"\nTotal configurations: {sum(len(v) for v in configs.values())}")
    print(f"Models: {len(configs)}")

    for model_type, model_configs in configs.items():
        print(f"\n{'─' * 100}")
        print(f"{model_type.upper()}: {len(model_configs)} configurations")
        print(f"{'─' * 100}")

        for config_name, config in model_configs.items():
            desc = config.pop('description', 'No description')
            feature_sel = config.pop('feature_selection', 'all')

            print(f"\n  {config_name}:")
            print(f"    Description: {desc}")
            print(f"    Features: {feature_sel} ({FEATURE_SELECTION_STRATEGIES.get(feature_sel, 'N/A')})")
            print(f"    Parameters: {config}")


if __name__ == '__main__':
    print_config_summary()
