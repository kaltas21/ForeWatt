"""
Baseline Grid Configuration Generator
=====================================
Pre-defined configurations for CatBoost, XGBoost, LightGBM, and Prophet.

15 configurations per model with feature tier distribution:
- minimal:  3 configs
- core:     5 configs
- extended: 4 configs
- full:     3 configs

Author: ForeWatt Team - New Experiment V2
Date: November 2025
"""

import sys
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# TARGETS
# =============================================================================
TARGETS = ['price_real', 'consumption']

TARGET_STRATEGIES = {
    'price_real': 'baseline_core',
    'consumption': 'consumption_baseline_core'
}


# =============================================================================
# CATBOOST CONFIGURATIONS (15 configs)
# =============================================================================
CATBOOST_CONFIGS = {
    # === MINIMAL TIER (3 configs) ===
    'catboost_ultra_light': {
        'iterations': 200,
        'depth': 4,
        'learning_rate': 0.1,
        'l2_leaf_reg': 3.0,
        'border_count': 64,
        'feature_tier': 'minimal',
        'description': 'Ultra-light CatBoost for fast iteration'
    },
    'catboost_light': {
        'iterations': 500,
        'depth': 5,
        'learning_rate': 0.08,
        'l2_leaf_reg': 3.0,
        'border_count': 128,
        'feature_tier': 'minimal',
        'description': 'Light CatBoost with minimal features'
    },
    'catboost_minimal_deep': {
        'iterations': 800,
        'depth': 6,
        'learning_rate': 0.05,
        'l2_leaf_reg': 5.0,
        'border_count': 128,
        'feature_tier': 'minimal',
        'description': 'Deeper CatBoost with minimal features'
    },

    # === CORE TIER (5 configs) ===
    'catboost_balanced': {
        'iterations': 1000,
        'depth': 6,
        'learning_rate': 0.05,
        'l2_leaf_reg': 3.0,
        'border_count': 128,
        'feature_tier': 'core',
        'description': 'Balanced CatBoost - recommended default'
    },
    'catboost_core_fast': {
        'iterations': 500,
        'depth': 6,
        'learning_rate': 0.1,
        'l2_leaf_reg': 3.0,
        'border_count': 128,
        'feature_tier': 'core',
        'description': 'Fast CatBoost with core features'
    },
    'catboost_core_deep': {
        'iterations': 1500,
        'depth': 8,
        'learning_rate': 0.03,
        'l2_leaf_reg': 5.0,
        'border_count': 254,
        'feature_tier': 'core',
        'description': 'Deep CatBoost with core features'
    },
    'catboost_core_regularized': {
        'iterations': 1000,
        'depth': 6,
        'learning_rate': 0.05,
        'l2_leaf_reg': 10.0,
        'border_count': 128,
        'feature_tier': 'core',
        'description': 'Regularized CatBoost with core features'
    },
    'catboost_core_wide': {
        'iterations': 800,
        'depth': 4,
        'learning_rate': 0.08,
        'l2_leaf_reg': 3.0,
        'border_count': 254,
        'feature_tier': 'core',
        'description': 'Wide shallow CatBoost with core features'
    },

    # === EXTENDED TIER (4 configs) ===
    'catboost_extended': {
        'iterations': 1500,
        'depth': 7,
        'learning_rate': 0.03,
        'l2_leaf_reg': 5.0,
        'border_count': 254,
        'feature_tier': 'extended',
        'description': 'CatBoost with extended features'
    },
    'catboost_extended_deep': {
        'iterations': 2000,
        'depth': 8,
        'learning_rate': 0.02,
        'l2_leaf_reg': 5.0,
        'border_count': 254,
        'feature_tier': 'extended',
        'description': 'Deep CatBoost with extended features'
    },
    'catboost_extended_fast': {
        'iterations': 800,
        'depth': 6,
        'learning_rate': 0.08,
        'l2_leaf_reg': 3.0,
        'border_count': 128,
        'feature_tier': 'extended',
        'description': 'Fast CatBoost with extended features'
    },
    'catboost_extended_reg': {
        'iterations': 1500,
        'depth': 6,
        'learning_rate': 0.03,
        'l2_leaf_reg': 15.0,
        'border_count': 254,
        'feature_tier': 'extended',
        'description': 'Regularized CatBoost with extended features'
    },

    # === FULL TIER (3 configs) ===
    'catboost_full': {
        'iterations': 2000,
        'depth': 8,
        'learning_rate': 0.02,
        'l2_leaf_reg': 5.0,
        'border_count': 254,
        'feature_tier': 'full',
        'description': 'Full CatBoost with all features'
    },
    'catboost_full_deep': {
        'iterations': 3000,
        'depth': 10,
        'learning_rate': 0.01,
        'l2_leaf_reg': 10.0,
        'border_count': 254,
        'feature_tier': 'full',
        'description': 'Deep CatBoost with all features'
    },
    'catboost_full_regularized': {
        'iterations': 2000,
        'depth': 7,
        'learning_rate': 0.02,
        'l2_leaf_reg': 20.0,
        'border_count': 254,
        'feature_tier': 'full',
        'description': 'Regularized CatBoost with all features'
    },
}


# =============================================================================
# XGBOOST CONFIGURATIONS (15 configs)
# =============================================================================
XGBOOST_CONFIGS = {
    # === MINIMAL TIER (3 configs) ===
    'xgboost_ultra_light': {
        'n_estimators': 200,
        'max_depth': 4,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.0,
        'reg_lambda': 1.0,
        'feature_tier': 'minimal',
        'description': 'Ultra-light XGBoost for fast iteration'
    },
    'xgboost_light': {
        'n_estimators': 500,
        'max_depth': 5,
        'learning_rate': 0.08,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'feature_tier': 'minimal',
        'description': 'Light XGBoost with minimal features'
    },
    'xgboost_minimal_deep': {
        'n_estimators': 800,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 2.0,
        'feature_tier': 'minimal',
        'description': 'Deeper XGBoost with minimal features'
    },

    # === CORE TIER (5 configs) ===
    'xgboost_balanced': {
        'n_estimators': 1000,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'feature_tier': 'core',
        'description': 'Balanced XGBoost - recommended default'
    },
    'xgboost_core_fast': {
        'n_estimators': 500,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'reg_alpha': 0.0,
        'reg_lambda': 1.0,
        'feature_tier': 'core',
        'description': 'Fast XGBoost with core features'
    },
    'xgboost_core_deep': {
        'n_estimators': 1500,
        'max_depth': 8,
        'learning_rate': 0.03,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.1,
        'reg_lambda': 2.0,
        'feature_tier': 'core',
        'description': 'Deep XGBoost with core features'
    },
    'xgboost_core_regularized': {
        'n_estimators': 1000,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 1.0,
        'reg_lambda': 5.0,
        'feature_tier': 'core',
        'description': 'Regularized XGBoost with core features'
    },
    'xgboost_core_wide': {
        'n_estimators': 800,
        'max_depth': 4,
        'learning_rate': 0.08,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'reg_alpha': 0.0,
        'reg_lambda': 1.0,
        'feature_tier': 'core',
        'description': 'Wide shallow XGBoost with core features'
    },

    # === EXTENDED TIER (4 configs) ===
    'xgboost_extended': {
        'n_estimators': 1500,
        'max_depth': 7,
        'learning_rate': 0.03,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.1,
        'reg_lambda': 2.0,
        'feature_tier': 'extended',
        'description': 'XGBoost with extended features'
    },
    'xgboost_extended_deep': {
        'n_estimators': 2000,
        'max_depth': 8,
        'learning_rate': 0.02,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.1,
        'reg_lambda': 3.0,
        'feature_tier': 'extended',
        'description': 'Deep XGBoost with extended features'
    },
    'xgboost_extended_fast': {
        'n_estimators': 800,
        'max_depth': 6,
        'learning_rate': 0.08,
        'subsample': 0.9,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.0,
        'reg_lambda': 1.0,
        'feature_tier': 'extended',
        'description': 'Fast XGBoost with extended features'
    },
    'xgboost_extended_reg': {
        'n_estimators': 1500,
        'max_depth': 6,
        'learning_rate': 0.03,
        'subsample': 0.7,
        'colsample_bytree': 0.6,
        'reg_alpha': 1.0,
        'reg_lambda': 10.0,
        'feature_tier': 'extended',
        'description': 'Regularized XGBoost with extended features'
    },

    # === FULL TIER (3 configs) ===
    'xgboost_full': {
        'n_estimators': 2000,
        'max_depth': 8,
        'learning_rate': 0.02,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.1,
        'reg_lambda': 3.0,
        'feature_tier': 'full',
        'description': 'Full XGBoost with all features'
    },
    'xgboost_full_deep': {
        'n_estimators': 3000,
        'max_depth': 10,
        'learning_rate': 0.01,
        'subsample': 0.8,
        'colsample_bytree': 0.6,
        'reg_alpha': 0.5,
        'reg_lambda': 5.0,
        'feature_tier': 'full',
        'description': 'Deep XGBoost with all features'
    },
    'xgboost_full_regularized': {
        'n_estimators': 2000,
        'max_depth': 7,
        'learning_rate': 0.02,
        'subsample': 0.7,
        'colsample_bytree': 0.6,
        'reg_alpha': 2.0,
        'reg_lambda': 10.0,
        'feature_tier': 'full',
        'description': 'Regularized XGBoost with all features'
    },
}


# =============================================================================
# LIGHTGBM CONFIGURATIONS (15 configs)
# =============================================================================
LIGHTGBM_CONFIGS = {
    # === MINIMAL TIER (3 configs) ===
    'lightgbm_ultra_light': {
        'n_estimators': 200,
        'max_depth': 4,
        'learning_rate': 0.1,
        'num_leaves': 15,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.0,
        'reg_lambda': 1.0,
        'feature_tier': 'minimal',
        'description': 'Ultra-light LightGBM for fast iteration'
    },
    'lightgbm_light': {
        'n_estimators': 500,
        'max_depth': 5,
        'learning_rate': 0.08,
        'num_leaves': 31,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'feature_tier': 'minimal',
        'description': 'Light LightGBM with minimal features'
    },
    'lightgbm_minimal_deep': {
        'n_estimators': 800,
        'max_depth': 6,
        'learning_rate': 0.05,
        'num_leaves': 63,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 2.0,
        'feature_tier': 'minimal',
        'description': 'Deeper LightGBM with minimal features'
    },

    # === CORE TIER (5 configs) ===
    'lightgbm_balanced': {
        'n_estimators': 1000,
        'max_depth': 6,
        'learning_rate': 0.05,
        'num_leaves': 63,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'feature_tier': 'core',
        'description': 'Balanced LightGBM - recommended default'
    },
    'lightgbm_core_fast': {
        'n_estimators': 500,
        'max_depth': 6,
        'learning_rate': 0.1,
        'num_leaves': 63,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'reg_alpha': 0.0,
        'reg_lambda': 1.0,
        'feature_tier': 'core',
        'description': 'Fast LightGBM with core features'
    },
    'lightgbm_core_deep': {
        'n_estimators': 1500,
        'max_depth': 8,
        'learning_rate': 0.03,
        'num_leaves': 127,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.1,
        'reg_lambda': 2.0,
        'feature_tier': 'core',
        'description': 'Deep LightGBM with core features'
    },
    'lightgbm_core_regularized': {
        'n_estimators': 1000,
        'max_depth': 6,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 1.0,
        'reg_lambda': 5.0,
        'feature_tier': 'core',
        'description': 'Regularized LightGBM with core features'
    },
    'lightgbm_core_wide': {
        'n_estimators': 800,
        'max_depth': 4,
        'learning_rate': 0.08,
        'num_leaves': 31,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'reg_alpha': 0.0,
        'reg_lambda': 1.0,
        'feature_tier': 'core',
        'description': 'Wide shallow LightGBM with core features'
    },

    # === EXTENDED TIER (4 configs) ===
    'lightgbm_extended': {
        'n_estimators': 1500,
        'max_depth': 7,
        'learning_rate': 0.03,
        'num_leaves': 127,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.1,
        'reg_lambda': 2.0,
        'feature_tier': 'extended',
        'description': 'LightGBM with extended features'
    },
    'lightgbm_extended_deep': {
        'n_estimators': 2000,
        'max_depth': 8,
        'learning_rate': 0.02,
        'num_leaves': 255,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.1,
        'reg_lambda': 3.0,
        'feature_tier': 'extended',
        'description': 'Deep LightGBM with extended features'
    },
    'lightgbm_extended_fast': {
        'n_estimators': 800,
        'max_depth': 6,
        'learning_rate': 0.08,
        'num_leaves': 63,
        'subsample': 0.9,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.0,
        'reg_lambda': 1.0,
        'feature_tier': 'extended',
        'description': 'Fast LightGBM with extended features'
    },
    'lightgbm_extended_reg': {
        'n_estimators': 1500,
        'max_depth': 6,
        'learning_rate': 0.03,
        'num_leaves': 31,
        'subsample': 0.7,
        'colsample_bytree': 0.6,
        'reg_alpha': 1.0,
        'reg_lambda': 10.0,
        'feature_tier': 'extended',
        'description': 'Regularized LightGBM with extended features'
    },

    # === FULL TIER (3 configs) ===
    'lightgbm_full': {
        'n_estimators': 2000,
        'max_depth': 8,
        'learning_rate': 0.02,
        'num_leaves': 255,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.1,
        'reg_lambda': 3.0,
        'feature_tier': 'full',
        'description': 'Full LightGBM with all features'
    },
    'lightgbm_full_deep': {
        'n_estimators': 3000,
        'max_depth': 10,
        'learning_rate': 0.01,
        'num_leaves': 511,
        'subsample': 0.8,
        'colsample_bytree': 0.6,
        'reg_alpha': 0.5,
        'reg_lambda': 5.0,
        'feature_tier': 'full',
        'description': 'Deep LightGBM with all features'
    },
    'lightgbm_full_regularized': {
        'n_estimators': 2000,
        'max_depth': 7,
        'learning_rate': 0.02,
        'num_leaves': 127,
        'subsample': 0.7,
        'colsample_bytree': 0.6,
        'reg_alpha': 2.0,
        'reg_lambda': 10.0,
        'feature_tier': 'full',
        'description': 'Regularized LightGBM with all features'
    },
}


# =============================================================================
# PROPHET CONFIGURATIONS (15 configs)
# =============================================================================
PROPHET_CONFIGS = {
    # === MINIMAL TIER (3 configs) ===
    'prophet_ultra_light': {
        'changepoint_prior_scale': 0.05,
        'seasonality_prior_scale': 10.0,
        'seasonality_mode': 'additive',
        'yearly_seasonality': True,
        'weekly_seasonality': True,
        'daily_seasonality': True,
        'n_changepoints': 15,
        'feature_tier': 'minimal',
        'description': 'Ultra-light Prophet for fast iteration'
    },
    'prophet_light': {
        'changepoint_prior_scale': 0.1,
        'seasonality_prior_scale': 10.0,
        'seasonality_mode': 'additive',
        'yearly_seasonality': True,
        'weekly_seasonality': True,
        'daily_seasonality': True,
        'n_changepoints': 25,
        'feature_tier': 'minimal',
        'description': 'Light Prophet with minimal features'
    },
    'prophet_minimal_flexible': {
        'changepoint_prior_scale': 0.2,
        'seasonality_prior_scale': 5.0,
        'seasonality_mode': 'multiplicative',
        'yearly_seasonality': True,
        'weekly_seasonality': True,
        'daily_seasonality': True,
        'n_changepoints': 25,
        'feature_tier': 'minimal',
        'description': 'Flexible Prophet with minimal features'
    },

    # === CORE TIER (5 configs) ===
    'prophet_balanced': {
        'changepoint_prior_scale': 0.1,
        'seasonality_prior_scale': 10.0,
        'seasonality_mode': 'additive',
        'yearly_seasonality': True,
        'weekly_seasonality': True,
        'daily_seasonality': True,
        'n_changepoints': 25,
        'feature_tier': 'core',
        'description': 'Balanced Prophet - recommended default'
    },
    'prophet_core_flexible': {
        'changepoint_prior_scale': 0.2,
        'seasonality_prior_scale': 5.0,
        'seasonality_mode': 'additive',
        'yearly_seasonality': True,
        'weekly_seasonality': True,
        'daily_seasonality': True,
        'n_changepoints': 35,
        'feature_tier': 'core',
        'description': 'Flexible Prophet with core features'
    },
    'prophet_core_multiplicative': {
        'changepoint_prior_scale': 0.1,
        'seasonality_prior_scale': 10.0,
        'seasonality_mode': 'multiplicative',
        'yearly_seasonality': True,
        'weekly_seasonality': True,
        'daily_seasonality': True,
        'n_changepoints': 25,
        'feature_tier': 'core',
        'description': 'Multiplicative Prophet with core features'
    },
    'prophet_core_smooth': {
        'changepoint_prior_scale': 0.05,
        'seasonality_prior_scale': 15.0,
        'seasonality_mode': 'additive',
        'yearly_seasonality': True,
        'weekly_seasonality': True,
        'daily_seasonality': True,
        'n_changepoints': 15,
        'feature_tier': 'core',
        'description': 'Smooth Prophet with core features'
    },
    'prophet_core_sensitive': {
        'changepoint_prior_scale': 0.3,
        'seasonality_prior_scale': 5.0,
        'seasonality_mode': 'additive',
        'yearly_seasonality': True,
        'weekly_seasonality': True,
        'daily_seasonality': True,
        'n_changepoints': 50,
        'feature_tier': 'core',
        'description': 'Sensitive Prophet with core features'
    },

    # === EXTENDED TIER (4 configs) ===
    'prophet_extended': {
        'changepoint_prior_scale': 0.15,
        'seasonality_prior_scale': 10.0,
        'seasonality_mode': 'additive',
        'yearly_seasonality': True,
        'weekly_seasonality': True,
        'daily_seasonality': True,
        'n_changepoints': 35,
        'feature_tier': 'extended',
        'description': 'Prophet with extended features'
    },
    'prophet_extended_flexible': {
        'changepoint_prior_scale': 0.25,
        'seasonality_prior_scale': 5.0,
        'seasonality_mode': 'multiplicative',
        'yearly_seasonality': True,
        'weekly_seasonality': True,
        'daily_seasonality': True,
        'n_changepoints': 50,
        'feature_tier': 'extended',
        'description': 'Flexible Prophet with extended features'
    },
    'prophet_extended_smooth': {
        'changepoint_prior_scale': 0.05,
        'seasonality_prior_scale': 15.0,
        'seasonality_mode': 'additive',
        'yearly_seasonality': True,
        'weekly_seasonality': True,
        'daily_seasonality': True,
        'n_changepoints': 25,
        'feature_tier': 'extended',
        'description': 'Smooth Prophet with extended features'
    },
    'prophet_extended_multiplicative': {
        'changepoint_prior_scale': 0.15,
        'seasonality_prior_scale': 10.0,
        'seasonality_mode': 'multiplicative',
        'yearly_seasonality': True,
        'weekly_seasonality': True,
        'daily_seasonality': True,
        'n_changepoints': 35,
        'feature_tier': 'extended',
        'description': 'Multiplicative Prophet with extended features'
    },

    # === FULL TIER (3 configs) ===
    'prophet_full': {
        'changepoint_prior_scale': 0.15,
        'seasonality_prior_scale': 10.0,
        'seasonality_mode': 'additive',
        'yearly_seasonality': True,
        'weekly_seasonality': True,
        'daily_seasonality': True,
        'n_changepoints': 50,
        'feature_tier': 'full',
        'description': 'Full Prophet with all features'
    },
    'prophet_full_flexible': {
        'changepoint_prior_scale': 0.3,
        'seasonality_prior_scale': 5.0,
        'seasonality_mode': 'multiplicative',
        'yearly_seasonality': True,
        'weekly_seasonality': True,
        'daily_seasonality': True,
        'n_changepoints': 75,
        'feature_tier': 'full',
        'description': 'Flexible Prophet with all features'
    },
    'prophet_full_smooth': {
        'changepoint_prior_scale': 0.05,
        'seasonality_prior_scale': 20.0,
        'seasonality_mode': 'additive',
        'yearly_seasonality': True,
        'weekly_seasonality': True,
        'daily_seasonality': True,
        'n_changepoints': 25,
        'feature_tier': 'full',
        'description': 'Smooth Prophet with all features'
    },
}


def _compute_config_hash(config: Dict[str, Any], target: str) -> str:
    """Compute unique hash for a configuration."""
    hash_dict = {**config, 'target': target}
    hash_str = json.dumps(hash_dict, sort_keys=True, default=str)
    return hashlib.md5(hash_str.encode()).hexdigest()[:12]


def get_baseline_grid(
    model_type: Optional[str] = None,
    target: str = 'price_real'
) -> List[Dict[str, Any]]:
    """
    Get grid of baseline configurations.

    Args:
        model_type: 'catboost', 'xgboost', 'lightgbm', 'prophet', or None for all
        target: Target variable

    Returns:
        List of configuration dictionaries
    """
    configs = []

    model_configs = {
        'catboost': CATBOOST_CONFIGS,
        'xgboost': XGBOOST_CONFIGS,
        'lightgbm': LIGHTGBM_CONFIGS,
        'prophet': PROPHET_CONFIGS,
    }

    # Feature tier mapping for targets
    tier_mapping = {
        'price_real': {
            'minimal': 'baseline_minimal',
            'core': 'baseline_core',
            'extended': 'baseline_extended',
            'full': 'baseline_full',
        },
        'consumption': {
            'minimal': 'consumption_baseline_minimal',
            'core': 'consumption_baseline_core',
            'full': 'consumption_baseline_full',
        }
    }

    models_to_process = [model_type] if model_type else list(model_configs.keys())

    for model in models_to_process:
        if model not in model_configs:
            continue

        for config_name, config in model_configs[model].items():
            config_copy = config.copy()
            config_copy['config_name'] = config_name
            config_copy['model_type'] = model
            config_copy['target'] = target

            # Map feature tier to strategy
            tier = config_copy.get('feature_tier', 'core')
            target_tiers = tier_mapping.get(target, tier_mapping['price_real'])

            # For consumption, map extended -> full if extended not available
            if target == 'consumption' and tier == 'extended':
                tier = 'full'

            config_copy['feature_strategy'] = target_tiers.get(tier, 'baseline_core')

            # Compute unique hash
            config_copy['config_hash'] = _compute_config_hash(config_copy, target)

            configs.append(config_copy)

    return configs


class BaselineGridConfigGenerator:
    """
    Generator for baseline model grid search configurations.
    """

    def __init__(self, targets: List[str] = None):
        """
        Initialize generator.

        Args:
            targets: List of targets (default: ['price_real', 'consumption'])
        """
        self.targets = targets or TARGETS

    def get_grid_summary(self) -> Dict[str, Any]:
        """Get summary of grid configuration."""
        return {
            'catboost': {'count': len(CATBOOST_CONFIGS)},
            'xgboost': {'count': len(XGBOOST_CONFIGS)},
            'lightgbm': {'count': len(LIGHTGBM_CONFIGS)},
            'prophet': {'count': len(PROPHET_CONFIGS)},
            'total_per_target': (
                len(CATBOOST_CONFIGS) +
                len(XGBOOST_CONFIGS) +
                len(LIGHTGBM_CONFIGS) +
                len(PROPHET_CONFIGS)
            ),
            'total_all_targets': (
                len(CATBOOST_CONFIGS) +
                len(XGBOOST_CONFIGS) +
                len(LIGHTGBM_CONFIGS) +
                len(PROPHET_CONFIGS)
            ) * len(self.targets),
            'targets': self.targets
        }

    def print_grid_summary(self):
        """Print grid summary."""
        summary = self.get_grid_summary()

        print(f"\n{'='*80}")
        print("BASELINE GRID CONFIGURATION SUMMARY")
        print(f"{'='*80}")
        print(f"\nModels:")
        print(f"  CatBoost:  {summary['catboost']['count']} configurations")
        print(f"  XGBoost:   {summary['xgboost']['count']} configurations")
        print(f"  LightGBM:  {summary['lightgbm']['count']} configurations")
        print(f"  Prophet:   {summary['prophet']['count']} configurations")
        print(f"\nTotal per target: {summary['total_per_target']} configurations")
        print(f"Targets: {summary['targets']}")
        print(f"Total all targets: {summary['total_all_targets']} configurations")
        print(f"{'='*80}\n")


def main():
    """Print grid summary."""
    generator = BaselineGridConfigGenerator()
    generator.print_grid_summary()

    # Print tier distribution
    print("\nFeature tier distribution per model:")
    for model in ['catboost', 'xgboost', 'lightgbm', 'prophet']:
        configs = get_baseline_grid(model, 'price_real')
        tiers = {}
        for c in configs:
            tier = c.get('feature_tier', 'core')
            tiers[tier] = tiers.get(tier, 0) + 1
        print(f"  {model:10s}: {tiers}")


if __name__ == "__main__":
    main()
