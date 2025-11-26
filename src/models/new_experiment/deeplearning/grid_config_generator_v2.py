"""
Hyperparameter Configurations for New Experiment V2
===================================================
Pre-defined configurations optimized for RTX 5090 (32GB VRAM) and
fundamental price forecasting features.

15 configurations per model with feature tier distribution:
- minimal:     3 configs (ultra_light, light, micro)
- ratios_only: 2 configs (ratios_only, ratios_deep)
- core:        3 configs (balanced, regularized, core_deep)
- extended:    4 configs (deep, wide, narrow_deep, extended_balanced)
- full:        3 configs (ultra_deep, full_comparison, full_regularized)

Author: ForeWatt Team - New Experiment V2
Date: November 2025
"""

import sys
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# SHARED PARAMETERS (RTX 5090 OPTIMIZED)
# ============================================================================

SHARED_PARAMS = {
    'input_size': 168,       # 1 week lookback
    'horizon': 24,           # 24h forecast
    'val_check_steps': 50,
    'early_stop_patience_steps': 100,
    'random_seed': 42,
}

# Target-specific DEFAULT feature strategies (D-1 SAFE)
# Individual configs can override with their own feature_strategy
TARGET_STRATEGIES = {
    'price_real': 'fundamental_v2',
    'consumption': 'consumption_v2',
}

# All targets to run
TARGETS = ['price_real', 'consumption']

# Feature strategy tiers for each target
PRICE_FEATURE_STRATEGIES = {
    'ratios_only': 'price_ratios_only',      # ~15 features - cleanest
    'minimal': 'fundamental_v2_minimal',      # ~20 features
    'core': 'fundamental_v2',                 # ~35 features (default)
    'extended': 'fundamental_v2_extended',    # ~50 features
    'full': 'fundamental_v2_full',            # ~70 features
}

CONSUMPTION_FEATURE_STRATEGIES = {
    'minimal': 'consumption_v2_minimal',      # ~15 features
    'core': 'consumption_v2',                 # ~25 features (default)
    'extended': 'consumption_v2_extended',    # ~40 features
    'full': 'consumption_v2_full',            # ~55 features
}


# ============================================================================
# PATCHTST CONFIGURATIONS (10 configs)
# ============================================================================

PATCHTST_CONFIGS = {
    # MINIMAL FEATURES (~15-20) - Fast iteration
    'patchtst_ultra_light': {
        'patch_len': 12,
        'stride': 12,
        'n_layers': 1,
        'd_model': 64,
        'n_heads': 4,
        'dropout': 0.1,
        'batch_size': 512,
        'learning_rate': 2e-3,
        'max_steps': 300,
        'input_size': 72,
        'feature_tier': 'minimal',
        'description': 'Ultra-fast PatchTST with minimal features (~20)'
    },

    'patchtst_light': {
        'patch_len': 12,
        'stride': 12,
        'n_layers': 2,
        'd_model': 128,
        'n_heads': 4,
        'dropout': 0.2,
        'batch_size': 256,
        'learning_rate': 1e-3,
        'max_steps': 500,
        'feature_tier': 'minimal',
        'description': 'Fast PatchTST with minimal features (~20)'
    },

    # RATIOS ONLY (~15) - Cleanest signal
    'patchtst_ratios_only': {
        'patch_len': 24,
        'stride': 12,
        'n_layers': 3,
        'd_model': 192,
        'n_heads': 6,
        'dropout': 0.2,
        'batch_size': 256,
        'learning_rate': 5e-4,
        'max_steps': 800,
        'feature_tier': 'ratios_only',
        'description': 'PatchTST with ratio features only (~15)'
    },

    # CORE FEATURES (~35) - Recommended default
    'patchtst_balanced': {
        'patch_len': 24,
        'stride': 12,
        'n_layers': 3,
        'd_model': 256,
        'n_heads': 8,
        'dropout': 0.2,
        'batch_size': 256,
        'learning_rate': 5e-4,
        'max_steps': 1000,
        'feature_tier': 'core',
        'description': 'Balanced PatchTST with core features (~35)'
    },

    'patchtst_regularized': {
        'patch_len': 24,
        'stride': 12,
        'n_layers': 4,
        'd_model': 256,
        'n_heads': 8,
        'dropout': 0.4,
        'batch_size': 256,
        'learning_rate': 3e-4,
        'max_steps': 1200,
        'feature_tier': 'core',
        'description': 'Regularized PatchTST with core features (~35)'
    },

    # EXTENDED FEATURES (~50) - More capacity needed
    'patchtst_deep': {
        'patch_len': 24,
        'stride': 12,
        'n_layers': 5,
        'd_model': 384,
        'n_heads': 12,
        'dropout': 0.2,
        'batch_size': 256,
        'learning_rate': 4e-4,
        'max_steps': 1200,
        'feature_tier': 'extended',
        'description': 'Deep PatchTST with extended features (~50)'
    },

    'patchtst_wide': {
        'patch_len': 24,
        'stride': 12,
        'n_layers': 3,
        'd_model': 384,
        'n_heads': 12,
        'dropout': 0.2,
        'batch_size': 256,
        'learning_rate': 5e-4,
        'max_steps': 1200,
        'feature_tier': 'extended',
        'description': 'Wide PatchTST with extended features (~50)'
    },

    'patchtst_narrow_deep': {
        'patch_len': 12,
        'stride': 6,
        'n_layers': 6,
        'd_model': 128,
        'n_heads': 4,
        'dropout': 0.3,
        'batch_size': 256,
        'learning_rate': 5e-4,
        'max_steps': 1200,
        'feature_tier': 'extended',
        'description': 'Narrow-deep PatchTST with extended features (~50)'
    },

    # FULL FEATURES (~70) - Maximum signal, needs capacity
    'patchtst_ultra_deep': {
        'patch_len': 24,
        'stride': 12,
        'n_layers': 6,
        'd_model': 384,
        'n_heads': 12,
        'dropout': 0.25,
        'batch_size': 256,
        'learning_rate': 3e-4,
        'max_steps': 1500,
        'feature_tier': 'full',
        'description': 'Deep PatchTST with full features (~70)'
    },

    'patchtst_full_comparison': {
        'patch_len': 24,
        'stride': 24,
        'n_layers': 4,
        'd_model': 256,
        'n_heads': 8,
        'dropout': 0.25,
        'batch_size': 256,
        'learning_rate': 5e-4,
        'max_steps': 1200,
        'feature_tier': 'full',
        'description': 'PatchTST baseline with full features (~70)'
    },

    # === NEW CONFIGS (5 more for 15 total) ===

    # MINIMAL - One more (3 total minimal)
    'patchtst_micro': {
        'patch_len': 8,
        'stride': 8,
        'n_layers': 1,
        'd_model': 48,
        'n_heads': 2,
        'dropout': 0.1,
        'batch_size': 512,
        'learning_rate': 3e-3,
        'max_steps': 200,
        'input_size': 48,
        'feature_tier': 'minimal',
        'description': 'Micro PatchTST for fastest iteration (~20)'
    },

    # RATIOS_ONLY - One more (2 total ratios_only)
    'patchtst_ratios_deep': {
        'patch_len': 24,
        'stride': 12,
        'n_layers': 4,
        'd_model': 256,
        'n_heads': 8,
        'dropout': 0.25,
        'batch_size': 256,
        'learning_rate': 4e-4,
        'max_steps': 1000,
        'feature_tier': 'ratios_only',
        'description': 'Deep PatchTST with ratio features only (~15)'
    },

    # CORE - One more (3 total core)
    'patchtst_core_deep': {
        'patch_len': 24,
        'stride': 12,
        'n_layers': 5,
        'd_model': 320,
        'n_heads': 8,
        'dropout': 0.25,
        'batch_size': 256,
        'learning_rate': 4e-4,
        'max_steps': 1100,
        'feature_tier': 'core',
        'description': 'Deep PatchTST with core features (~35)'
    },

    # EXTENDED - One more (4 total extended)
    'patchtst_extended_balanced': {
        'patch_len': 24,
        'stride': 12,
        'n_layers': 4,
        'd_model': 320,
        'n_heads': 8,
        'dropout': 0.3,
        'batch_size': 256,
        'learning_rate': 4e-4,
        'max_steps': 1000,
        'feature_tier': 'extended',
        'description': 'Balanced PatchTST with extended features (~50)'
    },

    # FULL - One more (3 total full)
    'patchtst_full_regularized': {
        'patch_len': 24,
        'stride': 12,
        'n_layers': 5,
        'd_model': 320,
        'n_heads': 8,
        'dropout': 0.35,
        'batch_size': 256,
        'learning_rate': 3e-4,
        'max_steps': 1300,
        'feature_tier': 'full',
        'description': 'Regularized PatchTST with full features (~70)'
    },
}


# ============================================================================
# N-HITS CONFIGURATIONS (15 configs)
# ============================================================================

NHITS_CONFIGS = {
    # MINIMAL FEATURES (~15-20) - Fast iteration
    'nhits_ultra_light': {
        'n_blocks': [1, 1],
        'hidden_size': 128,
        'n_mlp_layers': 2,
        'n_pool_kernel_size': [2, 1],
        'n_freq_downsample': [2, 1],
        'batch_size': 512,
        'learning_rate': 2e-3,
        'max_steps': 300,
        'input_size': 72,
        'feature_tier': 'minimal',
        'description': 'Ultra-fast N-HiTS with minimal features (~20)'
    },

    'nhits_light': {
        'n_blocks': [1, 1, 1],
        'hidden_size': 256,
        'n_mlp_layers': 2,
        'n_pool_kernel_size': [2, 2, 1],
        'n_freq_downsample': [2, 1, 1],
        'batch_size': 256,
        'learning_rate': 1e-3,
        'max_steps': 500,
        'feature_tier': 'minimal',
        'description': 'Fast N-HiTS with minimal features (~20)'
    },

    # RATIOS ONLY (~15) - Cleanest signal
    'nhits_ratios_only': {
        'n_blocks': [2, 2, 2],
        'hidden_size': 384,
        'n_mlp_layers': 2,
        'n_pool_kernel_size': [4, 4, 1],
        'n_freq_downsample': [4, 2, 1],
        'batch_size': 256,
        'learning_rate': 5e-4,
        'max_steps': 800,
        'feature_tier': 'ratios_only',
        'description': 'N-HiTS with ratio features only (~15)'
    },

    # CORE FEATURES (~35) - Recommended default
    'nhits_balanced': {
        'n_blocks': [2, 2, 2],
        'hidden_size': 512,
        'n_mlp_layers': 2,
        'n_pool_kernel_size': [4, 4, 1],
        'n_freq_downsample': [4, 2, 1],
        'batch_size': 256,
        'learning_rate': 5e-4,
        'max_steps': 1000,
        'feature_tier': 'core',
        'description': 'Balanced N-HiTS with core features (~35)'
    },

    'nhits_regularized': {
        'n_blocks': [2, 2, 2],
        'hidden_size': 384,
        'n_mlp_layers': 2,
        'n_pool_kernel_size': [4, 4, 1],
        'n_freq_downsample': [4, 2, 1],
        'batch_size': 256,
        'learning_rate': 3e-4,
        'max_steps': 1200,
        'feature_tier': 'core',
        'description': 'Regularized N-HiTS with core features (~35)'
    },

    # EXTENDED FEATURES (~50) - More capacity needed
    'nhits_deep': {
        'n_blocks': [2, 2, 2],
        'hidden_size': 768,
        'n_mlp_layers': 3,
        'n_pool_kernel_size': [4, 4, 1],
        'n_freq_downsample': [4, 2, 1],
        'batch_size': 256,
        'learning_rate': 4e-4,
        'max_steps': 1200,
        'feature_tier': 'extended',
        'description': 'Deep N-HiTS with extended features (~50)'
    },

    'nhits_wide': {
        'n_blocks': [2, 2, 2],
        'hidden_size': 768,
        'n_mlp_layers': 2,
        'n_pool_kernel_size': [4, 4, 1],
        'n_freq_downsample': [4, 2, 1],
        'batch_size': 256,
        'learning_rate': 5e-4,
        'max_steps': 1200,
        'feature_tier': 'extended',
        'description': 'Wide N-HiTS with extended features (~50)'
    },

    'nhits_narrow_deep': {
        'n_blocks': [2, 2, 2],
        'hidden_size': 256,
        'n_mlp_layers': 3,
        'n_pool_kernel_size': [4, 4, 1],
        'n_freq_downsample': [4, 2, 1],
        'batch_size': 256,
        'learning_rate': 5e-4,
        'max_steps': 1200,
        'feature_tier': 'extended',
        'description': 'Narrow-deep N-HiTS with extended features (~50)'
    },

    # FULL FEATURES (~70) - Maximum signal, needs capacity
    'nhits_ultra_deep': {
        'n_blocks': [2, 2, 2],
        'hidden_size': 768,
        'n_mlp_layers': 3,
        'n_pool_kernel_size': [4, 4, 1],
        'n_freq_downsample': [4, 2, 1],
        'batch_size': 256,
        'learning_rate': 3e-4,
        'max_steps': 1500,
        'feature_tier': 'full',
        'description': 'Deep N-HiTS with full features (~70)'
    },

    'nhits_full_comparison': {
        'n_blocks': [2, 2, 2],
        'hidden_size': 512,
        'n_mlp_layers': 2,
        'n_pool_kernel_size': [4, 4, 1],
        'n_freq_downsample': [4, 2, 1],
        'batch_size': 256,
        'learning_rate': 5e-4,
        'max_steps': 1200,
        'feature_tier': 'full',
        'description': 'N-HiTS baseline with full features (~70)'
    },

    # === NEW CONFIGS (5 more for 15 total) ===

    # MINIMAL - One more (3 total minimal)
    'nhits_micro': {
        'n_blocks': [1, 1],
        'hidden_size': 64,
        'n_mlp_layers': 1,
        'n_pool_kernel_size': [2, 1],
        'n_freq_downsample': [2, 1],
        'batch_size': 512,
        'learning_rate': 3e-3,
        'max_steps': 200,
        'input_size': 48,
        'feature_tier': 'minimal',
        'description': 'Micro N-HiTS for fastest iteration (~20)'
    },

    # RATIOS_ONLY - One more (2 total ratios_only)
    'nhits_ratios_deep': {
        'n_blocks': [2, 2, 2],
        'hidden_size': 512,
        'n_mlp_layers': 3,
        'n_pool_kernel_size': [4, 4, 1],
        'n_freq_downsample': [4, 2, 1],
        'batch_size': 256,
        'learning_rate': 4e-4,
        'max_steps': 1000,
        'feature_tier': 'ratios_only',
        'description': 'Deep N-HiTS with ratio features only (~15)'
    },

    # CORE - One more (3 total core)
    'nhits_core_deep': {
        'n_blocks': [2, 2, 2],
        'hidden_size': 640,
        'n_mlp_layers': 3,
        'n_pool_kernel_size': [4, 4, 1],
        'n_freq_downsample': [4, 2, 1],
        'batch_size': 256,
        'learning_rate': 4e-4,
        'max_steps': 1100,
        'feature_tier': 'core',
        'description': 'Deep N-HiTS with core features (~35)'
    },

    # EXTENDED - One more (4 total extended)
    'nhits_extended_balanced': {
        'n_blocks': [2, 2, 2],
        'hidden_size': 512,
        'n_mlp_layers': 2,
        'n_pool_kernel_size': [4, 4, 1],
        'n_freq_downsample': [4, 2, 1],
        'batch_size': 256,
        'learning_rate': 4e-4,
        'max_steps': 1000,
        'feature_tier': 'extended',
        'description': 'Balanced N-HiTS with extended features (~50)'
    },

    # FULL - One more (3 total full)
    'nhits_full_regularized': {
        'n_blocks': [2, 2, 2],
        'hidden_size': 640,
        'n_mlp_layers': 3,
        'n_pool_kernel_size': [4, 4, 1],
        'n_freq_downsample': [4, 2, 1],
        'batch_size': 256,
        'learning_rate': 3e-4,
        'max_steps': 1300,
        'feature_tier': 'full',
        'description': 'Regularized N-HiTS with full features (~70)'
    },
}


# ============================================================================
# TFT CONFIGURATIONS (15 configs)
# ============================================================================

TFT_CONFIGS = {
    # MINIMAL FEATURES (~15-20) - Fast iteration
    'tft_ultra_light': {
        'hidden_size': 32,
        'n_head': 2,
        'lstm_n_layers': 1,
        'dropout': 0.1,
        'batch_size': 512,
        'learning_rate': 2e-3,
        'max_steps': 300,
        'input_size': 72,
        'feature_tier': 'minimal',
        'description': 'Ultra-fast TFT with minimal features (~20)'
    },

    'tft_light': {
        'hidden_size': 64,
        'n_head': 4,
        'lstm_n_layers': 1,
        'dropout': 0.2,
        'batch_size': 256,
        'learning_rate': 1e-3,
        'max_steps': 500,
        'feature_tier': 'minimal',
        'description': 'Fast TFT with minimal features (~20)'
    },

    # RATIOS ONLY (~15) - Cleanest signal
    'tft_ratios_only': {
        'hidden_size': 96,
        'n_head': 4,
        'lstm_n_layers': 2,
        'dropout': 0.2,
        'batch_size': 256,
        'learning_rate': 5e-4,
        'max_steps': 800,
        'feature_tier': 'ratios_only',
        'description': 'TFT with ratio features only (~15)'
    },

    # CORE FEATURES (~35) - Recommended default
    'tft_balanced': {
        'hidden_size': 128,
        'n_head': 4,
        'lstm_n_layers': 2,
        'dropout': 0.2,
        'batch_size': 256,
        'learning_rate': 5e-4,
        'max_steps': 1000,
        'feature_tier': 'core',
        'description': 'Balanced TFT with core features (~35)'
    },

    'tft_regularized': {
        'hidden_size': 128,
        'n_head': 4,
        'lstm_n_layers': 2,
        'dropout': 0.4,
        'batch_size': 256,
        'learning_rate': 3e-4,
        'max_steps': 1200,
        'feature_tier': 'core',
        'description': 'Regularized TFT with core features (~35)'
    },

    # EXTENDED FEATURES (~50) - More capacity needed
    'tft_deep': {
        'hidden_size': 192,
        'n_head': 8,
        'lstm_n_layers': 2,
        'dropout': 0.2,
        'batch_size': 256,
        'learning_rate': 4e-4,
        'max_steps': 1200,
        'feature_tier': 'extended',
        'description': 'Deep TFT with extended features (~50)'
    },

    'tft_wide': {
        'hidden_size': 192,
        'n_head': 8,
        'lstm_n_layers': 2,
        'dropout': 0.2,
        'batch_size': 256,
        'learning_rate': 5e-4,
        'max_steps': 1200,
        'feature_tier': 'extended',
        'description': 'Wide TFT with extended features (~50)'
    },

    'tft_narrow_deep': {
        'hidden_size': 64,
        'n_head': 4,
        'lstm_n_layers': 3,
        'dropout': 0.2,
        'batch_size': 256,
        'learning_rate': 5e-4,
        'max_steps': 1200,
        'feature_tier': 'extended',
        'description': 'Narrow-deep TFT with extended features (~50)'
    },

    # FULL FEATURES (~70) - Maximum signal, needs capacity
    'tft_ultra_deep': {
        'hidden_size': 192,
        'n_head': 8,
        'lstm_n_layers': 3,
        'dropout': 0.25,
        'batch_size': 256,
        'learning_rate': 3e-4,
        'max_steps': 1500,
        'feature_tier': 'full',
        'description': 'Deep TFT with full features (~70)'
    },

    'tft_full_comparison': {
        'hidden_size': 128,
        'n_head': 4,
        'lstm_n_layers': 2,
        'dropout': 0.25,
        'batch_size': 256,
        'learning_rate': 5e-4,
        'max_steps': 1200,
        'feature_tier': 'full',
        'description': 'TFT baseline with full features (~70)'
    },

    # === NEW CONFIGS (5 more for 15 total) ===

    # MINIMAL - One more (3 total minimal)
    'tft_micro': {
        'hidden_size': 24,
        'n_head': 2,
        'lstm_n_layers': 1,
        'dropout': 0.1,
        'batch_size': 512,
        'learning_rate': 3e-3,
        'max_steps': 200,
        'input_size': 48,
        'feature_tier': 'minimal',
        'description': 'Micro TFT for fastest iteration (~20)'
    },

    # RATIOS_ONLY - One more (2 total ratios_only)
    'tft_ratios_deep': {
        'hidden_size': 128,
        'n_head': 4,
        'lstm_n_layers': 3,
        'dropout': 0.25,
        'batch_size': 256,
        'learning_rate': 4e-4,
        'max_steps': 1000,
        'feature_tier': 'ratios_only',
        'description': 'Deep TFT with ratio features only (~15)'
    },

    # CORE - One more (3 total core)
    'tft_core_deep': {
        'hidden_size': 160,
        'n_head': 8,
        'lstm_n_layers': 3,
        'dropout': 0.25,
        'batch_size': 256,
        'learning_rate': 4e-4,
        'max_steps': 1100,
        'feature_tier': 'core',
        'description': 'Deep TFT with core features (~35)'
    },

    # EXTENDED - One more (4 total extended)
    'tft_extended_balanced': {
        'hidden_size': 160,
        'n_head': 8,
        'lstm_n_layers': 2,
        'dropout': 0.3,
        'batch_size': 256,
        'learning_rate': 4e-4,
        'max_steps': 1000,
        'feature_tier': 'extended',
        'description': 'Balanced TFT with extended features (~50)'
    },

    # FULL - One more (3 total full)
    'tft_full_regularized': {
        'hidden_size': 160,
        'n_head': 8,
        'lstm_n_layers': 3,
        'dropout': 0.35,
        'batch_size': 256,
        'learning_rate': 3e-4,
        'max_steps': 1300,
        'feature_tier': 'full',
        'description': 'Regularized TFT with full features (~70)'
    },
}


# ============================================================================
# GRID CONFIG GENERATOR CLASS
# ============================================================================

class GridConfigGeneratorV2:
    """
    Generates pre-defined hyperparameter configurations for V2 experiment.

    Each model has 10 configurations optimized for RTX 5090.
    Supports both price_real and consumption targets with D-1 safe features.
    """

    def __init__(self, shared_params: Dict[str, Any] = None):
        """
        Initialize generator.

        Args:
            shared_params: Shared parameters for all configurations
        """
        self.shared_params = shared_params or SHARED_PARAMS.copy()

    @staticmethod
    def _config_hash(config: Dict[str, Any]) -> str:
        """Generate unique hash for configuration."""
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]

    def _build_config(
        self,
        config_name: str,
        config_params: Dict[str, Any],
        model_type: str,
        target: str = 'price_real'
    ) -> Dict[str, Any]:
        """Build full configuration with shared params and target."""
        config = self.shared_params.copy()
        config.update(config_params)
        config['config_name'] = config_name
        config['model_type'] = model_type
        config['target'] = target

        # Resolve feature_tier to actual feature_strategy
        feature_tier = config.pop('feature_tier', 'core')

        if target == 'price_real':
            strategy_map = PRICE_FEATURE_STRATEGIES
        else:
            strategy_map = CONSUMPTION_FEATURE_STRATEGIES
            # Consumption doesn't have ratios_only, map to minimal
            if feature_tier == 'ratios_only':
                feature_tier = 'minimal'

        config['feature_strategy'] = strategy_map.get(feature_tier, strategy_map['core'])
        config['feature_tier'] = feature_tier  # Keep for reference
        config['config_hash'] = self._config_hash(config)
        return config

    def get_patchtst_configs(self, target: str = 'price_real') -> List[Dict[str, Any]]:
        """Get all PatchTST configurations for a target."""
        configs = []
        for name, params in PATCHTST_CONFIGS.items():
            configs.append(self._build_config(name, params, 'patchtst', target))
        return configs

    def get_nhits_configs(self, target: str = 'price_real') -> List[Dict[str, Any]]:
        """Get all N-HiTS configurations for a target."""
        configs = []
        for name, params in NHITS_CONFIGS.items():
            configs.append(self._build_config(name, params, 'nhits', target))
        return configs

    def get_tft_configs(self, target: str = 'price_real') -> List[Dict[str, Any]]:
        """Get all TFT configurations for a target."""
        configs = []
        for name, params in TFT_CONFIGS.items():
            configs.append(self._build_config(name, params, 'tft', target))
        return configs

    def get_full_grid(
        self,
        model_type: Optional[str] = None,
        target: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get full grid for specified model type and target.

        Args:
            model_type: 'patchtst', 'nhits', 'tft', or None for all
            target: 'price_real', 'consumption', or None for all targets

        Returns:
            List of configuration dictionaries
        """
        targets = [target] if target else TARGETS
        all_configs = []

        for t in targets:
            if model_type == 'patchtst':
                all_configs.extend(self.get_patchtst_configs(t))
            elif model_type == 'nhits':
                all_configs.extend(self.get_nhits_configs(t))
            elif model_type == 'tft':
                all_configs.extend(self.get_tft_configs(t))
            elif model_type is None:
                all_configs.extend(self.get_patchtst_configs(t))
                all_configs.extend(self.get_nhits_configs(t))
                all_configs.extend(self.get_tft_configs(t))
            else:
                raise ValueError(f"Unknown model type: {model_type}")

        return all_configs

    def get_grid_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the grid."""
        configs_per_model = len(PATCHTST_CONFIGS)
        total_per_target = configs_per_model * 3  # 3 models
        total_all = total_per_target * len(TARGETS)

        return {
            'targets': TARGETS,
            'models': ['patchtst', 'nhits', 'tft'],
            'configs_per_model': configs_per_model,
            'patchtst': {
                'count': len(PATCHTST_CONFIGS),
                'configs': list(PATCHTST_CONFIGS.keys()),
            },
            'nhits': {
                'count': len(NHITS_CONFIGS),
                'configs': list(NHITS_CONFIGS.keys()),
            },
            'tft': {
                'count': len(TFT_CONFIGS),
                'configs': list(TFT_CONFIGS.keys()),
            },
            'total_per_target': total_per_target,
            'total_all_targets': total_all,
            'shared_params': self.shared_params,
        }

    def get_recommended_configs(self) -> Dict[str, str]:
        """Get recommended config for each model type."""
        return {
            'patchtst': 'patchtst_balanced',
            'nhits': 'nhits_balanced',
            'tft': 'tft_balanced',
        }


def get_full_grid(
    model_type: Optional[str] = None,
    target: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Convenience function to get full grid.

    Args:
        model_type: 'patchtst', 'nhits', 'tft', or None for all
        target: 'price_real', 'consumption', or None for all

    Returns:
        List of configuration dictionaries
    """
    generator = GridConfigGeneratorV2()
    return generator.get_full_grid(model_type, target)


def print_grid_summary():
    """Print comprehensive grid summary."""
    generator = GridConfigGeneratorV2()
    summary = generator.get_grid_summary()

    print("\n" + "="*80)
    print("NEW EXPERIMENT V2 - HYPERPARAMETER CONFIGURATIONS")
    print("Target Hardware: NVIDIA RTX 5090 (32GB VRAM)")
    print("D-1 Safe Features for Day-Ahead Forecasting")
    print("="*80)

    print(f"\nTargets: {', '.join(summary['targets'])}")

    print(f"\nFeature Tiers (Price Forecasting):")
    for tier, strategy in PRICE_FEATURE_STRATEGIES.items():
        print(f"  {tier:15s} -> {strategy}")

    print(f"\nFeature Tiers (Consumption Forecasting):")
    for tier, strategy in CONSUMPTION_FEATURE_STRATEGIES.items():
        print(f"  {tier:15s} -> {strategy}")

    for model_type in ['patchtst', 'nhits', 'tft']:
        print(f"\n{'-'*80}")
        print(f"{model_type.upper()}: {summary[model_type]['count']} configurations")
        print(f"{'-'*80}")

        configs = PATCHTST_CONFIGS if model_type == 'patchtst' else \
                  NHITS_CONFIGS if model_type == 'nhits' else TFT_CONFIGS

        # Group by feature tier
        by_tier = {}
        for name, params in configs.items():
            tier = params.get('feature_tier', 'core')
            if tier not in by_tier:
                by_tier[tier] = []
            by_tier[tier].append((name, params.get('description', '')))

        for tier in ['minimal', 'ratios_only', 'core', 'extended', 'full']:
            if tier in by_tier:
                print(f"\n  [{tier.upper()}]")
                for name, desc in by_tier[tier]:
                    print(f"    {name}: {desc}")

    print(f"\n{'='*80}")
    print(f"SUMMARY:")
    print(f"  Configs per model: {summary['configs_per_model']}")
    print(f"  Configs per target: {summary['total_per_target']} (3 models x {summary['configs_per_model']})")
    print(f"  Total configs (all targets): {summary['total_all_targets']}")
    print(f"{'='*80}")

    print(f"\nShared parameters:")
    for key, value in summary['shared_params'].items():
        print(f"  {key}: {value}")


def main():
    """Main entry point."""
    print_grid_summary()


if __name__ == "__main__":
    main()
