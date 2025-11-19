"""
Deep Learning Model Hyperparameter Configurations
=================================================
Multiple parameter combinations for N-HiTS, TFT, and PatchTST models
to test different architectures and feature subsets.

Targets:
- consumption: Electricity demand (MWh)
- price_real: Day-ahead market price (PTF, inflation-adjusted TL/MWh)

Author: ForeWatt Team
Date: November 2025
"""

from typing import Dict, List, Any


class DeepLearningHyperparameterConfigs:
    """
    Pre-defined hyperparameter configurations for deep learning models.

    Each model has 5-10 configurations:
    1. Light: Fast training, fewer parameters
    2. Balanced: Good trade-off (recommended)
    3. Deep: More complex architecture
    4. Wide: More hidden units, fewer layers
    5. Narrow-Deep: Fewer units, more layers
    6. Price-Optimized: Specialized for price forecasting
    7. Demand-Optimized: Specialized for demand forecasting
    8. Ultra-Light: Minimal model for rapid prototyping
    9. Ultra-Deep: Maximum capacity for best performance
    10. Regularized: Strong regularization for generalization
    """

    @staticmethod
    def get_nhits_configs() -> Dict[str, Dict[str, Any]]:
        """
        N-HiTS (Neural Hierarchical Interpolation for Time Series) configurations.

        N-HiTS uses stacked blocks with multi-rate sampling for hierarchical
        interpolation of seasonality patterns.

        Returns:
            Dictionary with 10 configurations (7 original + 3 advanced)
        """
        return {
            'nhits_light': {
                'stack_types': ['identity', 'identity', 'identity'],
                'n_blocks': [1, 1, 1],  # 1 block per stack
                'n_pool_kernel_size': [2, 2, 1],
                'n_freq_downsample': [2, 1, 1],
                'hidden_size': 256,
                'n_mlp_layers': 2,
                'learning_rate': 1e-3,
                'batch_size': 128,
                'max_steps': 500,
                'early_stop_patience_steps': 50,
                'input_size': 168,
                'horizon': 24,
                'feature_selection': 'fourier_lags_only',  # Minimal features
                'description': 'Fast N-HiTS with small architecture'
            },

            'nhits_balanced': {
                'stack_types': ['identity', 'identity', 'identity'],
                'n_blocks': [2, 2, 2],  # 2 blocks per stack
                'n_pool_kernel_size': [4, 4, 1],
                'n_freq_downsample': [4, 2, 1],
                'hidden_size': 512,
                'n_mlp_layers': 2,
                'learning_rate': 1e-3,
                'batch_size': 64,
                'max_steps': 1000,
                'early_stop_patience_steps': 100,
                'input_size': 168,
                'horizon': 24,
                'feature_selection': 'standard_dl',  # Fourier + lags + rolling
                'description': 'Balanced N-HiTS (recommended)'
            },

            'nhits_deep': {
                'stack_types': ['identity', 'identity', 'identity'],
                'n_blocks': [3, 3, 3],  # 3 blocks per stack
                'n_pool_kernel_size': [8, 4, 1],
                'n_freq_downsample': [8, 4, 1],
                'hidden_size': 1024,
                'n_mlp_layers': 3,
                'learning_rate': 5e-4,
                'batch_size': 16,  # Reduced from 32 to avoid OOM
                'max_steps': 2000,
                'early_stop_patience_steps': 150,
                'input_size': 336,  # 2 weeks lookback
                'horizon': 24,
                'feature_selection': 'all',
                'description': 'Deep N-HiTS with large lookback window'
            },

            'nhits_wide': {
                'stack_types': ['identity', 'identity', 'identity'],
                'n_blocks': [2, 2, 2],
                'n_pool_kernel_size': [4, 4, 1],
                'n_freq_downsample': [4, 2, 1],
                'hidden_size': 1024,  # Wide layers
                'n_mlp_layers': 2,    # Fewer layers
                'learning_rate': 8e-4,
                'batch_size': 64,
                'max_steps': 1500,
                'early_stop_patience_steps': 100,
                'input_size': 168,
                'horizon': 24,
                'feature_selection': 'standard_dl',
                'description': 'Wide N-HiTS with large hidden size'
            },

            'nhits_narrow_deep': {
                'stack_types': ['identity', 'identity', 'identity'],
                'n_blocks': [3, 3, 3],
                'n_pool_kernel_size': [8, 4, 1],
                'n_freq_downsample': [8, 4, 1],
                'hidden_size': 256,  # Narrow layers
                'n_mlp_layers': 4,   # More layers
                'learning_rate': 1e-3,
                'batch_size': 128,
                'max_steps': 1500,
                'early_stop_patience_steps': 100,
                'input_size': 168,
                'horizon': 24,
                'feature_selection': 'fourier_heavy',
                'description': 'Narrow-deep N-HiTS relying on Fourier features'
            },

            'nhits_price_optimized': {
                'stack_types': ['identity', 'identity', 'identity'],
                'n_blocks': [2, 2, 2],
                'n_pool_kernel_size': [4, 4, 1],
                'n_freq_downsample': [4, 2, 1],
                'hidden_size': 512,
                'n_mlp_layers': 3,
                'learning_rate': 8e-4,
                'batch_size': 64,
                'max_steps': 1500,
                'early_stop_patience_steps': 120,
                'input_size': 168,
                'horizon': 24,
                'feature_selection': 'price_focused_dl',  # Price lags + consumption + FX
                'description': 'N-HiTS optimized for price volatility'
            },

            'nhits_demand_optimized': {
                'stack_types': ['identity', 'identity', 'identity'],
                'n_blocks': [2, 2, 2],
                'n_pool_kernel_size': [4, 4, 1],
                'n_freq_downsample': [4, 2, 1],
                'hidden_size': 512,
                'n_mlp_layers': 2,
                'learning_rate': 1e-3,
                'batch_size': 64,
                'max_steps': 1200,
                'early_stop_patience_steps': 100,
                'input_size': 168,
                'horizon': 24,
                'feature_selection': 'demand_focused_dl',  # Consumption lags + weather + calendar
                'description': 'N-HiTS optimized for demand patterns'
            },

            'nhits_ultra_light': {
                'stack_types': ['identity', 'identity'],  # Two identity stacks for speed
                'n_blocks': [1, 1],
                'n_pool_kernel_size': [2, 1],
                'n_freq_downsample': [2, 1],
                'hidden_size': 128,
                'n_mlp_layers': 1,
                'learning_rate': 2e-3,  # Higher LR for faster convergence
                'batch_size': 256,  # Large batches for speed
                'max_steps': 300,
                'early_stop_patience_steps': 30,
                'input_size': 72,  # Shorter lookback (3 days)
                'horizon': 24,
                'feature_selection': 'fourier_lags_only',
                'description': 'Ultra-fast N-HiTS for rapid prototyping (< 2 min training)'
            },

            'nhits_ultra_deep': {
                'stack_types': ['identity', 'identity', 'identity'],
                'n_blocks': [4, 4, 4],  # Maximum blocks
                'n_pool_kernel_size': [16, 8, 1],  # Aggressive pooling
                'n_freq_downsample': [16, 8, 1],
                'hidden_size': 2048,  # Very large
                'n_mlp_layers': 4,
                'learning_rate': 3e-4,  # Lower LR for stability
                'batch_size': 16,  # Small batches for large model
                'max_steps': 3000,
                'early_stop_patience_steps': 200,
                'input_size': 504,  # 3 weeks lookback
                'horizon': 24,
                'feature_selection': 'all',
                'dropout': 0.1,  # Add dropout for regularization
                'description': 'Maximum capacity N-HiTS for best possible performance'
            },

            'nhits_regularized': {
                'stack_types': ['identity', 'identity', 'identity'],
                'n_blocks': [2, 2, 2],
                'n_pool_kernel_size': [4, 4, 1],
                'n_freq_downsample': [4, 2, 1],
                'hidden_size': 384,
                'n_mlp_layers': 2,
                'learning_rate': 5e-4,
                'batch_size': 64,
                'max_steps': 1500,
                'early_stop_patience_steps': 150,
                'input_size': 168,
                'horizon': 24,
                'feature_selection': 'standard_dl',
                'dropout': 0.2,  # Heavy dropout
                'weight_decay': 1e-4,  # L2 regularization
                'description': 'N-HiTS with strong regularization for better generalization'
            }
        }

    @staticmethod
    def get_tft_configs() -> Dict[str, Dict[str, Any]]:
        """
        TFT (Temporal Fusion Transformer) configurations.

        TFT uses attention mechanisms for interpretable multi-horizon forecasting
        with variable selection networks.

        Returns:
            Dictionary with 10 configurations (7 original + 3 advanced)
        """
        return {
            'tft_light': {
                'hidden_size': 32,
                'n_rnn_layers': 1,
                'n_head': 2,
                'dropout': 0.2,
                'learning_rate': 1e-3,
                'batch_size': 128,
                'max_steps': 500,
                'early_stop_patience_steps': 50,
                'input_size': 168,
                'horizon': 24,
                'feature_selection': 'fourier_lags_only',
                'description': 'Lightweight TFT for fast training'
            },

            'tft_balanced': {
                'hidden_size': 64,
                'n_rnn_layers': 2,
                'n_head': 4,
                'dropout': 0.1,
                'learning_rate': 1e-3,
                'batch_size': 64,
                'max_steps': 1000,
                'early_stop_patience_steps': 100,
                'input_size': 168,
                'horizon': 24,
                'feature_selection': 'standard_dl',
                'description': 'Balanced TFT (recommended)'
            },

            'tft_deep': {
                'hidden_size': 128,
                'n_rnn_layers': 3,
                'n_head': 8,
                'dropout': 0.1,
                'learning_rate': 5e-4,
                'batch_size': 16,  # Reduced from 32 to avoid OOM
                'max_steps': 2000,
                'early_stop_patience_steps': 150,
                'input_size': 336,  # 2 weeks
                'horizon': 24,
                'feature_selection': 'all',
                'description': 'Deep TFT with multi-head attention'
            },

            'tft_wide': {
                'hidden_size': 128,  # Wide
                'n_rnn_layers': 2,    # Fewer layers
                'n_head': 8,
                'dropout': 0.15,
                'learning_rate': 8e-4,
                'batch_size': 64,
                'max_steps': 1500,
                'early_stop_patience_steps': 100,
                'input_size': 168,
                'horizon': 24,
                'feature_selection': 'standard_dl',
                'description': 'Wide TFT with large hidden layers'
            },

            'tft_narrow_deep': {
                'hidden_size': 32,   # Narrow
                'n_rnn_layers': 4,    # More layers
                'n_head': 4,
                'dropout': 0.1,
                'learning_rate': 1e-3,
                'batch_size': 128,
                'max_steps': 1500,
                'early_stop_patience_steps': 100,
                'input_size': 168,
                'horizon': 24,
                'feature_selection': 'fourier_heavy',
                'description': 'Narrow-deep TFT with more LSTM layers'
            },

            'tft_price_optimized': {
                'hidden_size': 96,
                'n_rnn_layers': 3,
                'n_head': 6,
                'dropout': 0.15,
                'learning_rate': 8e-4,
                'batch_size': 64,
                'max_steps': 1500,
                'early_stop_patience_steps': 120,
                'input_size': 168,
                'horizon': 24,
                'feature_selection': 'price_focused_dl',
                'description': 'TFT for price with attention on volatility'
            },

            'tft_demand_optimized': {
                'hidden_size': 80,
                'n_rnn_layers': 2,
                'n_head': 5,
                'dropout': 0.1,
                'learning_rate': 1e-3,
                'batch_size': 64,
                'max_steps': 1200,
                'early_stop_patience_steps': 100,
                'input_size': 168,
                'horizon': 24,
                'feature_selection': 'demand_focused_dl',
                'description': 'TFT for demand with weather attention'
            },

            'tft_ultra_light': {
                'hidden_size': 16,  # Minimal hidden size
                'n_rnn_layers': 1,
                'n_head': 1,  # Single attention head
                'dropout': 0.1,
                'learning_rate': 2e-3,
                'batch_size': 256,
                'max_steps': 300,
                'early_stop_patience_steps': 30,
                'input_size': 72,  # 3 days
                'horizon': 24,
                'feature_selection': 'fourier_lags_only',
                'description': 'Ultra-fast TFT for rapid experimentation'
            },

            'tft_ultra_deep': {
                'hidden_size': 256,  # Very large
                'n_rnn_layers': 4,  # Deep LSTM stack
                'n_head': 16,  # Many attention heads
                'dropout': 0.15,
                'learning_rate': 3e-4,
                'batch_size': 16,
                'max_steps': 3000,
                'early_stop_patience_steps': 200,
                'input_size': 504,  # 3 weeks
                'horizon': 24,
                'feature_selection': 'all',
                'description': 'Maximum capacity TFT with deep attention'
            },

            'tft_regularized': {
                'hidden_size': 96,
                'n_rnn_layers': 2,
                'n_head': 6,
                'dropout': 0.3,  # Heavy dropout
                'learning_rate': 5e-4,
                'batch_size': 64,
                'max_steps': 1500,
                'early_stop_patience_steps': 150,
                'input_size': 168,
                'horizon': 24,
                'feature_selection': 'standard_dl',
                'weight_decay': 1e-4,
                'description': 'TFT with strong regularization and dropout'
            }
        }

    @staticmethod
    def get_patchtst_configs() -> Dict[str, Dict[str, Any]]:
        """
        PatchTST (Patched Time Series Transformer) configurations.

        PatchTST uses patch-based encoding for efficient long-sequence modeling
        with channel independence.

        Returns:
            Dictionary with 10 configurations (7 original + 3 advanced)
        """
        return {
            'patchtst_light': {
                'patch_len': 16,
                'stride': 8,
                'encoder_layers': 2,
                'hidden_size': 128,
                'n_heads': 4,
                'dropout': 0.2,
                'learning_rate': 1e-3,
                'batch_size': 128,
                'max_steps': 500,
                'early_stop_patience_steps': 50,
                'input_size': 168,
                'horizon': 24,
                'feature_selection': 'fourier_lags_only',
                'description': 'Lightweight PatchTST with small patches'
            },

            'patchtst_balanced': {
                'patch_len': 24,
                'stride': 12,
                'encoder_layers': 3,
                'hidden_size': 256,
                'n_heads': 8,
                'dropout': 0.1,
                'learning_rate': 1e-3,
                'batch_size': 64,
                'max_steps': 1000,
                'early_stop_patience_steps': 100,
                'input_size': 168,
                'horizon': 24,
                'feature_selection': 'standard_dl',
                'description': 'Balanced PatchTST (recommended)'
            },

            'patchtst_deep': {
                'patch_len': 24,
                'stride': 12,
                'encoder_layers': 6,
                'hidden_size': 512,
                'n_heads': 16,
                'dropout': 0.1,
                'learning_rate': 5e-4,
                'batch_size': 16,  # Reduced from 32 to avoid OOM
                'max_steps': 2000,
                'early_stop_patience_steps': 150,
                'input_size': 336,  # 2 weeks
                'horizon': 24,
                'feature_selection': 'all',
                'description': 'Deep PatchTST with many transformer layers'
            },

            'patchtst_wide': {
                'patch_len': 32,  # Larger patches
                'stride': 16,
                'encoder_layers': 3,
                'hidden_size': 512,  # Wide
                'n_heads': 16,
                'dropout': 0.15,
                'learning_rate': 8e-4,
                'batch_size': 64,
                'max_steps': 1500,
                'early_stop_patience_steps': 100,
                'input_size': 168,
                'horizon': 24,
                'feature_selection': 'standard_dl',
                'description': 'Wide PatchTST with large hidden size'
            },

            'patchtst_narrow_deep': {
                'patch_len': 12,  # Smaller patches
                'stride': 6,
                'encoder_layers': 6,   # More layers
                'hidden_size': 128,  # Narrow
                'n_heads': 4,
                'dropout': 0.1,
                'learning_rate': 1e-3,
                'batch_size': 128,
                'max_steps': 1500,
                'early_stop_patience_steps': 100,
                'input_size': 168,
                'horizon': 24,
                'feature_selection': 'fourier_heavy',
                'description': 'Narrow-deep PatchTST with small patches'
            },

            'patchtst_price_optimized': {
                'patch_len': 20,
                'stride': 10,
                'encoder_layers': 4,
                'hidden_size': 384,
                'n_heads': 12,
                'dropout': 0.15,
                'learning_rate': 8e-4,
                'batch_size': 64,
                'max_steps': 1500,
                'early_stop_patience_steps': 120,
                'input_size': 168,
                'horizon': 24,
                'feature_selection': 'price_focused_dl',
                'description': 'PatchTST for price with adaptive patching'
            },

            'patchtst_demand_optimized': {
                'patch_len': 24,
                'stride': 12,
                'encoder_layers': 3,
                'hidden_size': 320,
                'n_heads': 10,
                'dropout': 0.1,
                'learning_rate': 1e-3,
                'batch_size': 64,
                'max_steps': 1200,
                'early_stop_patience_steps': 100,
                'input_size': 168,
                'horizon': 24,
                'feature_selection': 'demand_focused_dl',
                'description': 'PatchTST for demand with seasonal patches'
            },

            'patchtst_ultra_light': {
                'patch_len': 8,  # Small patches
                'stride': 4,
                'encoder_layers': 1,  # Single transformer layer
                'hidden_size': 64,
                'n_heads': 2,
                'dropout': 0.1,
                'learning_rate': 2e-3,
                'batch_size': 256,
                'max_steps': 300,
                'early_stop_patience_steps': 30,
                'input_size': 72,  # 3 days
                'horizon': 24,
                'feature_selection': 'fourier_lags_only',
                'description': 'Ultra-fast PatchTST for quick iterations'
            },

            'patchtst_ultra_deep': {
                'patch_len': 48,  # Large patches
                'stride': 24,
                'encoder_layers': 12,  # Very deep transformer
                'hidden_size': 1024,  # Very wide
                'n_heads': 32,  # Many heads
                'dropout': 0.15,
                'learning_rate': 2e-4,
                'batch_size': 8,  # Small batches for huge model
                'max_steps': 3000,
                'early_stop_patience_steps': 200,
                'input_size': 672,  # 4 weeks
                'horizon': 24,
                'feature_selection': 'all',
                'description': 'Maximum capacity PatchTST with deep attention stack'
            },

            'patchtst_regularized': {
                'patch_len': 20,
                'stride': 10,
                'encoder_layers': 4,
                'hidden_size': 256,
                'n_heads': 8,
                'dropout': 0.25,  # Heavy dropout
                'learning_rate': 5e-4,
                'batch_size': 64,
                'max_steps': 1500,
                'early_stop_patience_steps': 150,
                'input_size': 168,
                'horizon': 24,
                'feature_selection': 'standard_dl',
                'weight_decay': 1e-4,
                'description': 'PatchTST with strong regularization for robustness'
            }
        }

    @staticmethod
    def get_all_deep_learning_configs() -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Get all deep learning model configurations.

        Returns:
            Dictionary with all configurations for all models
        """
        return {
            'nhits': DeepLearningHyperparameterConfigs.get_nhits_configs(),
            'tft': DeepLearningHyperparameterConfigs.get_tft_configs(),
            'patchtst': DeepLearningHyperparameterConfigs.get_patchtst_configs()
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
                'nhits': 'nhits_demand_optimized',
                'tft': 'tft_demand_optimized',
                'patchtst': 'patchtst_demand_optimized'
            }
        elif target == 'price_real':
            return {
                'nhits': 'nhits_price_optimized',
                'tft': 'tft_price_optimized',
                'patchtst': 'patchtst_price_optimized'
            }
        else:
            # Default to balanced
            return {
                'nhits': 'nhits_balanced',
                'tft': 'tft_balanced',
                'patchtst': 'patchtst_balanced'
            }


# Feature selection strategies for deep learning
DL_FEATURE_SELECTION_STRATEGIES = {
    # General strategies
    'all': 'All 106 features (Fourier + lags + rolling + weather + calendar + cross-domain)',
    'standard_dl': 'Fourier (24 features) + Lags (8) + Rolling (32) + Core weather (15) + Calendar (9) ≈ 88 features',
    'fourier_lags_only': 'Fourier seasonality (24) + Target lags (8) + Calendar (9) ≈ 41 features (fast)',
    'fourier_heavy': 'Extended Fourier (daily/weekly/yearly) + Lags + Minimal weather ≈ 50 features',

    # Target-specific strategies
    'price_focused_dl': '''
        - All Fourier features (24)
        - Price lags: 1h, 2h, 3h, 6h, 12h, 24h, 48h, 168h
        - Price rolling stats: 24h, 168h (mean, std, min, max)
        - Consumption features (cross-domain): consumption, lag_24h, lag_168h
        - FX features: USD_TRY, EUR_TRY, FX_basket (if available)
        - Calendar: hour_sin/cos, dow_sin/cos, is_weekend, is_holiday
        - Core weather: temp_national, HDD, CDD
        Total ≈ 60-70 features
    ''',

    'demand_focused_dl': '''
        - All Fourier features (24)
        - Consumption lags: 1h, 2h, 3h, 6h, 12h, 24h, 48h, 168h
        - Consumption rolling stats: 24h, 168h (mean, std, min, max)
        - Weather features (extended):
          * Current: temp, humidity, wind, precipitation
          * Lags: temp_lag_1h, temp_lag_24h, temp_lag_168h
          * Derived: HDD, CDD, heat_index, wind_chill
          * Binary: is_hot, is_cold, is_raining
        - Temperature rolling stats: 24h, 168h
        - Calendar: hour_sin/cos, dow_sin/cos, month_sin/cos, is_weekend, is_holiday
        - Price (cross-domain): price_real, price_lag_24h
        Total ≈ 70-80 features
    '''
}


# Optuna search spaces for Bayesian optimization
OPTUNA_SEARCH_SPACES = {
    'nhits': {
        'n_blocks': [1, 2, 3, 4],  # Added 4
        'hidden_size': [128, 256, 512, 1024, 2048],  # Added 2048
        'n_mlp_layers': [1, 2, 3, 4],
        'learning_rate': (1e-4, 2e-3, 'log-uniform'),  # Extended range
        'batch_size': [16, 32, 64, 128, 256],  # Added 16
        'max_steps': [300, 500, 1000, 1500, 2000, 3000],  # More options
        'pool_kernel_size': [[2, 2, 1], [4, 4, 1], [8, 4, 1], [16, 8, 1]],  # Added aggressive
        'freq_downsample': [[2, 1, 1], [4, 2, 1], [8, 4, 1], [16, 8, 1]],  # Added aggressive
        'dropout': (0.0, 0.3, 'uniform'),  # Added dropout
        'weight_decay': (0.0, 1e-3, 'log-uniform'),  # Added weight decay
        'input_size': [72, 168, 336, 504]  # Added input size variation
    },

    'tft': {
        'hidden_size': [16, 32, 64, 128, 256],  # Added 16 and 256
        'n_rnn_layers': [1, 2, 3, 4],
        'n_head': [1, 2, 4, 8, 16],  # Added 1
        'dropout': (0.05, 0.4, 'uniform'),  # Extended range
        'learning_rate': (1e-4, 2e-3, 'log-uniform'),
        'batch_size': [16, 32, 64, 128, 256],  # Added 16
        'max_steps': [300, 500, 1000, 1500, 2000, 3000],
        'weight_decay': (0.0, 1e-3, 'log-uniform'),  # Added weight decay
        'input_size': [72, 168, 336, 504]
    },

    'patchtst': {
        'patch_len': [8, 12, 16, 20, 24, 32, 48],  # Added 20 and 48
        'stride': [4, 6, 8, 10, 12, 16, 24],  # More granular
        'encoder_layers': [1, 2, 3, 4, 6, 8, 12],  # Added 1 and 12
        'hidden_size': [64, 128, 256, 384, 512, 1024],  # Added 64 and 1024
        'n_heads': [2, 4, 8, 12, 16, 32],  # Added 2 and 32
        'dropout': (0.05, 0.35, 'uniform'),
        'learning_rate': (1e-4, 2e-3, 'log-uniform'),
        'batch_size': [8, 16, 32, 64, 128, 256],  # Added 8
        'max_steps': [300, 500, 1000, 1500, 2000, 3000],
        'weight_decay': (0.0, 1e-3, 'log-uniform'),  # Added weight decay
        'input_size': [72, 168, 336, 504, 672]  # Added more options
    }
}


def print_config_summary():
    """Print summary of all available deep learning configurations."""
    configs = DeepLearningHyperparameterConfigs.get_all_deep_learning_configs()

    print("=" * 100)
    print("DEEP LEARNING MODEL HYPERPARAMETER CONFIGURATIONS")
    print("=" * 100)
    print(f"\nTotal configurations: {sum(len(v) for v in configs.values())}")
    print(f"Models: {len(configs)}")

    for model_type, model_configs in configs.items():
        print(f"\n{'─' * 100}")
        print(f"{model_type.upper()}: {len(model_configs)} configurations")
        print(f"{'─' * 100}")

        for config_name, config in model_configs.items():
            desc = config.get('description', 'No description')
            feature_sel = config.get('feature_selection', 'standard_dl')
            input_size = config.get('input_size', 168)
            horizon = config.get('horizon', 24)
            batch_size = config.get('batch_size', 64)
            max_steps = config.get('max_steps', 1000)

            print(f"\n  {config_name}:")
            print(f"    Description: {desc}")
            print(f"    Features: {feature_sel}")
            print(f"    Input/Horizon: {input_size}h → {horizon}h")
            print(f"    Training: {max_steps} steps, batch_size={batch_size}")

            # Model-specific parameters
            if model_type == 'nhits':
                print(f"    Architecture: blocks={config.get('n_blocks')}, hidden={config.get('hidden_size')}, mlp_layers={config.get('n_mlp_layers')}")
            elif model_type == 'tft':
                print(f"    Architecture: n_rnn_layers={config.get('n_rnn_layers')}, hidden={config.get('hidden_size')}, n_head={config.get('n_head')}")
            elif model_type == 'patchtst':
                print(f"    Architecture: patch={config.get('patch_len')}, stride={config.get('stride')}, encoder_layers={config.get('encoder_layers')}, hidden={config.get('hidden_size')}")

    print(f"\n{'=' * 100}")
    print("FEATURE SELECTION STRATEGIES")
    print(f"{'=' * 100}")
    for strategy_name, description in DL_FEATURE_SELECTION_STRATEGIES.items():
        print(f"\n{strategy_name}:")
        print(f"  {description}")


if __name__ == '__main__':
    print_config_summary()
