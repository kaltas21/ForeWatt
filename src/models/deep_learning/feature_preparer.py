"""
Feature Preparation for Deep Learning Models
===========================================
Prepares fixed, versioned feature set for N-HiTS, TFT, and PatchTST.

Features:
- Lagged loads (consumption/price)
- Rolling statistics (24h, 168h windows)
- Fourier seasonality (daily, weekly, yearly)
- Calendar encodings (cyclical + binary)
- Weather covariates with lags

Author: ForeWatt Team
Date: November 2025
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeepLearningFeaturePreparer:
    """
    Feature preparer for deep learning time series models.

    Creates a fixed, versioned feature set optimized for deep learning:
    - Lagged target variables (autoregressive features)
    - Rolling statistics (trends and volatility)
    - Fourier seasonality (smooth periodic patterns)
    - Calendar encodings (holidays, weekends, time of day/week/year)
    - Weather covariates with temporal lags
    """

    def __init__(
        self,
        target: str = 'consumption',
        max_lag: int = 168,
        rolling_windows: List[int] = None,
        fourier_orders: Dict[str, int] = None
    ):
        """
        Initialize feature preparer.

        Args:
            target: Target variable ('consumption' or 'price_real')
            max_lag: Maximum lag to include (default: 168h = 1 week)
            rolling_windows: Rolling window sizes (default: [24, 168])
            fourier_orders: Fourier series orders for each period
        """
        self.target = target
        self.max_lag = max_lag
        self.rolling_windows = rolling_windows or [24, 168]

        # Fourier orders: K terms for each seasonality
        # K determines smoothness (higher K = more flexible)
        self.fourier_orders = fourier_orders or {
            'daily': 5,      # 24h period, 5 terms
            'weekly': 3,     # 168h period, 3 terms
            'yearly': 4      # 8760h period, 4 terms
        }

        self.feature_names = None
        self.feature_version = "v2_deep_learning"

    def create_fourier_features(
        self,
        df: pd.DataFrame,
        period: int,
        order: int,
        prefix: str
    ) -> pd.DataFrame:
        """
        Create Fourier series features for a given period.

        Fourier features capture smooth periodic patterns:
        sin(2π * k * t / period) and cos(2π * k * t / period) for k=1,...,order

        Args:
            df: DataFrame with datetime index
            period: Period in hours (24 for daily, 168 for weekly, 8760 for yearly)
            order: Number of Fourier terms (higher = more flexible)
            prefix: Feature name prefix

        Returns:
            DataFrame with Fourier features
        """
        features = pd.DataFrame(index=df.index)

        # Time index (hours since start)
        t = np.arange(len(df))

        for k in range(1, order + 1):
            # Sine and cosine components
            features[f'{prefix}_sin_{k}'] = np.sin(2 * np.pi * k * t / period)
            features[f'{prefix}_cos_{k}'] = np.cos(2 * np.pi * k * t / period)

        return features

    def get_feature_set(self, df: pd.DataFrame) -> List[str]:
        """
        Define the fixed feature set for deep learning models.

        Feature groups:
        1. Lagged target (8 lags: 1h, 2h, 3h, 6h, 12h, 24h, 48h, 168h)
        2. Rolling statistics (24h, 168h windows: mean, std, min, max)
        3. Fourier seasonality (daily, weekly, yearly)
        4. Calendar encodings (hour, dow, month cyclical + binary flags)
        5. Weather features (current + lags: 1h, 24h, 168h)
        6. Cross-domain features (for price: consumption, for demand: price)

        Args:
            df: Master dataset

        Returns:
            List of feature names
        """
        features = []

        # 1. LAGGED TARGET (Autoregressive features)
        lag_horizons = [1, 2, 3, 6, 12, 24, 48, 168]
        for lag in lag_horizons:
            features.append(f'{self.target}_lag_{lag}h')

        # 2. ROLLING STATISTICS (Trend and volatility)
        for window in self.rolling_windows:
            features.extend([
                f'{self.target}_rolling_mean_{window}h',
                f'{self.target}_rolling_std_{window}h',
                f'{self.target}_rolling_min_{window}h',
                f'{self.target}_rolling_max_{window}h'
            ])

        # 3. FOURIER SEASONALITY (will be created if not present)
        for period_name, order in self.fourier_orders.items():
            for k in range(1, order + 1):
                features.extend([
                    f'fourier_{period_name}_sin_{k}',
                    f'fourier_{period_name}_cos_{k}'
                ])

        # 4. CALENDAR ENCODINGS
        # Cyclical encodings (smooth continuous)
        features.extend([
            'hour_sin', 'hour_cos',
            'dow_sin_x', 'dow_cos_x',
            'month_sin', 'month_cos'
        ])

        # Binary flags (discrete)
        features.extend([
            'is_weekend_x',
            'is_holiday_day',
            'is_holiday_hour'
        ])

        # 5. WEATHER FEATURES
        # Current weather
        weather_vars = [
            'temp_national',
            'humidity_national',
            'wind_speed_national',
            'precipitation_national'
        ]
        features.extend(weather_vars)

        # Weather lags (1h, 24h, 168h)
        weather_lags = [1, 24, 168]
        for var in ['temp_national', 'temperature']:
            for lag in weather_lags:
                col_name = f'{var}_lag_{lag}h'
                if col_name in df.columns:
                    features.append(col_name)
                    break  # Avoid duplicates

        # Weather derived
        features.extend([
            'HDD', 'CDD',
            'heat_index',
            'temp_change_24h',
            'is_hot', 'is_cold'
        ])

        # 6. CROSS-DOMAIN FEATURES
        if self.target == 'consumption':
            # For demand forecasting, use price features
            features.extend([
                'price_real',
                'price_ptf_lag_24h',
                'price_ptf_lag_168h'
            ])
        else:  # price_real
            # For price forecasting, use consumption features
            features.extend([
                'consumption',
                'consumption_lag_24h',
                'consumption_lag_168h'
            ])

        # Filter to available features
        available = [f for f in features if f in df.columns or 'fourier' in f]

        return available

    def add_fourier_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Fourier seasonality features to dataframe.

        Args:
            df: Input dataframe with datetime index

        Returns:
            DataFrame with added Fourier features
        """
        df = df.copy()

        # Daily seasonality (24h period)
        if 'daily' in self.fourier_orders:
            fourier_daily = self.create_fourier_features(
                df, period=24,
                order=self.fourier_orders['daily'],
                prefix='fourier_daily'
            )
            df = pd.concat([df, fourier_daily], axis=1)

        # Weekly seasonality (168h period)
        if 'weekly' in self.fourier_orders:
            fourier_weekly = self.create_fourier_features(
                df, period=168,
                order=self.fourier_orders['weekly'],
                prefix='fourier_weekly'
            )
            df = pd.concat([df, fourier_weekly], axis=1)

        # Yearly seasonality (8760h period)
        if 'yearly' in self.fourier_orders:
            fourier_yearly = self.create_fourier_features(
                df, period=8760,
                order=self.fourier_orders['yearly'],
                prefix='fourier_yearly'
            )
            df = pd.concat([df, fourier_yearly], axis=1)

        return df

    def prepare_features(
        self,
        df: pd.DataFrame,
        add_fourier: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Prepare features for deep learning models.

        Args:
            df: Master dataset
            add_fourier: Whether to add Fourier features (default: True)

        Returns:
            Tuple of (X, y, feature_names)
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"PREPARING DEEP LEARNING FEATURES: {self.target}")
        logger.info(f"{'='*80}")

        # Add Fourier features if requested
        if add_fourier:
            df = self.add_fourier_features(df)
            logger.info("✓ Fourier seasonality features added")

        # Get fixed feature set
        feature_names = self.get_feature_set(df)

        # Filter to available features
        available_features = [f for f in feature_names if f in df.columns]
        missing_features = set(feature_names) - set(available_features)

        if missing_features:
            logger.warning(f"Missing {len(missing_features)} features:")
            for f in sorted(missing_features)[:5]:
                logger.warning(f"  - {f}")
            if len(missing_features) > 5:
                logger.warning(f"  ... and {len(missing_features) - 5} more")

        # Extract features and target
        X = df[available_features].copy()
        y = df[self.target].copy()

        # Store feature names
        self.feature_names = available_features

        logger.info(f"✓ Feature set prepared: {len(available_features)} features")
        logger.info(f"  - Target: {self.target}")
        logger.info(f"  - Samples: {len(X)}")
        logger.info(f"  - Version: {self.feature_version}")

        # Feature breakdown
        fourier_count = len([f for f in available_features if 'fourier' in f])
        lag_count = len([f for f in available_features if 'lag' in f])
        rolling_count = len([f for f in available_features if 'rolling' in f])

        logger.info(f"\nFeature breakdown:")
        logger.info(f"  - Fourier seasonality: {fourier_count}")
        logger.info(f"  - Lag features: {lag_count}")
        logger.info(f"  - Rolling statistics: {rolling_count}")
        logger.info(f"  - Other: {len(available_features) - fourier_count - lag_count - rolling_count}")

        return X, y, available_features

    def prepare_sequences(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        input_size: int,
        horizon: int = 24,
        stride: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare sequences for deep learning models.

        Creates sliding windows of input_size for features and horizon for target.

        Args:
            X: Features dataframe
            y: Target series
            input_size: Number of past time steps to use (lookback window)
            horizon: Number of future time steps to predict (default: 24h)
            stride: Step size for sliding window (default: 1)

        Returns:
            Tuple of (X_sequences, y_sequences, timestamps)
            - X_sequences: (n_samples, input_size, n_features)
            - y_sequences: (n_samples, horizon)
            - timestamps: (n_samples,) end timestamps for each sequence
        """
        n_features = X.shape[1]
        n_samples = (len(X) - input_size - horizon + 1) // stride

        X_sequences = np.zeros((n_samples, input_size, n_features))
        y_sequences = np.zeros((n_samples, horizon))
        timestamps = []

        for i in range(n_samples):
            start_idx = i * stride
            end_idx = start_idx + input_size
            target_end = end_idx + horizon

            if target_end <= len(X):
                X_sequences[i] = X.iloc[start_idx:end_idx].values
                y_sequences[i] = y.iloc[end_idx:target_end].values
                timestamps.append(X.index[end_idx - 1])

        timestamps = np.array(timestamps)

        logger.info(f"\nSequences prepared:")
        logger.info(f"  - Input size: {input_size}h")
        logger.info(f"  - Horizon: {horizon}h")
        logger.info(f"  - Stride: {stride}")
        logger.info(f"  - Total sequences: {n_samples}")
        logger.info(f"  - X shape: {X_sequences.shape}")
        logger.info(f"  - y shape: {y_sequences.shape}")

        return X_sequences, y_sequences, timestamps

    def get_feature_groups(self) -> Dict[str, List[str]]:
        """
        Get feature groups for analysis.

        Returns:
            Dictionary mapping group names to feature lists
        """
        if self.feature_names is None:
            raise ValueError("Features not prepared yet. Call prepare_features first.")

        groups = {
            'fourier': [f for f in self.feature_names if 'fourier' in f],
            'lags': [f for f in self.feature_names if 'lag' in f],
            'rolling': [f for f in self.feature_names if 'rolling' in f],
            'calendar': [f for f in self.feature_names if any(x in f for x in ['hour', 'dow', 'month', 'weekend', 'holiday'])],
            'weather': [f for f in self.feature_names if any(x in f for x in ['temp', 'humidity', 'wind', 'precip', 'HDD', 'CDD', 'heat'])]
        }

        return groups

    def print_feature_summary(self):
        """Print feature summary."""
        if self.feature_names is None:
            logger.error("Features not prepared yet")
            return

        groups = self.get_feature_groups()

        print(f"\n{'='*80}")
        print(f"DEEP LEARNING FEATURE SET: {self.target}")
        print(f"{'='*80}")
        print(f"Version: {self.feature_version}")
        print(f"Total features: {len(self.feature_names)}")
        print(f"\nFeature groups:")
        for group_name, features in groups.items():
            print(f"  {group_name:20s}: {len(features):3d} features")
        print(f"{'='*80}\n")
