"""
Robust Deflator Index Builder (DID_index)
==========================================
Builds a Real Price Deflator Index resilient to Turkish hyperinflation
and structural breaks of 2021-2023.

Methodology:
1. Smart Drop: Spearman correlation filtering for unstable features
2. RobustScaler: Median/IQR scaling to handle Dec 2021 volatility spike
3. HuberRegressor: Robust regression to follow trend, not outliers

Key Improvements over StandardScaler + OLS:
- StandardScaler is dominated by extreme values (Dec 2021 spike)
- OLS fits outliers; HuberRegressor down-weights them
- Smart Drop removes features with sign-flip (unorthodox policy)

Author: ForeWatt Team
Date: December 2025
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import HuberRegressor
from sklearn.decomposition import FactorAnalysis, PCA
import statsmodels.api as sm

# Project setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# File paths
BRONZE_PATH_PARQUET = PROJECT_ROOT / "data/bronze/macro/macro_evds_2020-01-01_2025-10-31.parquet"
BRONZE_PATH_CSV = PROJECT_ROOT / "data/bronze/macro/macro_evds_raw.csv"
SILVER_DIR = PROJECT_ROOT / "data/silver/macro"
BASE_MONTH = "2022-01"


class RobustDeflatorBuilder:
    """
    Builds a robust Real Price Deflator Index (DID_index) for Turkish electricity prices.

    Handles:
    - Hyperinflation periods (2021-2023)
    - Structural breaks in monetary policy
    - Unorthodox interest rate policy effects
    """

    def __init__(
        self,
        anchor_col: str = 'TUFE',
        correlation_threshold: float = 0.3,
        sign_flip_window: int = 12,
        huber_epsilon: float = 1.35,
        random_state: int = 42
    ):
        """
        Initialize RobustDeflatorBuilder.

        Args:
            anchor_col: Primary inflation anchor (TUFE = CPI)
            correlation_threshold: Min Spearman correlation to keep feature
            sign_flip_window: Rolling window to detect correlation sign flips
            huber_epsilon: HuberRegressor epsilon (1.35 = default, lower = more robust)
            random_state: Random seed for reproducibility
        """
        self.anchor_col = anchor_col
        self.correlation_threshold = correlation_threshold
        self.sign_flip_window = sign_flip_window
        self.huber_epsilon = huber_epsilon
        self.random_state = random_state

        self.scaler: Optional[RobustScaler] = None
        self.regressor: Optional[HuberRegressor] = None
        self.kept_features: List[str] = []
        self.dropped_features: Dict[str, str] = {}

    def load_macro_data(self) -> pd.DataFrame:
        """Load macro economic data from bronze layer."""
        if BRONZE_PATH_PARQUET.exists():
            logger.info(f"Loading from Parquet: {BRONZE_PATH_PARQUET}")
            df = pd.read_parquet(BRONZE_PATH_PARQUET)
        elif BRONZE_PATH_CSV.exists():
            logger.info(f"Loading from CSV: {BRONZE_PATH_CSV}")
            df = pd.read_csv(BRONZE_PATH_CSV)
        else:
            raise FileNotFoundError(
                f"Bronze EVDS data not found. Run evds_fetcher.py first.\n"
                f"Expected: {BRONZE_PATH_PARQUET}"
            )

        logger.info(f"Loaded macro data: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        return df

    def smart_drop(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        anchor_col: str
    ) -> Tuple[List[str], Dict[str, str]]:
        """
        Smart feature dropping based on Spearman correlation analysis.

        Drops features that:
        1. Have low correlation with anchor (< threshold)
        2. Show correlation sign flip (unorthodox policy effects)

        Args:
            df: Input dataframe with macro features
            feature_cols: List of feature columns to evaluate
            anchor_col: Column to use as correlation anchor

        Returns:
            Tuple of (kept_features, dropped_reasons)
        """
        logger.info(f"\n{'='*60}")
        logger.info("SMART DROP: Spearman Correlation Filtering")
        logger.info(f"{'='*60}")
        logger.info(f"Anchor: {anchor_col}")
        logger.info(f"Threshold: {self.correlation_threshold}")

        anchor = df[anchor_col].dropna()
        kept_features = []
        dropped_reasons = {}

        for col in feature_cols:
            if col == anchor_col:
                continue

            feature = df[col].dropna()

            # Align indices
            common_idx = anchor.index.intersection(feature.index)
            if len(common_idx) < 10:
                dropped_reasons[col] = "Insufficient data"
                logger.warning(f"  DROPPED {col}: Insufficient data")
                continue

            # Calculate Spearman correlation
            corr, p_value = stats.spearmanr(
                anchor.loc[common_idx],
                feature.loc[common_idx]
            )

            # Check for sign flip
            has_sign_flip = self._detect_sign_flip(
                anchor.loc[common_idx],
                feature.loc[common_idx]
            )

            # Decision logic
            if abs(corr) < self.correlation_threshold:
                dropped_reasons[col] = f"Low correlation: {corr:.3f}"
                logger.warning(f"  DROPPED {col}: Low correlation ({corr:.3f})")
            elif has_sign_flip:
                dropped_reasons[col] = "Sign flip detected (policy break)"
                logger.warning(f"  DROPPED {col}: Correlation sign flip")
            else:
                kept_features.append(col)
                logger.info(f"  KEPT {col}: Spearman={corr:.3f}, p={p_value:.4f}")

        self.kept_features = kept_features
        self.dropped_features = dropped_reasons

        return kept_features, dropped_reasons

    def _detect_sign_flip(
        self,
        anchor: pd.Series,
        feature: pd.Series
    ) -> bool:
        """Detect if correlation sign flips between first/second half of data."""
        if len(anchor) < self.sign_flip_window * 2:
            return False

        mid = len(anchor) // 2

        corr_first, _ = stats.spearmanr(anchor.iloc[:mid], feature.iloc[:mid])
        corr_second, _ = stats.spearmanr(anchor.iloc[mid:], feature.iloc[mid:])

        # Check for significant sign flip
        if abs(corr_first) > 0.3 and abs(corr_second) > 0.3:
            if np.sign(corr_first) != np.sign(corr_second):
                logger.debug(f"Sign flip: first={corr_first:.3f}, second={corr_second:.3f}")
                return True

        return False

    def robust_scale(self, X: np.ndarray) -> np.ndarray:
        """
        Apply RobustScaler (Median/IQR) to handle hyperinflation volatility.

        CRITICAL: StandardScaler would be dominated by Dec 2021 spike,
        crushing the variance of normal years.
        """
        logger.info("Applying RobustScaler (Median/IQR normalization)")

        self.scaler = RobustScaler(
            with_centering=True,
            with_scaling=True,
            quantile_range=(25.0, 75.0)
        )

        return self.scaler.fit_transform(X)

    def calibrate_with_huber(
        self,
        factor: np.ndarray,
        target: np.ndarray
    ) -> Tuple[np.ndarray, float, float]:
        """
        Calibrate latent factor to inflation scale using HuberRegressor.

        HuberRegressor is robust to outliers - follows trend, not extreme spikes.

        Returns:
            Tuple of (calibrated_values, slope, intercept)
        """
        logger.info(f"\n{'='*60}")
        logger.info("HUBER CALIBRATION: Robust Regression")
        logger.info(f"{'='*60}")
        logger.info(f"Huber epsilon: {self.huber_epsilon}")

        self.regressor = HuberRegressor(
            epsilon=self.huber_epsilon,
            max_iter=1000,
            fit_intercept=True
        )

        X = factor.reshape(-1, 1)
        self.regressor.fit(X, target)

        slope = self.regressor.coef_[0]
        intercept = self.regressor.intercept_

        logger.info(f"  Slope: {slope:.6f}")
        logger.info(f"  Intercept: {intercept:.6f}")

        calibrated = self.regressor.predict(X)

        return calibrated, slope, intercept

    def build(self, save: bool = True) -> pd.DataFrame:
        """
        Build the robust Deflator Index.

        Steps:
        1. Load data
        2. Compute growth rates
        3. Smart drop unstable features
        4. RobustScaler normalization
        5. PCA for latent factor
        6. HuberRegressor calibration
        7. Build cumulative index

        Returns:
            DataFrame with DID_index
        """
        logger.info("\n" + "="*70)
        logger.info("ROBUST DEFLATOR INDEX BUILDER")
        logger.info("="*70)

        # 1. Load data
        df = self.load_macro_data()
        df = df.sort_values("DATE").reset_index(drop=True)

        # 2. Create output dataframe with growth rates
        out = pd.DataFrame({"DATE": df["DATE"]})

        if "TUFE" in df.columns:
            out["TUFE_mom"] = df["TUFE"].pct_change()
            out["TUFE_yoy"] = df["TUFE"].pct_change(12)
        if "UFE" in df.columns:
            out["UFE_mom"] = df["UFE"].pct_change()
        if "M2" in df.columns:
            out["M2_yoy"] = df["M2"].pct_change(12)
        if "TL_FAIZ" in df.columns:
            out["TL_FAIZ_lvl"] = df["TL_FAIZ"]

        # Drop rows with NaN
        out = out.dropna()

        # 3. Smart drop unstable features
        feature_cols = [c for c in out.columns if c not in ["DATE", "TUFE_mom"]]
        anchor_col = "TUFE_mom"

        kept_features, dropped = self.smart_drop(out, feature_cols, anchor_col)

        # Build feature matrix (anchor + kept features)
        all_features = [anchor_col] + kept_features
        X = out[all_features].values

        # 4. RobustScaler normalization
        Z = self.robust_scale(X)

        # 5. Extract latent factor using PCA
        logger.info("\nExtracting latent inflation factor (PCA)...")
        pca = PCA(n_components=1, random_state=self.random_state)
        factor_raw = pca.fit_transform(Z).ravel()

        logger.info(f"  Explained variance: {pca.explained_variance_ratio_[0]:.4f}")

        # Sign correction (should correlate positively with TUFE_mom)
        if np.corrcoef(factor_raw, Z[:, 0])[0, 1] < 0:
            factor_raw *= -1
            logger.info("  Applied sign correction")

        out["DID_factor_raw"] = np.nan
        out.loc[out.index[:len(factor_raw)], "DID_factor_raw"] = factor_raw

        # 6. HuberRegressor calibration
        target = out[anchor_col].iloc[:len(factor_raw)].values
        pi_hat, slope, intercept = self.calibrate_with_huber(factor_raw, target)

        out["pi_hat_monthly"] = np.nan
        out.loc[out.index[:len(pi_hat)], "pi_hat_monthly"] = pi_hat

        # 7. Build cumulative DID index
        pi = out["pi_hat_monthly"].fillna(0.0)
        did_cumulative = (1.0 + pi).cumprod()

        # Base at BASE_MONTH = 100
        base_mask = out["DATE"] == BASE_MONTH
        if base_mask.any():
            base_val = did_cumulative.loc[base_mask].iloc[0]
        else:
            base_val = did_cumulative.iloc[0]

        out["DID_index"] = 100.0 * did_cumulative / base_val

        logger.info(f"\n{'='*70}")
        logger.info("ROBUST DEFLATOR BUILD COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"DID_index range: {out['DID_index'].min():.2f} to {out['DID_index'].max():.2f}")
        logger.info(f"Features used: {all_features}")
        logger.info(f"Features dropped: {list(dropped.keys())}")

        if save:
            self._save_output(out)

        return out

    def _save_output(self, df: pd.DataFrame) -> None:
        """Save output to parquet and CSV."""
        SILVER_DIR.mkdir(parents=True, exist_ok=True)

        # CSV
        csv_path = SILVER_DIR / "deflator_did_robust.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved CSV: {csv_path}")

        # Parquet
        parquet_path = SILVER_DIR / "deflator_did_robust.parquet"
        df.to_parquet(parquet_path)
        logger.info(f"Saved Parquet: {parquet_path}")


# =============================================================================
# LEGACY FUNCTIONS (kept for backwards compatibility)
# =============================================================================

def _load_bronze_data():
    """Load bronze EVDS data (prefers Parquet, falls back to CSV)."""
    if BRONZE_PATH_PARQUET.exists():
        return pd.read_parquet(BRONZE_PATH_PARQUET)
    if BRONZE_PATH_CSV.exists():
        return pd.read_csv(BRONZE_PATH_CSV)
    raise FileNotFoundError(f"Bronze EVDS data not found.")


def build_did_baseline():
    """Build baseline DID deflator using Factor Analysis (legacy method)."""
    df = _load_bronze_data()
    df = df.sort_values("DATE").reset_index(drop=True)

    out = pd.DataFrame({"DATE": df["DATE"]})

    if "TUFE" in df.columns:
        out["TUFE_mom"] = df["TUFE"].pct_change()
    if "UFE" in df.columns:
        out["UFE_mom"] = df["UFE"].pct_change()
    if "M2" in df.columns:
        out["M2_yoy"] = df["M2"].pct_change(12)
    if "TL_FAIZ" in df.columns:
        out["TL_FAIZ_lvl"] = df["TL_FAIZ"]

    out = out.ffill().dropna()

    feature_cols = [c for c in ["TUFE_mom", "UFE_mom", "M2_yoy", "TL_FAIZ_lvl"] if c in out.columns]
    X = out[feature_cols].copy()

    # StandardScaler (legacy - less robust)
    scaler = StandardScaler()
    Z = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)

    # Factor Analysis
    fa = FactorAnalysis(n_components=1, random_state=42)
    out["DID_factor_raw"] = fa.fit_transform(Z).ravel()

    # Sign correction
    if "TUFE_mom" in Z.columns:
        if np.corrcoef(out["DID_factor_raw"], Z["TUFE_mom"])[0, 1] < 0:
            out["DID_factor_raw"] *= -1

    # OLS calibration (legacy - less robust)
    proxy = "TUFE_mom" if "TUFE_mom" in out.columns else "UFE_mom"
    y = out[proxy]
    X_ols = sm.add_constant(out["DID_factor_raw"])
    model = sm.OLS(y, X_ols, missing="drop").fit()
    a, b = model.params["const"], model.params["DID_factor_raw"]

    out["pi_hat_monthly"] = a + b * out["DID_factor_raw"]

    # DID index
    pi = out["pi_hat_monthly"].fillna(0.0)
    did = (1.0 + pi).cumprod()

    base_mask = out["DATE"] == BASE_MONTH
    base_val = did.loc[base_mask].iloc[0] if base_mask.any() else did.iloc[0]
    out["DID_index"] = 100.0 * did / base_val

    # Save
    SILVER_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = SILVER_DIR / "deflator_did_baseline.csv"
    out.to_csv(csv_path, index=False)
    parquet_path = SILVER_DIR / "deflator_did_baseline.parquet"
    out.to_parquet(parquet_path)

    print(f"Baseline DID saved to {csv_path}")


def build_did_dfm():
    """Build DFM/Kalman-smoothed DID deflator (legacy method)."""
    df = _load_bronze_data()
    df = df.sort_values("DATE").reset_index(drop=True)

    out = pd.DataFrame({"DATE": df["DATE"]})
    if "TUFE" in df.columns:
        out["TUFE_mom"] = df["TUFE"].pct_change()
    if "UFE" in df.columns:
        out["UFE_mom"] = df["UFE"].pct_change()
    if "M2" in df.columns:
        out["M2_yoy"] = df["M2"].pct_change(12)
    if "TL_FAIZ" in df.columns:
        out["TL_FAIZ_lvl"] = df["TL_FAIZ"]

    out = out.ffill()
    feats = [c for c in ["TUFE_mom", "UFE_mom", "M2_yoy", "TL_FAIZ_lvl"] if c in out.columns]
    Z = out[feats].copy()
    Z = (Z - Z.mean()) / Z.std()
    Z = Z.ffill().bfill()

    # Dynamic Factor Model
    mod = sm.tsa.DynamicFactor(endog=Z, k_factors=1, factor_order=1, error_cov_type='diagonal')
    res = mod.fit(maxiter=1000, disp=False)
    smoothed = res.factors.smoothed

    if isinstance(smoothed, np.ndarray):
        if smoothed.shape[0] == 1:
            f = pd.Series(smoothed[0, :], index=Z.index, name="f1")
        else:
            f = pd.Series(smoothed[:, 0], index=Z.index, name="f1")
    else:
        f = smoothed.iloc[:, 0].rename("f1")

    # Sign correction
    if "TUFE_mom" in Z.columns and np.corrcoef(f.loc[Z.index], Z["TUFE_mom"].loc[Z.index])[0, 1] < 0:
        f *= -1

    # Calibration
    proxy = "TUFE_mom" if "TUFE_mom" in Z.columns else "UFE_mom"
    y = out.loc[Z.index, proxy]
    X_ols = sm.add_constant(f)
    model = sm.OLS(y, X_ols, missing="drop").fit()
    a = model.params["const"]
    b = model.params.get("f1", model.params.iloc[1])

    out2 = out.loc[Z.index].copy()
    out2["pi_hat_monthly"] = a + b * f
    did = (1.0 + out2["pi_hat_monthly"].fillna(0)).cumprod()

    base_mask = out2["DATE"] == BASE_MONTH
    base_val = did.loc[base_mask].iloc[0] if base_mask.any() else did.iloc[0]
    out2["DID_index"] = 100.0 * did / base_val

    # Save
    SILVER_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = SILVER_DIR / "deflator_did_dfm.csv"
    out2.to_csv(csv_path, index=False)
    parquet_path = SILVER_DIR / "deflator_did_dfm.parquet"
    out2.to_parquet(parquet_path)

    print(f"DFM DID saved to {csv_path}")


def build_did_robust():
    """Build robust DID deflator using RobustScaler + HuberRegressor."""
    builder = RobustDeflatorBuilder(
        anchor_col='TUFE',
        correlation_threshold=0.3,
        huber_epsilon=1.35
    )
    return builder.build(save=True)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("BUILDING DEFLATOR INDICES")
    print("="*70)

    # Build all three versions
    print("\n1. Building ROBUST deflator (RobustScaler + Huber)...")
    build_did_robust()

    print("\n2. Building BASELINE deflator (StandardScaler + OLS)...")
    build_did_baseline()

    print("\n3. Building DFM deflator (Kalman smoothing)...")
    build_did_dfm()

    print("\n" + "="*70)
    print("ALL DEFLATOR INDICES BUILT SUCCESSFULLY")
    print("="*70)
