from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler


@dataclass
class IFConfig:
    contamination: float = 0.01
    n_estimators: int = 300
    random_state: int = 42
    max_samples: str | int = "auto"


class IsolationForestDetector:
    def __init__(self, cfg: IFConfig = IFConfig()):
        self.cfg = cfg
        self.scaler = RobustScaler()
        self.model = IsolationForest(
            n_estimators=cfg.n_estimators,
            contamination=cfg.contamination,
            random_state=cfg.random_state,
            max_samples=cfg.max_samples,
            n_jobs=-1,
        )
        self._fit = False

    @staticmethod
    def make_features(
        df: pd.DataFrame,
        ts_col: str,
        y_col: str,
        pred_col: Optional[str] = None,
        extra_cols: Optional[list[str]] = None,
        win: int = 24,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Zaman serisi için sağlam (robust) feature set:
        - residual (y - yhat) varsa
        - rolling mean/std (y, residual)
        - lag'ler
        - hour/day-of-week
        """
        d = df.copy()
        d = d.sort_values(ts_col)
        d[ts_col] = pd.to_datetime(d[ts_col])
        d = d.set_index(ts_col)

        y = d[y_col].astype(float)

        feats = {}

        # takvim
        feats["hour"] = d.index.hour
        feats["dow"] = d.index.dayofweek
        feats["month"] = d.index.month

        # lag
        for lag in [1, 2, 3, 6, 12, 24]:
            feats[f"y_lag_{lag}"] = y.shift(lag)

        # rolling
        feats[f"y_roll_mean_{win}"] = y.rolling(win).mean()
        feats[f"y_roll_std_{win}"] = y.rolling(win).std()

        if pred_col and pred_col in d.columns:
            yhat = d[pred_col].astype(float)
            resid = (y - yhat)
            feats["resid"] = resid
            feats[f"resid_abs"] = resid.abs()
            feats[f"resid_roll_mean_{win}"] = resid.rolling(win).mean()
            feats[f"resid_roll_std_{win}"] = resid.rolling(win).std()

        if extra_cols:
            for c in extra_cols:
                if c in d.columns:
                    feats[c] = pd.to_numeric(d[c], errors="coerce")

        X = pd.DataFrame(feats, index=d.index)
        X = X.replace([np.inf, -np.inf], np.nan).dropna()

        # align y
        y_aligned = y.loc[X.index]
        return X, y_aligned

    def fit(self, X: pd.DataFrame) -> "IsolationForestDetector":
        Xs = self.scaler.fit_transform(X.values)
        self.model.fit(Xs)
        self._fit = True
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fit:
            raise RuntimeError("Model is not fit yet.")
        Xs = self.scaler.transform(X.values)
        # sklearn: higher = more normal, lower = more anomalous
        decision = self.model.decision_function(Xs)
        # anomaly_score: higher = more anomalous
        return -decision

    def predict(self, X: pd.DataFrame, threshold: Optional[float] = None) -> Tuple[np.ndarray, float]:
        scores = self.score(X)
        if threshold is None:
            # contamination'a göre otomatik cutoff (quantile)
            q = 1.0 - self.cfg.contamination
            threshold = float(np.quantile(scores, q))
        is_anom = (scores >= threshold).astype(int)
        return is_anom, threshold
