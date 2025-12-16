from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

from src.anomaly.isolationForest import IFConfig, IsolationForestDetector


MODEL_DIR = Path("reports/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "iforest.joblib"


def train_iforest(
    df: pd.DataFrame,
    ts_col: str,
    y_col: str,
    pred_col: Optional[str] = None,
    extra_cols: Optional[list[str]] = None,
    cfg: IFConfig = IFConfig(),
) -> dict:
    det = IsolationForestDetector(cfg)
    X, _y = det.make_features(df, ts_col=ts_col, y_col=y_col, pred_col=pred_col, extra_cols=extra_cols)
    det.fit(X)

    payload = {"cfg": asdict(cfg), "detector": det}
    joblib.dump(payload, MODEL_PATH)
    return {"model_path": str(MODEL_PATH), "n_rows": int(len(X))}


def detect_anomalies(
    df: pd.DataFrame,
    ts_col: str,
    y_col: str,
    pred_col: Optional[str] = None,
    extra_cols: Optional[list[str]] = None,
    threshold: Optional[float] = None,
) -> pd.DataFrame:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Train it first.")

    payload = joblib.load(MODEL_PATH)
    det: IsolationForestDetector = payload["detector"]

    X, y_aligned = det.make_features(df, ts_col=ts_col, y_col=y_col, pred_col=pred_col, extra_cols=extra_cols)
    is_anom, used_thr = det.predict(X, threshold=threshold)
    scores = det.score(X)

    out = pd.DataFrame(
        {
            ts_col: X.index,
            y_col: y_aligned.values,
            "anomaly_score": scores,
            "is_anomaly": is_anom,
            "threshold": used_thr,
        }
    )
    return out
