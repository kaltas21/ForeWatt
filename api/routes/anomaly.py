from __future__ import annotations

from typing import Optional
import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel

from services.anomalyServices import train_iforest, detect_anomalies

router = APIRouter(prefix="/anomaly", tags=["anomaly"])


class TrainReq(BaseModel):
    ts_col: str = "timestamp"
    y_col: str = "y"
    pred_col: Optional[str] = None
    extra_cols: Optional[list[str]] = None
    contamination: float = 0.01


class DetectReq(BaseModel):
    ts_col: str = "timestamp"
    y_col: str = "y"
    pred_col: Optional[str] = None
    extra_cols: Optional[list[str]] = None
    threshold: Optional[float] = None


@router.post("/train")
def train(req: TrainReq):
    # TODO: burada df'yi projenin mevcut veri kaynağından çek
    # ör: df = load_latest_timeseries(...)
    raise NotImplementedError("Load df from your project pipeline, then call train_iforest().")


@router.post("/detect")
def detect(req: DetectReq):
    # TODO: df = load_latest_timeseries(...)
    raise NotImplementedError("Load df, then call detect_anomalies().")
