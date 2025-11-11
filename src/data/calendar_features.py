# src/data/calendar_features.py
from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any

TZ = "Europe/Istanbul"


def _to_datetime_index(idx: pd.Index) -> pd.DatetimeIndex:
    """Ensure a tz-aware Europe/Istanbul DatetimeIndex (for Pylance & runtime)."""
    di = pd.to_datetime(idx)
    if isinstance(di, pd.DatetimeIndex):
        if di.tz is None:
            di = di.tz_localize(TZ)
        else:
            di = di.tz_convert(TZ)
        return di
    # Fallback (shouldn't happen after pd.to_datetime)
    return pd.DatetimeIndex(di, tz=TZ)


def add_calendar_features_hourly(
    X: pd.DataFrame,
    calendar_days_path: str = "data/silver/calendar/calendar_days.csv",
) -> pd.DataFrame:
    """
    Add hour-level calendar features to an hourly DataFrame.

    Inputs:
      - X: hourly pd.DataFrame with a datetime-like index (tz-naive or tz-aware).
      - calendar_days_path: path to exploded holiday days CSV (has date_only, name, half_day, is_holiday_day).

    Outputs:
      - X joined with:
          dow, dom, month, weekofyear, is_weekend,
          is_holiday_day, is_holiday_hour, holiday_name
    """

    # --- 1) Normalize index to tz-aware Europe/Istanbul ---
    X = X.copy()
    di: pd.DatetimeIndex = _to_datetime_index(X.index)
    X.index = di

    feats = pd.DataFrame(index=di)
    # Use an explicit DatetimeIndex variable for attribute access (Pylance)
    dix: pd.DatetimeIndex = pd.DatetimeIndex(feats.index)

    feats["date_only"] = dix.date
    feats["dow"] = dix.dayofweek
    feats["dom"] = dix.day
    feats["month"] = dix.month
    # isocalendar() returns a DataFrame-like accessor in recent pandas
    feats["weekofyear"] = dix.isocalendar().week.astype(int)
    feats["is_weekend"] = feats["dow"].isin([5, 6]).astype("int8")

    # --- 2) Load holiday day table & normalize types ---
    path = Path(calendar_days_path)
    if not path.exists():
        raise FileNotFoundError(f"Calendar days file not found at: {path}")

    cal_days = pd.read_csv(path, parse_dates=["date_only"])
    # Normalize to pure Python date for stable dict-based lookups
    cal_days["date_only"] = cal_days["date_only"].dt.date

    # Safety: if is_holiday_day not present, derive from name non-null
    if "is_holiday_day" not in cal_days.columns:
        cal_days["is_holiday_day"] = cal_days["name"].notna().astype("int8")

    # --- 3) Day-level holiday flags via dict maps (robust across pandas versions) ---
    day_map: Dict[Any, Any] = cal_days.set_index("date_only")["is_holiday_day"].to_dict()
    feats["is_holiday_day"] = feats["date_only"].map(lambda d: int(bool(day_map.get(d, 0)))).astype("int8")

    # --- 4) Hour-level flag with half-day PM handling ---
    feats["is_holiday_hour"] = feats["is_holiday_day"].copy()

    if "half_day" in cal_days.columns:
        half = cal_days[cal_days["half_day"].notna()][["date_only", "half_day"]].drop_duplicates()
        if not half.empty:
            half_map: Dict[Any, Any] = half.set_index("date_only")["half_day"].to_dict()
            # For 'pm' half-days, AM hours (<13) are not holiday
            am_mask = dix.hour < 13
            feats.loc[
                feats["date_only"].map(lambda d: half_map.get(d) == "pm") & am_mask,
                "is_holiday_hour"
            ] = 0
            # If you ever add 'am' half-days, mirror the logic for hours >= 13

    # --- 5) Optional holiday name (useful for EDA / categorical models) ---
    name_map: Dict[Any, Any] = cal_days.set_index("date_only")["name"].to_dict() if "name" in cal_days.columns else {}
    feats["holiday_name"] = feats["date_only"].map(lambda d: name_map.get(d, "None"))

    # --- 6) Return X with features joined ---
    return X.join(feats.drop(columns=["date_only"]))
