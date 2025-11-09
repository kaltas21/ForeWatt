from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]  # project root
STATIC = ROOT / "src" / "data" / "static" / "tr_holidays_2020_2025.json"

OUT_BRONZE = ROOT / "data" / "bronze" / "calendar"
OUT_SILVER = ROOT / "data" / "silver" / "calendar"
OUT_BRONZE.mkdir(parents=True, exist_ok=True)
OUT_SILVER.mkdir(parents=True, exist_ok=True)

TZ = "Europe/Istanbul"

def _explode_spans(rows: list[dict]) -> pd.DataFrame:
    out = []
    for r in rows:
        s = pd.to_datetime(r["start_date"]).date()
        e = pd.to_datetime(r["end_date"]).date()
        for d in pd.date_range(s, e, freq="D"):
            out.append({
                "date_only": d.date(),
                "name": r["name"],
                "kind": r.get("kind", "official"),
                "half_day": r.get("half_day", None),
            })
    df = pd.DataFrame(out)
    if df.empty:
        return pd.DataFrame(columns=["date_only","name","kind","half_day"])
    return (df.sort_values(["date_only","kind"])
              .drop_duplicates(subset=["date_only"], keep="first"))

def build_calendar_tables():
    with open(STATIC, "r", encoding="utf-8") as f:
        payload = json.load(f)

    years = payload["metadata"]["years"]
    start = pd.Timestamp(f"{min(years)}-01-01", tz=TZ)
    end = pd.Timestamp(f"{max(years)}-12-31", tz=TZ)

    # ---- BRONZE: raw spans ----
    bronze = pd.DataFrame(payload["holidays"])
    bronze.to_csv(OUT_BRONZE / "calendar_raw.csv", index=False)
    try:
        bronze.to_parquet(OUT_BRONZE / "calendar_raw.parquet", index=False)
    except Exception:
        pass

    # ---- SILVER A: exploded holiday days (keeps half_day) ----
    days = _explode_spans(payload["holidays"])
    days["is_holiday_day"] = 1
    days.to_csv(OUT_SILVER / "calendar_days.csv", index=False)
    try:
        days.to_parquet(OUT_SILVER / "calendar_days.parquet", index=False)
    except Exception:
        pass

    # ---- SILVER B: full daily calendar with flags ----
    cal = pd.DataFrame(index=pd.date_range(start.normalize(), end.normalize(), freq="D", tz=TZ))
    idx: pd.DatetimeIndex = pd.DatetimeIndex(cal.index)
    cal["date_only"] = idx.date
    cal["dow"] = idx.dayofweek
    cal["is_weekend"] = cal["dow"].isin([5, 6]).astype("int8")
    cal["is_weekend"] = cal["dow"].isin([5, 6]).astype("int8")
    cal = cal.merge(days[["date_only","name"]], on="date_only", how="left")
    cal["is_holiday_day"] = cal["name"].notna().astype("int8")
    cal["holiday_name"] = cal["name"].fillna("None")
    cal.drop(columns=["name"], inplace=True)
    cal["is_holiday_weekend"] = ((cal["is_holiday_day"]==1) & (cal["is_weekend"]==1)).astype("int8")
    cal["is_holiday_weekday"] = ((cal["is_holiday_day"]==1) & (cal["is_weekend"]==0)).astype("int8")

    cal_out_csv = OUT_SILVER / "calendar_full_days.csv"
    cal.reset_index(drop=True).to_csv(cal_out_csv, index=False)
    try:
        cal.reset_index(drop=True).to_parquet(OUT_SILVER / "calendar_full_days.parquet", index=False)
    except Exception:
        pass

    print("Wrote:")
    print(" -", OUT_BRONZE / "calendar_raw.csv")
    print(" -", OUT_SILVER / "calendar_days.csv")
    print(" -", cal_out_csv)
    print(f"Full days rows: {len(cal)}")

if __name__ == "__main__":
    build_calendar_tables()
