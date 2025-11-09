# src/analysis/make_gold_calendar_features.py
import sys
from pathlib import Path
import pandas as pd

# Make sure package imports work when running with -m OR by path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.data.calendar_features import add_calendar_features_hourly  # noqa: E402

TZ = "Europe/Istanbul"
SILVER_DIR = Path("data/silver/demand_weather")
CAL_DAYS = Path("data/silver/calendar/calendar_days.csv")
GOLD_DIR = Path("data/gold/demand_features")

def _load_any(path: Path) -> pd.DataFrame:
    """Load parquet or csv; return a DataFrame with a datetime index."""
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        # parse first column as datetime and set as index
        df = pd.read_csv(path, parse_dates=[0])
        df = df.set_index(df.columns[0])
    else:
        raise ValueError(f"Unsupported file type: {path}")
    # ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        # try to find a likely datetime column
        candidates = [c for c in df.columns if str(c).lower() in ("timestamp","ts","datetime","date","time")]
        if candidates:
            df[candidates[0]] = pd.to_datetime(df[candidates[0]])
            df = df.set_index(candidates[0])
        else:
            # last resort: try to coerce current index
            df.index = pd.to_datetime(df.index)
    return df

def _ensure_tz_istanbul(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if idx.tz is None:
        return idx.tz_localize(TZ)
    return idx.tz_convert(TZ)

def main():
    if not CAL_DAYS.exists():
        raise FileNotFoundError(
            f"Missing {CAL_DAYS}. Run: python -m src.data.calendar_builder"
        )
    if not SILVER_DIR.exists():
        raise FileNotFoundError(f"Missing {SILVER_DIR} directory.")

    files = sorted(list(SILVER_DIR.glob("*.parquet")) + list(SILVER_DIR.glob("*.csv")))
    if not files:
        raise FileNotFoundError(f"No .parquet or .csv found in {SILVER_DIR}")

    GOLD_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(files)} silver file(s). Processing…")
    failures = []

    for f in files:
        try:
            print(f"→ Loading {f.name}")
            df = _load_any(f)
            df.index = _ensure_tz_istanbul(pd.to_datetime(df.index))

            df_out = add_calendar_features_hourly(
                df, calendar_days_path=str(CAL_DAYS)
            )

            out_base = f.stem + "_w_calendar"
            out_parquet = GOLD_DIR / f"{out_base}.parquet"
            try:
                df_out.to_parquet(out_parquet)
                print(f"   ✓ Wrote {out_parquet}")
            except Exception as e:
                out_csv = GOLD_DIR / f"{out_base}.csv"
                df_out.to_csv(out_csv)
                print(f"   • Parquet write failed ({e}); wrote CSV {out_csv}")

        except Exception as e:
            print(f"   ✗ Failed on {f.name}: {e}")
            failures.append((f.name, str(e)))

    if failures:
        print("\nSome files failed:")
        for name, msg in failures:
            print(f" - {name}: {msg}")
        sys.exit(1)

if __name__ == "__main__":
    main()
