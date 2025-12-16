from __future__ import annotations

import sys
from pathlib import Path
import types
import importlib.util

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parent.parent
DASHBOARD_DIR = ROOT_DIR / "dashboard"
UTILS_DIR = DASHBOARD_DIR / "utils"

if not UTILS_DIR.exists():
    raise RuntimeError(f"utils klasörü bulunamadı: {UTILS_DIR}")

print("✅ Using UTILS_DIR =", UTILS_DIR)


def load_utils_module(module_name: str, file_path: Path):
    """Load a module from a .py file WITHOUT executing utils/__init__.py."""
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Spec oluşturulamadı: {module_name} -> {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def bootstrap_fake_utils_package():
    """
    Create a fake 'utils' package in sys.modules so that
    relative imports like 'from .config import ...' work,
    but without importing utils/__init__.py.
    """
    if "utils" not in sys.modules:
        pkg = types.ModuleType("utils")
        pkg.__path__ = [str(UTILS_DIR)]  # mark as package
        sys.modules["utils"] = pkg


def pick_forecast_column(df: pd.DataFrame) -> str:
    candidates = [
        "consumption_forecast",
        "forecast",
        "yhat",
        "prediction",
        "pred",
        "consumption_pred",
        "y_pred",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        "Forecast column not found. Checked: "
        + ", ".join(candidates)
        + "\nAvailable columns: "
        + ", ".join(df.columns[:80].tolist())
        + (" ..." if len(df.columns) > 80 else "")
    )


def main():
    # 1) Fake package + load config + load data_loader (NO __init__.py)
    bootstrap_fake_utils_package()

    config_mod = load_utils_module("utils.config", UTILS_DIR / "config.py")
    data_loader_mod = load_utils_module("utils.data_loader", UTILS_DIR / "data_loader.py")

    # 2) Get functions/vars
    load_master_data = getattr(data_loader_mod, "load_master_data")
    TARGET_VARIABLE = getattr(config_mod, "TARGET_VARIABLE")

    # 3) Load master data
    df = load_master_data()
    if df is None or df.empty:
        raise RuntimeError("load_master_data() empty. MASTER_DATA path/file erişimini kontrol et.")

    y_col = TARGET_VARIABLE
    if y_col not in df.columns:
        raise KeyError(f"TARGET_VARIABLE='{y_col}' master data columns içinde yok.")

    forecast_col = pick_forecast_column(df)

    print("✅ Loaded master data")
    print("Rows:", len(df), "Cols:", len(df.columns))
    print("Target:", y_col, "Forecast:", forecast_col)

    # 4) Residuals
    df = df.copy().dropna(subset=[y_col, forecast_col])
    df["residual"] = df[y_col].astype(float) - df[forecast_col].astype(float)
    df["abs_residual"] = df["residual"].abs()

    # 5) Feature set (az + güçlü)
    desired_features = [
        "residual",
        "abs_residual",
        "consumption_rolling_std_24h",
        "consumption_range_24h",
        "consumption_cv_24h",
        "hour_sin",
        "hour_cos",
        "is_weekend_x",
        "consumption_lag_1h",
        "consumption_lag_24h",
        "consumption_lag_168h",
    ]
    features = [c for c in desired_features if c in df.columns]
    if len(features) < 4:
        print("Available columns:", df.columns.tolist())
        raise RuntimeError(f"Too few features found: {features}")

    X = df[features].replace([float("inf"), float("-inf")], pd.NA).dropna()
    df = df.loc[X.index]

    print("✅ Using features:", features)
    print("Rows after dropna:", len(df))

    # 6) Scale + IsolationForest
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)

    iso = IsolationForest(
        n_estimators=400,
        contamination=0.01,
        random_state=42,
        n_jobs=-1,
    )
    pred = iso.fit_predict(Xs)
    normality = iso.decision_function(Xs)

    df["is_anomaly"] = (pred == -1).astype(int)
    df["anomaly_score"] = (-normality).astype(float)

    # 7) Save
    out = df.reset_index()
    first_col = out.columns[0]
    if first_col not in ["timestamp", "datetime"]:
        out = out.rename(columns={first_col: "timestamp"})

    out_cols = ["timestamp", y_col, forecast_col, "residual", "anomaly_score", "is_anomaly"]
    out_cols = [c for c in out_cols if c in out.columns]

    (ROOT_DIR / "reports").mkdir(exist_ok=True)
    out_path = ROOT_DIR / "reports" / "anomaly_results.csv"
    out[out_cols].to_csv(out_path, index=False)

    print(f"\n✅ Saved: {out_path}")
    print("\nTop 20 anomalies:")
    print(out.sort_values("anomaly_score", ascending=False).head(20)[out_cols])


if __name__ == "__main__":
    main()
