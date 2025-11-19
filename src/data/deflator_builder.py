# src/data/deflator_builder.py
"""
Deflator Index Builder for Turkish Lira Normalization
======================================================
Builds real value deflators using EVDS macroeconomic indicators.

Methods:
- Baseline: Factor Analysis on TÜFE, ÜFE, M2, TL_FAIZ
- DFM: Dynamic Factor Model with Kalman smoothing

Output:
- DID_index: Deflator index (base=100 at BASE_MONTH)
- Use to convert nominal TL prices to real values

Aligned with EPİAŞ pipeline: 2020-01-01 to 2025-10-31
"""
import os
import pandas as pd
import numpy as np
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm  # OLS için
from dateutil.relativedelta import relativedelta
from pathlib import Path

# File paths (supports both CSV and Parquet)
BRONZE_PATH_CSV = "data/bronze/macro/macro_evds_raw.csv"
BRONZE_PATH_PARQUET = "data/bronze/macro/macro_evds_2020-01-01_2025-10-31.parquet"
SILVER_DIR = "data/silver/macro"
BASE_MONTH = "2022-01"  # DID base month (middle of dataset, YYYY-MM format)

def _safe_pct_change(s, periods=1):
    return s.pct_change(periods=periods)

def _load_bronze_data():
    """
    Load bronze EVDS data (prefers Parquet, falls back to CSV).

    Returns:
        DataFrame with DATE column in 'YYYY-MM' format
    """
    # Try Parquet first (faster, aligned with EPİAŞ pipeline)
    if Path(BRONZE_PATH_PARQUET).exists():
        print(f"📦 Loading bronze data from Parquet: {BRONZE_PATH_PARQUET}")
        return pd.read_parquet(BRONZE_PATH_PARQUET)

    # Fallback to CSV
    if Path(BRONZE_PATH_CSV).exists():
        print(f"📦 Loading bronze data from CSV: {BRONZE_PATH_CSV}")
        return pd.read_csv(BRONZE_PATH_CSV)

    # Try synthetic data (for testing)
    bronze_dir = Path("data/bronze/macro")
    if bronze_dir.exists():
        synthetic_files = list(bronze_dir.glob("macro_evds_*_SYNTHETIC.parquet"))
        if synthetic_files:
            synthetic_file = synthetic_files[0]
            print(f"📦 Loading SYNTHETIC bronze data: {synthetic_file}")
            return pd.read_parquet(synthetic_file)

    raise FileNotFoundError(
        f"Bronze EVDS data not found. Run evds_fetcher.py first.\n"
        f"Expected: {BRONZE_PATH_PARQUET} or {BRONZE_PATH_CSV}"
    )

def build_did_baseline():
    """
    Build baseline DID deflator using Factor Analysis.

    Steps:
    1. Load EVDS data (TÜFE, ÜFE, M2, TL_FAIZ)
    2. Compute growth rates (MoM, YoY)
    3. Extract single inflation factor via Factor Analysis
    4. Calibrate to TÜFE_mom using OLS
    5. Build cumulative deflator index (base=100 at BASE_MONTH)

    Output:
        Saves to: data/silver/macro/deflator_did_baseline.csv
    """
    # 1) Veriyi oku
    df = _load_bronze_data()

    # 2) Gerekli kolonlar
    base_cols = [c for c in ["TUFE", "UFE", "M2", "TL_FAIZ"] if c in df.columns]
    if not {"TUFE","UFE"}.intersection(base_cols):
        raise ValueError("En azından TUFE veya UFE kolonu gerekli.")

    out = pd.DataFrame({"DATE": df["DATE"]})

    # 3) Büyüme/transformasyonlar (pct_change fill_method=None)
    if "TUFE" in df:
        out["TUFE_mom"] = df["TUFE"].pct_change(fill_method=None)
    if "UFE" in df:
        out["UFE_mom"]  = df["UFE"].pct_change(fill_method=None)
    if "M2" in df:
        out["M2_yoy"]   = df["M2"].pct_change(periods=12, fill_method=None)
    if "TL_FAIZ" in df:
        out["TL_FAIZ_lvl"] = df["TL_FAIZ"]  # seviye

    # 4) Tamamen NaN olan kolonları at
    for c in list(out.columns):
        if c != "DATE" and out[c].isna().all():
            out.drop(columns=c, inplace=True)

    # 5) Tarih sıralaması ve eksik doldurma
    out = out.sort_values("DATE").reset_index(drop=True)
    out = out.ffill()   # lider NaN kalırsa aşağıda drop edeceğiz
    out = out.dropna()  # hala NaN olan satırları temizle

    # 6) Feature matrisi; mevcut olanlardan seç
    feature_cols = [c for c in ["TUFE_mom","UFE_mom","M2_yoy","TL_FAIZ_lvl"] if c in out.columns]
    if len(feature_cols) < 1:
        raise ValueError("Faktör analizi için en az bir geçerli gösterge gerekli (ör. TUFE_mom).")
    X = out[feature_cols].copy()

    # 7) Z-score ölçekleme
    scaler = StandardScaler()
    Z = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)

    # 8) Tek faktör çıkar
    fa = FactorAnalysis(n_components=1, random_state=42)
    out["DID_factor_raw"] = fa.fit_transform(Z).ravel()

    # 9) İşaret kontrolü (varsa TUFE_mom'a göre)
    if "TUFE_mom" in Z.columns:
        if np.corrcoef(out["DID_factor_raw"], Z["TUFE_mom"])[0,1] < 0:
            out["DID_factor_raw"] *= -1
    elif "UFE_mom" in Z.columns:
        if np.corrcoef(out["DID_factor_raw"], Z["UFE_mom"])[0,1] < 0:
            out["DID_factor_raw"] *= -1

    # 10) Kalibrasyon (TUFE_mom varsa ona, yoksa UFE_mom'a)
    proxy = "TUFE_mom" if "TUFE_mom" in Z.columns else ("UFE_mom" if "UFE_mom" in Z.columns else None)
    if proxy is None:
        raise ValueError("Kalibrasyon için TUFE_mom veya UFE_mom gerekli.")

    y = out[proxy]
    X_ols = sm.add_constant(out["DID_factor_raw"])
    model = sm.OLS(y, X_ols, missing="drop").fit()
    a, b = model.params["const"], model.params["DID_factor_raw"]

    out["pi_hat_monthly"] = a + b * out["DID_factor_raw"]

    # 11) DID indeksi (zincirle + bazla)
    pi = out["pi_hat_monthly"].fillna(0.0)
    did = (1.0 + pi).cumprod()
    out["DID_monthly"] = did

    base_mask = out["DATE"] == BASE_MONTH  # 'YYYY-MM' bekliyoruz
    base_val = out.loc[base_mask, "DID_monthly"].iloc[0] if base_mask.any() else out["DID_monthly"].iloc[0]
    out["DID_index"] = 100.0 * out["DID_monthly"] / base_val

    # 12) Kaydet (dual format: CSV + Parquet)
    os.makedirs(SILVER_DIR, exist_ok=True)

    # CSV (human-readable)
    csv_path = os.path.join(SILVER_DIR, "deflator_did_baseline.csv")
    out.to_csv(csv_path, index=False)
    print(f"✅ Baseline DID CSV saved → {csv_path}")

    # Parquet (efficient, aligned with EPİAŞ pipeline)
    parquet_path = os.path.join(SILVER_DIR, "deflator_did_baseline.parquet")
    out.to_parquet(parquet_path, engine='pyarrow', compression='snappy')
    print(f"✅ Baseline DID Parquet saved → {parquet_path}")


def build_did_dfm():
    """
    Build DFM/Kalman-smoothed DID deflator using Dynamic Factor Model.

    More sophisticated than baseline: uses Kalman filter to smooth factor estimates.

    Output:
        Saves to: data/silver/macro/deflator_did_dfm.csv
    """
    import statsmodels.api as sm
    df = _load_bronze_data()
    df = df.sort_values("DATE")

    out = pd.DataFrame({"DATE": df["DATE"]})
    if "TUFE" in df: out["TUFE_mom"] = df["TUFE"].pct_change()
    if "UFE" in df:  out["UFE_mom"]  = df["UFE"].pct_change()
    if "M2" in df:   out["M2_yoy"]   = df["M2"].pct_change(12)
    if "TL_FAIZ" in df: out["TL_FAIZ_lvl"] = df["TL_FAIZ"]

    out = out.ffill()
    feats = [c for c in ["TUFE_mom", "UFE_mom", "M2_yoy", "TL_FAIZ_lvl"] if c in out.columns]
    Z = out[feats].copy()

    # z-score
    Z = (Z - Z.mean()) / Z.std()
    Z = Z.ffill().bfill()


    # Dynamic Factor Model
    mod = sm.tsa.DynamicFactor(endog=Z, k_factors=1, factor_order=1, error_cov_type='diagonal')
    res = mod.fit(maxiter=1000, disp=False)
    smoothed = res.factors.smoothed
    if isinstance(smoothed, np.ndarray):
        # Eğer faktör uzunluğu 1 ise, transpoze etmeyi dene
        if smoothed.shape[0] == 1:
            f = pd.Series(smoothed[0, :], index=Z.index, name="f1")
        else:
            f = pd.Series(smoothed[:, 0], index=Z.index, name="f1")
    else:
        f = smoothed.iloc[:, 0].rename("f1")


    # işaret kontrolü
    if "TUFE_mom" in Z.columns and np.corrcoef(f.loc[Z.index], Z["TUFE_mom"].loc[Z.index])[0,1] < 0:
        f *= -1

    # kalibrasyon (TUFE_mom varsa)
    proxy = "TUFE_mom" if "TUFE_mom" in Z.columns else ("UFE_mom" if "UFE_mom" in Z.columns else None)
    if proxy is None:
        raise ValueError("Kalibrasyon için TUFE_mom veya UFE_mom gerekli.")

    y = Z.index.map(lambda i: out.loc[i, proxy])  # orijinal ölçekteki proxy'yi çek
    y = out.loc[Z.index, proxy]
    X_ols = sm.add_constant(f)
    model = sm.OLS(y, X_ols, missing="drop").fit()
    a, b = model.params["const"], model.params[0] if "f1" not in model.params else model.params["f1"]

    # pi_hat ve DID
    out2 = out.loc[Z.index].copy()
    out2["pi_hat_monthly"] = a + b * f
    did = (1.0 + out2["pi_hat_monthly"].fillna(0)).cumprod()
    # baz
    base_mask = out2["DATE"] == BASE_MONTH
    base_val = did.loc[base_mask].iloc[0] if base_mask.any() else did.iloc[0]
    out2["DID_index"] = 100.0 * did / base_val

    # Kaydet (dual format: CSV + Parquet)
    os.makedirs(SILVER_DIR, exist_ok=True)

    # CSV (human-readable)
    csv_path = os.path.join(SILVER_DIR, "deflator_did_dfm.csv")
    out2.to_csv(csv_path, index=False)
    print(f"✅ DFM/Kalman DID CSV saved → {csv_path}")

    # Parquet (efficient, aligned with EPİAŞ pipeline)
    parquet_path = os.path.join(SILVER_DIR, "deflator_did_dfm.parquet")
    out2.to_parquet(parquet_path, engine='pyarrow', compression='snappy')
    print(f"✅ DFM/Kalman DID Parquet saved → {parquet_path}")


if __name__ == "__main__":
    build_did_baseline()
    build_did_dfm()
