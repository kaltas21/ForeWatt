# src/data/deflator_builder.py
import os
import pandas as pd
import numpy as np
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm  # OLS için
from dateutil.relativedelta import relativedelta

BRONZE_PATH = "data/bronze/macro/macro_evds_raw.csv"
SILVER_DIR = "data/silver/macro"
BASE_MONTH = "2022-01"  # DID için baz (sonra değiştirilebilir)

def _safe_pct_change(s, periods=1):
    return s.pct_change(periods=periods)

def build_did_baseline():
    # 1) Veriyi oku
    df = pd.read_csv(BRONZE_PATH)

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

    # 12) Kaydet
    os.makedirs(SILVER_DIR, exist_ok=True)
    out.to_csv(os.path.join(SILVER_DIR, "deflator_did_baseline.csv"), index=False)
    print("✅ Baseline DID kaydedildi → data/silver/macro/deflator_did_baseline.csv")


def build_did_dfm():
    import statsmodels.api as sm
    df = pd.read_csv(BRONZE_PATH).sort_values("DATE")

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

    os.makedirs(SILVER_DIR, exist_ok=True)
    out2.to_csv(os.path.join(SILVER_DIR, "deflator_did_dfm.csv"), index=False)
    print("✅ DFM/Kalman DID kaydedildi → data/silver/macro/deflator_did_dfm.csv")


if __name__ == "__main__":
    build_did_baseline()
    build_did_dfm()
