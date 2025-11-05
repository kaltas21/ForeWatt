import pandas as pd
import matplotlib.pyplot as plt

baseline = pd.read_csv("data/silver/macro/deflator_did_baseline.csv")
dfm = pd.read_csv("data/silver/macro/deflator_did_dfm.csv")
evds = pd.read_csv("data/bronze/macro/macro_evds_raw.csv")

plt.figure(figsize=(10, 6))
plt.plot(dfm["DATE"], dfm["DID_index"], label="DFM/Kalman Deflatör", linewidth=2)
plt.plot(baseline["DATE"], baseline["DID_index"], label="Baseline (TÜFE bazlı)", linestyle="--")
plt.plot(evds["DATE"], evds["TUFE_rebased"], label="TÜFE (2003=100, rebased)", linestyle=":")
plt.title("ForeWatt Makroekonomik Deflatör Karşılaştırması")
plt.xlabel("Tarih")
plt.ylabel("Endeks (2022=100)")
plt.legend()
plt.grid(True)
plt.show()
