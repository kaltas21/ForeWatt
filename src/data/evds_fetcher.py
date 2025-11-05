from evds import evdsAPI
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

EVDS_API_KEY = os.getenv("EVDS_API_KEY")

def fetch_evds_data(start_date="01-01-2021", end_date="01-01-2025"):
    """
    TÜFE, ÜFE, M2, TL series will be fetched.
    """
    evds = evdsAPI(EVDS_API_KEY)

    series = {
        "TUFE": "TP.FG.J0",          # TÜFE (2003=100)
        "UFE": "TP.FG.J01",          # Yurtiçi ÜFE (2003=100)
        "M2": "TP.PBD.H09",           # Para arzı M2 (Milyon TL)
        "TL_FAIZ": "TP.TRY.MT06" # TL mevduat faizi (%)
    }

    df = evds.get_data(
        list(series.values()),
        startdate=start_date,
        enddate=end_date,
        frequency="5"  # Monthly
    )

    df.columns = ["DATE"] + list(series.keys())

    # 🔹 tarih biçimini normalize et
    df["DATE"] = pd.to_datetime(df["DATE"]).dt.strftime("%Y-%m")

    df = df.sort_values("DATE")

    return df


def rebase_index(df, column, base_date="2022-01"):
    """
    EVDS üzerinden baz otomatik olarak 2003 şeklinde. 
    Bu baz yılına göre ölçeklendirme yapmak için eklendi bu kod.
    """
    base_value = df.loc[df["DATE"] == base_date, column].values
    if len(base_value) == 0:
        print(f"⚠️ {base_date} bulunamadı, ilk gözlem baz alındı.")
        base_value = df[column].iloc[0]
    else:
        base_value = base_value[0]

    df[f"{column}_rebased"] = (df[column] / base_value) * 100
    return df


def save_bronze(df):
    bronze_path = os.path.join("data", "bronze", "macro")
    os.makedirs(bronze_path, exist_ok=True)

    csv_path = os.path.join(bronze_path, "macro_evds_raw.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ Bronze dataset saved to {csv_path}")


if __name__ == "__main__":
    # fetching
    df = fetch_evds_data()

    # rebasing for TÜFE and ÜFE 
    df = rebase_index(df, "TUFE", base_date="2022-01")
    df = rebase_index(df, "UFE", base_date="2022-01")

    save_bronze(df)
