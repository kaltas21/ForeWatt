from evds import evdsAPI
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

EVDS_API_KEY = os.getenv("EVDS_API_KEY")

def fetch_evds_data(start_date="2020-01-01", end_date="2024-12-31"):
    """
    TÜFE, ÜFE, M2, TL series will be fetched.

    Args:
        start_date: Start date in 'YYYY-MM-DD' format (aligned with EPİAŞ pipeline)
        end_date: End date in 'YYYY-MM-DD' format (aligned with EPİAŞ pipeline)

    Returns:
        DataFrame with DATE column in 'YYYY-MM' format and macro indicators
    """
    # Convert YYYY-MM-DD to DD-MM-YYYY format required by evdsAPI
    from datetime import datetime
    start_evds = datetime.strptime(start_date, "%Y-%m-%d").strftime("%d-%m-%Y")
    end_evds = datetime.strptime(end_date, "%Y-%m-%d").strftime("%d-%m-%Y")

    evds = evdsAPI(EVDS_API_KEY)

    series = {
        "TUFE": "TP.FG.J0",          # TÜFE (2003=100)
        "UFE": "TP.FG.J01",          # Yurtiçi ÜFE (2003=100)
        "M2": "TP.PBD.H09",           # Para arzı M2 (Milyon TL)
        "TL_FAIZ": "TP.TRY.MT06" # TL mevduat faizi (%)
    }

    df = evds.get_data(
        list(series.values()),
        startdate=start_evds,
        enddate=end_evds,
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


def save_bronze(df, start_date="2020-01-01", end_date="2024-12-31"):
    """
    Save bronze layer in dual format (CSV + Parquet) to match EPİAŞ pipeline.

    Args:
        df: DataFrame to save
        start_date: Start date for filename
        end_date: End date for filename
    """
    bronze_path = os.path.join("data", "bronze", "macro")
    os.makedirs(bronze_path, exist_ok=True)

    # Legacy CSV format (kept for backward compatibility)
    csv_path = os.path.join(bronze_path, "macro_evds_raw.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ Bronze CSV saved to {csv_path}")

    # Parquet format (aligned with EPİAŞ pipeline)
    parquet_path = os.path.join(bronze_path, f"macro_evds_{start_date}_{end_date}.parquet")
    df.to_parquet(parquet_path, engine='pyarrow', compression='snappy')
    print(f"✅ Bronze Parquet saved to {parquet_path}")


if __name__ == "__main__":
    # Date range aligned with EPİAŞ pipeline
    START_DATE = "2020-01-01"
    END_DATE = "2024-12-31"

    # fetching
    df = fetch_evds_data(start_date=START_DATE, end_date=END_DATE)

    # rebasing for TÜFE and ÜFE
    df = rebase_index(df, "TUFE", base_date="2022-01")
    df = rebase_index(df, "UFE", base_date="2022-01")

    save_bronze(df, start_date=START_DATE, end_date=END_DATE)
