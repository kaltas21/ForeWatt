import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CSV_PATH = Path("reports/anomaly_results.csv")

def load():
    df = pd.read_csv(CSV_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

def plot_window(df, start=None, end=None, title=""):
    d = df.copy()
    if start is not None:
        d = d[d["timestamp"] >= pd.to_datetime(start)]
    if end is not None:
        d = d[d["timestamp"] <= pd.to_datetime(end)]

    anom = d[d["is_anomaly"] == 1]

    # 1) consumption vs forecast (zoomed)
    plt.figure(figsize=(14, 5))
    plt.plot(d["timestamp"], d["consumption"], label="consumption", linewidth=1.2, alpha=0.9)
    plt.plot(d["timestamp"], d["consumption_forecast"], label="forecast", linewidth=1.0, alpha=0.7)

    # anomalies on top (big + contrasting)
    plt.scatter(
        anom["timestamp"],
        anom["consumption"],
        label=f"anomaly ({len(anom)})",
        s=22,
        alpha=0.9,
        zorder=5
    )

    plt.title(title or "Consumption vs Forecast (Zoomed)")
    plt.xlabel("time")
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2) residual plot (anomaly is clearest here)
    plt.figure(figsize=(14, 4))
    plt.plot(d["timestamp"], d["residual"], label="residual (actual-forecast)", linewidth=1.0, alpha=0.9)
    plt.axhline(0, linewidth=1, alpha=0.6)

    plt.scatter(
        anom["timestamp"],
        anom["residual"],
        label="anomaly residual",
        s=22,
        alpha=0.9,
        zorder=5
    )

    plt.title((title + " | Residual") if title else "Residual (Zoomed)")
    plt.xlabel("time")
    plt.ylabel("residual")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_last_days(df, days=30):
    end = df["timestamp"].max()
    start = end - pd.Timedelta(days=days)
    plot_window(df, start=start, end=end, title=f"Last {days} days")

def plot_top_k_anomaly_windows(df, k=8, hours_before=48, hours_after=48):
    # en “anomal” olanları skorla sırala
    top = df[df["is_anomaly"] == 1].sort_values("anomaly_score", ascending=False).head(k)

    if top.empty:
        print("No anomalies found.")
        return

    for i, row in enumerate(top.itertuples(index=False), start=1):
        center = row.timestamp
        start = center - pd.Timedelta(hours=hours_before)
        end = center + pd.Timedelta(hours=hours_after)
        plot_window(
            df,
            start=start,
            end=end,
            title=f"Top anomaly #{i} | {center} | score={row.anomaly_score:.4f}"
        )

if __name__ == "__main__":
    df = load()

    # 1) hızlı anlaşılır zoom’lar
    plot_last_days(df, 7)
    plot_last_days(df, 30)
    plot_last_days(df, 90)

    # 2) en güçlü anomaliler etrafında pencere pencere inceleme
    plot_top_k_anomaly_windows(df, k=6, hours_before=48, hours_after=48)
