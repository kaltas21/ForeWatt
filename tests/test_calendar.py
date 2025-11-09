import pandas as pd

def test_calendar_basic_flags():
    cal_days = pd.read_csv("data/silver/calendar/calendar_days.csv", parse_dates=["date_only"])
    dates = set(cal_days["date_only"].dt.strftime("%Y-%m-%d"))

    for d in ["2025-03-30","2025-03-31","2025-04-01"]:
        assert d in dates  # Ramazan Bayramı

    for d in ["2025-06-06","2025-06-07","2025-06-08","2025-06-09"]:
        assert d in dates  # Kurban Bayramı

    half = cal_days[cal_days["date_only"].dt.strftime("%Y-%m-%d") == "2025-10-28"]["half_day"].iloc[0]
    assert half == "pm"
