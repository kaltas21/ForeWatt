import pandas as pd
from pathlib import Path
from src.data.calendar_features import add_calendar_features_hourly

def test_daily_calendar_rowcount_and_weekend_flags():
    cal = pd.read_csv("data/silver/calendar/calendar_full_days.csv", parse_dates=["date_only"])
    assert len(cal) == 2192
    dow_map = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}
    cal["dow_name"] = cal["dow"].map(dow_map)
    assert ((cal["dow_name"].isin(["Mon","Tue","Wed","Thu","Fri"]) & (cal["is_weekend"]==1)).sum()) == 0
    assert ((cal["dow_name"].isin(["Sat","Sun"]) & (cal["is_weekend"]==0)).sum()) == 0

def test_half_day_pm_logic():
    days = pd.read_csv("data/silver/calendar/calendar_days.csv", parse_dates=["date_only"])
    row = days[days["date_only"].dt.strftime("%Y-%m-%d")=="2025-10-28"]
    assert not row.empty and row["half_day"].iloc[0] == "pm"

    # synthetic hourly frame for 2025-10-28
    idx = pd.date_range("2025-10-28 08:00", "2025-10-28 18:00", freq="h", tz="Europe/Istanbul")
    dummy = pd.DataFrame(index=idx)
    out = add_calendar_features_hourly(dummy, "data/silver/calendar/calendar_days.csv")
    assert (out.between_time("08:00","12:59")["is_holiday_hour"]==0).all()
    assert (out.between_time("13:00","18:00")["is_holiday_hour"]==1).all()
