import pandas as pd
from src.data.calendar_features import add_calendar_features_hourly

def test_add_calendar_features_half_day_pm():
    tz = "Europe/Istanbul"
    idx = pd.date_range("2025-10-28 08:00","2025-10-28 18:00", freq="h", tz=tz)
    df = pd.DataFrame(index=idx, data={"x": 0})
    out = add_calendar_features_hourly(df, "data/silver/calendar/calendar_days.csv")

    assert "is_holiday_day" in out.columns
    assert "is_holiday_hour" in out.columns
    # Half-day logic:
    assert (out.between_time("08:00","12:59")["is_holiday_hour"]==0).all()
    assert (out.between_time("13:00","18:00")["is_holiday_hour"]==1).all()
