# Create a daily calendar (2021-01-01 to 2025-12-31) with holiday flags for Türkiye.
import pandas as pd
from datetime import date, timedelta
import json

def daterange(start, end):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)

# Holiday ranges (inclusive) covering 2021–2025
HOLIDAYS = []

def add_fixed(y):
    HOLIDAYS.extend([
        (date(y,1,1), date(y,1,1), "Yılbaşı", "national"),
        (date(y,4,23), date(y,4,23), "Ulusal Egemenlik ve Çocuk Bayramı", "national"),
        (date(y,5,1), date(y,5,1), "Emek ve Dayanışma Günü", "national"),
        (date(y,5,19), date(y,5,19), "Atatürk'ü Anma, Gençlik ve Spor Bayramı", "national"),
        (date(y,7,15), date(y,7,15), "Demokrasi ve Milli Birlik Günü", "national"),
        (date(y,8,30), date(y,8,30), "Zafer Bayramı", "national"),
        # Republic Day is officially 29 Oct, with 28 Oct afternoon. Flag only 29 Oct here.
        (date(y,10,29), date(y,10,29), "Cumhuriyet Bayramı", "national"),
    ])

for y in range(2021, 2026):
    add_fixed(y)

# Religious holiday ranges (verified for each year)
# 2021
HOLIDAYS += [
    (date(2021,5,13), date(2021,5,15), "Ramazan Bayramı", "religious"),
    (date(2021,7,20), date(2021,7,23), "Kurban Bayramı", "religious"),
]
# 2022
HOLIDAYS += [
    (date(2022,5,2), date(2022,5,4), "Ramazan Bayramı", "religious"),
    (date(2022,7,9), date(2022,7,12), "Kurban Bayramı", "religious"),
]
# 2023
HOLIDAYS += [
    (date(2023,4,21), date(2023,4,23), "Ramazan Bayramı", "religious"),
    (date(2023,6,28), date(2023,7,1), "Kurban Bayramı", "religious"),
]
# 2024
HOLIDAYS += [
    (date(2024,4,10), date(2024,4,12), "Ramazan Bayramı", "religious"),
    (date(2024,6,16), date(2024,6,19), "Kurban Bayramı", "religious"),
]
# 2025 (from official/credible sources: MEB calendar & consolidated holiday sites)
HOLIDAYS += [
    (date(2025,3,30), date(2025,4,1), "Ramazan Bayramı", "religious"),
    (date(2025,6,6), date(2025,6,9), "Kurban Bayramı", "religious"),
]

# School semester breaks (Sömestr / Yarıyıl) - long ranges as in earlier file + MEB 2025
SEMESTER_BREAKS = [
    (date(2021,1,25), date(2021,2,15), "Sömestr Tatili (Yarıyıl)", "school"),
    (date(2022,1,24), date(2022,2,4), "Sömestr Tatili (Yarıyıl)", "school"),
    (date(2023,1,23), date(2023,2,3), "Sömestr Tatili (Yarıyıl)", "school"),
    (date(2024,1,20), date(2024,2,4), "Sömestr Tatili (Yarıyıl)", "school"),
    (date(2025,1,20), date(2025,1,31), "Sömestr Tatili (Yarıyıl)", "school"),
]
HOLIDAYS += SEMESTER_BREAKS

# Build daily frame
start = date(2021,1,1)
end = date(2025,12,31)

rows = []
# Build a lookup of holiday days
holiday_map = {}
for s,e,name,typ in HOLIDAYS:
    d = s
    while d <= e:
        # If multiple holidays overlap, prefer religious over school, then national, keep both names concatenated
        if d not in holiday_map:
            holiday_map[d] = {"name": name, "type": typ}
        else:
            # concatenate names and pick priority
            holiday_map[d]["name"] = holiday_map[d]["name"] + " | " + name
            order = {"religious":2,"national":1,"school":0}
            if order[typ] > order[holiday_map[d]["type"]]:
                holiday_map[d]["type"] = typ
        d += timedelta(days=1)

for d in daterange(start, end):
    rows.append({
        "date": d.isoformat(),
        "day_of_week": d.strftime("%A"),
        "is_weekend": d.weekday() >= 5,   # Sat/Sun
        "is_holiday": d in holiday_map,
        "holiday_name": holiday_map.get(d, {}).get("name"),
        "holiday_type": holiday_map.get(d, {}).get("type"),
    })

df = pd.DataFrame(rows)
csv_path = "./src/data/static/holiday/turkey_calendar_2021_2025_daily.csv"
json_path = "./src/data/static/holiday/turkey_calendar_2021_2025_daily.json"
df.to_csv(csv_path, index=False, encoding="utf-8")
df.to_json(json_path, orient="records", force_ascii=False, indent=2)


