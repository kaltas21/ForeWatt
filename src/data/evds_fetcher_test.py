from evds import evdsAPI

# API keyini buraya yapıştır
evds = evdsAPI('RiQOjxgjLA')

# Örnek: USD ve EUR alış kurları (2019 arası)
df = evds.get_data(
    ['TP.DK.USD.A.YTL', 'TP.DK.EUR.A.YTL'],
    startdate="01-01-2019",
    enddate="01-01-2020"
)

print(df.head())

print("Toplam satır:", len(df))
print("İlk tarih:", df["Tarih"].iloc[0])
print("Son tarih:", df["Tarih"].iloc[-1])
print(df.tail())  # son 5 satırı gösterir
