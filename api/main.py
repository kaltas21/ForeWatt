import os
from fastapi import FastAPI
from influxdb_client import InfluxDBClient, Point, WritePrecision

app = FastAPI()

client = InfluxDBClient(
    url="http://influxdb:8086",
    token=os.getenv("INFLUXDB_TOKEN"),
    org=os.getenv("INFLUXDB_ORG"),
)
write_api = client.write_api()
query_api = client.query_api()
bucket = os.getenv("INFLUXDB_BUCKET")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/write_test")
def write_test():
    p = Point("load").tag("region","TR").field("value", 12345).time(None, WritePrecision.S)
    write_api.write(bucket=bucket, record=p)
    return {"ok": True}

@app.get("/read_test")
def read_test():
    q = f'from(bucket:"{bucket}") |> range(start: -1h) |> limit(n:5)'
    tables = query_api.query(q)
    count = sum(1 for _t in tables for _r in _t.records)
    return {"rows": count}
