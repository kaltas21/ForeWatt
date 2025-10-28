# ForeWatt — Hourly Electricity Demand Forecasting, Uncertainty, and Anomaly Diagnostics

> **ForeWatt** is a fully reproducible, open-source platform for **1–24h** electricity demand forecasting with **calibrated prediction intervals**, **actionable anomaly diagnostics**, and an optional **EV load-shifting optimizer**.
> Stack: **FastAPI**, **MLflow**, **QuestDB**, **Streamlit**, **Docker Compose**.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#license)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](#requirements)
[![Reproducible](https://img.shields.io/badge/reproducible-mlflow%20%7C%20docker-informational)](#mlops--reproducibility)

---

## Table of Contents

* [Why ForeWatt?](#why-forewatt)
* [System Overview](#system-overview)
* [Repo Structure](#repo-structure)
* [Quickstart (Docker)](#quickstart-docker)
* [Local Dev (no Docker)](#local-dev-no-docker)
* [Data & Features](#data--features)
* [Models & Training](#models--training)
* [Uncertainty Calibration](#uncertainty-calibration)
* [Anomaly Detection](#anomaly-detection)
* [EV Load-Shifting Optimizer (Stretch)](#ev-load-shifting-optimizer-stretch)
* [API](#api)
* [Dashboard](#dashboard)
* [Evaluation & Targets](#evaluation--targets)
* [MLOps & Reproducibility](#mlops--reproducibility)
* [Roadmap](#roadmap)
* [Contributing](#contributing)
* [Cite / Acknowledgments](#cite--acknowledgments)

---

## Why ForeWatt?

* **Operational**: day-ahead forecasts with **90% split-conformal intervals**; single-horizon latency **≤ 300 ms** on CPU.
* **Auditable & open**: public data (EPİAŞ, PJM, Open-Meteo), zero license cost, full run traceability via **MLflow**.
* **Actionable**: anomaly alerts with **level-shift/weekly-drift** diagnostics and feature attributions.
* **Decision support**: optional LP optimizer turns forecasts into **cost-reducing EV charging schedules**.

---

## System Overview

```
EPİAŞ / PJM / Open-Meteo  --(APScheduler)-->  Bronze (parquet)
                                           ->  Silver (validated, tz: Europe/Istanbul)
                                           ->  Gold (features: lags/rolls/Fourier/weather)

Gold  ->  Models (N-HiTS, CatBoost, Prophet) -> Ensemble -> Conformal Calibration
     ->  Anomaly (IsolationForest + diagnostics)
     ->  Optimizer (LP, cost under tariffs)

FastAPI  <->  QuestDB (TS store)  <->  Streamlit Dashboard
MLflow: experiments, artifacts, registry (champion/challenger)
```

---

## Repo Structure

```
forewatt/
  ├─ apps/
  │   ├─ api/                  # FastAPI service (forecast/intervals/anomalies/optimize/health/metadata)
  │   └─ dashboard/            # Streamlit app
  ├─ dataflow/
  │   ├─ ingest/               # APScheduler jobs, EPİAŞ/PJM/Open-Meteo pullers
  │   ├─ bronze_silver/        # normalization, validation, tz handling
  │   └─ gold_features/        # feature engineering (lags/rolls/Fourier/weather)
  ├─ models/
  │   ├─ baselines/            # Prophet, CatBoost, XGBoost, SARIMA, DLinear
  │   ├─ nhits/                # N-HiTS core forecaster
  │   ├─ ensemble/             # weighted-median aggregator
  │   └─ calibrate/            # conformal wrappers (MAPIE/Darts)
  ├─ anomalies/
  │   └─ isolation_forest/     # training + diagnostics (level-shift, weekly-drift)
  ├─ optimizer/
  │   └─ ev_lp/                # PuLP/CBC linear program + I/O
  ├─ mlops/
  │   ├─ tracking/             # MLflow utils, run tags, registry helpers
  │   └─ ops/                  # health checks, seeds, smoke tests
  ├─ configs/
  │   ├─ creds.example.env     # copy to .env and fill (no secrets in Git)
  │   ├─ data.yaml             # endpoints, regions, calendars
  │   └─ model.yaml            # hyperparams, CV, calibration window
  ├─ docker/
  │   ├─ Dockerfile.api
  │   ├─ Dockerfile.dashboard
  │   └─ docker-compose.yml
  ├─ scripts/                  # seed replay, backfill, benchmarks
  ├─ tests/                    # pytest (data, models, api, dashboard)
  ├─ docs/                     # documentation
  ├─ requirements.txt
  ├─ Makefile
  └─ README.md
```
---

## Quickstart (Docker)

1.⁠ ⁠*Clone & configure*

⁠ bash
git clone https://github.com/<org-or-user>/forewatt.git
cd forewatt
cp configs/creds.example.env .env   # fill any required tokens/keys
 ⁠

2.⁠ ⁠*Bring the stack up*

⁠ bash
docker compose -f docker/docker-compose.yml up --build
 ⁠

3.⁠ ⁠*Services*

•⁠  ⁠API (OpenAPI): ⁠ http://localhost:8000/docs ⁠
•⁠  ⁠Dashboard: ⁠ http://localhost:8501 ⁠
•⁠  ⁠MLflow UI (optional): ⁠ http://localhost:5000 ⁠

	⁠*Freshness guard:* if upstream data are > 2 hours stale, ⁠ /forecast ⁠ returns *503* with a human-readable message.

---

## Local Dev (no Docker)

⁠ bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) Ingest → Bronze
python -m dataflow.ingest.run

# 2) Bronze → Silver
python -m dataflow.bronze_silver.to_silver

# 3) Silver → Gold (features)
python -m dataflow.gold_features.build

# 4) Train & log baselines
python -m models.baselines.train --config configs/model.yaml

# 5) Start API + Dashboard
uvicorn apps.api.main:app --reload --port 8000
streamlit run apps/dashboard/Home.py --server.port 8501
 ⁠

---

## Data & Features

•⁠  ⁠*Sources: EPİAŞ hourly load (TR national + DSO subsets), PJM hourly load (benchmark), **Open-Meteo* weather.
•⁠  ⁠*Timezone: **Europe/Istanbul (UTC+3)* end-to-end.
•⁠  ⁠*Medallion*: bronze (raw) → silver (validated schema/types/tz) → gold (features).
•⁠  ⁠*Features: lags (1,2,3,6,12,24,168h), rolling stats (3–168h), Fourier (24h, 168h), calendar (hour/day/month, **Ramadan*), weather + 1h lags.
•⁠  ⁠*Missing-data policy*: small gaps forward-fill (load ≤ 2h) / interpolate (weather ≤ 6h); longer gaps excluded from training.
•⁠  ⁠*Config*: ⁠ configs/data.yaml ⁠.

---

## Models & Training

•⁠  ⁠*Core: **N-HiTS* (multi-resolution), *CatBoost, **Prophet; optional **XGBoost/SARIMA/DLinear*.
•⁠  ⁠*Ensemble: weighted **median*; static weights ∝ inverse validation sMAPE (penalize under-coverage).
•⁠  ⁠*Validation*: expanding-window temporal CV; held-out test.
•⁠  ⁠*Tuning*: Optuna (30–50 trials caps).
•⁠  ⁠*Targets: day-ahead **sMAPE 4–6%* (EPİAŞ subsets), *MASE < 1.0*.
•⁠  ⁠*Repro: all runs tracked in **MLflow* with artifacts, params, and plots.
•⁠  ⁠*Config*: ⁠ configs/model.yaml ⁠.

---

## Uncertainty Calibration

•⁠  ⁠*Method: **Split Conformal Prediction* per horizon (1…24), rolling 28-day residual window.
•⁠  ⁠*Outputs*: ⁠ q05, q50, q95 ⁠ (90% nominal); coverage & width monitored.
•⁠  ⁠*Libs*: MAPIE / Darts wrappers.

---

## Anomaly Detection

•⁠  ⁠*Primary: **IsolationForest* on residuals + lags/rolls; hysteresis (≥ 2h) to reduce chatter.
•⁠  ⁠*Diagnostics*: level-shift test, weekly-drift check, basic feature attributions.
•⁠  ⁠*Targets*: ≥ 0.80 precision at ≤ 5% false positive rate on validation weeks.

---

## EV Load-Shifting Optimizer (Stretch)

•⁠  ⁠*Formulation: LP in **PuLP* (CBC solver) with hourly vars; minimize MCP-based cost.
•⁠  ⁠*Constraints*: power limits, energy-by-deadline, optional degradation penalty.
•⁠  ⁠*Output*: schedule, cost delta, feasibility slack; integrated in dashboard.

---

## API

Typed *FastAPI* endpoints (see ⁠ /docs ⁠):

| Endpoint          | Purpose                                   |
| ----------------- | ----------------------------------------- |
| ⁠ GET /health ⁠     | Freshness, last ingest, latency, uptime   |
| ⁠ GET /metadata ⁠   | Model + feature pipeline hashes           |
| ⁠ POST /forecast ⁠  | 1–24h forecasts                           |
| ⁠ POST /intervals ⁠ | Conformal intervals per horizon           |
| ⁠ GET /anomalies ⁠  | Recent anomaly events + diagnostics refs  |
| ⁠ POST /optimize ⁠  | EV schedule given price/limits (optional) |

*Example*

⁠ json
POST /forecast
{
  "region": "TR-National",
  "horizons": [1, 2, 3, 24]
}
 ⁠

---

## Dashboard

•⁠  ⁠*Forecasts: last 14 days + next 24h with shaded **90%* intervals.
•⁠  ⁠*Anomalies*: markers with drill-down diagnostics.
•⁠  ⁠*Status ribbon*: coverage, width, sharpness, freshness, model version.
•⁠  ⁠*Optimizer panel: baseline vs optimized schedule and **cost delta*.

Run:

⁠ bash
streamlit run apps/dashboard/Home.py
 ⁠

---

## Evaluation & Targets

•⁠  ⁠*Point*: sMAPE, MASE, MAE, RMSE (per horizon).
•⁠  ⁠*Probabilistic: Pinball, **CRPS, Winkler; **90% coverage* ± 5pp.
•⁠  ⁠*Latency: p95 single-horizon *≤ 300 ms** (CPU).
•⁠  ⁠*Staleness: serve only if *≤ 120 minutes** since last ingest.
•⁠  ⁠*Generalization*: EPİAŞ primary + PJM benchmarking.

Reproduce headline numbers:

⁠ bash
make backtest      # temporal CV on configured datasets
make benchmark     # aggregates sMAPE/MASE/coverage → reports/
 ⁠

---
