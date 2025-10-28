# ForeWatt
Energy demand forecasting and anomaly detection system with EPİAŞ + PJM datasets, weather covariates (Open-Meteo, NASA POWER), N-HiTS/PatchTST deep learning models, uncertainty quantification via conformal prediction, anomaly detection with Isolation Forest, and EV load-shifting optimization.

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
  ├─ requirements.txt
  ├─ Makefile
  └─ README.md
```

