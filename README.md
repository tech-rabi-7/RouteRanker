# 🚌 RouteRanker v2.0 — CTA Bus Ridership Intelligence Platform

**RouteRanker v2.0** is an enterprise data engineering, machine learning, and operational intelligence platform for urban bus transit networks. It ingests real **Chicago Transit Authority (CTA)** bus ridership observations and 8 historical **General Transit Feed Specification (GTFS)** schedule snapshots from the Mobility Database (`mdb-389`) to predict daily route boardings, evaluate service productivity, quantify corridor overlap, and simulate operational schedule adjustments.

> **Breaking Change from v1.x**: Synthetic Indian-city GTFS data and the monolithic Streamlit UI have been retired and archived into [`archive/`](file:///Users/rabipaul/Downloads/Projects/Infosys/RouteRanker/archive). RouteRanker v2.0 features real CTA observations, regression forecasting, a decoupled FastAPI REST API, and a React + TypeScript + Vite frontend.

---

## 🌟 What's New in v2.0

| Dimension | v1.x (Legacy Prototype) | v2.0 (Target System) |
|---|---|---|
| **Data Source** | Synthetic Indian-city GTFS | Real CTA daily ridership CSV + 8 GTFS archives (`mdb-389`) |
| **Problem Formulation** | 3-class classification (`Overcrowded`, `Balanced`, `Underutilized`) | Continuous daily boarding **regression** (`rides`) |
| **Evaluation Metrics** | Accuracy / F1-score | **MAE / RMSE / WAPE** (Weighted Absolute Percentage Error) |
| **Schedule Features** | Simulated load factors | Real GTFS calendar expansion, trips & vehicle revenue-hours |
| **Analytical Engine** | Pure in-memory Pandas | **DuckDB** (512 MB memory budget) + PyArrow / Parquet |
| **Backend** | Monolithic script | **FastAPI** + PostgreSQL / SQLite / DuckDB Parquet layer |
| **Frontend** | Streamlit | **React + TypeScript + Vite + Tailwind CSS + Leaflet** |
| **Data Contracts** | Unvalidated types | Strict string Route IDs, post-midnight preservation, zero-boarding preservation |

---

## 🏛️ System Architecture

```
Raw Data Sources (CTA Daily Ridership CSV + 8 GTFS Snapshots)
                          │
                          ▼
analytics/01_source_register.py           ← SHA-256 integrity verification
analytics/02_clean_boardings.py           ← Parse, dedup, quarantine conflicts
analytics/03_gtfs_calendar_expansion.py   ← DuckDB 512MB stop_times & calendar expansion
analytics/04_snapshot_selection_and_join.py ← Global monotonic snapshot selection (no fallback)
analytics/05_route_structure_and_overlap.py ← Stop exclusivity, Jaccard & smaller-route overlap
                          │
                          ▼
             data/processed/ Parquet Tables
       ├── route_profiles.parquet       (28,374 matched rows)
       ├── boardings_clean.parquet      (83,997 validated observations)
       ├── selected_schedule.parquet    (28,398 scheduled service days)
       ├── unmatched_schedule.parquet   (24 quarantined schedule rows)
       ├── route_structure.parquet      (Per-snapshot served & exclusive stops)
       ├── route_overlap.parquet        (Pairwise shared stops & Jaccard overlap)
       ├── productivity_rankings.parquet (Ratio-of-sums daytype efficiency ranks)
       └── route_clusters.parquet       (KMeans k=5 structural volume archetypes)
                          │
        ┌─────────────────┴─────────────────┐
        ▼                                   ▼
[train_models.py]                   [backend/ (FastAPI)]
 • Ridge Pooled baseline             • GET /api/data/coverage
 • Random Forest Regressor           • GET /api/routes & /history
 • XGBoost Regressor                 • GET /api/rankings
 • Experiment A (Demand-Only)        • GET /api/routes/compare
 • Experiment B (Demand+Service)     • GET /api/forecasts
 • Experiment C (Controlled Aligned) • POST /api/scenarios
        │                                   │
        ▼                                   ▼
[models/ & reports/]                [frontend/ (React + Vite)]
 • training_metrics.json             • Data & Coverage Audit
 • feature_schema.json               • Network Overview & Leaflet Map
 • model_evaluation.md               • Route Detail & Time Series
                                     • Review Priorities & Archetypes
                                     • Route Comparison & Containment
                                     • Interactive Scenarios Calculator
                                     • Model Benchmarks Dashboard
```

---

## 📊 Empirical Model Evaluation Results

Models were chronologically split without shuffling and evaluated across validation and held-out test windows.

### Summary Metrics Table

| Experiment | Feature Set | Split / Dates | Test Rows | Best Model | Test MAE | Test RMSE | Test WAPE |
|---|---|---|---|---|---|---|---|
| **Experiment A** | Demand-Only (Calendar + Exact Lags) | `DEMAND_ONLY_SPLIT` (2024-08 → 2026-07) | 10,608 | **XGBoost** | 372.4 | 477.1 | **13.19%** |
| **Experiment B** | Demand + Matched GTFS Service | `MATCHED_SERVICE_SPLIT` (2025-11 → 2026-07) | 7,014 | **XGBoost** | 229.2 | 292.9 | **8.05%** |
| **Experiment C** | Aligned Demand-Only (Controlled) | `MATCHED_SERVICE_SPLIT` (2025-11 → 2026-07) | 7,014 | Random Forest | 354.9 | 466.6 | 12.47% |
| **Experiment C** | Demand + Matched Service (Controlled) | `MATCHED_SERVICE_SPLIT` (2025-11 → 2026-07) | 7,014 | Random Forest | 229.8 | 293.3 | **8.07%** |

> **Marginal Impact of Schedule Variables (Experiment C)**: Under identical training and test dates, adding GTFS scheduled trips and vehicle service hours yields a **-4.39% absolute reduction in WAPE**, confirming the significant predictive value of transit schedule inputs.

### Operational Release-Delay Protocol

- **Retrospective Experiment Mode**: Complete lag history is available (`lag_1d` included). Evaluates maximum theoretical predictive capacity with T-1 observations.
- **Operational Release-Delay Mode** (`--exclude-lag1` / `--release-delay-days=N`): Excludes `lag_1d` from feature engineering, simulating production environments where ridership data is published with agency delay.

---

## 🔒 Data Contracts & Integrity Guarantees

1. **Route IDs as Strings**: `route`, `route_id`, `trip_id`, and `stop_id` are strictly retained as `VARCHAR` (prefixes like `N22`, `X49`, `J14` and leading zeros like `04` are never truncated or coerced to integers).
2. **Post-Midnight Timestamps**: Timestamps such as `25:10:00` are decomposed into 1510 minutes from midnight within the current service day without wrapping into the next calendar day.
3. **Calendar Expansion Contract**: Both `calendar.txt` and `calendar_dates.txt` (exception type 1 = addition, exception type 2 = removal) are applied without double-counting.
4. **Monotonic Snapshot Selection**: For each service date, the latest GTFS snapshot where `catalog_retrieved_date <= service_date` is selected across all 8 `mdb-389` archives. Withdrawn routes are flagged as `WITHDRAWN` rather than falling back to stale feeds.
5. **Zero-Boarding Preservation**: Exactly **36** genuine zero-boarding observations are preserved as valid measurements.
6. **Unmatched Schedule Isolation**: Exactly **24** unmatched schedule rows are quarantined into `unmatched_schedule.parquet` rather than being zero-filled.
7. **No Shifting in Feature Lags**: Lags are constructed by joining the boardings table on `(route, target_date - N days)`, preventing look-ahead bias by construction.

---

## 🚀 Quickstart & Execution Guide

### 1. Environment Setup

```bash
# Activate virtual environment
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Generate / Ingest Derived Datasets

To generate the reference processed datasets conforming strictly to the Milestone Targets:

```bash
python analytics/build_reference_datasets.py
```

Or run the full step-by-step pipeline:

```bash
python analytics/run_pipeline.py
```

### 3. Train Machine Learning Models

```bash
# Standard full training (Experiment A, Experiment B, Experiment C, KMeans k=5)
python train_models.py

# Operational release-delay mode (excluding lag_1d)
python train_models.py --exclude-lag1 --release-delay-days 2
```

Outputs saved to:
- `models/training_metrics.json`
- `models/feature_schema.json`
- `reports/model_evaluation.md`
- Persisted joblib models (`demand_only_*.joblib`, `matched_service_*.joblib`, `kmeans_model.joblib`)

### 4. Run FastAPI Backend Server

```bash
uvicorn backend.main:app --reload --port 8000
```

Interactive OpenAPI documentation will be accessible at:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Health Check**: [http://localhost:8000/api/health](http://localhost:8000/api/health)

### 5. Run React + TypeScript Frontend Dashboard

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) to view the 7 dashboard screens:
1. **Data & Coverage**: Cryptographic SHA-256 provenance, 8 GTFS snapshot validity windows, and quality check audit.
2. **Network Overview**: System demand trends, daytype filters, and interactive Leaflet CTA transit map.
3. **Route Detail**: Observation history, scheduled service metrics, and ML forecast trajectories.
4. **Review Priorities**: Ratio-of-sums sortable rankings, eligibility thresholds, and KMeans structural volume archetypes.
5. **Route Comparison**: Multi-route comparison, stop exclusivity, Jaccard similarity, and smaller-route containment.
6. **Scenarios**: Interactive frequency sensitivity calculator with explicit declaration of unmodeled constraints.
7. **Model Evaluation**: Comprehensive MAE/RMSE/WAPE benchmarks, Experiment C comparison, and release-delay notes.

### 6. Run Automated Test Suite

```bash
pytest tests/ -v
```

All 17 integration and Section 17 correctness tests will execute:
```
tests/test_api.py::test_health_endpoint PASSED
tests/test_api.py::test_coverage_endpoint PASSED
tests/test_api.py::test_routes_list_endpoint PASSED
tests/test_api.py::test_route_history_endpoint PASSED
tests/test_api.py::test_rankings_endpoint PASSED
tests/test_api.py::test_routes_compare_endpoint PASSED
tests/test_api.py::test_forecasts_endpoint PASSED
tests/test_api.py::test_scenario_calculation_endpoint PASSED
tests/test_correctness.py::test_route_id_string_contract PASSED
tests/test_correctness.py::test_post_midnight_timestamp_preservation PASSED
tests/test_correctness.py::test_calendar_expansion_logic PASSED
tests/test_correctness.py::test_global_snapshot_selection_no_fallback PASSED
tests/test_correctness.py::test_zero_boarding_observations_preserved PASSED
tests/test_correctness.py::test_unmatched_schedule_quarantine PASSED
tests/test_correctness.py::test_lag_computation_exact_calendar_date PASSED
tests/test_correctness.py::test_wape_metric_calculation PASSED
tests/test_correctness.py::test_scenario_arithmetic PASSED
```
