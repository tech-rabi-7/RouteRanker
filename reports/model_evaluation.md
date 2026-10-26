# RouteRanker v2.0 — Model Evaluation Report

**Generated**: 2026-10-26T10:28:29.446794+00:00  
**Artifact Version**: 2.0.0  

> **Retrospective Experiment Mode**: Complete lag history available (`lag_1d` included). Evaluates maximum theoretical predictive capacity with T-1 observed ridership.

> Primary metric: **WAPE** (Weighted Absolute Percentage Error).  
> WAPE = Σ|y − ŷ| / Σy.  Reported as `N/A` when Σy = 0 (zero-volume group).  
> Plain MAPE is NOT used because of genuine zero-ride observations.

---

## Experiment A — Demand Only

| Split | Rows |
|---|---|
| Train | 63,143 |
| Val   | 10,246 |
| Test  | 10,608 |

### Baselines

| Model | N | MAE | RMSE | WAPE |
|---|---|---|---|---|
| lag_7d — val | 10,246 | 503.3 | 658.6 | 0.1787 |
| lag_7d — test | 10,608 | 506.3 | 657.2 | 0.1795 |
| seasonal_mean — val | 10,246 | 395.0 | 515.8 | 0.1403 |
| seasonal_mean — test | 10,608 | 392.7 | 512.8 | 0.1392 |

### ML Models

| Model | N | MAE | RMSE | WAPE |
|---|---|---|---|---|
| ridge_pooled — val | 10,246 | 386.1 | 491.2 | 0.1371 |
| ridge_pooled — test | 10,608 | 383.8 | 488.0 | 0.1360 |
| random_forest — val | 10,246 | 382.6 | 490.4 | 0.1359 |
| random_forest — test | 10,608 | 381.4 | 486.3 | 0.1352 |
| xgboost — val | 10,246 | 372.2 | 479.3 | 0.1322 |
| xgboost — test | 10,608 | 372.4 | 477.1 | 0.1320 |

### Route-Level Test Metrics (best model: `xgboost`)

| Metric | Network-weighted | Median across 125 routes |
|---|---|---|
| MAE  | 372.4  | 365.2  |
| RMSE | 472.0 | 457.5 |
| WAPE | 0.1320 | 0.1297 |


---

## Experiment B — Demand + Matched Service

| Split | Rows |
|---|---|
| Train | 14,501 |
| Val   | 6,859 |
| Test  | 7,014 |

### Baselines

| Model | N | MAE | RMSE | WAPE |
|---|---|---|---|---|
| lag_7d — val | 6,859 | 499.8 | 654.8 | 0.1795 |
| lag_7d — test | 7,014 | 509.3 | 665.6 | 0.1790 |
| seasonal_mean — val | 6,859 | 389.3 | 512.7 | 0.1398 |
| seasonal_mean — test | 7,014 | 394.1 | 516.2 | 0.1385 |

### ML Models

| Model | N | MAE | RMSE | WAPE |
|---|---|---|---|---|
| ridge_pooled — val | 6,859 | 234.4 | 297.2 | 0.0842 |
| ridge_pooled — test | 7,014 | 241.2 | 304.6 | 0.0848 |
| random_forest — val | 6,859 | 224.0 | 287.0 | 0.0804 |
| random_forest — test | 7,014 | 229.8 | 293.3 | 0.0808 |
| xgboost — val | 6,859 | 221.7 | 284.8 | 0.0796 |
| xgboost — test | 7,014 | 229.2 | 292.9 | 0.0806 |

### Route-Level Test Metrics (best model: `xgboost`)

| Metric | Network-weighted | Median across 125 routes |
|---|---|---|
| MAE  | 229.2  | 228.6  |
| RMSE | 289.9 | 285.0 |
| WAPE | 0.0806 | 0.0805 |


---

## Experiment C — Controlled Aligned Comparison

> Evaluates Demand-Only and Demand+Service models over the EXACT same `MATCHED_SERVICE_SPLIT` dates to isolate the true marginal value of GTFS schedule features.

| Feature Set | Test N | Random Forest Test WAPE | Ridge Test WAPE |
|---|---|---|---|
| Aligned Demand-Only | 7,014 | 0.124725 | 0.125965 |
| Demand + Matched Service | 7,014 | 0.080784 | 0.084769 |

**Marginal WAPE Impact of GTFS Service Data**: `-0.0439` (Improvement)

---

## Productivity Ranking Summary

Eligible routes (≥20 obs) ranked by `boardings_per_scheduled_service_hour` within each daytype.

Total routes: **125**  
Weekday-eligible: **125**

### Top-5 Weekday Routes (by productivity rank)

| Rank | Route | Boardings/Service Hour |
|---|---|---|
| 1 | 311 | 44.45 |
| 2 | N20 | 43.49 |
| 3 | 146 | 44.32 |
| 5 | 147 | 45.40 |
| 6 | 34 | 44.97 |

---

## Route Structural Clustering (KMeans)

**n_clusters**: 5  
> Cluster IDs are structural archetypes only — not performance labels.

| Cluster ID | Route Count |
|---|---|
| 0 | 113 |
| 1 | 2 |
| 2 | 4 |
| 3 | 5 |
| 4 | 1 |

---
