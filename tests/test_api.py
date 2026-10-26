"""
tests/test_api.py
=================
Integration tests validating all RouteRanker v2.0 FastAPI REST API endpoints.
"""

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import app

client = TestClient(app)


def test_health_endpoint():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["version"] == "2.0.0"


def test_coverage_endpoint():
    res = client.get("/api/data/coverage")
    assert res.status_code == 200
    data = res.json()
    assert data["matched_rows_count"] == 28374
    assert data["unmatched_schedule_rows_count"] == 24
    assert data["total_routes"] == 125
    assert data["total_service_dates"] == 251
    assert data["zero_rides_count"] == 36
    assert len(data["snapshot_windows"]) == 8


def test_routes_list_endpoint():
    res = client.get("/api/routes")
    assert res.status_code == 200
    data = res.json()
    assert data["total_routes"] >= 100
    # Verify string route preserving prefixes
    routes = [r["route"] for r in data["routes"]]
    assert any(r.startswith("N") or r.startswith("X") or r.startswith("J") for r in routes)


def test_route_history_endpoint():
    res = client.get("/api/routes/20/history")
    assert res.status_code == 200
    data = res.json()
    assert data["route"] == "20"
    assert data["total_observations"] > 0
    assert len(data["history"]) > 0
    first_obs = data["history"][0]
    assert "rides" in first_obs
    assert "scheduled_trips" in first_obs


def test_rankings_endpoint():
    res = client.get("/api/rankings?daytype=W&sort_by=service_productivity_metric_rank")
    assert res.status_code == 200
    data = res.json()
    assert data["daytype"] == "W"
    assert data["eligible_routes_count"] > 0
    top1 = data["rankings"][0]
    assert top1["service_productivity_metric_rank"] == 1
    assert top1["boardings_per_scheduled_service_hour"] > 0


def test_routes_compare_endpoint():
    res = client.get("/api/routes/compare?routes=49&routes=X49")
    assert res.status_code == 200
    data = res.json()
    assert len(data["routes"]) == 2
    assert len(data["pairwise_overlaps"]) == 1
    ov = data["pairwise_overlaps"][0]
    assert ov["shared_stops_count"] > 0
    assert ov["smaller_route_overlap"] > 0


def test_forecasts_endpoint():
    res = client.get("/api/forecasts?route=20")
    assert res.status_code == 200
    data = res.json()
    assert data["route"] == "20"
    assert data["model_version"] == "2.0.0"
    assert "mae" in data["pooled_metrics"]
    assert "wape" in data["pooled_metrics"]
    assert len(data["data_points"]) > 0


def test_scenario_calculation_endpoint():
    payload = {
        "route": "20",
        "daytype": "W",
        "frequency_change_pct": 25.0,
    }
    res = client.post("/api/scenarios", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["route"] == "20"
    assert data["baseline"]["scheduled_trips"] > 0
    assert data["scenario"]["scheduled_trips"] > data["baseline"]["scheduled_trips"]
    assert len(data["missing_operational_constraints"]) == 5
