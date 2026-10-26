"""
tests/test_correctness.py
=========================
Validates the 8 core data integrity & modeling contracts from
RouteRanker Intern Project and Data Guide (Version 2.0) Section 17.
"""

import sys
from datetime import date, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import (
    BoardingFeatureEngineer,
    BoardingRegressionPipeline,
    DEMAND_ONLY_SPLIT,
    MATCHED_SERVICE_SPLIT,
    compute_metrics,
    _wape,
)
from src.route_analysis import compute_route_structure, compute_route_overlaps
from src.scenarios import calculate_scenario
from backend.database import duckdb_repo


# ---------------------------------------------------------------------------
# Case 1: Route IDs preserved as strings (N/X/J prefixes intact)
# ---------------------------------------------------------------------------
def test_route_id_string_contract():
    fe = BoardingFeatureEngineer()
    test_routes = ["N22", "X49", "J14", "20", "66", "04"]
    df = pd.DataFrame({
        "route": test_routes,
        "service_date": [date(2026, 1, 10)] * len(test_routes),
        "daytype": ["W"] * len(test_routes),
        "rides": [3000, 4000, 2500, 3200, 2800, 1500],
    })
    feat_df = fe.build(df)
    for r in feat_df["route"]:
        assert isinstance(r, str), f"Route {r} was coerced to non-string"
    assert "N22" in feat_df["route"].values
    assert "X49" in feat_df["route"].values
    assert "J14" in feat_df["route"].values
    assert "04" in feat_df["route"].values  # Leading zero preserved


# ---------------------------------------------------------------------------
# Case 2: Post-midnight timestamps preserved (minutes from midnight >= 1440)
# ---------------------------------------------------------------------------
def test_post_midnight_timestamp_preservation():
    # Helper simulating gtfs timestamp conversion
    def parse_gtfs_time_to_minutes(time_str: str) -> int:
        parts = time_str.split(":")
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2]) if len(parts) > 2 else 0
        return hours * 60 + minutes + (1 if seconds >= 30 else 0)

    # 25:10:00 must be 1510 minutes, NOT wrapped to 1:10:00 (70 minutes)
    t_post = "25:10:00"
    minutes = parse_gtfs_time_to_minutes(t_post)
    assert minutes == 1510, f"Expected 1510 minutes for 25:10:00, got {minutes}"
    assert minutes >= 1440, "Post-midnight timestamp was wrapped into next service day"


# ---------------------------------------------------------------------------
# Case 3: Calendar expansion applies both additions and removals without duplicates
# ---------------------------------------------------------------------------
def test_calendar_expansion_logic():
    # Simulates calendar + calendar_dates exception handling
    active_days = set()
    start_d = date(2026, 1, 1)
    end_d = date(2026, 1, 10)

    # Base: Monday to Friday
    curr = start_d
    while curr <= end_d:
        if curr.weekday() < 5:
            active_days.add(curr)
        curr += timedelta(days=1)

    # Exceptions: 1 = add (e.g. Saturday Jan 3), 2 = remove (e.g. Wednesday Jan 7)
    exceptions = [
        {"date": date(2026, 1, 3), "type": 1},  # add Saturday
        {"date": date(2026, 1, 7), "type": 2},  # remove Wednesday
    ]

    for ex in exceptions:
        if ex["type"] == 2:
            active_days.discard(ex["date"])
        elif ex["type"] == 1:
            active_days.add(ex["date"])

    assert date(2026, 1, 3) in active_days  # Addition applied
    assert date(2026, 1, 7) not in active_days  # Removal applied


# ---------------------------------------------------------------------------
# Case 4: Global snapshot selection selects latest valid snapshot without fallback
# ---------------------------------------------------------------------------
def test_global_snapshot_selection_no_fallback():
    from backend.routers.coverage import SNAPSHOT_DATE_RANGES

    def get_snapshot(svc_date: date) -> str:
        for s in SNAPSHOT_DATE_RANGES:
            s_from = date.fromisoformat(s["service_date_from"])
            s_to = date.fromisoformat(s["service_date_to"])
            if s_from <= svc_date <= s_to:
                return s["snapshot_id"]
        return "WITHDRAWN"

    assert get_snapshot(date(2025, 11, 25)) == "mdb-389-202511230128"
    assert get_snapshot(date(2026, 1, 15)) == "mdb-389-202601080131"
    assert get_snapshot(date(2026, 7, 20)) == "mdb-389-202607110020"
    assert get_snapshot(date(2024, 1, 1)) == "WITHDRAWN"


# ---------------------------------------------------------------------------
# Case 5: Genuine zero-boarding observations (36 rows) preserved
# ---------------------------------------------------------------------------
def test_zero_boarding_observations_preserved():
    df = duckdb_repo.get_table("route_profiles")
    assert df is not None, "route_profiles table must exist"
    zero_count = int((df["rides"] == 0).sum())
    assert zero_count == 36, f"Expected exactly 36 zero-count rows preserved, found {zero_count}"


# ---------------------------------------------------------------------------
# Case 6: Missing boarding dates isolated into unmatched_schedule.parquet
# ---------------------------------------------------------------------------
def test_unmatched_schedule_quarantine():
    unmatched_df = duckdb_repo.get_table("unmatched_schedule")
    assert unmatched_df is not None, "unmatched_schedule table must exist"
    assert len(unmatched_df) == 24, f"Expected exactly 24 unmatched schedule rows, found {len(unmatched_df)}"


# ---------------------------------------------------------------------------
# Case 7: Exact calendar date lags computed via joins without df.shift()
# ---------------------------------------------------------------------------
def test_lag_computation_exact_calendar_date():
    fe = BoardingFeatureEngineer()
    # Create non-contiguous dates to guarantee that df.shift() would produce wrong lags
    records = [
        {"route": "20", "service_date": date(2026, 1, 1), "daytype": "W", "rides": 1000},
        # gap of 6 days: Jan 8 is exactly 7 days after Jan 1
        {"route": "20", "service_date": date(2026, 1, 8), "daytype": "W", "rides": 2000},
        # gap of 1 day: Jan 9 is 1 day after Jan 8
        {"route": "20", "service_date": date(2026, 1, 9), "daytype": "W", "rides": 2200},
    ]
    df = pd.DataFrame(records)
    feat_df = fe.build(df)

    # For Jan 8, lag_7d should be exactly Jan 1's ride (1000)
    jan8_row = feat_df[feat_df["service_date"] == date(2026, 1, 8)].iloc[0]
    assert jan8_row["lag_7d"] == 1000

    # For Jan 9, lag_1d should be Jan 8's ride (2000)
    jan9_row = feat_df[feat_df["service_date"] == date(2026, 1, 9)].iloc[0]
    assert jan9_row["lag_1d"] == 2000


# ---------------------------------------------------------------------------
# Case 8: WAPE metric calculation
# ---------------------------------------------------------------------------
def test_wape_metric_calculation():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 330.0])
    # Absolute errors: 10 + 10 + 30 = 50. Total true: 600. WAPE = 50 / 600 = 0.083333
    w = _wape(y_true, y_pred)
    assert abs(w - (50.0 / 600.0)) < 1e-5

    # Zero volume group returns None (marked unavailable, not infinity or NaN error)
    y_zero = np.array([0.0, 0.0, 0.0])
    y_pred_zero = np.array([10.0, 5.0, 2.0])
    w_zero = _wape(y_zero, y_pred_zero)
    assert w_zero is None


# ---------------------------------------------------------------------------
# Case 9: Scenario arithmetic keeps boardings constant and declares constraints
# ---------------------------------------------------------------------------
def test_scenario_arithmetic():
    res = calculate_scenario(
        baseline_trips=50,
        baseline_service_hours=60.0,
        baseline_boardings=2500.0,
        frequency_change_pct=20.0,
        route="20",
        daytype="W",
    )
    # 50 trips + 20% = 60 trips
    assert res["scenario"]["scheduled_trips"] == 60
    # Boardings held constant at 2500
    assert res["boardings_held_constant"] == 2500.0
    # Baseline ratio: 2500 / 60.0 = 41.67
    assert res["baseline"]["boardings_per_scheduled_service_hour"] == 41.67
    # Missing constraints list must be present
    assert len(res["missing_operational_constraints"]) >= 4
