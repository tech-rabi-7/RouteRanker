"""
backend/routers/scenarios.py
============================
POST /api/scenarios
Explicit scenario arithmetic calculator comparing baseline vs proposed scheduled trips
holding forecast/observed boardings constant, with transparent declaration of unmodeled constraints.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
import pandas as pd

from backend.database import duckdb_repo
from backend.schemas import ScenarioRequest, ScenarioResponse, ScenarioMetricBlock, ScenarioDiffBlock
from src.scenarios import calculate_scenario

router = APIRouter(prefix="/scenarios", tags=["scenarios"])


@router.post("", response_model=ScenarioResponse)
def run_scenario(request: ScenarioRequest) -> ScenarioResponse:
    df = duckdb_repo.get_table("route_profiles")
    if df is None:
        raise HTTPException(status_code=404, detail="route_profiles.parquet not found")

    route_col = "route" if "route" in df.columns else "route_short_name"
    df[route_col] = df[route_col].astype(str)
    r_df = df[df[route_col] == str(request.route)]

    if r_df.empty:
        raise HTTPException(status_code=404, detail=f"Route '{request.route}' not found in route profiles")

    # Filter by service_date or daytype
    if request.service_date:
        match_df = r_df[r_df["service_date"].astype(str) == request.service_date]
        if match_df.empty:
            match_df = r_df
    else:
        match_df = r_df[r_df["daytype"] == request.daytype.upper()]
        if match_df.empty:
            match_df = r_df

    # Extract baseline
    b_trips = int(match_df["scheduled_trips"].mean()) if "scheduled_trips" in match_df else 50
    b_hours = float(match_df["scheduled_service_hours"].mean()) if "scheduled_service_hours" in match_df else 60.0
    b_rides = float(match_df["rides"].mean()) if "rides" in match_df else 2500.0

    res = calculate_scenario(
        baseline_trips=b_trips,
        baseline_service_hours=b_hours,
        baseline_boardings=b_rides,
        proposed_trips=request.proposed_trips,
        frequency_change_pct=request.frequency_change_pct,
        duration_assumption=request.duration_assumption,
        route=request.route,
        daytype=request.daytype,
        service_date=request.service_date,
    )

    return ScenarioResponse(
        route=res["route"],
        daytype=res["daytype"],
        service_date=res["service_date"],
        trip_duration_hours_assumed=res["trip_duration_hours_assumed"],
        boardings_held_constant=res["boardings_held_constant"],
        baseline=ScenarioMetricBlock(**res["baseline"]),
        scenario=ScenarioMetricBlock(**res["scenario"]),
        difference=ScenarioDiffBlock(**res["difference"]),
        missing_operational_constraints=res["missing_operational_constraints"],
    )
