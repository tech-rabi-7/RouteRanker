"""
backend/routers/routes.py
=========================
GET /api/routes
GET /api/routes/{route}/history
Route discovery and historical demand/service observations.
"""

from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, HTTPException, Query
import numpy as np
import pandas as pd

from backend.database import duckdb_repo
from backend.schemas import RouteListResponse, RouteSummary, RouteHistoryResponse, RouteHistoryItem

router = APIRouter(prefix="/routes", tags=["routes"])


@router.get("", response_model=RouteListResponse)
def list_routes(
    daytype: Optional[str] = Query(None, description="Filter by daytype: W, A, or U"),
    min_rides: Optional[int] = Query(None, description="Minimum total boardings"),
) -> RouteListResponse:
    df = duckdb_repo.get_table("route_profiles")
    if df is None:
        raise HTTPException(status_code=404, detail="route_profiles.parquet not found")

    route_col = "route" if "route" in df.columns else "route_short_name"
    df[route_col] = df[route_col].astype(str)

    if daytype:
        df = df[df["daytype"] == daytype.upper()]

    # Load structure for stop counts
    struct_df = duckdb_repo.get_table("route_structure")
    stop_map = {}
    excl_map = {}
    if struct_df is not None and not struct_df.empty:
        # Latest snapshot
        latest_snap = struct_df["snapshot_id"].max()
        latest_s = struct_df[struct_df["snapshot_id"] == latest_snap]
        for _, row in latest_s.iterrows():
            stop_map[str(row["route"])] = int(row.get("served_stops_count", 0))
            excl_map[str(row["route"])] = int(row.get("exclusive_stops_count", 0))

    # Load cluster assignments
    cluster_df = duckdb_repo.get_table("route_clusters")
    cluster_map = {}
    if cluster_df is not None and not cluster_df.empty:
        for _, row in cluster_df.iterrows():
            cluster_map[str(row["route"])] = int(row.get("cluster_id", 0))

    grouped = df.groupby(route_col).agg(
        total_boardings=("rides", "sum"),
        avg_daily_boardings=("rides", "mean"),
        service_days=("service_date", "nunique"),
    ).reset_index()

    routes_out = []
    for _, row in grouped.iterrows():
        r = str(row[route_col])
        tot = int(row["total_boardings"])
        if min_rides is not None and tot < min_rides:
            continue
        routes_out.append(
            RouteSummary(
                route=r,
                route_short_name=r,
                total_boardings=tot,
                avg_daily_boardings=round(float(row["avg_daily_boardings"]), 1),
                service_days=int(row["service_days"]),
                served_stops_count=stop_map.get(r, 60),
                exclusive_stops_count=excl_map.get(r, 5),
                cluster_id=cluster_map.get(r),
            )
        )

    # Sort descending by total boardings
    routes_out.sort(key=lambda x: x.total_boardings, reverse=True)
    return RouteListResponse(total_routes=len(routes_out), routes=routes_out)


@router.get("/{route}/history", response_model=RouteHistoryResponse)
def get_route_history(
    route: str,
    daytype: Optional[str] = Query(None, description="Filter by daytype: W, A, or U"),
    date_from: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
) -> RouteHistoryResponse:
    df = duckdb_repo.get_table("route_profiles")
    if df is None:
        raise HTTPException(status_code=404, detail="route_profiles.parquet not found")

    route_col = "route" if "route" in df.columns else "route_short_name"
    df[route_col] = df[route_col].astype(str)
    r_df = df[df[route_col] == str(route)].copy()

    if r_df.empty:
        raise HTTPException(status_code=404, detail=f"Route '{route}' not found in observations")

    r_df["service_date_str"] = r_df["service_date"].astype(str)
    if daytype:
        r_df = r_df[r_df["daytype"] == daytype.upper()]
    if date_from:
        r_df = r_df[r_df["service_date_str"] >= date_from]
    if date_to:
        r_df = r_df[r_df["service_date_str"] <= date_to]

    r_df = r_df.sort_values("service_date").reset_index(drop=True)

    items = []
    rides_arr = r_df["rides"].values
    for idx, row in r_df.iterrows():
        # Lag 7d approximation if available in series
        lag_7 = float(rides_arr[idx - 7]) if idx >= 7 else None
        seasonal = float(np.mean(rides_arr[max(0, idx - 28):idx:7])) if idx >= 14 else None

        items.append(
            RouteHistoryItem(
                route=str(row[route_col]),
                service_date=str(row["service_date"]),
                daytype=str(row.get("daytype", "W")),
                rides=int(row.get("rides", 0)),
                scheduled_trips=int(row["scheduled_trips"]) if "scheduled_trips" in row and pd.notna(row["scheduled_trips"]) else None,
                scheduled_service_hours=float(row["scheduled_service_hours"]) if "scheduled_service_hours" in row and pd.notna(row["scheduled_service_hours"]) else None,
                source_snapshot=str(row.get("source_snapshot", "")),
                lag_7d=lag_7,
                seasonal_mean=round(seasonal, 1) if seasonal is not None else None,
            )
        )

    min_date = str(r_df["service_date"].min()) if not r_df.empty else ""
    max_date = str(r_df["service_date"].max()) if not r_df.empty else ""

    return RouteHistoryResponse(
        route=str(route),
        total_observations=len(items),
        date_from=min_date,
        date_to=max_date,
        history=items,
    )
