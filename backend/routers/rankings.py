"""
backend/routers/rankings.py
===========================
GET /api/rankings
Service productivity rankings, ratio-of-sums indicators, and transparent review reasons.
"""

from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, HTTPException, Query
import numpy as np
import pandas as pd

from backend.database import duckdb_repo
from backend.schemas import RankingsResponse, RankingItem
from src.models import PRODUCTIVITY_MIN_OBS

router = APIRouter(prefix="/rankings", tags=["rankings"])


@router.get("", response_model=RankingsResponse)
def get_rankings(
    daytype: str = Query("W", description="Daytype code: W (weekday), A (Saturday), U (Sunday)"),
    sort_by: str = Query("service_productivity_metric_rank", description="Indicator to sort by"),
    order: str = Query("asc", description="Sort order: asc or desc"),
    eligible_only: bool = Query(False, description="Filter only routes with >= 20 observations"),
) -> RankingsResponse:
    daytype_code = daytype.upper()
    prod_df = duckdb_repo.get_table("productivity_rankings")
    if prod_df is None or prod_df.empty:
        # Fallback to computing from route_profiles
        rp_df = duckdb_repo.get_table("route_profiles")
        if rp_df is None:
            raise HTTPException(status_code=404, detail="No route data available for rankings")
        from src.models import RouteProductivityRanker
        ranker = RouteProductivityRanker()
        rp_df = rp_df.rename(columns={"route": "route_short_name"} if "route" in rp_df.columns else {})
        prod_df = ranker.build_profile_table(rp_df)

    # Filter to requested daytype
    dt_df = prod_df[prod_df["daytype"] == daytype_code].copy()
    if dt_df.empty:
        return RankingsResponse(daytype=daytype_code, total_routes=0, eligible_routes_count=0, rankings=[])

    # Cluster mapping
    cluster_df = duckdb_repo.get_table("route_clusters")
    cluster_map = {}
    if cluster_df is not None and not cluster_df.empty:
        for _, row in cluster_df.iterrows():
            cluster_map[str(row["route"])] = int(row.get("cluster_id", 0))

    # Overlap flags
    overlap_df = duckdb_repo.get_table("route_overlap")
    high_overlap_routes = set()
    if overlap_df is not None and not overlap_df.empty:
        heavy = overlap_df[overlap_df["smaller_route_overlap"] >= 0.50]
        high_overlap_routes = set(heavy["route_a"].astype(str)).union(set(heavy["route_b"].astype(str)))

    # Median productivity for explaining review reasons
    eligible_mask = dt_df["n_obs"] >= PRODUCTIVITY_MIN_OBS
    median_prod_hour = (
        float(dt_df.loc[eligible_mask, "boardings_per_scheduled_service_hour"].median())
        if eligible_mask.any()
        else 45.0
    )
    median_demand = float(dt_df["total_rides"].median()) if not dt_df.empty else 100000

    items = []
    for _, row in dt_df.iterrows():
        r = str(row["route_short_name"])
        n_obs = int(row["n_obs"])
        is_elig = bool(n_obs >= PRODUCTIVITY_MIN_OBS)
        tot_rides = int(row["total_rides"])
        tot_trips = int(row["total_scheduled_trips"])
        tot_hours = float(row["total_scheduled_service_hours"])
        b_per_trip = round(float(row["boardings_per_scheduled_trip"]), 2)
        b_per_hour = round(float(row["boardings_per_scheduled_service_hour"]), 2)
        rank_val = (
            int(row["service_productivity_metric_rank"])
            if pd.notna(row["service_productivity_metric_rank"])
            else None
        )

        overlap_flag = r in high_overlap_routes

        # Explainable planner review reasons
        reasons = []
        if not is_elig:
            reasons.append(f"Insufficient sample coverage ({n_obs} < {PRODUCTIVITY_MIN_OBS} observations)")
        else:
            if tot_rides > median_demand and b_per_hour < median_prod_hour:
                reasons.append("High passenger volume but below-median vehicle-hour productivity")
            elif tot_rides < (median_demand * 0.4) and b_per_hour < (median_prod_hour * 0.6):
                reasons.append("Low demand feeder with depressed service productivity")
            if overlap_flag:
                reasons.append("Corridor redundancy: shares ≥50% stops with partner route")
            if rank_val and rank_val <= 10:
                reasons.append("Top-tier network productivity anchor")

        items.append(
            RankingItem(
                route=r,
                daytype=daytype_code,
                observed_demand=tot_rides,
                n_obs=n_obs,
                scheduled_trips=tot_trips,
                scheduled_service_hours=round(tot_hours, 1),
                boardings_per_scheduled_trip=b_per_trip,
                boardings_per_scheduled_service_hour=b_per_hour,
                service_productivity_metric_rank=rank_val,
                is_eligible=is_elig,
                demand_trend=0.0,
                overlap_review_flag=overlap_flag,
                planner_review_reasons=reasons,
                cluster_id=cluster_map.get(r),
            )
        )

    if eligible_only:
        items = [it for it in items if it.is_eligible]

    # Sort
    reverse = order.lower() == "desc"
    if sort_by in ["service_productivity_metric_rank", "rank"]:
        # Put None ranks at end
        items.sort(key=lambda x: (x.service_productivity_metric_rank is None, x.service_productivity_metric_rank or 999999), reverse=reverse)
    elif sort_by in ["observed_demand", "total_rides"]:
        items.sort(key=lambda x: x.observed_demand, reverse=reverse)
    elif sort_by in ["boardings_per_scheduled_trip", "trip_productivity"]:
        items.sort(key=lambda x: x.boardings_per_scheduled_trip, reverse=reverse)
    elif sort_by in ["boardings_per_scheduled_service_hour", "hour_productivity"]:
        items.sort(key=lambda x: x.boardings_per_scheduled_service_hour, reverse=reverse)
    else:
        items.sort(key=lambda x: x.route, reverse=reverse)

    eligible_count = sum(1 for it in items if it.is_eligible)
    return RankingsResponse(
        daytype=daytype_code,
        total_routes=len(items),
        eligible_routes_count=eligible_count,
        rankings=items,
    )
