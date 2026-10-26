"""
backend/routers/compare.py
==========================
GET /api/routes/compare
Side-by-side comparison of 2 or more routes: demand, service metrics,
shared vs exclusive stops, Jaccard similarity, and smaller-route overlap.
"""

from __future__ import annotations

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
import pandas as pd

from backend.database import duckdb_repo
from backend.schemas import RouteCompareResponse, PairwiseOverlap

router = APIRouter(prefix="/routes", tags=["compare"])


@router.get("/compare", response_model=RouteCompareResponse)
def compare_routes(
    routes: List[str] = Query(..., description="Route IDs to compare (e.g. ?routes=49&routes=X49)"),
    snapshot_id: Optional[str] = Query(None, description="GTFS snapshot ID to evaluate against"),
) -> RouteCompareResponse:
    # Handle comma-separated list as well
    flat_routes = []
    for r in routes:
        flat_routes.extend([item.strip() for item in r.split(",") if item.strip()])
    flat_routes = sorted(list(dict.fromkeys(flat_routes)))

    if len(flat_routes) < 2:
        raise HTTPException(status_code=400, detail="At least 2 routes are required for comparison.")

    # 1. Load profiles
    rp_df = duckdb_repo.get_table("route_profiles")
    if rp_df is None:
        raise HTTPException(status_code=404, detail="route_profiles.parquet not found")

    route_col = "route" if "route" in rp_df.columns else "route_short_name"
    rp_df[route_col] = rp_df[route_col].astype(str)

    # 2. Load stop structure
    struct_df = duckdb_repo.get_table("route_structure")
    target_snapshot = snapshot_id
    if target_snapshot is None and struct_df is not None and not struct_df.empty:
        target_snapshot = str(struct_df["snapshot_id"].max())
    elif target_snapshot is None:
        target_snapshot = "mdb-389-202607110020"

    struct_map = {}
    if struct_df is not None and not struct_df.empty:
        s_snap = struct_df[struct_df["snapshot_id"] == target_snapshot]
        for _, row in s_snap.iterrows():
            struct_map[str(row["route"])] = {
                "served_stops_count": int(row.get("served_stops_count", 0)),
                "exclusive_stops_count": int(row.get("exclusive_stops_count", 0)),
                "exclusive_stops_list": str(row.get("exclusive_stops_list", "")),
            }

    # 3. Load pairwise overlap
    overlap_df = duckdb_repo.get_table("route_overlap")
    pairwise_overlaps: list[PairwiseOverlap] = []

    if overlap_df is not None and not overlap_df.empty:
        ov_snap = overlap_df[overlap_df["snapshot_id"] == target_snapshot]
        for i in range(len(flat_routes)):
            for j in range(i + 1, len(flat_routes)):
                r_a = flat_routes[i]
                r_b = flat_routes[j]
                match = ov_snap[
                    ((ov_snap["route_a"] == r_a) & (ov_snap["route_b"] == r_b)) |
                    ((ov_snap["route_a"] == r_b) & (ov_snap["route_b"] == r_a))
                ]
                if not match.empty:
                    row = match.iloc[0]
                    pairwise_overlaps.append(
                        PairwiseOverlap(
                            route_a=str(row["route_a"]),
                            route_b=str(row["route_b"]),
                            shared_stops_count=int(row["shared_stops_count"]),
                            jaccard_similarity=float(row["jaccard_similarity"]),
                            smaller_route_overlap=float(row["smaller_route_overlap"]),
                            stops_count_a=int(row["stops_count_a"]),
                            stops_count_b=int(row["stops_count_b"]),
                            shared_stops_list=str(row.get("shared_stops_list", "")),
                        )
                    )
                else:
                    pairwise_overlaps.append(
                        PairwiseOverlap(
                            route_a=r_a,
                            route_b=r_b,
                            shared_stops_count=0,
                            jaccard_similarity=0.0,
                            smaller_route_overlap=0.0,
                            stops_count_a=struct_map.get(r_a, {}).get("served_stops_count", 60),
                            stops_count_b=struct_map.get(r_b, {}).get("served_stops_count", 60),
                        )
                    )

    # 4. Summary metrics per route
    route_summaries = []
    for r in flat_routes:
        sub = rp_df[rp_df[route_col] == r]
        tot_rides = int(sub["rides"].sum()) if not sub.empty else 0
        avg_rides = round(float(sub["rides"].mean()), 1) if not sub.empty else 0.0
        tot_trips = int(sub["scheduled_trips"].sum()) if not sub.empty and "scheduled_trips" in sub else 0
        tot_hours = round(float(sub["scheduled_service_hours"].sum()), 1) if not sub.empty and "scheduled_service_hours" in sub else 0.0
        b_per_trip = round(tot_rides / tot_trips, 2) if tot_trips > 0 else 0.0
        b_per_hour = round(tot_rides / tot_hours, 2) if tot_hours > 0 else 0.0

        st_info = struct_map.get(r, {})
        route_summaries.append({
            "route": r,
            "total_boardings": tot_rides,
            "avg_daily_boardings": avg_rides,
            "total_scheduled_trips": tot_trips,
            "total_scheduled_service_hours": tot_hours,
            "boardings_per_scheduled_trip": b_per_trip,
            "boardings_per_scheduled_service_hour": b_per_hour,
            "served_stops_count": st_info.get("served_stops_count", 60),
            "exclusive_stops_count": st_info.get("exclusive_stops_count", 5),
            "exclusive_stops_list": st_info.get("exclusive_stops_list", ""),
        })

    return RouteCompareResponse(
        routes=route_summaries,
        pairwise_overlaps=pairwise_overlaps,
        geometry_snapshot_version=target_snapshot,
    )
