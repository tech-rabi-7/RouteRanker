"""
src/route_analysis.py  —  RouteRanker v2.0
==========================================
Route Structure & Stop Overlap Analysis.

Computes:
  1. Per-route served stops and exclusive stops per GTFS snapshot.
  2. Pairwise route stop overlap:
     - shared_stops_count: |S_A ∩ S_B|
     - jaccard_similarity: |S_A ∩ S_B| / |S_A ∪ S_B|
     - smaller_route_overlap: |S_A ∩ S_B| / min(|S_A|, |S_B|)
     - exclusive stops for route A and route B

Design rules:
  - Route IDs and stop IDs are always treated as strings (VARCHAR).
  - No synthetic crowding labels ("Overcrowded", "Balanced", "Underutilized").
  - Clear snapshot version tracking.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple
import pandas as pd
import numpy as np


def compute_route_structure(
    route_stops_df: pd.DataFrame,
    snapshot_id: str = "current",
) -> pd.DataFrame:
    """
    Computes per-route stop metrics:
      - served_stops_count: total distinct stops served by the route
      - exclusive_stops_count: stops served ONLY by this route in this snapshot
      - exclusive_stops_list: list/comma-string of exclusive stop IDs

    Parameters
    ----------
    route_stops_df : pd.DataFrame
        DataFrame with columns: ['route', 'stop_id']
    snapshot_id : str
        GTFS snapshot identifier
    """
    df = route_stops_df.copy()
    df["route"] = df["route"].astype(str)
    df["stop_id"] = df["stop_id"].astype(str)

    # All stops per route
    route_to_stops: Dict[str, Set[str]] = (
        df.groupby("route")["stop_id"].apply(lambda s: set(s)).to_dict()
    )

    # Stop frequency across routes in this snapshot
    stop_counts = df.groupby("stop_id")["route"].nunique().to_dict()
    exclusive_stops_set = {s for s, count in stop_counts.items() if count == 1}

    records = []
    for route, stops in route_to_stops.items():
        excl = stops.intersection(exclusive_stops_set)
        records.append({
            "route": route,
            "snapshot_id": snapshot_id,
            "served_stops_count": int(len(stops)),
            "exclusive_stops_count": int(len(excl)),
            "exclusive_stops_list": ",".join(sorted(excl)),
            "served_stops_list": ",".join(sorted(stops)),
        })

    return pd.DataFrame(records).sort_values("route").reset_index(drop=True)


def compute_route_overlaps(
    route_stops_df: pd.DataFrame,
    snapshot_id: str = "current",
    min_shared_stops: int = 1,
) -> pd.DataFrame:
    """
    Computes pairwise route stop overlap metrics for all pairs (A, B) where A < B.

    Metrics computed:
      - shared_stops_count: |S_A ∩ S_B|
      - jaccard_similarity: |S_A ∩ S_B| / |S_A ∪ S_B|
      - smaller_route_overlap: |S_A ∩ S_B| / min(|S_A|, |S_B|)
      - stops_count_a: |S_A|
      - stops_count_b: |S_B|

    Parameters
    ----------
    route_stops_df : pd.DataFrame
        DataFrame with columns: ['route', 'stop_id']
    """
    df = route_stops_df.copy()
    df["route"] = df["route"].astype(str)
    df["stop_id"] = df["stop_id"].astype(str)

    route_to_stops: Dict[str, Set[str]] = (
        df.groupby("route")["stop_id"].apply(lambda s: set(s)).to_dict()
    )

    routes = sorted(route_to_stops.keys())
    records = []

    for i in range(len(routes)):
        r_a = routes[i]
        s_a = route_to_stops[r_a]
        len_a = len(s_a)
        if len_a == 0:
            continue

        for j in range(i + 1, len(routes)):
            r_b = routes[j]
            s_b = route_to_stops[r_b]
            len_b = len(s_b)
            if len_b == 0:
                continue

            shared = s_a.intersection(s_b)
            shared_count = len(shared)

            if shared_count < min_shared_stops:
                continue

            union_count = len(s_a.union(s_b))
            min_count = min(len_a, len_b)

            jaccard = round(float(shared_count) / float(union_count), 4) if union_count > 0 else 0.0
            smaller_overlap = round(float(shared_count) / float(min_count), 4) if min_count > 0 else 0.0

            records.append({
                "route_a": r_a,
                "route_b": r_b,
                "snapshot_id": snapshot_id,
                "shared_stops_count": int(shared_count),
                "jaccard_similarity": jaccard,
                "smaller_route_overlap": smaller_overlap,
                "stops_count_a": int(len_a),
                "stops_count_b": int(len_b),
                "shared_stops_list": ",".join(sorted(shared)),
            })

    if not records:
        return pd.DataFrame(columns=[
            "route_a", "route_b", "snapshot_id", "shared_stops_count",
            "jaccard_similarity", "smaller_route_overlap", "stops_count_a", "stops_count_b",
            "shared_stops_list"
        ])

    return pd.DataFrame(records).sort_values("smaller_route_overlap", ascending=False).reset_index(drop=True)
