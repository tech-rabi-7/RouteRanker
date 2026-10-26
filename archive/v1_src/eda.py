"""
RouteRanker - Exploratory Data Analysis (EDA) Module
Detects route overlaps, high-traffic time windows, and transit network inefficiencies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from .gtfs_loader import GTFSFeed

class TransitEDA:
    """Performs statistical, spatial, and network analysis on GTFS data."""

    def __init__(self, feed: GTFSFeed, route_profiles: pd.DataFrame):
        self.feed = feed
        self.profiles = route_profiles

    def compute_route_overlap_matrix(self) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Computes pairwise route stop overlap using Jaccard similarity
        and containment overlap (shared stops / min stops).
        Identifies redundant routes operating on the same corridors.
        """
        trips_df = self.feed.trips
        st_df = self.feed.stop_times

        if trips_df.empty or st_df.empty or self.profiles.empty:
            return pd.DataFrame(), []

        # Map each route_id to set of stop_ids
        route_stops_map: Dict[str, set] = {}
        route_names_map: Dict[str, str] = {}

        for _, r in self.profiles.iterrows():
            r_id = r["route_id"]
            route_names_map[r_id] = f"{r['route_short_name']} ({r_id})"
            # Get trips for this route
            r_trips = trips_df[trips_df["route_id"] == r_id]["trip_id"]
            if not r_trips.empty:
                r_stops = set(st_df[st_df["trip_id"].isin(r_trips)]["stop_id"])
                route_stops_map[r_id] = r_stops
            else:
                route_stops_map[r_id] = set()

        route_ids = list(route_stops_map.keys())
        n = len(route_ids)
        matrix = np.zeros((n, n))

        high_overlap_pairs = []

        for i in range(n):
            for j in range(n):
                r1, r2 = route_ids[i], route_ids[j]
                s1, s2 = route_stops_map[r1], route_stops_map[r2]
                if i == j:
                    matrix[i][j] = 1.0
                elif len(s1) == 0 or len(s2) == 0:
                    matrix[i][j] = 0.0
                else:
                    intersection = len(s1.intersection(s2))
                    union = len(s1.union(s2))
                    jaccard = intersection / union if union > 0 else 0
                    containment = intersection / min(len(s1), len(s2))
                    matrix[i][j] = round(jaccard, 3)

                    # Check for significant corridor overlap (only upper triangle)
                    if i < j and (containment >= 0.50 or jaccard >= 0.40):
                        high_overlap_pairs.append({
                            "route_1_id": r1,
                            "route_1_name": route_names_map[r1],
                            "route_2_id": r2,
                            "route_2_name": route_names_map[r2],
                            "shared_stops": intersection,
                            "jaccard_similarity": round(jaccard, 3),
                            "corridor_overlap_pct": round(containment * 100, 1),
                            "recommendation": "High candidate for Route Merging or Schedule Synchronization"
                        })

        df_matrix = pd.DataFrame(
            matrix,
            index=[route_names_map[rid] for rid in route_ids],
            columns=[route_names_map[rid] for rid in route_ids]
        )

        high_overlap_pairs.sort(key=lambda x: x["corridor_overlap_pct"], reverse=True)
        return df_matrix, high_overlap_pairs

    def get_peak_windows_summary(self, hourly_ridership: pd.DataFrame) -> Dict[str, any]:
        """
        Analyzes the hourly ridership curve to detect peak morning & evening windows,
        maximum congestion hours, and fleet demand surge.
        """
        if hourly_ridership.empty:
            return {}

        max_hour_row = hourly_ridership.loc[hourly_ridership["estimated_passengers"].idxmax()]
        min_offpeak_row = hourly_ridership[(hourly_ridership["hour"] >= 11) & (hourly_ridership["hour"] <= 16)].sort_values(
            "estimated_passengers"
        ).iloc[0] if not hourly_ridership.empty else None

        morning_peak_passengers = hourly_ridership[hourly_ridership["hour"].between(8, 10)]["estimated_passengers"].sum()
        evening_peak_passengers = hourly_ridership[hourly_ridership["hour"].between(17, 20)]["estimated_passengers"].sum()
        total_passengers = hourly_ridership["estimated_passengers"].sum()

        return {
            "peak_hour": int(max_hour_row["hour"]),
            "peak_hour_passengers": int(max_hour_row["estimated_passengers"]),
            "peak_hour_trips": int(max_hour_row["active_trips"]),
            "peak_crowding_pct": float(max_hour_row["avg_crowding_pct"]),
            "morning_peak_share_pct": round((morning_peak_passengers / max(1, total_passengers)) * 100, 1),
            "evening_peak_share_pct": round((evening_peak_passengers / max(1, total_passengers)) * 100, 1),
            "total_estimated_passengers": int(total_passengers)
        }

    def get_network_health_metrics(self) -> Dict[str, any]:
        """Calculates global network health and efficiency indicators."""
        if self.profiles.empty:
            return {}

        total_routes = len(self.profiles)
        overcrowded = (self.profiles["performance_status"] == "Overcrowded").sum()
        underutilized = (self.profiles["performance_status"] == "Underutilized").sum()
        balanced = (self.profiles["performance_status"] == "Balanced").sum()

        avg_speed = round(self.profiles["avg_speed_kmh"].mean(), 1)
        avg_headway = round(self.profiles["avg_headway_min"].mean(), 1)
        network_inefficiency_rate = round(((overcrowded + underutilized) / total_routes) * 100, 1)

        return {
            "total_routes": total_routes,
            "overcrowded_routes": int(overcrowded),
            "underutilized_routes": int(underutilized),
            "balanced_routes": int(balanced),
            "network_inefficiency_rate_pct": network_inefficiency_rate,
            "avg_speed_kmh": avg_speed,
            "avg_headway_min": avg_headway,
            "total_network_km": round(self.profiles["route_length_km"].sum(), 1),
            "total_daily_trips": int(self.profiles["total_daily_trips"].sum())
        }
