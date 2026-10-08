"""
RouteRanker - Feature Engineering Module
Transforms raw GTFS tables into operational transit features, route profiles,
and ridership/demand metrics.
"""

import math
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from .gtfs_loader import GTFSFeed

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes great circle distance between two points in kilometers."""
    if any(pd.isna([lat1, lon1, lat2, lon2])):
        return 0.0
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c

def parse_time_to_minutes(time_str: str) -> Optional[int]:
    """Converts HH:MM:SS or H:MM:SS to total minutes from midnight."""
    if not isinstance(time_str, str) or ":" not in time_str:
        return None
    try:
        parts = time_str.strip().split(":")
        h = int(parts[0])
        m = int(parts[1])
        return h * 60 + m
    except (ValueError, IndexError):
        return None

def parse_time_to_hour(time_str: str) -> Optional[int]:
    """Extracts integer hour from time string."""
    mins = parse_time_to_minutes(time_str)
    return (mins // 60) if mins is not None else None

class RouteFeatureEngineer:
    """Engineers spatial, temporal, frequency and demand features from GTFS."""

    def __init__(self, feed: GTFSFeed, default_bus_capacity: int = 60):
        self.feed = feed
        self.bus_capacity = default_bus_capacity

    def build_route_profiles(self) -> pd.DataFrame:
        """
        Builds a comprehensive route profile DataFrame with aggregated spatial,
        temporal, frequency, and ridership performance metrics.
        """
        routes_df = self.feed.routes.copy()
        trips_df = self.feed.trips.copy()
        stop_times_df = self.feed.stop_times.copy()
        stops_df = self.feed.stops.copy()

        if routes_df.empty or trips_df.empty or stop_times_df.empty:
            return pd.DataFrame()

        # 1. Parse stop_times arrival/departure
        stop_times_df["dep_mins"] = stop_times_df["departure_time"].apply(parse_time_to_minutes)
        stop_times_df["dep_hour"] = stop_times_df["departure_time"].apply(parse_time_to_hour)

        # Merge stop coordinates onto stop_times
        st_merged = stop_times_df.merge(
            stops_df[["stop_id", "stop_name", "stop_lat", "stop_lon"]],
            on="stop_id",
            how="left"
        ).sort_values(by=["trip_id", "stop_sequence"])

        # 2. Trip-level metrics
        trip_metrics = []
        for trip_id, group in st_merged.groupby("trip_id"):
            n_stops = len(group)
            if n_stops < 2:
                continue

            # Calculate total trip distance by summing consecutive stop distances
            lats = group["stop_lat"].values
            lons = group["stop_lon"].values
            total_dist_km = sum(
                haversine_km(lats[i], lons[i], lats[i+1], lons[i+1])
                for i in range(len(lats) - 1)
            )

            # Duration
            start_min = group["dep_mins"].iloc[0]
            end_min = group["dep_mins"].iloc[-1]
            if start_min is not None and end_min is not None and end_min >= start_min:
                duration_min = max(5, end_min - start_min)
            else:
                duration_min = max(10, total_dist_km * 3.5) # approximate city speed

            dep_hour = group["dep_hour"].iloc[0] if group["dep_hour"].iloc[0] is not None else 8

            trip_metrics.append({
                "trip_id": trip_id,
                "n_stops": n_stops,
                "dist_km": total_dist_km,
                "duration_min": duration_min,
                "dep_hour": dep_hour,
                "speed_kmh": round((total_dist_km / (duration_min / 60.0)), 2) if duration_min > 0 else 18.0
            })

        df_trip_metrics = pd.DataFrame(trip_metrics)
        if df_trip_metrics.empty:
            return pd.DataFrame()

        # Merge with route_id from trips
        trip_with_route = df_trip_metrics.merge(
            trips_df[["trip_id", "route_id", "service_id"]],
            on="trip_id",
            how="inner"
        )

        # 3. Route-level aggregation
        route_profiles = []
        for route_id, r_trips in trip_with_route.groupby("route_id"):
            route_meta = routes_df[routes_df["route_id"] == route_id].iloc[0]
            total_trips = len(r_trips)
            avg_stops = r_trips["n_stops"].mean()
            avg_dist = r_trips["dist_km"].mean()
            avg_dur = r_trips["duration_min"].mean()
            avg_speed = r_trips["speed_kmh"].mean()

            # Peak time trips
            # Morning Peak: 8:00 - 11:00 (hours 8, 9, 10)
            morning_peak = ((r_trips["dep_hour"] >= 8) & (r_trips["dep_hour"] <= 10)).sum()
            # Evening Peak: 17:00 - 20:00 (hours 17, 18, 19, 20)
            evening_peak = ((r_trips["dep_hour"] >= 17) & (r_trips["dep_hour"] <= 20)).sum()
            # Off peak
            off_peak = total_trips - morning_peak - evening_peak

            # Operating hours window
            min_hour = r_trips["dep_hour"].min()
            max_hour = r_trips["dep_hour"].max()
            span_hours = max(1, (max_hour - min_hour + 1))
            avg_headway_min = round((span_hours * 60) / max(1, total_trips), 1)

            # Average stop spacing
            avg_stop_spacing = round(avg_dist / max(1, (avg_stops - 1)), 2)

            # 4. Realistic Passenger Demand & Load Factors
            # Route name or identifier characteristics
            r_name = str(route_meta.get("route_short_name", ""))
            r_long = str(route_meta.get("route_long_name", "")).lower()

            # Simulate base ridership density
            np.random.seed(abs(hash(route_id)) % 100000)
            
            # Identify route characteristics
            is_redundant = "dup" in r_name.lower() or "parallel" in r_long or "redundant" in r_long
            is_trunk_heavy = any(k in r_long for k in ["ring", "express", "highway", "orr", "isbt", "itpl", "bkc"])
            is_feeder = "feeder" in r_long or avg_dist < 10

            if is_redundant:
                # Underperforming duplicate route
                base_trip_pax = np.random.uniform(15, 25)
                peak_pax = np.random.uniform(25, 35)
                perf_status = "Underutilized"
            elif is_trunk_heavy or ("overcrowded" in r_long):
                # Heavy trunk route
                base_trip_pax = np.random.uniform(55, 75)
                peak_pax = np.random.uniform(75, 105) # Severely overcrowded (>100% capacity)
                perf_status = "Overcrowded"
            elif is_feeder:
                base_trip_pax = np.random.uniform(18, 30)
                peak_pax = np.random.uniform(30, 45)
                perf_status = "Underutilized" if base_trip_pax < 22 else "Balanced"
            else:
                base_trip_pax = np.random.uniform(38, 52)
                peak_pax = np.random.uniform(52, 62)
                perf_status = "Balanced"

            # Compute load factors (ratio to bus capacity)
            avg_load_factor = round((base_trip_pax / self.bus_capacity) * 100.0, 1)
            peak_load_factor = round((peak_pax / self.bus_capacity) * 100.0, 1)
            est_daily_passengers = int(base_trip_pax * total_trips * 1.3)

            # Inefficiency Score (0 to 100, where 0 is perfect equilibrium)
            # Penalizes underutilization (<40% load) and severe overcrowding (>100% load)
            if peak_load_factor > 100:
                inefficiency_score = min(100, round((peak_load_factor - 100) * 1.8 + 30, 1))
                perf_status = "Overcrowded"
            elif avg_load_factor < 45:
                inefficiency_score = min(100, round((45 - avg_load_factor) * 2.0 + 20, 1))
                perf_status = "Underutilized"
            else:
                inefficiency_score = round(abs(avg_load_factor - 75) * 0.5, 1)
                perf_status = "Balanced"

            route_profiles.append({
                "route_id": route_id,
                "route_short_name": route_meta.get("route_short_name", route_id),
                "route_long_name": route_meta.get("route_long_name", ""),
                "route_color": route_meta.get("route_color", "1F77B4"),
                "num_stops": int(round(avg_stops)),
                "route_length_km": round(avg_dist, 2),
                "avg_trip_duration_min": round(avg_dur, 1),
                "avg_speed_kmh": round(avg_speed, 1),
                "avg_stop_spacing_km": avg_stop_spacing,
                "total_daily_trips": total_trips,
                "peak_morning_trips": morning_peak,
                "peak_evening_trips": evening_peak,
                "off_peak_trips": off_peak,
                "avg_headway_min": avg_headway_min,
                "estimated_daily_passengers": est_daily_passengers,
                "avg_trip_passengers": int(round(base_trip_pax)),
                "peak_trip_passengers": int(round(peak_pax)),
                "avg_load_factor": avg_load_factor,
                "peak_load_factor": peak_load_factor,
                "inefficiency_score": inefficiency_score,
                "performance_status": perf_status
            })

        df_profiles = pd.DataFrame(route_profiles)
        return df_profiles

    def build_hourly_ridership(self) -> pd.DataFrame:
        """
        Aggregates trip frequency and estimated ridership across each hour of the day (00 to 23).
        """
        stop_times_df = self.feed.stop_times.copy()
        if stop_times_df.empty:
            return pd.DataFrame()

        stop_times_df["dep_hour"] = stop_times_df["departure_time"].apply(parse_time_to_hour)
        hourly_trips = stop_times_df.dropna(subset=["dep_hour"]).groupby("dep_hour")["trip_id"].nunique().reset_index()
        hourly_trips.columns = ["hour", "active_trips"]

        # Fill all 24 hours
        full_hours = pd.DataFrame({"hour": list(range(24))})
        df_hourly = full_hours.merge(hourly_trips, on="hour", how="left").fillna(0)
        df_hourly["active_trips"] = df_hourly["active_trips"].astype(int)

        # Passenger demand curve modeling:
        # Peak hours 8-10 AM and 17-20 PM have high boarding factors
        def hourly_demand_multiplier(h):
            if 8 <= h <= 10:
                return 1.8
            elif 17 <= h <= 20:
                return 2.0
            elif 11 <= h <= 16:
                return 1.1
            elif 6 <= h <= 7:
                return 0.9
            elif 21 <= h <= 22:
                return 0.7
            else:
                return 0.2

        df_hourly["demand_factor"] = df_hourly["hour"].apply(hourly_demand_multiplier)
        df_hourly["estimated_passengers"] = (df_hourly["active_trips"] * 35 * df_hourly["demand_factor"]).round().astype(int)
        df_hourly["avg_crowding_pct"] = (df_hourly["estimated_passengers"] / np.maximum(1, df_hourly["active_trips"] * self.bus_capacity) * 100).round(1)

        return df_hourly

    def build_stop_usage_profiles(self) -> pd.DataFrame:
        """
        Aggregates stop activity (number of routes serving stop, total daily departures,
        estimated daily passenger footfall).
        """
        st_df = self.feed.stop_times.copy()
        stops_df = self.feed.stops.copy()
        trips_df = self.feed.trips.copy()

        if st_df.empty or stops_df.empty or trips_df.empty:
            return pd.DataFrame()

        st_with_route = st_df.merge(trips_df[["trip_id", "route_id"]], on="trip_id", how="left")

        stop_stats = st_with_route.groupby("stop_id").agg(
            routes_count=("route_id", "nunique"),
            total_departures=("trip_id", "nunique")
        ).reset_index()

        merged_stops = stops_df.merge(stop_stats, on="stop_id", how="left").fillna(0)
        merged_stops["routes_count"] = merged_stops["routes_count"].astype(int)
        merged_stops["total_departures"] = merged_stops["total_departures"].astype(int)

        # Estimated passenger footfall based on departures and hub connectivity
        merged_stops["est_daily_footfall"] = (
            merged_stops["total_departures"] * 18 + merged_stops["routes_count"] * 450
        ).astype(int)

        return merged_stops.sort_values(by="est_daily_footfall", ascending=False)
