"""
RouteRanker - GTFS Data Loader & Validator
Handles ingestion, structural validation, and querying of standard GTFS feeds.
"""

import os
import zipfile
import tempfile
import pandas as pd
from typing import Dict, Tuple, Optional, List

REQUIRED_GTFS_FILES = [
    "routes.txt",
    "trips.txt",
    "stop_times.txt",
    "stops.txt",
]

OPTIONAL_GTFS_FILES = [
    "calendar.txt",
    "shapes.txt",
    "frequencies.txt"
]

class GTFSFeed:
    """Represents a validated GTFS Transit Feed."""

    def __init__(self, feed_source: str):
        """
        Initialize GTFS feed from directory path or zip archive.
        """
        self.source = feed_source
        self.tables: Dict[str, pd.DataFrame] = {}
        self.validation_report: Dict[str, any] = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "file_counts": {}
        }
        self._load()

    def _load(self):
        if zipfile.is_zipfile(self.source):
            with zipfile.ZipFile(self.source, 'r') as zf:
                file_names = zf.namelist()
                # Check required
                for req in REQUIRED_GTFS_FILES:
                    matched = [f for f in file_names if os.path.basename(f) == req]
                    if not matched:
                        self.validation_report["errors"].append(f"Missing required file: {req}")
                        self.validation_report["valid"] = False
                    else:
                        with zf.open(matched[0]) as f:
                            self.tables[req.replace(".txt", "")] = pd.read_csv(f, dtype=str)
                # Load optional
                for opt in OPTIONAL_GTFS_FILES:
                    matched = [f for f in file_names if os.path.basename(f) == opt]
                    if matched:
                        with zf.open(matched[0]) as f:
                            self.tables[opt.replace(".txt", "")] = pd.read_csv(f, dtype=str)
        elif os.path.isdir(self.source):
            for req in REQUIRED_GTFS_FILES:
                file_path = os.path.join(self.source, req)
                if not os.path.isfile(file_path):
                    self.validation_report["errors"].append(f"Missing required file: {req}")
                    self.validation_report["valid"] = False
                else:
                    self.tables[req.replace(".txt", "")] = pd.read_csv(file_path, dtype=str)
            for opt in OPTIONAL_GTFS_FILES:
                file_path = os.path.join(self.source, opt)
                if os.path.isfile(file_path):
                    self.tables[opt.replace(".txt", "")] = pd.read_csv(file_path, dtype=str)
        else:
            self.validation_report["errors"].append(f"Invalid path or format: {self.source}")
            self.validation_report["valid"] = False
            return

        self._clean_and_type_convert()
        self._validate_relationships()

    def _clean_and_type_convert(self):
        if not self.validation_report["valid"]:
            return

        # Numeric conversions
        if "stops" in self.tables:
            stops = self.tables["stops"]
            stops["stop_lat"] = pd.to_numeric(stops["stop_lat"], errors="coerce")
            stops["stop_lon"] = pd.to_numeric(stops["stop_lon"], errors="coerce")
            self.tables["stops"] = stops.dropna(subset=["stop_id", "stop_lat", "stop_lon"])

        if "stop_times" in self.tables:
            st = self.tables["stop_times"]
            st["stop_sequence"] = pd.to_numeric(st["stop_sequence"], errors="coerce").fillna(0).astype(int)
            self.tables["stop_times"] = st

        if "shapes" in self.tables:
            shapes = self.tables["shapes"]
            shapes["shape_pt_lat"] = pd.to_numeric(shapes["shape_pt_lat"], errors="coerce")
            shapes["shape_pt_lon"] = pd.to_numeric(shapes["shape_pt_lon"], errors="coerce")
            shapes["shape_pt_sequence"] = pd.to_numeric(shapes["shape_pt_sequence"], errors="coerce").fillna(0).astype(int)
            self.tables["shapes"] = shapes.sort_values(by=["shape_id", "shape_pt_sequence"])

        # Store summary counts
        for k, df in self.tables.items():
            self.validation_report["file_counts"][k] = len(df)

    def _validate_relationships(self):
        if not self.validation_report["valid"]:
            return

        routes = self.tables.get("routes")
        trips = self.tables.get("trips")
        stop_times = self.tables.get("stop_times")
        stops = self.tables.get("stops")

        if trips is not None and routes is not None:
            orphan_trips = (~trips["route_id"].isin(routes["route_id"])).sum()
            if orphan_trips > 0:
                self.validation_report["warnings"].append(f"Found {orphan_trips} trips with non-existent route_id.")

        if stop_times is not None and trips is not None:
            orphan_stop_times = (~stop_times["trip_id"].isin(trips["trip_id"])).sum()
            if orphan_stop_times > 0:
                self.validation_report["warnings"].append(f"Found {orphan_stop_times} stop_times with non-existent trip_id.")

        if stop_times is not None and stops is not None:
            orphan_stops = (~stop_times["stop_id"].isin(stops["stop_id"])).sum()
            if orphan_stops > 0:
                self.validation_report["warnings"].append(f"Found {orphan_stops} stop_times referencing non-existent stop_id.")

    @property
    def routes(self) -> pd.DataFrame:
        return self.tables.get("routes", pd.DataFrame())

    @property
    def trips(self) -> pd.DataFrame:
        return self.tables.get("trips", pd.DataFrame())

    @property
    def stop_times(self) -> pd.DataFrame:
        return self.tables.get("stop_times", pd.DataFrame())

    @property
    def stops(self) -> pd.DataFrame:
        return self.tables.get("stops", pd.DataFrame())

    @property
    def calendar(self) -> pd.DataFrame:
        return self.tables.get("calendar", pd.DataFrame())

    @property
    def shapes(self) -> pd.DataFrame:
        return self.tables.get("shapes", pd.DataFrame())

    def get_route_stops(self, route_id: str) -> pd.DataFrame:
        """Returns ordered list of stops for a route."""
        trips_for_route = self.trips[self.trips["route_id"] == route_id]
        if trips_for_route.empty:
            return pd.DataFrame()
        # Take the representative trip (e.g. first trip)
        sample_trip_id = trips_for_route.iloc[0]["trip_id"]
        route_st = self.stop_times[self.stop_times["trip_id"] == sample_trip_id].sort_values("stop_sequence")
        merged = route_st.merge(self.stops, on="stop_id", how="left")
        return merged

    def get_route_coordinates(self, route_id: str) -> List[Tuple[float, float]]:
        """Returns ordered [(lat, lon), ...] for map polylines."""
        trips_for_route = self.trips[self.trips["route_id"] == route_id]
        if trips_for_route.empty:
            return []

        # Try shapes.txt first if present
        if not self.shapes.empty and "shape_id" in trips_for_route.columns:
            shape_id = trips_for_route.iloc[0]["shape_id"]
            if pd.notna(shape_id):
                shape_pts = self.shapes[self.shapes["shape_id"] == shape_id].sort_values("shape_pt_sequence")
                if not shape_pts.empty:
                    return list(zip(shape_pts["shape_pt_lat"], shape_pts["shape_pt_lon"]))

        # Fallback to stop sequence coords
        stops_df = self.get_route_stops(route_id)
        if not stops_df.empty and "stop_lat" in stops_df.columns:
            return list(zip(stops_df["stop_lat"], stops_df["stop_lon"]))

        return []
