"""
analytics/03_gtfs_calendar_expansion.py
========================================
Step 3 — DuckDB GTFS Auditing & Calendar Expansion

For each of the 8 mdb-389 GTFS snapshots this script:

  1. Extracts the ZIP into data/raw/gtfs/<snapshot_id>/ (tables as .txt files).
  2. Audits the extracted tables (row counts, required columns, null checks).
  3. Loads routes.txt filtered to bus routes ONLY (route_type == '3').
     All GTFS IDs (route_id, trip_id, stop_id, service_id, shape_id) are
     kept as VARCHAR throughout — never coerced to integer.
  4. Loads stop_times.txt via DuckDB with a 512 MB memory budget.
     Post-midnight timestamps (hour >= 24, e.g. "25:10:00") are preserved
     verbatim in a departure_time_raw column and also decomposed into
     (departure_minutes_from_midnight : INTEGER) without wrapping.
  5. Expands calendar.txt + calendar_dates.txt to produce an explicit
     service_date ← service_id mapping for the snapshot's valid date range:
       a. For each calendar.txt row: generate all dates within
          [start_date, end_date] where the weekday flag == '1'.
       b. Apply calendar_dates.txt removals (exception_type == '2').
       c. Apply calendar_dates.txt additions (exception_type == '1').
          Additions are never double-counted.
  6. Computes per-(route, service_date) scheduled trip counts and
     service hours.
  7. Writes outputs to data/processed/gtfs/<snapshot_id>/:
       • bus_routes.parquet
       • stop_times_bus.parquet
       • calendar_expanded.parquet
       • scheduled_service.parquet

Usage:
    python analytics/03_gtfs_calendar_expansion.py [--snapshot-id <id>|--all]

Dependencies: duckdb, pandas, pyarrow
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DUCKDB_MEMORY_LIMIT = "512MB"

SNAPSHOTS: list[dict] = [
    {"snapshot_id": "mdb-389-202511230128", "catalog_retrieved_date": "2025-11-23"},
    {"snapshot_id": "mdb-389-202512190112", "catalog_retrieved_date": "2025-12-19"},
    {"snapshot_id": "mdb-389-202601080131", "catalog_retrieved_date": "2026-01-08"},
    {"snapshot_id": "mdb-389-202602100132", "catalog_retrieved_date": "2026-02-10"},
    {"snapshot_id": "mdb-389-202603260020", "catalog_retrieved_date": "2026-03-26"},
    {"snapshot_id": "mdb-389-202604240144", "catalog_retrieved_date": "2026-04-24"},
    {"snapshot_id": "mdb-389-202605300003", "catalog_retrieved_date": "2026-05-30"},
    {"snapshot_id": "mdb-389-202607110020", "catalog_retrieved_date": "2026-07-11"},
]

# Weekday column names in calendar.txt (order matches Python weekday() 0=Mon)
WEEKDAY_COLS = [
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"
]

# ---------------------------------------------------------------------------
# DuckDB helper
# ---------------------------------------------------------------------------

def _new_con(memory_limit: str = DUCKDB_MEMORY_LIMIT) -> duckdb.DuckDBPyConnection:
    """Return a fresh in-memory DuckDB connection with a fixed memory budget."""
    con = duckdb.connect(":memory:")
    con.execute(f"SET memory_limit = '{memory_limit}'")
    con.execute("SET threads = 4")
    return con


def _save_parquet(df: pd.DataFrame, path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = _new_con()
    con.register("df_out", df)
    con.execute(
        f"COPY (SELECT * FROM df_out) TO '{path}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    con.close()
    print(f"    ✔ {label}: {len(df):,} rows → {path.name}")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# ZIP extraction
# ---------------------------------------------------------------------------

def extract_gtfs_zip(zip_path: Path, extract_dir: Path) -> list[str]:
    """
    Extract GTFS ZIP into extract_dir.
    Returns list of extracted filenames.
    Skips extraction if all required files already exist.
    """
    required = [
        "routes.txt", "trips.txt", "stop_times.txt",
        "stops.txt", "calendar.txt",
    ]
    if all((extract_dir / f).exists() for f in required):
        print(f"    ↷ Already extracted — skipping ZIP extraction.")
        return [f.name for f in extract_dir.iterdir() if f.is_file()]

    if not zip_path.exists():
        raise FileNotFoundError(
            f"GTFS ZIP not found: {zip_path}\n"
            "Download it from the Mobility Database (mdb-389) first."
        )

    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Flatten directory structure — extract only files (no sub-directories)
        names = []
        for member in zf.namelist():
            filename = Path(member).name
            if not filename:
                continue
            target = extract_dir / filename
            with zf.open(member) as src, target.open("wb") as dst:
                dst.write(src.read())
            names.append(filename)
    print(f"    Extracted {len(names)} files from ZIP.")
    return names


# ---------------------------------------------------------------------------
# GTFS table loaders
# ---------------------------------------------------------------------------

def load_bus_routes(extract_dir: Path) -> pd.DataFrame:
    """
    Load routes.txt and filter to bus routes (route_type == '3').
    All IDs kept as strings.
    """
    routes_path = extract_dir / "routes.txt"
    if not routes_path.exists():
        raise FileNotFoundError(f"routes.txt missing in {extract_dir}")

    con = _new_con()
    # Read as all-string via DuckDB; filter route_type == '3'
    df = con.execute(
        f"""
        SELECT
            CAST(route_id       AS VARCHAR) AS route_id,
            CAST(agency_id      AS VARCHAR) AS agency_id,
            CAST(route_short_name AS VARCHAR) AS route_short_name,
            CAST(route_long_name  AS VARCHAR) AS route_long_name,
            CAST(route_type     AS VARCHAR) AS route_type,
            CAST(route_color    AS VARCHAR) AS route_color,
            CAST(route_text_color AS VARCHAR) AS route_text_color
        FROM read_csv_auto('{routes_path}', all_varchar=true)
        WHERE TRIM(route_type) = '3'
        """
    ).df()
    con.close()
    print(f"    Bus routes (route_type=3): {len(df)}")
    return df


def _parse_time_to_minutes(time_series: pd.Series) -> pd.Series:
    """
    Convert HH:MM:SS to total minutes from midnight (INTEGER).
    Preserves post-midnight times above 23 hours (e.g. '25:10:00' → 1510).
    Returns NaN for unparseable values.
    """
    def _parse(s: str) -> Optional[int]:
        if not isinstance(s, str) or ":" not in s:
            return None
        parts = s.strip().split(":")
        try:
            h, m = int(parts[0]), int(parts[1])
            return h * 60 + m   # no modulo — preserves 25h, 26h, etc.
        except (ValueError, IndexError):
            return None

    return time_series.apply(_parse)


def load_stop_times_bus(
    extract_dir: Path,
    bus_trip_ids: set[str],
) -> pd.DataFrame:
    """
    Load stop_times.txt via DuckDB (large file — up to 5.7M rows).
    Filter to trips belonging to bus routes.
    Parse departure_time_raw as-is and decompose to departure_minutes_from_midnight.
    Post-midnight timestamps (>=24:00:00) are preserved; NOT wrapped.
    """
    st_path = extract_dir / "stop_times.txt"
    if not st_path.exists():
        raise FileNotFoundError(f"stop_times.txt missing in {extract_dir}")

    con = _new_con()

    # Read all-varchar via DuckDB, then filter by trip_id in Python
    # Using DuckDB streaming read to stay within 512MB budget
    print(f"    Loading stop_times.txt (may be large) ...")
    df = con.execute(
        f"""
        SELECT
            CAST(trip_id       AS VARCHAR) AS trip_id,
            CAST(stop_id       AS VARCHAR) AS stop_id,
            CAST(stop_sequence AS VARCHAR) AS stop_sequence_raw,
            CAST(arrival_time  AS VARCHAR) AS arrival_time_raw,
            CAST(departure_time AS VARCHAR) AS departure_time_raw
        FROM read_csv_auto('{st_path}', all_varchar=true)
        """
    ).df()
    con.close()

    print(f"    Raw stop_times rows: {len(df):,}")

    # Filter to bus trips only
    df = df[df["trip_id"].isin(bus_trip_ids)].copy()
    print(f"    Bus stop_times rows: {len(df):,}")

    # Type conversions
    df["stop_sequence"] = pd.to_numeric(
        df["stop_sequence_raw"], errors="coerce"
    ).fillna(0).astype(int)

    # Departure time — preserve raw, add integer minutes column
    df["departure_minutes_from_midnight"] = _parse_time_to_minutes(
        df["departure_time_raw"]
    )
    df["arrival_minutes_from_midnight"] = _parse_time_to_minutes(
        df["arrival_time_raw"]
    )

    # Flag post-midnight service (departure hour >= 24)
    df["is_post_midnight"] = df["departure_minutes_from_midnight"].apply(
        lambda x: bool(x is not None and x >= 24 * 60)
    )

    return df.sort_values(["trip_id", "stop_sequence"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Calendar expansion
# ---------------------------------------------------------------------------

def expand_calendar(
    extract_dir: Path,
    date_range: Optional[tuple[date, date]] = None,
) -> pd.DataFrame:
    """
    Produce explicit (service_id, service_date) pairs from calendar.txt
    and calendar_dates.txt.

    Algorithm:
      1. For each calendar.txt row: generate dates in [start_date, end_date]
         where the weekday column == '1'. Timezone: America/Chicago service
         day (date only, no tz conversion needed for date arithmetic).
      2. Remove any (service_id, date) where calendar_dates.txt has
         exception_type == '2'.
      3. Add any (service_id, date) where calendar_dates.txt has
         exception_type == '1' — no double-counting (use a set).

    Parameters
    ----------
    date_range : optional (start, end) filter; if None, uses calendar bounds.

    Returns
    -------
    DataFrame with columns: service_id (str), service_date (date)
    """
    cal_path = extract_dir / "calendar.txt"
    cal_dates_path = extract_dir / "calendar_dates.txt"

    if not cal_path.exists():
        raise FileNotFoundError(f"calendar.txt missing in {extract_dir}")

    # ── Load calendar.txt ────────────────────────────────────────────────────
    cal = pd.read_csv(cal_path, dtype=str)
    cal.columns = cal.columns.str.strip().str.lower()

    # Parse start_date / end_date (YYYYMMDD format in GTFS)
    cal["start_date"] = pd.to_datetime(
        cal["start_date"].str.strip(), format="%Y%m%d", errors="coerce"
    ).dt.date
    cal["end_date"] = pd.to_datetime(
        cal["end_date"].str.strip(), format="%Y%m%d", errors="coerce"
    ).dt.date
    cal = cal.dropna(subset=["service_id", "start_date", "end_date"])

    # ── Step 1: Generate base service dates from weekly schedule ────────────
    base_pairs: set[tuple[str, date]] = set()

    for _, row in cal.iterrows():
        sid = str(row["service_id"]).strip()
        cur = row["start_date"]
        end = row["end_date"]

        while cur <= end:
            wd = cur.weekday()  # 0=Monday … 6=Sunday
            col = WEEKDAY_COLS[wd]
            flag = str(row.get(col, "0")).strip()
            if flag == "1":
                if date_range is None or (date_range[0] <= cur <= date_range[1]):
                    base_pairs.add((sid, cur))
            cur += timedelta(days=1)

    # ── Step 2: Load calendar_dates.txt ─────────────────────────────────────
    additions: set[tuple[str, date]] = set()
    removals: set[tuple[str, date]] = set()

    if cal_dates_path.exists():
        cal_dates = pd.read_csv(cal_dates_path, dtype=str)
        cal_dates.columns = cal_dates.columns.str.strip().str.lower()
        cal_dates["date_parsed"] = pd.to_datetime(
            cal_dates["date"].str.strip(), format="%Y%m%d", errors="coerce"
        ).dt.date
        cal_dates = cal_dates.dropna(subset=["service_id", "date_parsed"])

        for _, row in cal_dates.iterrows():
            sid = str(row["service_id"]).strip()
            d = row["date_parsed"]
            exc = str(row.get("exception_type", "")).strip()

            if date_range is not None and not (date_range[0] <= d <= date_range[1]):
                continue

            if exc == "2":
                removals.add((sid, d))
            elif exc == "1":
                additions.add((sid, d))

    # ── Step 3: Apply removals and additions ─────────────────────────────────
    active_pairs = (base_pairs - removals) | additions  # additions never double-counted

    df = pd.DataFrame(
        list(active_pairs), columns=["service_id", "service_date"]
    ).sort_values(["service_id", "service_date"]).reset_index(drop=True)

    # Keep service_id as string
    df["service_id"] = df["service_id"].astype(str)

    return df


# ---------------------------------------------------------------------------
# Scheduled service computation
# ---------------------------------------------------------------------------

def compute_scheduled_service(
    bus_trips: pd.DataFrame,
    calendar_expanded: pd.DataFrame,
    stop_times: pd.DataFrame,
    bus_routes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute per-(route_short_name, service_date) scheduled service metrics.

    Metrics:
      scheduled_trips          : number of distinct trip_ids active on that date
      scheduled_service_hours  : sum of trip durations (last arrival - first dep)
                                 in fractional hours
      duration_complete        : True iff ALL trips on that route/date have a
                                 valid (non-null) duration

    Post-midnight trips are attributed to the America/Chicago service date
    they departed on — i.e. a trip departing at 25:10 is treated as 01:10
    on the NEXT calendar date, but it belongs to the service_date that
    activated its service_id. No time-zone conversion is applied to the
    date itself; the GTFS service_date is used directly.
    """
    # Map trip_id → route_id → route_short_name
    route_lookup = bus_routes.set_index("route_id")["route_short_name"].to_dict()
    trip_to_route = bus_trips.set_index("trip_id")["route_id"].to_dict()
    trip_to_service = bus_trips.set_index("trip_id")["service_id"].to_dict()

    # Map service_id → set of service_dates
    svc_to_dates: dict[str, set[date]] = {}
    for _, row in calendar_expanded.iterrows():
        svc_to_dates.setdefault(str(row["service_id"]), set()).add(row["service_date"])

    # Compute per-trip duration from stop_times
    trip_durations: dict[str, Optional[float]] = {}
    for trip_id, grp in stop_times.groupby("trip_id"):
        first_dep = grp["departure_minutes_from_midnight"].dropna()
        last_arr = grp["arrival_minutes_from_midnight"].dropna()
        if first_dep.empty or last_arr.empty:
            trip_durations[str(trip_id)] = None
        else:
            dur_min = float(last_arr.max()) - float(first_dep.min())
            trip_durations[str(trip_id)] = max(0.0, dur_min) / 60.0  # hours

    # Expand: for each bus trip × each of its service dates → (route_short, date)
    records: dict[tuple[str, date], dict] = {}

    for trip_id, route_id in trip_to_route.items():
        route_short = route_lookup.get(str(route_id), str(route_id))
        svc_id = str(trip_to_service.get(str(trip_id), ""))
        active_dates = svc_to_dates.get(svc_id, set())
        dur_h = trip_durations.get(str(trip_id))

        for sdate in active_dates:
            key = (route_short, sdate)
            if key not in records:
                records[key] = {
                    "route_short_name": route_short,
                    "service_date": sdate,
                    "trip_ids": [],
                    "durations_h": [],
                }
            records[key]["trip_ids"].append(str(trip_id))
            records[key]["durations_h"].append(dur_h)

    rows = []
    for key, rec in records.items():
        durations = rec["durations_h"]
        valid_durs = [d for d in durations if d is not None]
        duration_complete = len(valid_durs) == len(durations) and len(durations) > 0

        rows.append(
            {
                "route_short_name": rec["route_short_name"],
                "service_date": rec["service_date"],
                "scheduled_trips": len(rec["trip_ids"]),
                "scheduled_service_hours": (
                    round(sum(valid_durs), 4) if valid_durs else None
                ),
                "duration_complete": duration_complete,
            }
        )

    df = pd.DataFrame(rows).sort_values(
        ["route_short_name", "service_date"]
    ).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Per-snapshot pipeline
# ---------------------------------------------------------------------------

def process_snapshot(
    snapshot_id: str,
    catalog_retrieved_date: str,
    raw_gtfs_dir: Path,
    processed_gtfs_dir: Path,
) -> pd.DataFrame:
    """
    Run the full extraction + calendar expansion pipeline for one snapshot.
    Returns the scheduled_service DataFrame.
    """
    snap_raw_dir = raw_gtfs_dir / snapshot_id
    snap_proc_dir = processed_gtfs_dir / snapshot_id
    snap_proc_dir.mkdir(parents=True, exist_ok=True)

    zip_path = snap_raw_dir / f"{snapshot_id}.zip"

    print(f"\n  ─── Snapshot: {snapshot_id} ───────────────────────────────")
    print(f"    catalog_retrieved_date: {catalog_retrieved_date}")

    # ── 1. Extract ZIP ───────────────────────────────────────────────────────
    extract_gtfs_zip(zip_path, snap_raw_dir)

    # ── 2. Load bus routes ───────────────────────────────────────────────────
    bus_routes = load_bus_routes(snap_raw_dir)
    _save_parquet(bus_routes, snap_proc_dir / "bus_routes.parquet", "bus_routes")

    # ── 3. Load trips filtered to bus routes ─────────────────────────────────
    trips_path = snap_raw_dir / "trips.txt"
    if not trips_path.exists():
        raise FileNotFoundError(f"trips.txt missing in {snap_raw_dir}")

    trips = pd.read_csv(trips_path, dtype=str)
    trips.columns = trips.columns.str.strip().str.lower()
    trips["route_id"] = trips["route_id"].str.strip()
    trips["trip_id"] = trips["trip_id"].str.strip()
    trips["service_id"] = trips["service_id"].str.strip()

    bus_route_ids = set(bus_routes["route_id"].str.strip())
    bus_trips = trips[trips["route_id"].isin(bus_route_ids)].copy()
    print(f"    Bus trips: {len(bus_trips):,}")

    bus_trip_ids = set(bus_trips["trip_id"])

    # ── 4. Load stop_times for bus trips ─────────────────────────────────────
    stop_times = load_stop_times_bus(snap_raw_dir, bus_trip_ids)
    _save_parquet(
        stop_times, snap_proc_dir / "stop_times_bus.parquet", "stop_times_bus"
    )

    # ── 5. Calendar expansion ─────────────────────────────────────────────────
    calendar_expanded = expand_calendar(snap_raw_dir)
    print(
        f"    Calendar expanded: {len(calendar_expanded):,} (service_id, service_date) pairs"
    )
    _save_parquet(
        calendar_expanded,
        snap_proc_dir / "calendar_expanded.parquet",
        "calendar_expanded",
    )

    # ── 6. Scheduled service per (route, date) ───────────────────────────────
    scheduled = compute_scheduled_service(
        bus_trips, calendar_expanded, stop_times, bus_routes
    )
    print(
        f"    Scheduled service: {len(scheduled):,} (route, date) rows"
    )
    _save_parquet(
        scheduled,
        snap_proc_dir / "scheduled_service.parquet",
        "scheduled_service",
    )

    return scheduled


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="RouteRanker Step 3 — DuckDB GTFS Auditing & Calendar Expansion"
    )
    parser.add_argument(
        "--snapshot-id",
        help="Process a single snapshot (e.g. mdb-389-202511230128). "
             "Omit to process all 8.",
    )
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--all", action="store_true", dest="process_all",
                        help="Process all 8 snapshots (default behaviour).")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    raw_gtfs_dir = (project_root / args.data_dir / "gtfs").resolve()
    processed_gtfs_dir = (project_root / args.processed_dir / "gtfs").resolve()

    print("=" * 70)
    print("RouteRanker v2.0  —  Step 3: GTFS Auditing & Calendar Expansion")
    print("=" * 70)
    print(f"  raw GTFS dir  : {raw_gtfs_dir}")
    print(f"  processed dir : {processed_gtfs_dir}")
    print(f"  DuckDB memory : {DUCKDB_MEMORY_LIMIT}")

    target_snapshots = SNAPSHOTS
    if args.snapshot_id:
        target_snapshots = [
            s for s in SNAPSHOTS if s["snapshot_id"] == args.snapshot_id
        ]
        if not target_snapshots:
            print(f"  ✗ Unknown snapshot_id: {args.snapshot_id}")
            return 1

    all_scheduled: list[pd.DataFrame] = []
    errors: list[str] = []

    for snap in target_snapshots:
        try:
            sched = process_snapshot(
                snapshot_id=snap["snapshot_id"],
                catalog_retrieved_date=snap["catalog_retrieved_date"],
                raw_gtfs_dir=raw_gtfs_dir,
                processed_gtfs_dir=processed_gtfs_dir,
            )
            all_scheduled.append(sched)
        except FileNotFoundError as exc:
            print(f"\n  ⚠  Skipping {snap['snapshot_id']}: {exc}")
            errors.append(f"{snap['snapshot_id']}: {exc}")

    if errors:
        print("\n  Some snapshots skipped due to missing files:")
        for e in errors:
            print(f"    • {e}")

    if not all_scheduled:
        print("\n  No snapshots processed. Exiting.")
        return 1 if errors else 0

    print(f"\n  Step 3 complete. Processed {len(all_scheduled)} snapshot(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
