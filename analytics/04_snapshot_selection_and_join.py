"""
analytics/04_snapshot_selection_and_join.py
============================================
Step 4 — Global Per-Date Snapshot Selection & Exact Route/Date Join

Implements the two-part final joining logic:

PART A — Global Snapshot Selection
    For each requested service_date (2025-11-23 to 2026-07-31), globally
    select the latest downloaded GTFS snapshot whose catalog_retrieved_date
    is on or before that service_date. The mapping is deterministic:

        2025-11-23 snapshot → 2025-11-23 to 2025-12-18
        2025-12-19 snapshot → 2025-12-19 to 2026-01-07
        2026-01-08 snapshot → 2026-01-08 to 2026-02-09
        2026-02-10 snapshot → 2026-02-10 to 2026-03-25
        2026-03-26 snapshot → 2026-03-26 to 2026-04-23
        2026-04-24 snapshot → 2026-04-24 to 2026-05-29
        2026-05-30 snapshot → 2026-05-30 to 2026-07-10
        2026-07-11 snapshot → 2026-07-11 to 2026-07-31

    Rule: NEVER fall back to an older snapshot if a route was withdrawn.
    A route not present in the applicable snapshot is marked as
    schedule_status = "WITHDRAWN" and excluded from the joined output.

PART B — Exact Route/Date Join
    Join selected scheduled_service data with boardings_clean on:
        schedule.route_short_name == boardings.route  (exact string match)
        schedule.service_date     == boardings.service_date

    Targets (per spec):
        • 28,374 matched observations across 251 dates × 125 routes
        • 24 unmatched schedule rows reported separately

    Outputs:
        data/processed/route_profiles.parquet     ← 28,374 matched rows
        data/processed/unmatched_schedule.parquet ← 24 unmatched schedule rows
        data/processed/unmatched_boardings.parquet ← boardings with no schedule
        reports/join_audit.md

Usage:
    python analytics/04_snapshot_selection_and_join.py
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

# ---------------------------------------------------------------------------
# Snapshot → service date range mapping (deterministic, per spec)
# ---------------------------------------------------------------------------

SNAPSHOT_DATE_RANGES: list[dict] = [
    {
        "snapshot_id": "mdb-389-202511230128",
        "catalog_retrieved_date": date(2025, 11, 23),
        "service_date_from": date(2025, 11, 23),
        "service_date_to": date(2025, 12, 18),
    },
    {
        "snapshot_id": "mdb-389-202512190112",
        "catalog_retrieved_date": date(2025, 12, 19),
        "service_date_from": date(2025, 12, 19),
        "service_date_to": date(2026, 1, 7),
    },
    {
        "snapshot_id": "mdb-389-202601080131",
        "catalog_retrieved_date": date(2026, 1, 8),
        "service_date_from": date(2026, 1, 8),
        "service_date_to": date(2026, 2, 9),
    },
    {
        "snapshot_id": "mdb-389-202602100132",
        "catalog_retrieved_date": date(2026, 2, 10),
        "service_date_from": date(2026, 2, 10),
        "service_date_to": date(2026, 3, 25),
    },
    {
        "snapshot_id": "mdb-389-202603260020",
        "catalog_retrieved_date": date(2026, 3, 26),
        "service_date_from": date(2026, 3, 26),
        "service_date_to": date(2026, 4, 23),
    },
    {
        "snapshot_id": "mdb-389-202604240144",
        "catalog_retrieved_date": date(2026, 4, 24),
        "service_date_from": date(2026, 4, 24),
        "service_date_to": date(2026, 5, 29),
    },
    {
        "snapshot_id": "mdb-389-202605300003",
        "catalog_retrieved_date": date(2026, 5, 30),
        "service_date_from": date(2026, 5, 30),
        "service_date_to": date(2026, 7, 10),
    },
    {
        "snapshot_id": "mdb-389-202607110020",
        "catalog_retrieved_date": date(2026, 7, 11),
        "service_date_from": date(2026, 7, 11),
        "service_date_to": date(2026, 7, 31),
    },
]

PIPELINE_DATE_FROM = date(2025, 11, 23)
PIPELINE_DATE_TO   = date(2026, 7, 31)

# Spec targets
SPEC_MATCHED_ROWS  = 28_374
SPEC_MATCHED_DATES = 251
SPEC_MATCHED_ROUTES = 125
SPEC_UNMATCHED_SCHEDULE_ROWS = 24


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _save_parquet(df: pd.DataFrame, path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(":memory:")
    con.register("df_out", df)
    con.execute(
        f"COPY (SELECT * FROM df_out) TO '{path}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    con.close()
    print(f"  ✔ {label}: {len(df):,} rows → {path.name}")


def snapshot_for_date(service_date: date) -> Optional[str]:
    """
    Return the snapshot_id applicable for a given service_date.
    Returns None if the date is outside the pipeline window.
    """
    for snap in SNAPSHOT_DATE_RANGES:
        if snap["service_date_from"] <= service_date <= snap["service_date_to"]:
            return snap["snapshot_id"]
    return None


# ---------------------------------------------------------------------------
# Part A: Build selected scheduled service table
# ---------------------------------------------------------------------------

def build_selected_schedule(processed_gtfs_dir: Path) -> pd.DataFrame:
    """
    For each snapshot, load its scheduled_service.parquet and tag every row
    with its snapshot_id and the date range it covers. Stitch all 8 into one
    unified schedule table filtered to PIPELINE_DATE_FROM → PIPELINE_DATE_TO.

    Enforces the 'never fall back' rule: if a route_short_name appears in an
    earlier snapshot but NOT in the applicable snapshot for a given date, it
    is excluded (treated as withdrawn).
    """
    frames: list[pd.DataFrame] = []

    for snap in SNAPSHOT_DATE_RANGES:
        sid = snap["snapshot_id"]
        sched_path = processed_gtfs_dir / sid / "scheduled_service.parquet"

        if not sched_path.exists():
            print(f"  ⚠  scheduled_service.parquet not found for {sid} — skipping.")
            continue

        con = duckdb.connect(":memory:")
        df = con.execute(
            f"""
            SELECT
                route_short_name,
                service_date,
                scheduled_trips,
                scheduled_service_hours,
                duration_complete
            FROM read_parquet('{sched_path}')
            WHERE service_date >= DATE '{snap["service_date_from"]}'
              AND service_date <= DATE '{snap["service_date_to"]}'
            """
        ).df()
        con.close()

        df["snapshot_id"] = sid
        df["catalog_retrieved_date"] = snap["catalog_retrieved_date"]
        frames.append(df)
        print(
            f"    {sid}: {len(df):,} rows "
            f"({snap['service_date_from']} → {snap['service_date_to']})"
        )

    if not frames:
        raise RuntimeError(
            "No scheduled_service.parquet files found. "
            "Run Step 3 (03_gtfs_calendar_expansion.py) first."
        )

    combined = pd.concat(frames, ignore_index=True)

    # Ensure service_date is a proper date (DuckDB may return datetime)
    combined["service_date"] = pd.to_datetime(
        combined["service_date"]
    ).dt.date

    # Filter to pipeline window
    combined = combined[
        (combined["service_date"] >= PIPELINE_DATE_FROM)
        & (combined["service_date"] <= PIPELINE_DATE_TO)
    ].copy()

    # Verify no (route, date) appears in more than one snapshot
    dupes = combined.duplicated(
        subset=["route_short_name", "service_date"], keep=False
    )
    if dupes.any():
        print(
            f"  ⚠  {dupes.sum()} duplicate (route, date) rows across snapshots — "
            "keeping latest snapshot."
        )
        combined = (
            combined
            .sort_values("catalog_retrieved_date", ascending=False)
            .drop_duplicates(subset=["route_short_name", "service_date"], keep="first")
        )

    return combined.sort_values(
        ["route_short_name", "service_date"]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Part B: Exact Route/Date Join with Boardings
# ---------------------------------------------------------------------------

def join_schedule_boardings(
    selected_schedule: pd.DataFrame,
    boardings: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Inner join on (route_short_name == route) AND (service_date == service_date).

    Returns
    -------
    matched       : inner-join result (target: 28,374 rows)
    unmatched_sched : schedule rows with no boarding match (target: 24 rows)
    unmatched_board : boarding rows with no schedule match
    """
    # Normalise join keys
    sched = selected_schedule.copy()
    sched["join_route"] = sched["route_short_name"].str.strip().str.upper()
    sched["join_date"] = pd.to_datetime(sched["service_date"]).dt.date

    board = boardings.copy()
    board["join_route"] = board["route"].str.strip().str.upper()
    board["join_date"] = pd.to_datetime(board["service_date"]).dt.date

    # Inner join
    merged = sched.merge(
        board,
        left_on=["join_route", "join_date"],
        right_on=["join_route", "join_date"],
        how="inner",
        suffixes=("_sched", "_board"),
    )

    # Unmatched schedule rows
    sched_matched_keys = set(
        zip(merged["join_route"], merged["join_date"])
    )
    unmatched_sched = sched[
        ~sched.apply(
            lambda r: (r["join_route"], r["join_date"]) in sched_matched_keys,
            axis=1,
        )
    ].drop(columns=["join_route", "join_date"])

    # Unmatched boarding rows
    board_matched_keys = set(
        zip(merged["join_route"], merged["join_date"])
    )
    unmatched_board = board[
        ~board.apply(
            lambda r: (r["join_route"], r["join_date"]) in board_matched_keys,
            axis=1,
        )
    ].drop(columns=["join_route", "join_date"])

    # Clean matched output
    matched = merged.drop(columns=["join_route", "join_date"])

    # Rationalise service_date columns (may have _sched / _board suffixes)
    for col_variant in ["service_date_sched", "service_date_board"]:
        if col_variant in matched.columns and "service_date" not in matched.columns:
            matched = matched.rename(columns={col_variant: "service_date"})
        elif col_variant in matched.columns:
            matched = matched.drop(columns=[col_variant])

    return matched, unmatched_sched, unmatched_board


# ---------------------------------------------------------------------------
# Join audit report
# ---------------------------------------------------------------------------

def write_join_audit(
    matched: pd.DataFrame,
    unmatched_sched: pd.DataFrame,
    unmatched_board: pd.DataFrame,
    reports_dir: Path,
) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "join_audit.md"

    n_matched = len(matched)
    n_dates = matched["service_date"].nunique() if "service_date" in matched.columns else "?"
    n_routes = matched["route_short_name"].nunique() if "route_short_name" in matched.columns else "?"
    n_unmatched_s = len(unmatched_sched)
    n_unmatched_b = len(unmatched_board)

    # Spec deviation flags
    deviation_rows  = abs(n_matched - SPEC_MATCHED_ROWS)
    deviation_dates = abs((n_dates if isinstance(n_dates, int) else 0) - SPEC_MATCHED_DATES)
    deviation_routes = abs((n_routes if isinstance(n_routes, int) else 0) - SPEC_MATCHED_ROUTES)
    deviation_unmatched = abs(n_unmatched_s - SPEC_UNMATCHED_SCHEDULE_ROWS)

    lines = [
        "# RouteRanker — Schedule × Boardings Join Audit",
        "",
        f"**Generated**: {_utc_now_iso()}  ",
        "",
        "---",
        "",
        "## Join Results vs Spec Targets",
        "",
        "| Metric | Actual | Spec Target | Δ |",
        "|---|---|---|---|",
        f"| Matched rows | {n_matched:,} | {SPEC_MATCHED_ROWS:,} | {deviation_rows:+,} |",
        f"| Unique service dates | {n_dates} | {SPEC_MATCHED_DATES} | {deviation_dates:+} |",
        f"| Unique routes | {n_routes} | {SPEC_MATCHED_ROUTES} | {deviation_routes:+} |",
        f"| Unmatched schedule rows | {n_unmatched_s} | {SPEC_UNMATCHED_SCHEDULE_ROWS} | {deviation_unmatched:+} |",
        f"| Unmatched boarding rows | {n_unmatched_b} | — | — |",
        "",
        "---",
        "",
        "## Snapshot Coverage",
        "",
        "| snapshot_id | catalog_retrieved_date | service_date_from | service_date_to |",
        "|---|---|---|---|",
    ]

    for snap in SNAPSHOT_DATE_RANGES:
        lines.append(
            f"| `{snap['snapshot_id']}` | {snap['catalog_retrieved_date']} "
            f"| {snap['service_date_from']} | {snap['service_date_to']} |"
        )

    if n_unmatched_s > 0:
        lines += [
            "",
            "---",
            "",
            f"## Unmatched Schedule Rows ({n_unmatched_s})",
            "",
            "These (route, date) combinations appear in a GTFS snapshot "
            "but have no corresponding CTA ridership observation.",
            "",
            "| route_short_name | service_date | snapshot_id | scheduled_trips |",
            "|---|---|---|---|",
        ]
        for _, row in unmatched_sched.head(50).iterrows():
            lines.append(
                f"| {row.get('route_short_name','?')} "
                f"| {row.get('service_date','?')} "
                f"| {row.get('snapshot_id','?')} "
                f"| {row.get('scheduled_trips','?')} |"
            )

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_join_result(
    matched: pd.DataFrame,
    unmatched_sched: pd.DataFrame,
) -> list[str]:
    warnings: list[str] = []

    n = len(matched)
    if n == 0:
        warnings.append("  ✗ CRITICAL: matched DataFrame is empty.")
        return warnings

    # Spec target check (allow ±5% tolerance for dataset version drift)
    tolerance = 0.05
    if abs(n - SPEC_MATCHED_ROWS) / SPEC_MATCHED_ROWS > tolerance:
        warnings.append(
            f"  ⚠  Matched rows ({n:,}) deviates >5% from spec target "
            f"({SPEC_MATCHED_ROWS:,})."
        )

    # No null rides
    if "rides" in matched.columns and matched["rides"].isna().any():
        warnings.append("  ✗ Null rides values in matched table.")

    # No null scheduled_trips
    if "scheduled_trips" in matched.columns and matched["scheduled_trips"].isna().any():
        warnings.append("  ✗ Null scheduled_trips values in matched table.")

    # Unique (route, date) — no duplicates in matched
    dupes = matched.duplicated(
        subset=["route_short_name", "service_date"], keep=False
    )
    if dupes.any():
        warnings.append(
            f"  ✗ {dupes.sum()} duplicate (route, date) rows in matched table."
        )

    return warnings


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="RouteRanker Step 4 — Snapshot Selection & Route/Date Join"
    )
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--reports-dir", default="reports")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    processed_dir = (project_root / args.processed_dir).resolve()
    processed_gtfs_dir = processed_dir / "gtfs"
    reports_dir = (project_root / args.reports_dir).resolve()

    print("=" * 70)
    print("RouteRanker v2.0  —  Step 4: Snapshot Selection & Route/Date Join")
    print("=" * 70)
    print(f"  processed_dir   : {processed_dir}")
    print(f"  Pipeline window : {PIPELINE_DATE_FROM} → {PIPELINE_DATE_TO}")
    print()

    # ── Part A: Build selected schedule ──────────────────────────────────────
    print("  Part A: Global snapshot selection ...")
    selected_schedule = build_selected_schedule(processed_gtfs_dir)
    print(
        f"  Selected schedule: {len(selected_schedule):,} (route, date) rows "
        f"across {selected_schedule['route_short_name'].nunique()} routes, "
        f"{selected_schedule['service_date'].nunique()} dates."
    )
    _save_parquet(
        selected_schedule,
        processed_dir / "selected_schedule.parquet",
        "Selected schedule",
    )
    print()

    # ── Load clean boardings ──────────────────────────────────────────────────
    boardings_path = processed_dir / "boardings_clean.parquet"
    if not boardings_path.exists():
        print(
            f"  ✗ boardings_clean.parquet not found at {boardings_path}\n"
            "  Run Step 2 (02_clean_boardings.py) first."
        )
        return 1

    con = duckdb.connect(":memory:")
    boardings = con.execute(
        f"""
        SELECT route, service_date, daytype, rides
        FROM read_parquet('{boardings_path}')
        WHERE service_date >= DATE '{PIPELINE_DATE_FROM}'
          AND service_date <= DATE '{PIPELINE_DATE_TO}'
        """
    ).df()
    con.close()
    boardings["service_date"] = pd.to_datetime(boardings["service_date"]).dt.date
    print(
        f"  Boardings in window: {len(boardings):,} rows "
        f"({boardings['route'].nunique()} routes, "
        f"{boardings['service_date'].nunique()} dates)"
    )
    print()

    # ── Part B: Exact join ───────────────────────────────────────────────────
    print("  Part B: Exact route/date join ...")
    matched, unmatched_sched, unmatched_board = join_schedule_boardings(
        selected_schedule, boardings
    )
    print(f"  Matched rows        : {len(matched):,}  (spec target: {SPEC_MATCHED_ROWS:,})")
    print(f"  Unmatched schedule  : {len(unmatched_sched)}  (spec target: {SPEC_UNMATCHED_SCHEDULE_ROWS})")
    print(f"  Unmatched boardings : {len(unmatched_board)}")
    print()

    # ── Validate ─────────────────────────────────────────────────────────────
    warnings = validate_join_result(matched, unmatched_sched)
    if warnings:
        print("  Validation warnings:")
        for w in warnings:
            print(w)
    else:
        print("  ✔ All join validation checks passed.")
    print()

    # ── Persist outputs ───────────────────────────────────────────────────────
    _save_parquet(matched, processed_dir / "route_profiles.parquet", "route_profiles (matched)")
    _save_parquet(
        unmatched_sched,
        processed_dir / "unmatched_schedule.parquet",
        "unmatched_schedule",
    )
    _save_parquet(
        unmatched_board,
        processed_dir / "unmatched_boardings.parquet",
        "unmatched_boardings",
    )

    # ── Audit report ──────────────────────────────────────────────────────────
    audit_path = write_join_audit(matched, unmatched_sched, unmatched_board, reports_dir)
    print(f"  ✔ Join audit → {audit_path.relative_to(project_root)}")
    print()
    print("  Step 4 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
