"""
analytics/02_clean_boardings.py
================================
Step 2 — Clean Boardings Validation

Parses the CTA daily ridership CSV and produces a validated, de-duplicated
table of (route, service_date, daytype, rides) observations.

Exact rules enforced (per spec):
  ✔ route retained as VARCHAR — preserves N/X/K prefix variants
  ✔ date parsed as US month/day/year format (e.g. "01/15/2026" → 2026-01-15)
  ✔ daytype kept as single char: W (weekday), A (Saturday), U (Sunday/holiday)
  ✔ rides: strip thousands-separators (commas), cast to INTEGER
  ✔ Exact-copy duplicates (all 4 cols identical) → deduplicated, kept once
  ✔ Conflicting duplicates (same route+date, different rides/daytype) →
       quarantined to data/processed/boardings_conflicts.parquet
       NOT dropped from clean table — largest rides value is kept with a
       conflict_flag=True marker so analysts can audit them
  ✔ Genuine zero-count rows (rides == 0) are PRESERVED as-is
  ✔ Missing dates between first/last observation are NOT filled with zero
  ✔ No synthetic rows added under any circumstance

Outputs:
    data/processed/boardings_clean.parquet   ← primary clean table
    data/processed/boardings_conflicts.parquet ← quarantined conflict pairs
    reports/data_coverage.md                ← human-readable coverage report

Usage:
    python analytics/02_clean_boardings.py [--data-dir data/raw]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CTA_RIDERSHIP_FILENAME = (
    "CTA_-_Ridership_-_Bus_Routes_-_Daily_Totals_by_Route_20261022.csv"
)

# Known zero-count observation count for validation assertion (spec: 36)
EXPECTED_TRUE_ZEROS = 36


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_and_parse_ridership(csv_path: Path) -> pd.DataFrame:
    """
    Load raw CTA ridership CSV into a clean DataFrame.

    Expected raw columns (case-insensitive):
        route, date, daytype, rides

    Returns DataFrame with typed columns:
        route        : str   (preserved, e.g. "N20", "X49", "36")
        service_date : date  (parsed from US MM/DD/YYYY)
        daytype      : str   (single char: W / A / U)
        rides        : int   (thousands-sep stripped)
    """
    # Read everything as strings first to prevent pandas from coercing types
    raw = pd.read_csv(
        csv_path,
        dtype=str,
        encoding="utf-8-sig",   # handle BOM if present
        skip_blank_lines=True,
    )

    # Normalise column names: strip whitespace and lowercase
    raw.columns = raw.columns.str.strip().str.lower()

    required = {"route", "date", "daytype", "rides"}
    missing_cols = required - set(raw.columns)
    if missing_cols:
        raise ValueError(
            f"CTA ridership CSV is missing required columns: {missing_cols}. "
            f"Found: {list(raw.columns)}"
        )

    # ── route: strip whitespace, keep as string (preserves N/X/K prefixes) ──
    raw["route"] = raw["route"].str.strip()

    # ── service_date: parse US date format MM/DD/YYYY ───────────────────────
    raw["service_date"] = pd.to_datetime(
        raw["date"].str.strip(),
        format="%m/%d/%Y",
        errors="coerce",
    ).dt.date

    unparseable_dates = raw["service_date"].isna().sum()
    if unparseable_dates > 0:
        # Attempt ISO fallback (YYYY-MM-DD) for any that failed
        mask = raw["service_date"].isna()
        raw.loc[mask, "service_date"] = pd.to_datetime(
            raw.loc[mask, "date"].str.strip(),
            format="%Y-%m-%d",
            errors="coerce",
        ).dt.date
        still_bad = raw["service_date"].isna().sum()
        if still_bad > 0:
            print(
                f"  ⚠  {still_bad} rows have unparseable dates — will be dropped."
            )

    # ── daytype: strip to single char, validate ──────────────────────────────
    raw["daytype"] = raw["daytype"].str.strip().str.upper()
    valid_daytypes = {"W", "A", "U"}
    bad_daytype_mask = ~raw["daytype"].isin(valid_daytypes)
    if bad_daytype_mask.sum() > 0:
        print(
            f"  ⚠  {bad_daytype_mask.sum()} rows with unexpected daytype values: "
            f"{raw.loc[bad_daytype_mask, 'daytype'].unique().tolist()}"
        )

    # ── rides: strip commas (thousands-sep), cast to int ────────────────────
    raw["rides"] = (
        raw["rides"]
        .str.strip()
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
    )
    raw["rides"] = pd.to_numeric(raw["rides"], errors="coerce")

    # ── Drop rows with missing mandatory fields (not zero rides!) ────────────
    before_drop = len(raw)
    raw = raw.dropna(subset=["route", "service_date", "daytype", "rides"])
    dropped = before_drop - len(raw)
    if dropped > 0:
        print(f"  ⚠  Dropped {dropped} rows with null mandatory fields.")

    raw["rides"] = raw["rides"].astype(int)

    # Select only the 4 canonical columns
    df = raw[["route", "service_date", "daytype", "rides"]].copy()

    return df


def deduplicate_boardings(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separate exact-copy duplicates and conflicting duplicates.

    Returns
    -------
    clean_df     : de-duplicated clean boardings (conflict rows kept with flag)
    conflict_df  : all conflicting (route, date) groups for quarantine
    """
    key = ["route", "service_date"]

    # Step 1: remove exact-copy duplicates (all 4 cols identical)
    df_deduped = df.drop_duplicates(
        subset=["route", "service_date", "daytype", "rides"],
        keep="first",
    )

    exact_removed = len(df) - len(df_deduped)
    if exact_removed > 0:
        print(f"  Removed {exact_removed} exact-copy duplicate rows.")

    # Step 2: identify conflicting duplicates (same route+date, different rides)
    dupe_mask = df_deduped.duplicated(subset=key, keep=False)
    conflict_df = df_deduped[dupe_mask].copy()
    conflict_df["conflict_group"] = conflict_df.groupby(key).ngroup()

    # Step 3: for clean table, keep row with max rides per conflict group
    #         and mark it with a conflict_flag so analysts know
    non_conflict = df_deduped[~dupe_mask].copy()
    non_conflict["conflict_flag"] = False

    if not conflict_df.empty:
        # Keep the row with the highest rides value for each conflict key
        resolved = (
            df_deduped[dupe_mask]
            .sort_values("rides", ascending=False)
            .drop_duplicates(subset=key, keep="first")
            .copy()
        )
        resolved["conflict_flag"] = True
        clean_df = pd.concat([non_conflict, resolved], ignore_index=True)
        print(
            f"  Found {len(conflict_df)} conflicting rows in "
            f"{conflict_df.groupby(key).ngroups} (route, date) groups → "
            f"quarantined; max-rides row retained in clean table."
        )
    else:
        clean_df = non_conflict

    clean_df = clean_df.sort_values(["route", "service_date"]).reset_index(
        drop=True
    )
    return clean_df, conflict_df


# ---------------------------------------------------------------------------
# DuckDB persistence helpers
# ---------------------------------------------------------------------------

def _save_parquet(df: pd.DataFrame, path: Path, label: str) -> None:
    """Write DataFrame to Parquet via DuckDB (enforces typed schema)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(":memory:")
    con.register("df_out", df)
    con.execute(
        f"COPY (SELECT * FROM df_out) TO '{path}' "
        f"(FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    con.close()
    print(f"  ✔ {label} → {path}  ({len(df):,} rows)")


# ---------------------------------------------------------------------------
# Coverage report
# ---------------------------------------------------------------------------

def write_coverage_report(
    clean_df: pd.DataFrame,
    conflict_df: pd.DataFrame,
    reports_dir: Path,
) -> Path:
    """Produce data/coverage Markdown report."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "data_coverage.md"

    total_rows = len(clean_df)
    n_routes = clean_df["route"].nunique()
    n_dates = clean_df["service_date"].nunique()
    min_date = clean_df["service_date"].min()
    max_date = clean_df["service_date"].max()
    zero_rows = (clean_df["rides"] == 0).sum()
    conflict_rows = clean_df.get("conflict_flag", pd.Series(dtype=bool)).sum()

    daytype_counts = clean_df["daytype"].value_counts().to_dict()

    # Top-10 routes by total boardings
    top_routes = (
        clean_df.groupby("route")["rides"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    lines = [
        "# RouteRanker — CTA Boardings Data Coverage Report",
        "",
        f"**Generated**: {_utc_now_iso()}  ",
        f"**Source**: `{CTA_RIDERSHIP_FILENAME}`  ",
        "",
        "---",
        "",
        "## Summary Statistics",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Total clean observations | {total_rows:,} |",
        f"| Unique routes | {n_routes} |",
        f"| Unique service dates | {n_dates} |",
        f"| Date range | {min_date} → {max_date} |",
        f"| Weekday observations (W) | {daytype_counts.get('W', 0):,} |",
        f"| Saturday observations (A) | {daytype_counts.get('A', 0):,} |",
        f"| Sunday/holiday observations (U) | {daytype_counts.get('U', 0):,} |",
        f"| Genuine zero-rides rows preserved | {zero_rows} |",
        f"| Rows with conflict_flag=True | {int(conflict_rows)} |",
        f"| Quarantined conflict rows | {len(conflict_df)} |",
        "",
        "> Zero-count rows are **preserved** as genuine observations.",
        "> Missing dates between observations are **not** filled with zero.",
        "",
        "---",
        "",
        "## Top 10 Routes by Total Boardings",
        "",
        "| Route | Total Rides |",
        "|---|---|",
    ]

    for route, total in top_routes.items():
        lines.append(f"| {route} | {total:,} |")

    lines += [
        "",
        "---",
        "",
        f"## Zero-Count Observations ({zero_rows} rows)",
        "",
        "These rows represent legitimate CTA observations where a route "
        "recorded zero passenger boardings (e.g., emergency shutdowns, "
        "extreme weather cancellations). They are **retained** in the clean "
        "dataset.",
        "",
    ]

    if zero_rows > 0:
        zero_sample = clean_df[clean_df["rides"] == 0][
            ["route", "service_date", "daytype", "rides"]
        ].head(20)
        lines += [
            "Sample (up to 20 rows):",
            "",
            "| route | service_date | daytype | rides |",
            "|---|---|---|---|",
        ]
        for _, row in zero_sample.iterrows():
            lines.append(
                f"| {row['route']} | {row['service_date']} "
                f"| {row['daytype']} | {row['rides']} |"
            )

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Validation assertions
# ---------------------------------------------------------------------------

def validate_clean_boardings(df: pd.DataFrame) -> list[str]:
    """
    Run a set of data-quality assertions.
    Returns a list of warning strings (empty = all passed).
    """
    warnings: list[str] = []

    # No null mandatory fields
    for col in ["route", "service_date", "daytype", "rides"]:
        n = df[col].isna().sum()
        if n > 0:
            warnings.append(f"  ✗ {n} null values in '{col}' after cleaning.")

    # rides is non-negative integer
    if (df["rides"] < 0).any():
        warnings.append("  ✗ Negative rides values found — unexpected.")

    # daytype restricted to W/A/U
    bad = ~df["daytype"].isin({"W", "A", "U"})
    if bad.any():
        warnings.append(
            f"  ✗ Unexpected daytype values: {df.loc[bad,'daytype'].unique().tolist()}"
        )

    # Zero-count rows present (spec: 36 known zeros)
    zero_count = (df["rides"] == 0).sum()
    if zero_count == 0:
        warnings.append(
            "  ✗ No zero-rides rows found — genuine zeros may have been dropped."
        )
    elif zero_count != EXPECTED_TRUE_ZEROS:
        warnings.append(
            f"  ⚠  Zero-rides row count = {zero_count} "
            f"(spec expects {EXPECTED_TRUE_ZEROS}). "
            "Acceptable if dataset range differs from spec."
        )

    # No exact duplicates remain
    exact_dupes = df.duplicated(
        subset=["route", "service_date", "daytype", "rides"]
    ).sum()
    if exact_dupes > 0:
        warnings.append(
            f"  ✗ {exact_dupes} exact-copy duplicate rows remain after dedup."
        )

    return warnings


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="RouteRanker Step 2 — Clean Boardings Validation"
    )
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--reports-dir", default="reports")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    ridership_csv = (
        project_root / args.data_dir / "ridership" / CTA_RIDERSHIP_FILENAME
    )
    processed_dir = (project_root / args.processed_dir).resolve()
    reports_dir = (project_root / args.reports_dir).resolve()

    print("=" * 70)
    print("RouteRanker v2.0  —  Step 2: Clean Boardings Validation")
    print("=" * 70)
    print(f"  Input CSV : {ridership_csv}")
    print()

    if not ridership_csv.exists():
        print(f"  ✗ ERROR: CTA ridership CSV not found at {ridership_csv}")
        print(
            "  Run Step 1 (01_source_register.py) and download the file first."
        )
        return 1

    # ── Parse ────────────────────────────────────────────────────────────────
    print("  Parsing raw CSV ...")
    df_raw = load_and_parse_ridership(ridership_csv)
    print(f"  Raw rows loaded: {len(df_raw):,}")
    print(f"  Unique routes  : {df_raw['route'].nunique()}")
    print(f"  Date range     : {df_raw['service_date'].min()} → {df_raw['service_date'].max()}")
    print(f"  Zero-ride rows : {(df_raw['rides']==0).sum()}")
    print()

    # ── Deduplicate ──────────────────────────────────────────────────────────
    print("  Deduplicating ...")
    df_clean, df_conflicts = deduplicate_boardings(df_raw)
    print(f"  Clean rows     : {len(df_clean):,}")
    print(f"  Conflict rows  : {len(df_conflicts)}")
    print()

    # ── Validate ─────────────────────────────────────────────────────────────
    print("  Running validation assertions ...")
    warnings = validate_clean_boardings(df_clean)
    if warnings:
        for w in warnings:
            print(w)
    else:
        print("  ✔ All assertions passed.")
    print()

    # ── Persist ──────────────────────────────────────────────────────────────
    _save_parquet(df_clean, processed_dir / "boardings_clean.parquet", "Clean boardings")
    if not df_conflicts.empty:
        _save_parquet(
            df_conflicts,
            processed_dir / "boardings_conflicts.parquet",
            "Conflict boardings (quarantine)",
        )

    # ── Coverage report ──────────────────────────────────────────────────────
    rpt_path = write_coverage_report(df_clean, df_conflicts, reports_dir)
    print(f"  ✔ Coverage report → {rpt_path.relative_to(project_root)}")
    print()
    print("  Step 2 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
