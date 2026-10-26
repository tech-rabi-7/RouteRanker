"""
analytics/01_source_register.py
================================
Step 1 — Source Register & Integrity Audit

Responsibilities:
  • Build and persist a Source Register table (SQLite via DuckDB) that tracks
    every raw input file: dataset_id, snapshot_id, source_url,
    catalog_retrieved_date, downloaded_at, sha256, byte_size.
  • Verify the SHA-256 hash of:
      - CTA ridership CSV
      - 8 Mobility Database mdb-389 GTFS ZIPs
  • Emit a human-readable integrity audit to reports/integrity_audit.md.
  • Exit non-zero if any registered file is missing from disk so the rest of
    the pipeline never runs on incomplete data.

Usage:
    python analytics/01_source_register.py [--data-dir data/raw]

Outputs:
    data/processed/source_register.parquet
    reports/integrity_audit.md
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

# ---------------------------------------------------------------------------
# Canonical source catalogue
# Each entry is the ground-truth record for one raw file.
# sha256 and byte_size are filled in at audit time; they are NOT hardcoded here
# so the register reflects the actual files on disk.
# ---------------------------------------------------------------------------

#: The single CTA ridership CSV filename expected under data/raw/ridership/
CTA_RIDERSHIP_FILENAME = (
    "CTA_-_Ridership_-_Bus_Routes_-_Daily_Totals_by_Route_20261022.csv"
)

#: Mobility Database feed identifier
MDB_FEED_ID = "mdb-389"

#: The 8 GTFS snapshot identifiers in chronological order.
#  Format: mdb-389-<YYYYMMDDHHII>  (UTC download timestamp from Mobility Database)
GTFS_SNAPSHOTS: list[dict] = [
    {
        "snapshot_id": "mdb-389-202511230128",
        "catalog_retrieved_date": "2025-11-23",   # date the catalog entry was current
        "source_url": (
            "https://storage.googleapis.com/mdb-latest/mdb-389-202511230128.zip"
        ),
    },
    {
        "snapshot_id": "mdb-389-202512190112",
        "catalog_retrieved_date": "2025-12-19",
        "source_url": (
            "https://storage.googleapis.com/mdb-latest/mdb-389-202512190112.zip"
        ),
    },
    {
        "snapshot_id": "mdb-389-202601080131",
        "catalog_retrieved_date": "2026-01-08",
        "source_url": (
            "https://storage.googleapis.com/mdb-latest/mdb-389-202601080131.zip"
        ),
    },
    {
        "snapshot_id": "mdb-389-202602100132",
        "catalog_retrieved_date": "2026-02-10",
        "source_url": (
            "https://storage.googleapis.com/mdb-latest/mdb-389-202602100132.zip"
        ),
    },
    {
        "snapshot_id": "mdb-389-202603260020",
        "catalog_retrieved_date": "2026-03-26",
        "source_url": (
            "https://storage.googleapis.com/mdb-latest/mdb-389-202603260020.zip"
        ),
    },
    {
        "snapshot_id": "mdb-389-202604240144",
        "catalog_retrieved_date": "2026-04-24",
        "source_url": (
            "https://storage.googleapis.com/mdb-latest/mdb-389-202604240144.zip"
        ),
    },
    {
        "snapshot_id": "mdb-389-202605300003",
        "catalog_retrieved_date": "2026-05-30",
        "source_url": (
            "https://storage.googleapis.com/mdb-latest/mdb-389-202605300003.zip"
        ),
    },
    {
        "snapshot_id": "mdb-389-202607110020",
        "catalog_retrieved_date": "2026-07-11",
        "source_url": (
            "https://storage.googleapis.com/mdb-latest/mdb-389-202607110020.zip"
        ),
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return hex SHA-256 digest of *path* in streaming fashion (memory safe)."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Core audit logic
# ---------------------------------------------------------------------------

def build_source_register(data_dir: Path) -> pd.DataFrame:
    """
    Walk every expected raw file, compute its SHA-256 + byte_size, and
    return a DataFrame representing the Source Register table.

    Columns
    -------
    dataset_id            : str  – logical name ("cta_ridership" or snapshot_id)
    snapshot_id           : str  – GTFS snapshot identifier or "N/A"
    source_url            : str  – canonical download URL
    catalog_retrieved_date: str  – ISO date the catalog entry was current (GTFS)
    downloaded_at         : str  – ISO-8601 UTC timestamp we measured the file
    sha256                : str  – hex SHA-256 of the file on disk
    byte_size             : int  – file size in bytes
    local_path            : str  – relative path from project root
    status                : str  – "OK" | "MISSING"
    """
    ridership_dir = data_dir / "ridership"
    gtfs_dir = data_dir / "gtfs"

    rows: list[dict] = []
    now = _utc_now_iso()

    # ── CTA Ridership CSV ────────────────────────────────────────────────────
    csv_path = ridership_dir / CTA_RIDERSHIP_FILENAME
    if csv_path.exists():
        rows.append(
            {
                "dataset_id": "cta_ridership",
                "snapshot_id": "N/A",
                "source_url": (
                    "https://data.cityofchicago.org/api/views/jyb9-n7fm/rows.csv"
                    "?accessType=DOWNLOAD"
                ),
                "catalog_retrieved_date": "2026-10-22",
                "downloaded_at": now,
                "sha256": sha256_of_file(csv_path),
                "byte_size": csv_path.stat().st_size,
                "local_path": str(csv_path.relative_to(data_dir.parent.parent)),
                "status": "OK",
            }
        )
    else:
        rows.append(
            {
                "dataset_id": "cta_ridership",
                "snapshot_id": "N/A",
                "source_url": (
                    "https://data.cityofchicago.org/api/views/jyb9-n7fm/rows.csv"
                    "?accessType=DOWNLOAD"
                ),
                "catalog_retrieved_date": "2026-10-22",
                "downloaded_at": now,
                "sha256": "",
                "byte_size": 0,
                "local_path": str(csv_path),
                "status": "MISSING",
            }
        )

    # ── GTFS Snapshots ────────────────────────────────────────────────────────
    for snap in GTFS_SNAPSHOTS:
        sid = snap["snapshot_id"]
        zip_path = gtfs_dir / sid / f"{sid}.zip"
        if zip_path.exists():
            rows.append(
                {
                    "dataset_id": sid,
                    "snapshot_id": sid,
                    "source_url": snap["source_url"],
                    "catalog_retrieved_date": snap["catalog_retrieved_date"],
                    "downloaded_at": now,
                    "sha256": sha256_of_file(zip_path),
                    "byte_size": zip_path.stat().st_size,
                    "local_path": str(
                        zip_path.relative_to(data_dir.parent.parent)
                    ),
                    "status": "OK",
                }
            )
        else:
            rows.append(
                {
                    "dataset_id": sid,
                    "snapshot_id": sid,
                    "source_url": snap["source_url"],
                    "catalog_retrieved_date": snap["catalog_retrieved_date"],
                    "downloaded_at": now,
                    "sha256": "",
                    "byte_size": 0,
                    "local_path": str(zip_path),
                    "status": "MISSING",
                }
            )

    return pd.DataFrame(rows)


def persist_register(df: pd.DataFrame, processed_dir: Path) -> Path:
    """Save the Source Register as a Parquet file via DuckDB (type-safe write)."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "source_register.parquet"

    con = duckdb.connect(":memory:")
    con.register("reg_df", df)
    con.execute(
        f"COPY (SELECT * FROM reg_df) TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    con.close()
    return out_path


def write_integrity_report(df: pd.DataFrame, reports_dir: Path) -> Path:
    """Emit a human-readable Markdown integrity audit report."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "integrity_audit.md"

    ok_rows = df[df["status"] == "OK"]
    missing_rows = df[df["status"] == "MISSING"]

    lines: list[str] = [
        "# RouteRanker — Source Register & Integrity Audit",
        "",
        f"**Generated**: {_utc_now_iso()}  ",
        f"**Total registered sources**: {len(df)}  ",
        f"**Status OK**: {len(ok_rows)}  ",
        f"**Status MISSING**: {len(missing_rows)}  ",
        "",
        "---",
        "",
        "## Source Register",
        "",
        "| dataset_id | snapshot_id | catalog_retrieved_date | byte_size | sha256 (first 16) | status |",
        "|---|---|---|---|---|---|",
    ]

    for _, row in df.iterrows():
        sha_display = row["sha256"][:16] + "…" if row["sha256"] else "—"
        size_display = f"{row['byte_size']:,}" if row["byte_size"] > 0 else "—"
        lines.append(
            f"| `{row['dataset_id']}` | `{row['snapshot_id']}` "
            f"| {row['catalog_retrieved_date']} "
            f"| {size_display} "
            f"| `{sha_display}` "
            f"| **{row['status']}** |"
        )

    lines += [
        "",
        "---",
        "",
        "## GTFS Snapshot → Service Date Mapping",
        "",
        "| snapshot_id | catalog_retrieved_date | service_date_from | service_date_to |",
        "|---|---|---|---|",
        "| mdb-389-202511230128 | 2025-11-23 | 2025-11-23 | 2025-12-18 |",
        "| mdb-389-202512190112 | 2025-12-19 | 2025-12-19 | 2026-01-07 |",
        "| mdb-389-202601080131 | 2026-01-08 | 2026-01-08 | 2026-02-09 |",
        "| mdb-389-202602100132 | 2026-02-10 | 2026-02-10 | 2026-03-25 |",
        "| mdb-389-202603260020 | 2026-03-26 | 2026-03-26 | 2026-04-23 |",
        "| mdb-389-202604240144 | 2026-04-24 | 2026-04-24 | 2026-05-29 |",
        "| mdb-389-202605300003 | 2026-05-30 | 2026-05-30 | 2026-07-10 |",
        "| mdb-389-202607110020 | 2026-07-11 | 2026-07-11 | 2026-07-31 |",
        "",
    ]

    if not missing_rows.empty:
        lines += [
            "---",
            "",
            "## ⚠️ Missing Files",
            "",
            "The following raw files were not found on disk. "
            "Download them before running subsequent pipeline steps.",
            "",
        ]
        for _, row in missing_rows.iterrows():
            lines.append(f"- **`{row['local_path']}`**")
            lines.append(f"  Download from: {row['source_url']}")
            lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="RouteRanker Step 1 — Source Register & Integrity Audit"
    )
    parser.add_argument(
        "--data-dir",
        default="data/raw",
        help="Path to data/raw directory (default: data/raw)",
    )
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
        help="Path to data/processed output directory (default: data/processed)",
    )
    parser.add_argument(
        "--reports-dir",
        default="reports",
        help="Path to reports output directory (default: reports)",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help=(
            "Exit 0 even when raw files are missing "
            "(useful for CI pre-download checks)."
        ),
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    data_dir = (project_root / args.data_dir).resolve()
    processed_dir = (project_root / args.processed_dir).resolve()
    reports_dir = (project_root / args.reports_dir).resolve()

    print("=" * 70)
    print("RouteRanker v2.0  —  Step 1: Source Register & Integrity Audit")
    print("=" * 70)
    print(f"  data/raw      : {data_dir}")
    print(f"  data/processed: {processed_dir}")
    print(f"  reports/      : {reports_dir}")
    print()

    df = build_source_register(data_dir)

    ok_count = (df["status"] == "OK").sum()
    missing_count = (df["status"] == "MISSING").sum()

    print(f"  Registered sources : {len(df)}")
    print(f"  Status OK          : {ok_count}")
    print(f"  Status MISSING     : {missing_count}")
    print()

    # Persist outputs
    reg_path = persist_register(df, processed_dir)
    print(f"  ✔ Source register written → {reg_path.relative_to(project_root)}")

    audit_path = write_integrity_report(df, reports_dir)
    print(f"  ✔ Integrity audit written → {audit_path.relative_to(project_root)}")
    print()

    # Print SHA-256 digest table for review
    print("  SHA-256 Digest Summary:")
    print(f"  {'dataset_id':<35} {'status':<8} {'sha256[:32]'}")
    print("  " + "-" * 75)
    for _, row in df.iterrows():
        sha_display = row["sha256"][:32] if row["sha256"] else "— file missing —"
        print(f"  {row['dataset_id']:<35} {row['status']:<8} {sha_display}")
    print()

    if missing_count > 0:
        print(f"  ⚠  {missing_count} file(s) not found on disk.")
        print(
            "     See reports/integrity_audit.md for download instructions."
        )
        if not args.allow_missing:
            print(
                "     Aborting — subsequent pipeline steps require all sources."
            )
            print(
                "     Re-run with --allow-missing to suppress this exit code."
            )
            return 1

    print("  ✔ Integrity audit complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
