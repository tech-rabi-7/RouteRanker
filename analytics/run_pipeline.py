"""
analytics/run_pipeline.py
==========================
Master pipeline runner for RouteRanker v2.0.

Runs all numbered analytics steps in order. Individual steps can be
skipped by name using --skip or run alone using --only.

Usage:
    # Run the full pipeline
    python analytics/run_pipeline.py

    # Run only steps 1 and 2
    python analytics/run_pipeline.py --only 1 2

    # Skip step 3 (GTFS expansion — may take time without data)
    python analytics/run_pipeline.py --skip 3

    # Allow missing raw files (step 1 integrity check won't abort)
    python analytics/run_pipeline.py --allow-missing

Environment
-----------
Set DATA_DIR, PROCESSED_DIR, REPORTS_DIR env vars to override defaults.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

PIPELINE_STEPS = [
    {"n": 1, "module": "analytics.01_source_register",         "label": "Source Register & Integrity Audit"},
    {"n": 2, "module": "analytics.02_clean_boardings",          "label": "Clean Boardings Validation"},
    {"n": 3, "module": "analytics.03_gtfs_calendar_expansion",  "label": "GTFS Auditing & Calendar Expansion"},
    {"n": 4, "module": "analytics.04_snapshot_selection_and_join", "label": "Snapshot Selection & Route/Date Join"},
    {"n": 5, "module": "analytics.05_route_structure_and_overlap", "label": "Route Structure & Stop Overlap Computation"},
]


def _run_module_main(module_name: str, extra_argv: list[str]) -> int:
    """Import a module by dotted name and call its main() with extra argv."""
    spec = importlib.util.spec_from_file_location(
        module_name,
        Path(__file__).parent / f"{module_name.split('.')[-1]}.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    old_argv = sys.argv[:]
    sys.argv = [module_name] + extra_argv
    try:
        return mod.main()
    finally:
        sys.argv = old_argv


def main() -> int:
    parser = argparse.ArgumentParser(
        description="RouteRanker v2.0 — Master Pipeline Runner"
    )
    parser.add_argument(
        "--only", nargs="+", type=int, metavar="N",
        help="Run only these step numbers (e.g. --only 1 2)."
    )
    parser.add_argument(
        "--skip", nargs="+", type=int, metavar="N",
        help="Skip these step numbers."
    )
    parser.add_argument("--allow-missing", action="store_true",
                        help="Pass --allow-missing to step 1.")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--reports-dir", default="reports")
    args = parser.parse_args()

    skip_set = set(args.skip or [])
    only_set = set(args.only or [])

    steps_to_run = [
        s for s in PIPELINE_STEPS
        if (not only_set or s["n"] in only_set) and s["n"] not in skip_set
    ]

    print("=" * 70)
    print("RouteRanker v2.0  —  Master Pipeline Runner")
    print("=" * 70)
    print(f"  Steps to run: {[s['n'] for s in steps_to_run]}")
    print()

    results: list[dict] = []
    for step in steps_to_run:
        n = step["n"]
        label = step["label"]
        print(f"{'─'*70}")
        print(f"  Running Step {n}: {label}")
        print(f"{'─'*70}")

        extra: list[str] = [
            f"--data-dir={args.data_dir}",
            f"--processed-dir={args.processed_dir}",
            f"--reports-dir={args.reports_dir}",
        ]
        if n == 1 and args.allow_missing:
            extra.append("--allow-missing")

        t0 = time.perf_counter()
        try:
            rc = _run_module_main(step["module"], extra)
        except SystemExit as e:
            rc = int(e.code) if e.code is not None else 1
        except Exception as exc:
            print(f"\n  ✗ Step {n} raised an exception: {exc}")
            rc = 2

        elapsed = time.perf_counter() - t0
        status = "✔ OK" if rc == 0 else f"✗ FAILED (rc={rc})"
        results.append({"step": n, "label": label, "rc": rc, "elapsed_s": elapsed})
        print(f"\n  Step {n} finished in {elapsed:.1f}s — {status}")
        print()

        if rc != 0:
            print("  Pipeline aborted at failed step.")
            break

    print("=" * 70)
    print("  Pipeline Summary")
    print("=" * 70)
    for r in results:
        icon = "✔" if r["rc"] == 0 else "✗"
        print(f"  {icon} Step {r['step']:1d}: {r['label']:<45} {r['elapsed_s']:>6.1f}s")
    print()

    return 0 if all(r["rc"] == 0 for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
