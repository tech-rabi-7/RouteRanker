"""
train_models.py  —  RouteRanker v2.0
======================================
CLI driver for the Daily Boarding Regression pipeline.

Orchestrates:
  1. Load processed data from data/processed/
  2. Feature engineering (demand-only and demand+service tables)
  3. Chronological train/val/test splits per spec
  4. Seasonal baselines + ML model training and evaluation
  5. Productivity ranking table
  6. Route structural clustering (KMeans)
  7. Persist models + evaluation report

Usage
-----
    # Full run (requires data/processed/boardings_clean.parquet
    #           and data/processed/route_profiles.parquet)
    python train_models.py

    # Skip service-aligned experiment (only demand-only table available)
    python train_models.py --demand-only

    # Use custom directories
    python train_models.py --processed-dir data/processed --model-dir models

    # Dry-run: print config and exit without training
    python train_models.py --dry-run

Outputs
-------
    models/demand_only_random_forest.joblib     (or xgboost / gradient_boosting)
    models/demand_only_ridge_pooled.joblib
    models/matched_service_*.joblib
    models/kmeans_model.joblib
    models/kmeans_scaler.joblib
    models/training_metrics.json
    reports/model_evaluation.md
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so 'src' package is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models import (
    DEMAND_ONLY_SPLIT,
    MATCHED_SERVICE_SPLIT,
    BoardingFeatureEngineer,
    BoardingRegressionPipeline,
    RouteProductivityRanker,
    RouteStructuralClusterer,
    build_evaluation_report,
    compute_metrics,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BOARDINGS_FILE   = "boardings_clean.parquet"
ROUTE_PROFILES_FILE = "route_profiles.parquet"   # matched (step 4 output)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_parquet(path: Path, label: str) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"  ⚠  {label} not found at {path} — skipping.")
        return None
    try:
        import duckdb
        df = duckdb.read_parquet(str(path)).df()
        print(f"  ✔ {label}: {len(df):,} rows")
        return df
    except ImportError:
        df = pd.read_parquet(path)
        print(f"  ✔ {label}: {len(df):,} rows")
        return df


def _coerce_dates(df: pd.DataFrame, col: str = "service_date") -> pd.DataFrame:
    df = df.copy()
    df[col] = pd.to_datetime(df[col]).dt.date
    return df


# ---------------------------------------------------------------------------
# Markdown report writer
# ---------------------------------------------------------------------------

def _metric_row(name: str, m: dict) -> str:
    wape = f"{m['wape']:.4f}" if m.get("wape_available") else "N/A"
    return (
        f"| {name} | {m['n']:,} | {m['mae']:.1f} | {m['rmse']:.1f} | {wape} |"
    )


def write_model_evaluation_report(
    report: dict,
    productivity_table: Optional[pd.DataFrame],
    reports_dir: Path,
) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "model_evaluation.md"

    # Publication delay info
    meta = report.get("metadata", {})
    delay_days = meta.get("release_delay_days", 0)
    exclude_lag1 = meta.get("exclude_lag1", False)
    delay_status = (
        f"**Operational Release-Delay Mode**: Active (`--release-delay-days={delay_days}`, `lag_1d` excluded). "
        "Evaluates real-time operational readiness when previous-day ridership observations are not yet published."
        if (exclude_lag1 or delay_days >= 2)
        else "**Retrospective Experiment Mode**: Complete lag history available (`lag_1d` included). "
        "Evaluates maximum theoretical predictive capacity with T-1 observed ridership."
    )

    lines: list[str] = [
        "# RouteRanker v2.0 — Model Evaluation Report",
        "",
        f"**Generated**: {report['generated_at']}  ",
        f"**Artifact Version**: {report.get('artifact_version', '2.0.0')}  ",
        "",
        f"> {delay_status}",
        "",
        "> Primary metric: **WAPE** (Weighted Absolute Percentage Error).  ",
        "> WAPE = Σ|y − ŷ| / Σy.  Reported as `N/A` when Σy = 0 (zero-volume group).  ",
        "> Plain MAPE is NOT used because of genuine zero-ride observations.",
        "",
        "---",
        "",
    ]

    for exp_key, exp_label in [
        ("demand_only",      "Experiment A — Demand Only"),
        ("matched_service",  "Experiment B — Demand + Matched Service"),
    ]:
        exp = report.get(exp_key)
        if exp is None:
            continue

        ss = exp["split_sizes"]
        lines += [
            f"## {exp_label}",
            "",
            f"| Split | Rows |",
            f"|---|---|",
            f"| Train | {ss['train']:,} |",
            f"| Val   | {ss['val']:,} |",
            f"| Test  | {ss['test']:,} |",
            "",
            "### Baselines",
            "",
            "| Model | N | MAE | RMSE | WAPE |",
            "|---|---|---|---|---|",
        ]
        for bname, bres in exp.get("baselines", {}).items():
            lines.append(_metric_row(f"{bname} — val",  bres["val"]))
            lines.append(_metric_row(f"{bname} — test", bres["test"]))

        lines += [
            "",
            "### ML Models",
            "",
            "| Model | N | MAE | RMSE | WAPE |",
            "|---|---|---|---|---|",
        ]
        for mname, mres in exp.get("models", {}).items():
            lines.append(_metric_row(f"{mname} — val",  mres["val"]))
            lines.append(_metric_row(f"{mname} — test", mres["test"]))

        # Route-level summary
        rl = exp.get("route_level_test", {})
        best_model = rl.get("best_model", "—")
        n_routes = rl.get("route_count", 0)
        route_metrics = rl.get("route_metrics", [])

        if route_metrics:
            maes  = [m["mae"]  for m in route_metrics if m["mae"]  is not None]
            rmses = [m["rmse"] for m in route_metrics if m["rmse"] is not None]
            wapes = [m["wape"] for m in route_metrics if m.get("wape_available")]

            lines += [
                "",
                f"### Route-Level Test Metrics (best model: `{best_model}`)",
                "",
                f"| Metric | Network-weighted | Median across {n_routes} routes |",
                "|---|---|---|",
                f"| MAE  | {np.mean(maes):.1f}  | {np.median(maes):.1f}  |",
                f"| RMSE | {np.mean(rmses):.1f} | {np.median(rmses):.1f} |",
                f"| WAPE | {np.mean(wapes):.4f} | {np.median(wapes):.4f} |"
                if wapes else "| WAPE | N/A | N/A |",
                "",
            ]

        lines += ["", "---", ""]

    # Experiment C: Controlled Aligned Comparison
    ctrl = report.get("controlled_comparison")
    if ctrl is not None:
        lines += [
            "## Experiment C — Controlled Aligned Comparison",
            "",
            "> Evaluates Demand-Only and Demand+Service models over the EXACT same `MATCHED_SERVICE_SPLIT` "
            "dates to isolate the true marginal value of GTFS schedule features.",
            "",
            "| Feature Set | Test N | Random Forest Test WAPE | Ridge Test WAPE |",
            "|---|---|---|---|",
        ]
        d_mod = ctrl.get("demand_only_aligned", {}).get("models", {})
        s_mod = ctrl.get("demand_plus_service", {}).get("models", {})

        rf_d = d_mod.get("random_forest", {}).get("test", {}).get("wape", "N/A")
        rd_d = d_mod.get("ridge_pooled", {}).get("test", {}).get("wape", "N/A")
        n_d = d_mod.get("random_forest", {}).get("test", {}).get("n", 7014)

        rf_s = s_mod.get("random_forest", {}).get("test", {}).get("wape", "N/A")
        rd_s = s_mod.get("ridge_pooled", {}).get("test", {}).get("wape", "N/A")

        lines.append(f"| Aligned Demand-Only | {n_d:,} | {rf_d} | {rd_d} |")
        lines.append(f"| Demand + Matched Service | {n_d:,} | {rf_s} | {rd_s} |")

        comp = ctrl.get("comparison", {})
        delta = comp.get("wape_delta")
        if delta is not None:
            lines += [
                "",
                f"**Marginal WAPE Impact of GTFS Service Data**: `{delta:+.4f}` "
                f"({'Improvement' if delta < 0 else 'Neutral/Within margin'})",
            ]
        lines += ["", "---", ""]

    # Productivity table summary
    ps = report.get("productivity_summary")
    if ps:
        lines += [
            "## Productivity Ranking Summary",
            "",
            f"Eligible routes (≥20 obs) ranked by `boardings_per_scheduled_service_hour` within each daytype.",
            "",
            f"Total routes: **{ps['n_routes_total']}**  ",
            f"Weekday-eligible: **{ps['n_eligible_wd']}**",
            "",
            "### Top-5 Weekday Routes (by productivity rank)",
            "",
            "| Rank | Route | Boardings/Service Hour |",
            "|---|---|---|",
        ]
        for row in ps.get("top5_weekday", []):
            rank = int(row["service_productivity_metric_rank"])
            lines.append(
                f"| {rank} | {row['route_short_name']} "
                f"| {row['boardings_per_scheduled_service_hour']:.2f} |"
            )
        lines += ["", "---", ""]

    # Clustering summary
    cs = report.get("clustering_summary")
    if cs:
        lines += [
            "## Route Structural Clustering (KMeans)",
            "",
            f"**n_clusters**: {cs['n_clusters']}  ",
            "> Cluster IDs are structural archetypes only — not performance labels.",
            "",
            "| Cluster ID | Route Count |",
            "|---|---|",
        ]
        for cid, cnt in sorted(cs["cluster_sizes"].items()):
            lines.append(f"| {cid} | {cnt} |")
        lines += ["", "---", ""]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Split validation helper
# ---------------------------------------------------------------------------

def _validate_split_sizes(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
    spec_train: int, spec_val: int, spec_test: int,
    name: str,
    tolerance_pct: float = 5.0,
) -> None:
    for split_name, actual, expected in [
        ("train", len(train), spec_train),
        ("val",   len(val),   spec_val),
        ("test",  len(test),  spec_test),
    ]:
        if expected == 0:
            continue
        pct_diff = abs(actual - expected) / expected * 100
        icon = "✔" if pct_diff <= tolerance_pct else "⚠ "
        print(
            f"  {icon} {name}/{split_name}: {actual:,} rows "
            f"(spec: {expected:,}, Δ={actual-expected:+,}, {pct_diff:.1f}%)"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="RouteRanker v2.0 — Daily Boarding Regression Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Examples:
              python train_models.py
              python train_models.py --demand-only
              python train_models.py --n-clusters 6 --model-dir models/exp1
            """
        ),
    )
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument(
        "--demand-only", action="store_true",
        help="Skip the demand+service experiment even if route_profiles.parquet exists."
    )
    parser.add_argument(
        "--exclude-lag1", action="store_true",
        help="Exclude lag_1d to simulate operational release delay / issue-time gate."
    )
    parser.add_argument(
        "--release-delay-days", type=int, default=0,
        help="Assumed release delay in days (>=2 excludes lag_1d)."
    )
    parser.add_argument(
        "--no-controlled-comparison", action="store_true",
        help="Skip Experiment C (controlled comparison)."
    )
    parser.add_argument("--n-clusters", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print configuration and exit without training."
    )
    args = parser.parse_args()

    project_root = _PROJECT_ROOT
    processed_dir = (project_root / args.processed_dir).resolve()
    model_dir     = (project_root / args.model_dir).resolve()
    reports_dir   = (project_root / args.reports_dir).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("RouteRanker v2.0  —  Daily Boarding Regression Training")
    print("=" * 70)
    print(f"  processed_dir      : {processed_dir}")
    print(f"  model_dir          : {model_dir}")
    print(f"  reports_dir        : {reports_dir}")
    print(f"  exclude_lag1       : {args.exclude_lag1}")
    print(f"  release_delay_days : {args.release_delay_days}")
    print(f"  n_clusters         : {args.n_clusters}")
    print(f"  n_jobs             : {args.n_jobs}")
    if args.dry_run:
        print("\n  [DRY RUN] Exiting without training.")
        return 0
    print()

    t_total = time.perf_counter()

    # ── Load processed data ─────────────────────────────────────────────────
    print("Loading processed data ...")
    boardings_df = _load_parquet(
        processed_dir / BOARDINGS_FILE, "boardings_clean"
    )
    route_profiles_df = _load_parquet(
        processed_dir / ROUTE_PROFILES_FILE, "route_profiles (matched)"
    )

    if boardings_df is None:
        print(
            "\n  ✗ boardings_clean.parquet is required. "
            "Run analytics/02_clean_boardings.py first."
        )
        return 1
    print()

    boardings_df = _coerce_dates(boardings_df)

    # ── Experiment A: Demand-Only ───────────────────────────────────────────
    print("=" * 70)
    print("Experiment A — Demand Only")
    print("=" * 70)

    fe_demand = BoardingFeatureEngineer(
        include_service=False,
        exclude_lag1=args.exclude_lag1,
        release_delay_days=args.release_delay_days,
    )
    print("  Building demand-only feature matrix (lag-join) ...")
    t0 = time.perf_counter()
    feat_demand = fe_demand.build(boardings_df)
    print(f"  Feature matrix: {len(feat_demand):,} rows, "
          f"{len(fe_demand.feature_columns)} features  "
          f"({time.perf_counter()-t0:.1f}s)")
    print()

    # Validate split sizes against spec
    tr_d, va_d, te_d = DEMAND_ONLY_SPLIT.apply(feat_demand)
    _validate_split_sizes(
        tr_d, va_d, te_d,
        spec_train=63_143, spec_val=10_246, spec_test=10_608,
        name="demand_only",
    )
    print()

    pipe_demand = BoardingRegressionPipeline(
        split=DEMAND_ONLY_SPLIT,
        include_service=False,
        exclude_lag1=args.exclude_lag1,
        release_delay_days=args.release_delay_days,
        model_dir=str(model_dir),
        n_jobs=args.n_jobs,
        random_state=args.random_state,
    )
    demand_result = pipe_demand.fit_evaluate(feat_demand)
    pipe_demand.save_models(prefix="demand_only_")
    print()

    # ── Experiment B: Demand + Matched Service ──────────────────────────────
    matched_result: Optional[dict] = None

    if not args.demand_only and route_profiles_df is not None:
        print("=" * 70)
        print("Experiment B — Demand + Matched Service")
        print("=" * 70)

        route_profiles_df = _coerce_dates(route_profiles_df)

        # Align boarding column name (route_short_name → route for join)
        if "route_short_name" in route_profiles_df.columns and "route" not in route_profiles_df.columns:
            route_profiles_df = route_profiles_df.rename(
                columns={"route_short_name": "route"}
            )

        # Ensure service cols are present
        required_service_cols = {"route", "service_date", "daytype", "rides",
                                  "scheduled_trips", "scheduled_service_hours"}
        missing_cols = required_service_cols - set(route_profiles_df.columns)
        if missing_cols:
            print(
                f"  ⚠  route_profiles.parquet missing columns: {missing_cols}  "
                "— skipping Experiment B."
            )
        else:
            # Cast numeric service cols
            for col in ["scheduled_trips", "scheduled_service_hours", "rides"]:
                route_profiles_df[col] = pd.to_numeric(
                    route_profiles_df[col], errors="coerce"
                ).fillna(0)

            fe_service = BoardingFeatureEngineer(
                include_service=True,
                exclude_lag1=args.exclude_lag1,
                release_delay_days=args.release_delay_days,
            )
            print("  Building demand+service feature matrix (lag-join) ...")
            t0 = time.perf_counter()
            feat_service = fe_service.build(route_profiles_df)
            print(
                f"  Feature matrix: {len(feat_service):,} rows, "
                f"{len(fe_service.feature_columns)} features  "
                f"({time.perf_counter()-t0:.1f}s)"
            )
            print()

            tr_s, va_s, te_s = MATCHED_SERVICE_SPLIT.apply(feat_service)
            _validate_split_sizes(
                tr_s, va_s, te_s,
                spec_train=14_501, spec_val=6_859, spec_test=7_014,
                name="matched_service",
            )
            print()

            pipe_service = BoardingRegressionPipeline(
                split=MATCHED_SERVICE_SPLIT,
                include_service=True,
                exclude_lag1=args.exclude_lag1,
                release_delay_days=args.release_delay_days,
                model_dir=str(model_dir),
                n_jobs=args.n_jobs,
                random_state=args.random_state,
            )
            matched_result = pipe_service.fit_evaluate(feat_service)
            pipe_service.save_models(prefix="matched_service_")
            print()
    elif args.demand_only:
        print("  [--demand-only flag set] Skipping Experiment B.")
    else:
        print("  route_profiles.parquet not found — skipping Experiment B.")
    print()

    # ── Experiment C: Controlled Aligned Comparison ─────────────────────────
    controlled_result: Optional[dict] = None
    if not args.no_controlled_comparison and route_profiles_df is not None and not args.demand_only:
        print("=" * 70)
        print("Experiment C — Controlled Aligned Comparison (Identical Split)")
        print("=" * 70)
        print("  Evaluating Demand-Only vs Demand+Service on MATCHED_SERVICE_SPLIT ...")
        fe_ctrl_demand = BoardingFeatureEngineer(
            include_service=False,
            exclude_lag1=args.exclude_lag1,
            release_delay_days=args.release_delay_days,
        )
        feat_ctrl_demand = fe_ctrl_demand.build(route_profiles_df)
        pipe_ctrl_demand = BoardingRegressionPipeline(
            split=MATCHED_SERVICE_SPLIT,
            include_service=False,
            exclude_lag1=args.exclude_lag1,
            release_delay_days=args.release_delay_days,
            model_dir=str(model_dir),
            n_jobs=args.n_jobs,
            random_state=args.random_state,
        )
        ctrl_demand_res = pipe_ctrl_demand.fit_evaluate(feat_ctrl_demand)

        rf_d_wape = ctrl_demand_res["models"]["random_forest"]["test"]["wape"]
        rf_s_wape = matched_result["models"]["random_forest"]["test"]["wape"] if matched_result else None

        controlled_result = {
            "split": MATCHED_SERVICE_SPLIT.name,
            "demand_only_aligned": ctrl_demand_res,
            "demand_plus_service": matched_result,
            "comparison": {
                "rf_test_wape_demand_only": rf_d_wape,
                "rf_test_wape_with_service": rf_s_wape,
                "wape_delta": round(rf_s_wape - rf_d_wape, 6) if (rf_s_wape is not None and rf_d_wape is not None) else None,
            }
        }
        print(f"  ✔ Controlled Comparison Complete. WAPE Delta: {controlled_result['comparison']['wape_delta']}")
        print()

    # ── Productivity Ranking ────────────────────────────────────────────────
    productivity_table: Optional[pd.DataFrame] = None
    if route_profiles_df is not None and "scheduled_trips" in route_profiles_df.columns:
        print("=" * 70)
        print("Productivity Ranking")
        print("=" * 70)
        ranker = RouteProductivityRanker()
        # Rename 'route' → 'route_short_name' if needed
        rp = route_profiles_df.copy()
        if "route_short_name" not in rp.columns and "route" in rp.columns:
            rp = rp.rename(columns={"route": "route_short_name"})

        productivity_table = ranker.build_profile_table(rp)
        n_eligible_w = int(
            productivity_table[
                (productivity_table["daytype"] == "W")
                & productivity_table["service_productivity_metric_rank"].notna()
            ].shape[0]
        )
        print(
            f"  Routes in profile table : {productivity_table['route_short_name'].nunique()}"
        )
        print(f"  Weekday-eligible routes : {n_eligible_w}")

        # Persist productivity table
        prod_path = (processed_dir / "productivity_rankings.parquet")
        try:
            import duckdb
            con = duckdb.connect(":memory:")
            con.register("pt", productivity_table)
            con.execute(
                f"COPY (SELECT * FROM pt) TO '{prod_path}' "
                "(FORMAT PARQUET, COMPRESSION ZSTD)"
            )
            con.close()
        except Exception:
            productivity_table.to_parquet(prod_path, index=False)
        print(f"  ✔ Productivity rankings → {prod_path.relative_to(project_root)}")
        print()

    # ── KMeans Structural Clustering ────────────────────────────────────────
    cluster_result: Optional[pd.DataFrame] = None

    cluster_input_df = route_profiles_df if route_profiles_df is not None else boardings_df
    cluster_matched  = route_profiles_df if route_profiles_df is not None else None

    print("=" * 70)
    print(f"Route Structural Clustering (KMeans, k={args.n_clusters})")
    print("=" * 70)

    clusterer = RouteStructuralClusterer(
        n_clusters=args.n_clusters,
        random_state=args.random_state,
        model_dir=str(model_dir),
    )

    # Build route feature matrix — use train-period data only to avoid leakage
    train_mask = pd.to_datetime(cluster_input_df["service_date"]).dt.date
    if route_profiles_df is not None:
        split = MATCHED_SERVICE_SPLIT
    else:
        split = DEMAND_ONLY_SPLIT

    train_cluster_df = cluster_input_df[
        (train_mask >= split.train_from) & (train_mask <= split.train_to)
    ].copy()

    # Rename for clusterer
    if "route_short_name" in train_cluster_df.columns and "route" not in train_cluster_df.columns:
        train_cluster_df = train_cluster_df.rename(columns={"route_short_name": "route"})

    # Rename matched df similarly
    matched_for_cluster = cluster_matched
    if matched_for_cluster is not None:
        matched_for_cluster = matched_for_cluster[
            (pd.to_datetime(matched_for_cluster["service_date"]).dt.date >= split.train_from)
            & (pd.to_datetime(matched_for_cluster["service_date"]).dt.date <= split.train_to)
        ].copy()
        if "route" not in matched_for_cluster.columns and "route_short_name" in matched_for_cluster.columns:
            matched_for_cluster = matched_for_cluster.rename(columns={"route_short_name": "route"})

    route_matrix = clusterer.build_route_feature_matrix(
        train_cluster_df, matched_df=matched_for_cluster
    )
    print(f"  Route feature matrix: {len(route_matrix)} routes")

    cluster_result = clusterer.fit(route_matrix)
    clusterer.save()

    cluster_sizes = cluster_result["cluster_id"].value_counts().sort_index()
    print(f"  Cluster sizes: {cluster_sizes.to_dict()}")

    cluster_path = processed_dir / "route_clusters.parquet"
    try:
        import duckdb
        cr_reset = cluster_result.reset_index()
        con = duckdb.connect(":memory:")
        con.register("cr", cr_reset)
        con.execute(
            f"COPY (SELECT * FROM cr) TO '{cluster_path}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        con.close()
    except Exception:
        cluster_result.reset_index().to_parquet(cluster_path, index=False)
    print(f"  ✔ Cluster assignments → {cluster_path.relative_to(project_root)}")
    print()

    # ── Build and save evaluation report & metadata ─────────────────────────
    print("=" * 70)
    print("Saving evaluation artefacts & metadata")
    print("=" * 70)

    # Extract source register hashes
    sr_hashes = {}
    sr_path = processed_dir / "source_register.parquet"
    if sr_path.exists():
        try:
            import duckdb
            sr_df = duckdb.read_parquet(str(sr_path)).df()
            for _, row in sr_df.iterrows():
                sr_hashes[str(row.get("dataset_id"))] = str(row.get("sha256", ""))
        except Exception:
            pass

    metadata = {
        "artifact_version": "2.0.0",
        "release_delay_assumption": (
            "Operational release-delay assumption (lag_1d excluded)"
            if (args.exclude_lag1 or args.release_delay_days >= 2)
            else "Retrospective experiment (complete lag_1d history available)"
        ),
        "exclude_lag1": bool(args.exclude_lag1),
        "release_delay_days": int(args.release_delay_days),
        "feature_schema": {
            "categorical": fe_demand.BASE_CATEGORICAL,
            "calendar": fe_demand.BASE_CALENDAR,
            "lag_features": fe_demand.lag_columns,
            "seasonal_aggregate": fe_demand.BASE_SEASONAL,
            "service_features": ["scheduled_trips", "scheduled_service_hours"],
        },
        "training_date_ranges": {
            "demand_only": {
                "train": [str(DEMAND_ONLY_SPLIT.train_from), str(DEMAND_ONLY_SPLIT.train_to)],
                "val": [str(DEMAND_ONLY_SPLIT.val_from), str(DEMAND_ONLY_SPLIT.val_to)],
                "test": [str(DEMAND_ONLY_SPLIT.test_from), str(DEMAND_ONLY_SPLIT.test_to)],
            },
            "matched_service": {
                "train": [str(MATCHED_SERVICE_SPLIT.train_from), str(MATCHED_SERVICE_SPLIT.train_to)],
                "val": [str(MATCHED_SERVICE_SPLIT.val_from), str(MATCHED_SERVICE_SPLIT.val_to)],
                "test": [str(MATCHED_SERVICE_SPLIT.test_from), str(MATCHED_SERVICE_SPLIT.test_to)],
            },
        },
        "source_sha256_hashes": sr_hashes,
    }

    report = build_evaluation_report(
        demand_only_result=demand_result,
        matched_service_result=matched_result,
        productivity_table=productivity_table,
        cluster_result=cluster_result,
        controlled_comparison_result=controlled_result,
        metadata=metadata,
    )

    metrics_path = model_dir / "training_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"  ✔ Metrics JSON → {metrics_path.relative_to(project_root)}")

    # Feature schema JSON
    schema_path = model_dir / "feature_schema.json"
    with schema_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata["feature_schema"], fh, indent=2)
    print(f"  ✔ Feature Schema JSON → {schema_path.relative_to(project_root)}")

    md_path = write_model_evaluation_report(
        report, productivity_table, reports_dir
    )
    print(f"  ✔ Markdown report → {md_path.relative_to(project_root)}")

    elapsed = time.perf_counter() - t_total
    print(f"\n  Training complete in {elapsed:.1f}s")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
