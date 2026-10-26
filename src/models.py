"""
src/models.py  —  RouteRanker v2.0
====================================
Daily Boarding Regression Pipeline, Productivity Ranking & Route Clustering.

Public API
----------
BoardingFeatureEngineer          — builds lag-join feature matrices (no df.shift)
BoardingRegressionPipeline       — trains baselines + RF + XGBoost regressors
                                   and evaluates with MAE / RMSE / WAPE
RouteProductivityRanker          — computes ratio-of-sums productivity metrics
                                   and eligibility-filtered ranks per daytype
RouteStructuralClusterer         — KMeans on ~125 route-level profiles
                                   (no performance-class labels)

Design rules enforced
---------------------
* Lags are computed by joining the boardings table on (route, target_date - N)
  — never via df.shift() — so look-ahead is impossible by construction.
* Same-day target-derived quantities (boardings_per_scheduled_trip, ranks) are
  NEVER included in predictor inputs.
* Imputers / encoders / scalers are fit exclusively on the training split and
  applied identically to validation and test splits.
* WAPE = sum(|y - ŷ|) / sum(y) is the primary metric; it is marked
  'unavailable' for any sub-group whose sum(y) == 0.  Plain MAPE is not used.
* KMeans cluster IDs are treated as structural archetypes; the labels
  'Overcrowded', 'Balanced', 'Underutilized' do not appear anywhere.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LAG_DAYS: list[int] = [1, 7, 14, 21, 28]
SEASONAL_LAG_DAYS: list[int] = [7, 14, 21, 28]   # used for seasonal mean
MIN_SEASONAL_LAGS: int = 2                          # need ≥2 for seasonal mean

PRODUCTIVITY_MIN_OBS: int = 20   # eligibility threshold for ranking

# KMeans profile features (no target-derived ratios, no performance labels)
CLUSTER_FEATURES: list[str] = [
    "mean_rides",
    "std_rides",
    "cv_rides",          # coefficient of variation
    "median_rides",
    "p10_rides",
    "p90_rides",
    "wd_mean_rides",     # weekday mean
    "we_mean_rides",     # weekend mean
    "wd_we_ratio",       # weekday/weekend demand ratio
    "n_obs",
]
# Extended cluster features when schedule data is available
CLUSTER_FEATURES_SERVICE: list[str] = CLUSTER_FEATURES + [
    "mean_scheduled_trips",
    "mean_scheduled_service_hours",
]


# ---------------------------------------------------------------------------
# 1. Feature Engineering — lag-join, NO df.shift()
# ---------------------------------------------------------------------------

class BoardingFeatureEngineer:
    """
    Constructs the predictor matrix for daily boarding regression.

    Parameters
    ----------
    include_service : bool
        If True, expect ``scheduled_trips`` and ``scheduled_service_hours``
        columns in the input DataFrame and add them as features.

    Feature set produced
    --------------------
    Categorical
        route, daytype
    Calendar
        day_of_week (0=Mon … 6=Sun), month (1–12), is_weekend (0/1)
    Lag features  (joined on route + exact date — never shifted)
        lag_1d, lag_7d, lag_14d, lag_21d, lag_28d
    Seasonal aggregate
        seasonal_mean_lag  — mean of [lag_7d, lag_14d, lag_21d, lag_28d]
                             when ≥ MIN_SEASONAL_LAGS are non-null
        n_seasonal_lags_available — count of non-null seasonal lags (0–4)
    Service features  (only when include_service=True)
        scheduled_trips, scheduled_service_hours
    """

    BASE_CATEGORICAL = ["route", "daytype"]
    BASE_CALENDAR = ["day_of_week", "month", "is_weekend"]
    BASE_LAG = [f"lag_{d}d" for d in LAG_DAYS]
    BASE_SEASONAL = ["seasonal_mean_lag", "n_seasonal_lags_available"]

    def __init__(
        self,
        include_service: bool = False,
        exclude_lag1: bool = False,
        release_delay_days: int = 0,
    ) -> None:
        self.include_service = include_service
        self.exclude_lag1 = exclude_lag1 or (release_delay_days >= 2)
        self.release_delay_days = release_delay_days

    @property
    def lag_columns(self) -> list[str]:
        lags = [d for d in LAG_DAYS if not (self.exclude_lag1 and d == 1)]
        return [f"lag_{d}d" for d in lags]

    @property
    def feature_columns(self) -> list[str]:
        cols = (
            self.BASE_CATEGORICAL
            + self.BASE_CALENDAR
            + self.lag_columns
            + self.BASE_SEASONAL
        )
        if self.include_service:
            cols += ["scheduled_trips", "scheduled_service_hours"]
        return cols

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build the full predictor + target matrix from a boardings DataFrame.

        Parameters
        ----------
        df : DataFrame with at minimum:
               route (str), service_date (date/datetime), daytype (str),
               rides (int)
             Optional (when include_service=True):
               scheduled_trips (int), scheduled_service_hours (float)

        Returns
        -------
        DataFrame with one row per (route, service_date), columns =
        feature_columns + ["rides"] (target).
        Rows with no lag information at all are retained (lags = NaN);
        the caller decides on imputation strategy.
        """
        df = df.copy().reset_index(drop=True)   # guarantee unique integer index
        df["service_date"] = pd.to_datetime(df["service_date"]).dt.date

        # ── Calendar features ────────────────────────────────────────────────
        df["_dt"] = pd.to_datetime(df["service_date"])
        df["day_of_week"] = df["_dt"].dt.dayofweek          # 0=Mon, 6=Sun
        df["month"] = df["_dt"].dt.month
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df = df.drop(columns=["_dt"])

        # ── Build a lookup: (route, date) → rides  for lag joins ─────────────
        #  This lookup will only contain historical observations; the join
        #  naturally returns NaN for any date before the series starts.
        lookup = (
            df[["route", "service_date", "rides"]]
            .drop_duplicates(subset=["route", "service_date"])
            .set_index(["route", "service_date"])["rides"]
        )

        # ── Lag features — joined, never shifted ─────────────────────────────
        for lag in LAG_DAYS:
            col = f"lag_{lag}d"
            lag_dates = df["service_date"].apply(
                lambda d: d - timedelta(days=lag)
            )
            df[col] = [
                lookup.get((route, ld), np.nan)
                for route, ld in zip(df["route"], lag_dates)
            ]

        # ── Seasonal mean and availability count (vectorised, no apply) ──────
        seasonal_cols = [f"lag_{d}d" for d in SEASONAL_LAG_DAYS]
        seasonal_arr = df[seasonal_cols].to_numpy(dtype=float, na_value=np.nan)
        n_avail = np.sum(~np.isnan(seasonal_arr), axis=1)
        df["n_seasonal_lags_available"] = n_avail
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            row_means = np.nanmean(seasonal_arr, axis=1)
        df["seasonal_mean_lag"] = np.where(n_avail >= MIN_SEASONAL_LAGS, row_means, np.nan)

        # ── Service features (optional) ───────────────────────────────────────
        if self.include_service:
            for col in ["scheduled_trips", "scheduled_service_hours"]:
                if col not in df.columns:
                    raise ValueError(
                        f"include_service=True but column '{col}' not found. "
                        "Pass the matched service table."
                    )

        # ── Final column selection ────────────────────────────────────────────
        raw_out_cols = self.feature_columns + ["rides", "service_date", "route"]
        out_cols = list(dict.fromkeys(raw_out_cols))
        # Keep only columns that exist (service cols may not be present)
        existing = [c for c in out_cols if c in df.columns]
        return df[existing].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Chronological splits (no shuffling)
# ---------------------------------------------------------------------------

@dataclass
class SplitSpec:
    """Defines chronological train / validation / test boundaries."""
    train_from: date
    train_to: date
    val_from: date
    val_to: date
    test_from: date
    test_to: date
    name: str = ""

    def apply(
        self, df: pd.DataFrame, date_col: str = "service_date"
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Return (train, val, test) DataFrames with reset index."""
        df = df.reset_index(drop=True)  # ensure unique integer index
        d = pd.to_datetime(df[date_col]).dt.date
        train = df[(d >= self.train_from) & (d <= self.train_to)].copy().reset_index(drop=True)
        val   = df[(d >= self.val_from)   & (d <= self.val_to)  ].copy().reset_index(drop=True)
        test  = df[(d >= self.test_from)  & (d <= self.test_to) ].copy().reset_index(drop=True)
        return train, val, test


# Per-spec splits
DEMAND_ONLY_SPLIT = SplitSpec(
    name="demand_only",
    train_from=date(2024, 8, 1),   train_to=date(2026, 1, 31),
    val_from  =date(2026, 2, 1),   val_to  =date(2026, 4, 30),
    test_from =date(2026, 5, 1),   test_to =date(2026, 7, 31),
)

MATCHED_SERVICE_SPLIT = SplitSpec(
    name="matched_service",
    train_from=date(2025, 11, 23), train_to=date(2026, 3, 31),
    val_from  =date(2026, 4, 1),   val_to  =date(2026, 5, 31),
    test_from =date(2026, 6, 1),   test_to =date(2026, 7, 31),
)


# ---------------------------------------------------------------------------
# 3. Metrics — MAE / RMSE / WAPE
# ---------------------------------------------------------------------------

def _wape(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    """
    Weighted Absolute Percentage Error = sum(|y - ŷ|) / sum(y).
    Returns None when sum(y) == 0 (zero-volume group).
    """
    denom = float(np.sum(np.abs(y_true)))
    if denom == 0.0:
        return None
    return float(np.sum(np.abs(y_true - y_pred))) / denom


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str = "",
) -> dict:
    """Compute MAE, RMSE, WAPE for a prediction array."""
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    wape = _wape(y_true, y_pred)
    return {
        "label": label,
        "n": int(len(y_true)),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "wape": round(wape, 6) if wape is not None else None,
        "wape_available": wape is not None,
    }


def compute_route_level_metrics(
    df: pd.DataFrame,
    y_pred_col: str,
    target_col: str = "rides",
    route_col: str = "route",
) -> pd.DataFrame:
    """
    Per-route MAE / RMSE / WAPE summary.
    WAPE is set to None for routes with sum(rides) == 0.
    """
    records = []
    for route, grp in df.groupby(route_col):
        y_true = grp[target_col].values.astype(float)
        y_pred = grp[y_pred_col].values.astype(float)
        m = compute_metrics(y_true, y_pred, label=str(route))
        records.append(m)
    return pd.DataFrame(records).sort_values("route_id", errors="ignore")


# ---------------------------------------------------------------------------
# 4. Seasonal baselines (no ML, pure lag lookups)
# ---------------------------------------------------------------------------

def baseline_7d_lag(df: pd.DataFrame) -> np.ndarray:
    """Predict using the 7-day same-date lag. Fills missing with global mean."""
    col = "lag_7d"
    if col not in df.columns:
        raise KeyError("lag_7d feature not found — run feature engineering first.")
    global_mean = df[col].mean(skipna=True)
    return df[col].fillna(global_mean).values.astype(float)


def baseline_seasonal_mean(df: pd.DataFrame) -> np.ndarray:
    """
    Predict using the mean of the past-four same-weekday lags (7/14/21/28 days).
    If seasonal_mean_lag is available, use it; otherwise fall back to lag_7d,
    then to global mean of seasonal_mean_lag.
    """
    if "seasonal_mean_lag" not in df.columns:
        raise KeyError("seasonal_mean_lag not found — run feature engineering first.")
    fallback = df["seasonal_mean_lag"].mean(skipna=True)
    if pd.isna(fallback):
        fallback = df["lag_7d"].mean(skipna=True) if "lag_7d" in df.columns else 0.0
    seasonal = df["seasonal_mean_lag"].values.astype(float)
    lag7 = df["lag_7d"].values.astype(float) if "lag_7d" in df.columns else np.full(len(df), fallback)
    # Fill: seasonal → lag_7d → global fallback (all via numpy, index-safe)
    result = np.where(np.isnan(seasonal), lag7, seasonal)
    result = np.where(np.isnan(result), fallback, result)
    return result


# ---------------------------------------------------------------------------
# 5. Sklearn pipeline builder (fit-on-train only)
# ---------------------------------------------------------------------------

def _build_sklearn_pipeline(
    model,
    categorical_features: list[str],
    numeric_features: list[str],
) -> Pipeline:
    """
    Construct a ColumnTransformer → model Pipeline.
    Imputer and encoder are part of the pipeline so they are fit only
    on the training split when pipeline.fit(X_train, y_train) is called.
    """
    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_transformer, categorical_features),
            ("num", num_transformer, numeric_features),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


# ---------------------------------------------------------------------------
# 6. BoardingRegressionPipeline — main training orchestrator
# ---------------------------------------------------------------------------

class BoardingRegressionPipeline:
    """
    Trains and evaluates the full daily boarding regression pipeline.

    Workflow
    --------
    1. Accept feature DataFrame (output of BoardingFeatureEngineer.build()).
    2. Apply the appropriate SplitSpec chronologically.
    3. Fit baselines (7-day lag, seasonal mean) on training split only
       (they reference only lag columns already computed).
    4. Fit pooled Ridge baseline, Random Forest, XGBoost/GradBoost on
       training split. Imputers/encoders/scalers inside sklearn Pipelines.
    5. Evaluate all models on val and test splits; return metric dicts.
    6. Persist fitted models to disk via joblib.
    """

    def __init__(
        self,
        split: SplitSpec,
        include_service: bool = False,
        exclude_lag1: bool = False,
        release_delay_days: int = 0,
        model_dir: str = "models",
        n_jobs: int = -1,
        random_state: int = 42,
    ) -> None:
        self.split = split
        self.include_service = include_service
        self.exclude_lag1 = exclude_lag1 or (release_delay_days >= 2)
        self.release_delay_days = release_delay_days
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.n_jobs = n_jobs
        self.random_state = random_state
        self._fitted_models: dict[str, Pipeline] = {}

    # ── Feature column bookkeeping ────────────────────────────────────────────

    @staticmethod
    def _split_feature_cols(
        feature_cols: list[str],
    ) -> tuple[list[str], list[str]]:
        cat = ["route", "daytype"]
        num = [c for c in feature_cols if c not in cat]
        return cat, num

    # ── Training ──────────────────────────────────────────────────────────────

    def fit_evaluate(
        self,
        feature_df: pd.DataFrame,
    ) -> dict:
        """
        Fit all models on the training split and evaluate on val + test.

        Parameters
        ----------
        feature_df : output of BoardingFeatureEngineer.build() —
                     must contain feature_columns + ["rides", "service_date"].

        Returns
        -------
        dict with keys: split_sizes, baselines, models, route_level_test
        """
        TARGET = "rides"
        DATE_COL = "service_date"

        # ── 1. Chronological split ────────────────────────────────────────────
        train, val, test = self.split.apply(feature_df, date_col=DATE_COL)

        fe = BoardingFeatureEngineer(
            include_service=self.include_service,
            exclude_lag1=self.exclude_lag1,
            release_delay_days=self.release_delay_days,
        )
        feature_cols = fe.feature_columns
        cat_cols, num_cols = self._split_feature_cols(feature_cols)

        # Restrict to columns that actually exist (service cols may be absent)
        feature_cols_present = [c for c in feature_cols if c in feature_df.columns]
        cat_cols = [c for c in cat_cols if c in feature_cols_present]
        num_cols = [c for c in num_cols if c in feature_cols_present]

        X_train, y_train = train[feature_cols_present], train[TARGET].values.astype(float)
        X_val,   y_val   = val[feature_cols_present],   val[TARGET].values.astype(float)
        X_test,  y_test  = test[feature_cols_present],  test[TARGET].values.astype(float)

        print(f"\n  Split '{self.split.name}':")
        print(f"    Train : {len(X_train):,} rows")
        print(f"    Val   : {len(X_val):,} rows")
        print(f"    Test  : {len(X_test):,} rows")

        result: dict = {
            "split": self.split.name,
            "split_sizes": {
                "train": len(X_train), "val": len(X_val), "test": len(X_test)
            },
        }

        # ── 2. Seasonal baselines (no ML) ────────────────────────────────────
        print("  Evaluating seasonal baselines ...")
        b7_val_pred   = baseline_7d_lag(X_val)
        bsm_val_pred  = baseline_seasonal_mean(X_val)
        b7_test_pred  = baseline_7d_lag(X_test)
        bsm_test_pred = baseline_seasonal_mean(X_test)

        result["baselines"] = {
            "lag_7d": {
                "val":  compute_metrics(y_val,  b7_val_pred,  "lag_7d_val"),
                "test": compute_metrics(y_test, b7_test_pred, "lag_7d_test"),
            },
            "seasonal_mean": {
                "val":  compute_metrics(y_val,  bsm_val_pred,  "seasonal_mean_val"),
                "test": compute_metrics(y_test, bsm_test_pred, "seasonal_mean_test"),
            },
        }

        # ── 3. ML models ─────────────────────────────────────────────────────
        models_to_train = {
            "ridge_pooled": Ridge(alpha=10.0),
            "random_forest": RandomForestRegressor(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=5,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            ),
        }

        if _HAS_XGB:
            models_to_train["xgboost"] = XGBRegressor(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                verbosity=0,
            )
        else:
            models_to_train["gradient_boosting"] = GradientBoostingRegressor(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                random_state=self.random_state,
            )

        result["models"] = {}
        for name, estimator in models_to_train.items():
            print(f"  Training {name} ...")
            pipe = _build_sklearn_pipeline(estimator, cat_cols, num_cols)
            pipe.fit(X_train, y_train)
            self._fitted_models[name] = pipe

            val_pred  = np.clip(pipe.predict(X_val),  0, None)
            test_pred = np.clip(pipe.predict(X_test), 0, None)

            result["models"][name] = {
                "val":  compute_metrics(y_val,  val_pred,  f"{name}_val"),
                "test": compute_metrics(y_test, test_pred, f"{name}_test"),
            }
            print(
                f"    val  RMSE={result['models'][name]['val']['rmse']:.1f} "
                f"MAE={result['models'][name]['val']['mae']:.1f} "
                f"WAPE={result['models'][name]['val']['wape']}"
            )
            print(
                f"    test RMSE={result['models'][name]['test']['rmse']:.1f} "
                f"MAE={result['models'][name]['test']['mae']:.1f} "
                f"WAPE={result['models'][name]['test']['wape']}"
            )

        # ── 4. Route-level test metrics (best model) ──────────────────────────
        best_name = self._best_model_name(result["models"])
        best_pipe = self._fitted_models[best_name]
        test_pred_best = np.clip(best_pipe.predict(X_test), 0, None)

        test_aug = test.copy()
        test_aug["_pred"] = test_pred_best
        route_metrics = []
        for route, grp in test_aug.groupby("route"):
            yt = grp[TARGET].values.astype(float)
            yp = grp["_pred"].values.astype(float)
            m = compute_metrics(yt, yp, label=str(route))
            route_metrics.append(m)

        result["route_level_test"] = {
            "best_model": best_name,
            "route_count": len(route_metrics),
            "route_metrics": route_metrics,
        }

        return result

    @staticmethod
    def _best_model_name(models_dict: dict) -> str:
        """Pick the model with the lowest test RMSE."""
        return min(
            models_dict,
            key=lambda k: models_dict[k]["test"]["rmse"],
        )

    def save_models(self, prefix: str = "") -> dict[str, Path]:
        """Serialize all fitted sklearn Pipelines with joblib."""
        saved = {}
        for name, pipe in self._fitted_models.items():
            fname = f"{prefix}{name}.joblib" if prefix else f"{name}.joblib"
            path = self.model_dir / fname
            joblib.dump(pipe, path, compress=3)
            saved[name] = path
            print(f"  ✔ Saved {name} → {path}")
        return saved

    def load_model(self, name: str, prefix: str = "") -> Pipeline:
        """Load a previously saved sklearn Pipeline."""
        fname = f"{prefix}{name}.joblib" if prefix else f"{name}.joblib"
        path = self.model_dir / fname
        pipe = joblib.load(path)
        self._fitted_models[name] = pipe
        return pipe

    def predict(self, name: str, X: pd.DataFrame) -> np.ndarray:
        """Run inference using a named fitted pipeline."""
        if name not in self._fitted_models:
            raise KeyError(f"Model '{name}' not loaded. Call load_model() first.")
        return np.clip(self._fitted_models[name].predict(X), 0, None)


# ---------------------------------------------------------------------------
# 7. RouteProductivityRanker — ratio-of-sums, per daytype, ≥20 obs threshold
# ---------------------------------------------------------------------------

class RouteProductivityRanker:
    """
    Computes explainable productivity metrics from the matched joined table
    (route_profiles.parquet output of Step 4).

    Metrics are computed as ratios of SUMS — not averages of daily ratios —
    to avoid Jensen-inequality distortion from small daily denominators.

    Columns produced
    ----------------
    route_short_name, daytype,
    n_obs,
    total_rides,
    total_scheduled_trips,
    total_scheduled_service_hours,
    boardings_per_scheduled_trip,
    boardings_per_scheduled_service_hour,
    service_productivity_metric_rank     (within daytype, desc; NaN if not eligible)
    """

    def __init__(self, min_obs: int = PRODUCTIVITY_MIN_OBS) -> None:
        self.min_obs = min_obs

    def build_profile_table(
        self,
        matched_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build the route × daytype productivity profile table.

        Parameters
        ----------
        matched_df : output of analytics step 4 route_profiles.parquet.
                     Required columns: route_short_name, service_date,
                     daytype, rides, scheduled_trips, scheduled_service_hours.
        """
        required = {
            "route_short_name", "service_date", "daytype",
            "rides", "scheduled_trips", "scheduled_service_hours",
        }
        missing = required - set(matched_df.columns)
        if missing:
            raise ValueError(f"Matched DataFrame missing columns: {missing}")

        df = matched_df.copy()
        df["rides"] = pd.to_numeric(df["rides"], errors="coerce").fillna(0)
        df["scheduled_trips"] = pd.to_numeric(
            df["scheduled_trips"], errors="coerce"
        ).fillna(0)
        df["scheduled_service_hours"] = pd.to_numeric(
            df["scheduled_service_hours"], errors="coerce"
        ).fillna(0)

        # ── Aggregate: sum per (route, daytype) ──────────────────────────────
        agg = (
            df.groupby(["route_short_name", "daytype"], as_index=False)
            .agg(
                n_obs=("rides", "count"),
                total_rides=("rides", "sum"),
                total_scheduled_trips=("scheduled_trips", "sum"),
                total_scheduled_service_hours=("scheduled_service_hours", "sum"),
            )
        )

        # ── Ratio of sums (not average of ratios) ────────────────────────────
        agg["boardings_per_scheduled_trip"] = np.where(
            agg["total_scheduled_trips"] > 0,
            agg["total_rides"] / agg["total_scheduled_trips"],
            np.nan,
        )
        agg["boardings_per_scheduled_service_hour"] = np.where(
            agg["total_scheduled_service_hours"] > 0,
            agg["total_rides"] / agg["total_scheduled_service_hours"],
            np.nan,
        )

        # ── Eligibility filter and within-daytype rank ────────────────────────
        eligible_mask = agg["n_obs"] >= self.min_obs

        def _rank_within_daytype(grp: pd.DataFrame) -> pd.Series:
            eligible = grp["n_obs"] >= self.min_obs
            ranks = pd.Series(np.nan, index=grp.index)
            eligible_sub = grp[eligible].copy()
            if not eligible_sub.empty:
                # Descending rank of boardings_per_scheduled_service_hour
                ranks[eligible] = (
                    eligible_sub["boardings_per_scheduled_service_hour"]
                    .rank(ascending=False, method="min", na_option="bottom")
                    .values
                )
            return ranks

        agg["service_productivity_metric_rank"] = (
            agg.groupby("daytype", group_keys=False)
            .apply(_rank_within_daytype)
            .values
        )

        agg = agg.sort_values(
            ["daytype", "service_productivity_metric_rank", "route_short_name"]
        ).reset_index(drop=True)

        return agg

    def top_n(
        self,
        profile_table: pd.DataFrame,
        daytype: str,
        n: int = 20,
    ) -> pd.DataFrame:
        """Return the top-N most productive routes for a given daytype."""
        sub = profile_table[
            (profile_table["daytype"] == daytype)
            & profile_table["service_productivity_metric_rank"].notna()
        ]
        return (
            sub.sort_values("service_productivity_metric_rank")
            .head(n)
            .reset_index(drop=True)
        )


# ---------------------------------------------------------------------------
# 8. RouteStructuralClusterer — KMeans on route profiles, no perf labels
# ---------------------------------------------------------------------------

class RouteStructuralClusterer:
    """
    Clusters the ~125 matched CTA bus routes by structural demand and
    service characteristics using KMeans.

    Rules
    -----
    * Input features are demand statistics and day-type variation — NOT
      productivity ratios or target-derived labels.
    * Cluster IDs are integers (0, 1, 2, …); they are NEVER relabelled as
      'Overcrowded', 'Balanced', or 'Underutilized'.
    * Scaler and KMeans are fit on the provided DataFrame; the caller must
      ensure it represents the full stable route set (e.g. training period).
    """

    def __init__(
        self,
        n_clusters: int = 5,
        random_state: int = 42,
        model_dir: str = "models",
    ) -> None:
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._scaler: Optional[StandardScaler] = None
        self._kmeans: Optional[KMeans] = None

    def build_route_feature_matrix(
        self,
        boardings_df: pd.DataFrame,
        matched_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Aggregate the per-(route, date) boardings table into one row per route
        with structural demand statistics.

        Parameters
        ----------
        boardings_df : daily boardings with (route, service_date, daytype, rides)
        matched_df   : optional matched table with scheduled_trips and
                       scheduled_service_hours for the service-aware features

        Returns
        -------
        DataFrame indexed by route with CLUSTER_FEATURES columns.
        """
        df = boardings_df.copy()
        df["rides"] = pd.to_numeric(df["rides"], errors="coerce").fillna(0)

        agg = df.groupby("route")["rides"].agg(
            mean_rides="mean",
            std_rides="std",
            median_rides="median",
            p10_rides=lambda x: float(np.percentile(x, 10)),
            p90_rides=lambda x: float(np.percentile(x, 90)),
            n_obs="count",
        ).reset_index()

        # Fill std NaN (single-obs routes)
        agg["std_rides"] = agg["std_rides"].fillna(0.0)

        # CV = std / mean (0 if mean == 0)
        agg["cv_rides"] = np.where(
            agg["mean_rides"] > 0,
            agg["std_rides"] / agg["mean_rides"],
            0.0,
        )

        # Weekday vs weekend mean
        wd = (
            df[df["daytype"] == "W"]
            .groupby("route")["rides"]
            .mean()
            .rename("wd_mean_rides")
        )
        we = (
            df[df["daytype"].isin(["A", "U"])]
            .groupby("route")["rides"]
            .mean()
            .rename("we_mean_rides")
        )
        agg = agg.merge(wd, on="route", how="left")
        agg = agg.merge(we, on="route", how="left")
        agg["wd_mean_rides"] = agg["wd_mean_rides"].fillna(agg["mean_rides"])
        agg["we_mean_rides"] = agg["we_mean_rides"].fillna(agg["mean_rides"])
        agg["wd_we_ratio"] = np.where(
            agg["we_mean_rides"] > 0,
            agg["wd_mean_rides"] / agg["we_mean_rides"],
            1.0,
        )

        # Optional service features
        if matched_df is not None:
            svc = (
                matched_df.groupby("route_short_name")
                .agg(
                    mean_scheduled_trips=("scheduled_trips", "mean"),
                    mean_scheduled_service_hours=("scheduled_service_hours", "mean"),
                )
                .reset_index()
                .rename(columns={"route_short_name": "route"})
            )
            agg = agg.merge(svc, on="route", how="left")
            agg["mean_scheduled_trips"] = agg["mean_scheduled_trips"].fillna(0)
            agg["mean_scheduled_service_hours"] = agg[
                "mean_scheduled_service_hours"
            ].fillna(0)

        return agg.set_index("route")

    def fit(self, route_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the StandardScaler and KMeans on the route feature matrix.

        Returns
        -------
        route_matrix with a 'cluster_id' (int) column added.
        No cluster labels or performance categories are assigned.
        """
        feature_cols = [
            c for c in CLUSTER_FEATURES if c in route_matrix.columns
        ]
        if "mean_scheduled_trips" in route_matrix.columns:
            feature_cols += [
                c for c in CLUSTER_FEATURES_SERVICE
                if c in route_matrix.columns and c not in feature_cols
            ]

        X = route_matrix[feature_cols].fillna(0).values

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        n_clusters = min(self.n_clusters, len(route_matrix))
        self._kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=20,
        )
        labels = self._kmeans.fit_predict(X_scaled)

        result = route_matrix.copy()
        result["cluster_id"] = labels
        return result

    def predict(self, route_matrix: pd.DataFrame) -> np.ndarray:
        """Assign cluster IDs to new route profiles."""
        if self._kmeans is None or self._scaler is None:
            raise RuntimeError("Call fit() before predict().")
        feature_cols = [
            c for c in CLUSTER_FEATURES_SERVICE
            if c in route_matrix.columns
        ]
        X = route_matrix[feature_cols].fillna(0).values
        X_scaled = self._scaler.transform(X)
        return self._kmeans.predict(X_scaled)

    def save(self, prefix: str = "") -> dict[str, Path]:
        """Persist scaler and KMeans model to disk."""
        saved = {}
        for name, obj in [("kmeans_scaler", self._scaler), ("kmeans_model", self._kmeans)]:
            if obj is None:
                continue
            fname = f"{prefix}{name}.joblib" if prefix else f"{name}.joblib"
            path = self.model_dir / fname
            joblib.dump(obj, path)
            saved[name] = path
        return saved

    def load(self, prefix: str = "") -> None:
        """Load persisted scaler and KMeans model from disk."""
        for name, attr in [("kmeans_scaler", "_scaler"), ("kmeans_model", "_kmeans")]:
            fname = f"{prefix}{name}.joblib" if prefix else f"{name}.joblib"
            path = self.model_dir / fname
            if path.exists():
                setattr(self, attr, joblib.load(path))


# ---------------------------------------------------------------------------
# 9. Consolidated model evaluation report builder
# ---------------------------------------------------------------------------

def build_evaluation_report(
    demand_only_result: Optional[dict],
    matched_service_result: Optional[dict],
    productivity_table: Optional[pd.DataFrame],
    cluster_result: Optional[pd.DataFrame],
    controlled_comparison_result: Optional[dict] = None,
    metadata: Optional[dict] = None,
) -> dict:
    """
    Assemble a structured evaluation report dict suitable for JSON serialisation
    and Markdown rendering.
    """
    report: dict = {
        "artifact_version": "2.0.0",
        "generated_at": str(pd.Timestamp.utcnow().isoformat()),
        "metadata": metadata or {},
        "demand_only": demand_only_result,
        "matched_service": matched_service_result,
        "controlled_comparison": controlled_comparison_result,
        "productivity_summary": None,
        "clustering_summary": None,
    }

    if productivity_table is not None and not productivity_table.empty:
        report["productivity_summary"] = {
            "n_routes_total": int(productivity_table["route_short_name"].nunique()),
            "n_eligible_wd": int(
                productivity_table[
                    (productivity_table["daytype"] == "W")
                    & productivity_table["service_productivity_metric_rank"].notna()
                ].shape[0]
            ),
            "top5_weekday": (
                productivity_table[productivity_table["daytype"] == "W"]
                .sort_values("service_productivity_metric_rank")
                .head(5)[
                    [
                        "route_short_name",
                        "daytype",
                        "boardings_per_scheduled_service_hour",
                        "service_productivity_metric_rank",
                    ]
                ]
                .to_dict(orient="records")
            ),
        }

    if cluster_result is not None and not cluster_result.empty:
        report["clustering_summary"] = {
            "n_clusters": int(cluster_result["cluster_id"].nunique()),
            "cluster_sizes": cluster_result["cluster_id"]
            .value_counts()
            .sort_index()
            .to_dict(),
        }

    return report
