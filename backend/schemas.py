"""
backend/schemas.py
==================
Pydantic API models and contracts for RouteRanker v2.0.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Coverage & Provenance Schemas
# ---------------------------------------------------------------------------

class SourceRecord(BaseModel):
    dataset_id: str
    snapshot_id: str
    catalog_retrieved_date: str
    sha256: Optional[str] = None
    status: str
    byte_size: Optional[int] = 0
    source_url: Optional[str] = None


class SnapshotWindow(BaseModel):
    snapshot_id: str
    catalog_retrieved_date: str
    service_date_from: str
    service_date_to: str


class DataCoverageResponse(BaseModel):
    total_sources: int
    sources: List[SourceRecord]
    snapshot_windows: List[SnapshotWindow]
    matched_rows_count: int
    unmatched_schedule_rows_count: int
    total_service_dates: int
    total_routes: int
    zero_rides_count: int
    quality_checks: Dict[str, Any]


# ---------------------------------------------------------------------------
# Route & History Schemas
# ---------------------------------------------------------------------------

class RouteSummary(BaseModel):
    route: str
    route_short_name: str
    total_boardings: int
    avg_daily_boardings: float
    service_days: int
    served_stops_count: int = 0
    exclusive_stops_count: int = 0
    cluster_id: Optional[int] = None


class RouteListResponse(BaseModel):
    total_routes: int
    routes: List[RouteSummary]


class RouteHistoryItem(BaseModel):
    route: str
    service_date: str
    daytype: str
    rides: int
    scheduled_trips: Optional[int] = None
    scheduled_service_hours: Optional[float] = None
    source_snapshot: Optional[str] = None
    lag_7d: Optional[float] = None
    seasonal_mean: Optional[float] = None


class RouteHistoryResponse(BaseModel):
    route: str
    total_observations: int
    date_from: str
    date_to: str
    history: List[RouteHistoryItem]


# ---------------------------------------------------------------------------
# Ranking & Review Priorities Schemas
# ---------------------------------------------------------------------------

class RankingItem(BaseModel):
    route: str
    daytype: str
    observed_demand: int
    n_obs: int
    scheduled_trips: int
    scheduled_service_hours: float
    boardings_per_scheduled_trip: float
    boardings_per_scheduled_service_hour: float
    service_productivity_metric_rank: Optional[int] = None
    is_eligible: bool
    demand_trend: float = 0.0
    overlap_review_flag: bool = False
    planner_review_reasons: List[str] = Field(default_factory=list)
    cluster_id: Optional[int] = None


class RankingsResponse(BaseModel):
    daytype: str
    total_routes: int
    eligible_routes_count: int
    rankings: List[RankingItem]


# ---------------------------------------------------------------------------
# Route Comparison Schemas
# ---------------------------------------------------------------------------

class PairwiseOverlap(BaseModel):
    route_a: str
    route_b: str
    shared_stops_count: int
    jaccard_similarity: float
    smaller_route_overlap: float
    stops_count_a: int
    stops_count_b: int
    shared_stops_list: Optional[str] = None


class RouteCompareResponse(BaseModel):
    routes: List[Dict[str, Any]]
    pairwise_overlaps: List[PairwiseOverlap]
    geometry_snapshot_version: str


# ---------------------------------------------------------------------------
# Forecasts Schemas
# ---------------------------------------------------------------------------

class ForecastDataPoint(BaseModel):
    service_date: str
    daytype: str
    actual_rides: Optional[float] = None
    predicted_rides: Optional[float] = None
    lag_7d: Optional[float] = None
    seasonal_mean: Optional[float] = None


class ForecastsResponse(BaseModel):
    route: str
    model_version: str
    release_delay_assumption: str
    pooled_metrics: Dict[str, Any]
    route_metrics: Optional[Dict[str, Any]] = None
    data_points: List[ForecastDataPoint]


# ---------------------------------------------------------------------------
# Scenario Arithmetic Schemas
# ---------------------------------------------------------------------------

class ScenarioRequest(BaseModel):
    route: str
    daytype: str = "W"
    service_date: Optional[str] = None
    proposed_trips: Optional[int] = None
    frequency_change_pct: Optional[float] = None
    duration_assumption: Optional[float] = None


class ScenarioMetricBlock(BaseModel):
    scheduled_trips: int
    scheduled_service_hours: float
    boardings_per_scheduled_trip: float
    boardings_per_scheduled_service_hour: float


class ScenarioDiffBlock(BaseModel):
    trips_delta: int
    service_hours_delta: float
    frequency_change_pct: float
    boardings_per_trip_delta: float
    boardings_per_hour_delta: float


class ScenarioResponse(BaseModel):
    route: str
    daytype: str
    service_date: Optional[str] = None
    trip_duration_hours_assumed: float
    boardings_held_constant: float
    baseline: ScenarioMetricBlock
    scenario: ScenarioMetricBlock
    difference: ScenarioDiffBlock
    missing_operational_constraints: List[str]
