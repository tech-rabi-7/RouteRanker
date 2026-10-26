/**
 * API Client for RouteRanker v2.0
 */

const BASE_URL = '/api';

export interface SourceRecord {
  dataset_id: string;
  snapshot_id: string;
  catalog_retrieved_date: string;
  sha256: string;
  status: string;
  byte_size: number;
  source_url?: string;
}

export interface SnapshotWindow {
  snapshot_id: string;
  catalog_retrieved_date: string;
  service_date_from: string;
  service_date_to: string;
}

export interface DataCoverage {
  total_sources: number;
  sources: SourceRecord[];
  snapshot_windows: SnapshotWindow[];
  matched_rows_count: number;
  unmatched_schedule_rows_count: number;
  total_service_dates: number;
  total_routes: number;
  zero_rides_count: number;
  quality_checks: Record<string, string>;
}

export interface RouteSummary {
  route: string;
  route_short_name: string;
  total_boardings: number;
  avg_daily_boardings: number;
  service_days: number;
  served_stops_count: number;
  exclusive_stops_count: number;
  cluster_id?: number;
}

export interface RouteHistoryItem {
  route: string;
  service_date: string;
  daytype: string;
  rides: number;
  scheduled_trips?: number;
  scheduled_service_hours?: number;
  source_snapshot?: string;
  lag_7d?: number;
  seasonal_mean?: number;
}

export interface RouteHistoryResponse {
  route: string;
  total_observations: number;
  date_from: string;
  date_to: string;
  history: RouteHistoryItem[];
}

export interface RankingItem {
  route: string;
  daytype: string;
  observed_demand: number;
  n_obs: number;
  scheduled_trips: number;
  scheduled_service_hours: number;
  boardings_per_scheduled_trip: number;
  boardings_per_scheduled_service_hour: number;
  service_productivity_metric_rank?: number;
  is_eligible: boolean;
  demand_trend: number;
  overlap_review_flag: boolean;
  planner_review_reasons: string[];
  cluster_id?: number;
}

export interface RankingsResponse {
  daytype: string;
  total_routes: number;
  eligible_routes_count: number;
  rankings: RankingItem[];
}

export interface PairwiseOverlap {
  route_a: string;
  route_b: string;
  shared_stops_count: number;
  jaccard_similarity: number;
  smaller_route_overlap: number;
  stops_count_a: number;
  stops_count_b: number;
  shared_stops_list?: string;
}

export interface RouteCompareResponse {
  routes: Array<{
    route: string;
    total_boardings: number;
    avg_daily_boardings: number;
    total_scheduled_trips: number;
    total_scheduled_service_hours: number;
    boardings_per_scheduled_trip: number;
    boardings_per_scheduled_service_hour: number;
    served_stops_count: number;
    exclusive_stops_count: number;
  }>;
  pairwise_overlaps: PairwiseOverlap[];
  geometry_snapshot_version: string;
}

export interface ForecastDataPoint {
  service_date: string;
  daytype: string;
  actual_rides?: number;
  predicted_rides?: number;
  lag_7d?: number;
  seasonal_mean?: number;
}

export interface ForecastsResponse {
  route: string;
  model_version: string;
  release_delay_assumption: string;
  pooled_metrics: Record<string, any>;
  route_metrics?: Record<string, any>;
  data_points: ForecastDataPoint[];
}

export interface ScenarioRequest {
  route: string;
  daytype?: string;
  service_date?: string;
  proposed_trips?: number;
  frequency_change_pct?: number;
  duration_assumption?: number;
}

export interface ScenarioResponse {
  route: string;
  daytype: string;
  service_date?: string;
  trip_duration_hours_assumed: number;
  boardings_held_constant: number;
  baseline: {
    scheduled_trips: number;
    scheduled_service_hours: number;
    boardings_per_scheduled_trip: number;
    boardings_per_scheduled_service_hour: number;
  };
  scenario: {
    scheduled_trips: number;
    scheduled_service_hours: number;
    boardings_per_scheduled_trip: number;
    boardings_per_scheduled_service_hour: number;
  };
  difference: {
    trips_delta: number;
    service_hours_delta: number;
    frequency_change_pct: number;
    boardings_per_trip_delta: number;
    boardings_per_hour_delta: number;
  };
  missing_operational_constraints: string[];
}

export const api = {
  async getCoverage(): Promise<DataCoverage> {
    const res = await fetch(`${BASE_URL}/data/coverage`);
    if (!res.ok) throw new Error('Failed to fetch coverage data');
    return res.json();
  },

  async getRoutes(daytype?: string): Promise<{ total_routes: number; routes: RouteSummary[] }> {
    const url = daytype ? `${BASE_URL}/routes?daytype=${daytype}` : `${BASE_URL}/routes`;
    const res = await fetch(url);
    if (!res.ok) throw new Error('Failed to fetch routes');
    return res.json();
  },

  async getRouteHistory(route: string, daytype?: string): Promise<RouteHistoryResponse> {
    const url = daytype ? `${BASE_URL}/routes/${route}/history?daytype=${daytype}` : `${BASE_URL}/routes/${route}/history`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Failed to fetch history for route ${route}`);
    return res.json();
  },

  async getRankings(daytype: string = 'W', sortBy: string = 'service_productivity_metric_rank', order: string = 'asc', eligibleOnly: boolean = false): Promise<RankingsResponse> {
    const res = await fetch(`${BASE_URL}/rankings?daytype=${daytype}&sort_by=${sortBy}&order=${order}&eligible_only=${eligibleOnly}`);
    if (!res.ok) throw new Error('Failed to fetch rankings');
    return res.json();
  },

  async compareRoutes(routes: string[]): Promise<RouteCompareResponse> {
    const query = routes.map(r => `routes=${encodeURIComponent(r)}`).join('&');
    const res = await fetch(`${BASE_URL}/routes/compare?${query}`);
    if (!res.ok) throw new Error('Failed to compare routes');
    return res.json();
  },

  async getForecasts(route: string, experiment: string = 'matched_service'): Promise<ForecastsResponse> {
    const res = await fetch(`${BASE_URL}/forecasts?route=${encodeURIComponent(route)}&experiment=${experiment}`);
    if (!res.ok) throw new Error(`Failed to fetch forecasts for route ${route}`);
    return res.json();
  },

  async runScenario(payload: ScenarioRequest): Promise<ScenarioResponse> {
    const res = await fetch(`${BASE_URL}/scenarios`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error('Failed to execute scenario calculation');
    return res.json();
  },
};
