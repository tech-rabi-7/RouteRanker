"""
src/scenarios.py  —  RouteRanker v2.0
====================================
Explicit Scenario Arithmetic & Service Sensitivity Calculator.

Implements the deterministic operational scenario model:
  - Takes route, daytype/date, and proposed trips (or frequency change %).
  - Evaluates baseline scheduled service and ridership.
  - Keeps forecast/observed boardings CONSTANT to isolate service sensitivity.
  - Computes new scheduled trips, scheduled service hours,
    conditional boardings per trip, and conditional boardings per service hour.
  - Transparently returns unmodeled operational constraints (fleet, crew, cost, latent demand).
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


# Explicit declaration of constraints omitted from the simplified arithmetic
OPERATIONAL_CONSTRAINTS_OMITTED = [
    "Peak fleet / vehicle availability constraint (depot allocation not modeled)",
    "Operator / crew duty scheduling & shift length limits not modeled",
    "Marginal operating cost per vehicle-revenue-hour not accounted for",
    "Actual on-board passenger load profile along route segments not modeled",
    "Latent commuter demand induced by higher frequencies not modeled",
]


def calculate_scenario(
    baseline_trips: int,
    baseline_service_hours: float,
    baseline_boardings: float,
    proposed_trips: Optional[int] = None,
    frequency_change_pct: Optional[float] = None,
    duration_assumption: Optional[float] = None,
    route: str = "",
    daytype: str = "W",
    service_date: Optional[str] = None,
) -> Dict[str, any]:
    """
    Computes deterministic scenario metrics while keeping boardings constant.

    Parameters
    ----------
    baseline_trips : int
        Current scheduled trips for this route and service day.
    baseline_service_hours : float
        Current scheduled vehicle service hours.
    baseline_boardings : float
        Current or forecasted passenger boardings.
    proposed_trips : Optional[int]
        Target number of scheduled trips.
    frequency_change_pct : Optional[float]
        Percentage change in trips (e.g. +20.0 for 20% increase, -15.0 for decrease).
    duration_assumption : Optional[float]
        Trip duration in hours (defaults to baseline_service_hours / baseline_trips).
    route : str
        Route identifier.
    daytype : str
        Daytype code ('W'=weekday, 'A'=Saturday, 'U'=Sunday/Holiday).
    service_date : Optional[str]
        Reference date (YYYY-MM-DD).

    Returns
    -------
    Dict with baseline vs scenario comparison and operational caveats.
    """
    b_trips = max(1, int(baseline_trips))
    b_hours = max(0.1, float(baseline_service_hours))
    b_rides = max(0.0, float(baseline_boardings))

    avg_trip_duration = b_hours / b_trips
    d_trip = float(duration_assumption) if duration_assumption and duration_assumption > 0 else avg_trip_duration

    if proposed_trips is not None and proposed_trips > 0:
        new_trips = int(proposed_trips)
    elif frequency_change_pct is not None:
        delta = 1.0 + (float(frequency_change_pct) / 100.0)
        new_trips = max(1, int(round(b_trips * delta)))
    else:
        new_trips = b_trips

    new_hours = round(float(new_trips * d_trip), 2)

    # Ratios
    b_rides_per_trip = round(b_rides / b_trips, 2)
    b_rides_per_hour = round(b_rides / b_hours, 2)

    scenario_rides_per_trip = round(b_rides / new_trips, 2)
    scenario_rides_per_hour = round(b_rides / new_hours, 2)

    delta_trips = new_trips - b_trips
    delta_hours = round(new_hours - b_hours, 2)
    pct_trip_change = round(((new_trips - b_trips) / b_trips) * 100.0, 1)

    return {
        "route": str(route),
        "daytype": str(daytype),
        "service_date": service_date,
        "trip_duration_hours_assumed": round(d_trip, 2),
        "boardings_held_constant": round(b_rides, 1),
        "baseline": {
            "scheduled_trips": b_trips,
            "scheduled_service_hours": round(b_hours, 2),
            "boardings_per_scheduled_trip": b_rides_per_trip,
            "boardings_per_scheduled_service_hour": b_rides_per_hour,
        },
        "scenario": {
            "scheduled_trips": new_trips,
            "scheduled_service_hours": new_hours,
            "boardings_per_scheduled_trip": scenario_rides_per_trip,
            "boardings_per_scheduled_service_hour": scenario_rides_per_hour,
        },
        "difference": {
            "trips_delta": delta_trips,
            "service_hours_delta": delta_hours,
            "frequency_change_pct": pct_trip_change,
            "boardings_per_trip_delta": round(scenario_rides_per_trip - b_rides_per_trip, 2),
            "boardings_per_hour_delta": round(scenario_rides_per_hour - b_rides_per_hour, 2),
        },
        "missing_operational_constraints": OPERATIONAL_CONSTRAINTS_OMITTED,
    }
