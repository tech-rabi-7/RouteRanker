import React, { useEffect, useState } from 'react';
import { api, RouteSummary, RouteHistoryItem, ForecastsResponse } from '../api/client';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, Legend } from 'recharts';
import { Bus, Clock, Calendar, TrendingUp, CheckCircle, Award } from 'lucide-react';

export const RouteDetailPage: React.FC = () => {
  const [routes, setRoutes] = useState<RouteSummary[]>([]);
  const [selectedRoute, setSelectedRoute] = useState<string>('20');
  const [history, setHistory] = useState<RouteHistoryItem[]>([]);
  const [forecasts, setForecasts] = useState<ForecastsResponse | null>(null);
  const [daytype, setDaytype] = useState<string>('All');
  const [loading, setLoading] = useState(false);

  // Load route list
  useEffect(() => {
    api.getRoutes().then(res => {
      setRoutes(res.routes);
      if (res.routes.length > 0 && !res.routes.some(r => r.route === selectedRoute)) {
        setSelectedRoute(res.routes[0].route);
      }
    });
  }, []);

  // Load history and forecasts for selected route
  useEffect(() => {
    if (!selectedRoute) return;
    setLoading(true);

    Promise.all([
      api.getRouteHistory(selectedRoute, daytype === 'All' ? undefined : daytype),
      api.getForecasts(selectedRoute).catch(() => null),
    ])
      .then(([histRes, foreRes]) => {
        setHistory(histRes.history);
        setForecasts(foreRes);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [selectedRoute, daytype]);

  const currRouteInfo = routes.find(r => r.route === selectedRoute);

  // Compute summary stats from history
  const totalRides = history.reduce((sum, h) => sum + h.rides, 0);
  const totalTrips = history.reduce((sum, h) => sum + (h.scheduled_trips || 0), 0);
  const totalHours = history.reduce((sum, h) => sum + (h.scheduled_service_hours || 0), 0);
  const bPerTrip = totalTrips > 0 ? (totalRides / totalTrips).toFixed(2) : '—';
  const bPerHour = totalHours > 0 ? (totalRides / totalHours).toFixed(2) : '—';

  // Prepare chart series: combine history and forecasts
  const chartData = history.slice(-60).map(h => {
    const fPoint = forecasts?.data_points.find(p => p.service_date === h.service_date);
    return {
      date: h.service_date,
      rides: h.rides,
      trips: (h.scheduled_trips || 0) * 35, // scaled for visualization
      lag_7: h.lag_7d,
      forecast: fPoint?.predicted_rides,
    };
  });

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header & Route Selection */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-3">
            <span className="px-3 py-1 rounded-lg font-mono text-xl font-bold bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">
              Route {selectedRoute}
            </span>
            <h1 className="text-3xl font-extrabold text-white tracking-tight">Ridership & Service Profile</h1>
          </div>
          <p className="text-slate-400 mt-1">Detailed history, schedule matching, and machine learning forecast trajectory.</p>
        </div>

        {/* Controls */}
        <div className="flex flex-wrap items-center gap-3">
          <select
            value={selectedRoute}
            onChange={(e) => setSelectedRoute(e.target.value)}
            className="bg-slate-900 border border-slate-700 text-white rounded-lg px-4 py-2 text-sm font-semibold focus:outline-none focus:border-cyan-400"
          >
            {routes.map(r => (
              <option key={r.route} value={r.route}>
                Route {r.route} ({r.total_boardings.toLocaleString()} rides)
              </option>
            ))}
          </select>

          <div className="flex bg-slate-900 border border-slate-800 p-1 rounded-lg">
            {['All', 'W', 'A', 'U'].map(dt => (
              <button
                key={dt}
                onClick={() => setDaytype(dt)}
                className={`px-2.5 py-1 text-xs font-semibold rounded ${
                  daytype === dt ? 'bg-cyan-500 text-white' : 'text-slate-400 hover:text-white'
                }`}
              >
                {dt}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
        <div className="bg-slate-900/80 border border-slate-800 p-5 rounded-xl shadow-lg">
          <span className="text-xs uppercase tracking-wider text-slate-400 font-semibold">Total Rides</span>
          <p className="text-2xl font-bold text-white mt-1">{totalRides.toLocaleString()}</p>
          <span className="text-xs text-cyan-400">Recorded Observations</span>
        </div>

        <div className="bg-slate-900/80 border border-slate-800 p-5 rounded-xl shadow-lg">
          <span className="text-xs uppercase tracking-wider text-slate-400 font-semibold">Scheduled Service</span>
          <p className="text-2xl font-bold text-white mt-1">{totalTrips.toLocaleString()}</p>
          <span className="text-xs text-slate-400">{totalHours.toLocaleString()} Service Hours</span>
        </div>

        <div className="bg-slate-900/80 border border-slate-800 p-5 rounded-xl shadow-lg">
          <span className="text-xs uppercase tracking-wider text-slate-400 font-semibold">Boardings / Trip</span>
          <p className="text-2xl font-bold text-emerald-400 mt-1">{bPerTrip}</p>
          <span className="text-xs text-slate-400">Ratio-of-sums metric</span>
        </div>

        <div className="bg-slate-900/80 border border-slate-800 p-5 rounded-xl shadow-lg">
          <span className="text-xs uppercase tracking-wider text-slate-400 font-semibold">Boardings / Hour</span>
          <p className="text-2xl font-bold text-indigo-400 mt-1">{bPerHour}</p>
          <span className="text-xs text-slate-400">Vehicle-hour productivity</span>
        </div>

        <div className="bg-slate-900/80 border border-slate-800 p-5 rounded-xl shadow-lg">
          <span className="text-xs uppercase tracking-wider text-slate-400 font-semibold">Sample Coverage</span>
          <p className="text-2xl font-bold text-white mt-1">{history.length}</p>
          <span className="text-xs text-emerald-400 font-medium">Eligible (≥20 obs)</span>
        </div>
      </div>

      {/* Main Chart */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-6 shadow-lg">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-6">
          <div>
            <h2 className="text-lg font-bold text-white">Daily Boardings vs. ML Regressor Forecasts</h2>
            <p className="text-xs text-slate-400 mt-0.5">Showing recent 60 observations with historical actuals, 7-day lag baseline, and model prediction.</p>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-xs px-2.5 py-1 rounded bg-slate-800 text-slate-300 font-mono">
              Model: {forecasts?.model_version || 'v2.0'}
            </span>
          </div>
        </div>

        <div className="w-full h-[400px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 10, right: 30, left: 10, bottom: 25 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="date" stroke="#64748b" tick={{ fill: '#94a3b8', fontSize: 11 }} />
              <YAxis stroke="#64748b" tick={{ fill: '#94a3b8', fontSize: 11 }} />
              <Tooltip
                contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '0.5rem', color: '#fff' }}
              />
              <Legend verticalAlign="top" height={36} />
              <Line type="monotone" dataKey="rides" stroke="#38bdf8" strokeWidth={2.5} name="Actual Rides" dot={false} />
              <Line type="monotone" dataKey="forecast" stroke="#f43f5e" strokeWidth={2} strokeDasharray="4 4" name="ML Predicted Rides" dot={false} />
              <Line type="monotone" dataKey="lag_7" stroke="#94a3b8" strokeWidth={1.5} strokeDasharray="2 2" name="7-Day Lag Baseline" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};
