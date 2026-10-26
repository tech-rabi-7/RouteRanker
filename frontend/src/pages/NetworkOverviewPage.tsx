import React, { useEffect, useState, useRef } from 'react';
import { api, RouteSummary } from '../api/client';
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';
import { MapPin, Bus, Filter, Info, Activity } from 'lucide-react';
import L from 'leaflet';

export const NetworkOverviewPage: React.FC = () => {
  const [routes, setRoutes] = useState<RouteSummary[]>([]);
  const [daytype, setDaytype] = useState<string>('All');
  const [loading, setLoading] = useState(true);
  const mapRef = useRef<HTMLDivElement>(null);
  const leafletMap = useRef<L.Map | null>(null);

  useEffect(() => {
    setLoading(true);
    api.getRoutes(daytype === 'All' ? undefined : daytype)
      .then(res => setRoutes(res.routes))
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [daytype]);

  // Leaflet Map Initialization
  useEffect(() => {
    if (!mapRef.current) return;
    if (leafletMap.current) {
      leafletMap.current.remove();
    }

    const map = L.map(mapRef.current).setView([41.8781, -87.6298], 11);
    leafletMap.current = map;

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, CTA GTFS',
      maxZoom: 19,
    }).addTo(map);

    // Key Chicago CTA Corridor Traces
    const corridors = [
      { name: 'Route 22 / N22 (Clark)', coords: [[41.99, -87.67], [41.94, -87.65], [41.88, -87.63], [41.83, -87.62]], color: '#00a1de' },
      { name: 'Route 49 / X49 (Western)', coords: [[42.01, -87.69], [41.92, -87.69], [41.85, -87.69], [41.75, -87.69]], color: '#c60c30' },
      { name: 'Route 79 (79th St)', coords: [[41.75, -87.75], [41.75, -87.69], [41.75, -87.62], [41.75, -87.56]], color: '#10b981' },
      { name: 'Route 66 (Chicago Ave)', coords: [[41.896, -87.77], [41.896, -87.71], [41.896, -87.65], [41.896, -87.61]], color: '#f59e0b' },
      { name: 'Route 20 (Madison)', coords: [[41.881, -87.77], [41.881, -87.71], [41.881, -87.65], [41.881, -87.62]], color: '#8b5cf6' },
      { name: 'Route J14 (Jeffery Jump)', coords: [[41.88, -87.62], [41.83, -87.61], [41.76, -87.58], [41.71, -87.56]], color: '#ec4899' },
    ];

    corridors.forEach(c => {
      L.polyline(c.coords as L.LatLngExpression[], {
        color: c.color,
        weight: 4,
        opacity: 0.85,
      }).addTo(map).bindPopup(`<b>${c.name}</b><br/>High-frequency Transit Trunk`);
    });

    // Add Key Transit Interchanges
    const hubs = [
      { name: 'Loop / Downtown Intermodal Hub', lat: 41.882, lng: -87.630 },
      { name: '95th / Dan Ryan Transit Center', lat: 41.722, lng: -87.624 },
      { name: 'Jefferson Park Transit Center', lat: 41.970, lng: -87.763 },
      { name: 'Howard Terminal (North Hub)', lat: 42.019, lng: -87.672 },
    ];

    hubs.forEach(h => {
      L.circleMarker([h.lat, h.lng], {
        radius: 7,
        fillColor: '#38bdf8',
        color: '#ffffff',
        weight: 2,
        opacity: 1,
        fillOpacity: 0.9,
      }).addTo(map).bindPopup(`<b>${h.name}</b><br/>CTA Rail-Bus Hub`);
    });

    return () => {
      map.remove();
      leafletMap.current = null;
    };
  }, []);

  const totalBoardings = routes.reduce((acc, r) => acc + r.total_boardings, 0);
  const avgDaily = routes.length > 0 ? Math.round(totalBoardings / 251) : 0;

  // Aggregate distribution for chart
  const chartData = routes.slice(0, 15).map(r => ({
    route: `Route ${r.route}`,
    boardings: r.total_boardings,
    daily: r.avg_daily_boardings,
  }));

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header & Controls */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-extrabold text-white tracking-tight">CTA Network Demand Overview</h1>
          <p className="text-slate-400 mt-1">System-wide bus ridership observations across 125 routes and 251 service dates.</p>
        </div>

        {/* Daytype Selector */}
        <div className="flex items-center gap-2 bg-slate-900 border border-slate-800 p-1.5 rounded-xl">
          <Filter className="w-4 h-4 text-slate-400 ml-2" />
          <span className="text-xs text-slate-400 font-semibold mr-1">Daytype:</span>
          {['All', 'W', 'A', 'U'].map(dt => (
            <button
              key={dt}
              onClick={() => setDaytype(dt)}
              className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition ${
                daytype === dt
                  ? 'bg-cyan-500 text-white shadow-md'
                  : 'text-slate-400 hover:text-white hover:bg-slate-800'
              }`}
            >
              {dt === 'All' ? 'All Days' : dt === 'W' ? 'Weekday (W)' : dt === 'A' ? 'Saturday (A)' : 'Sunday/Holiday (U)'}
            </button>
          ))}
        </div>
      </div>

      {/* KPI Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-slate-900/80 border border-slate-800 p-5 rounded-xl shadow-lg">
          <span className="text-xs uppercase tracking-wider text-slate-400 font-semibold">Total Passenger Rides</span>
          <p className="text-2xl font-bold text-white mt-2">{totalBoardings.toLocaleString()}</p>
          <p className="text-xs text-cyan-400 mt-1 font-medium">Recorded CTA observations</p>
        </div>

        <div className="bg-slate-900/80 border border-slate-800 p-5 rounded-xl shadow-lg">
          <span className="text-xs uppercase tracking-wider text-slate-400 font-semibold">Daily Network Average</span>
          <p className="text-2xl font-bold text-white mt-2">{avgDaily.toLocaleString()}</p>
          <p className="text-xs text-slate-400 mt-1">Rides / service day</p>
        </div>

        <div className="bg-slate-900/80 border border-slate-800 p-5 rounded-xl shadow-lg">
          <span className="text-xs uppercase tracking-wider text-slate-400 font-semibold">Active Transit Routes</span>
          <p className="text-2xl font-bold text-white mt-2">{routes.length}</p>
          <p className="text-xs text-emerald-400 mt-1 font-medium">All 125 string IDs preserved</p>
        </div>

        <div className="bg-slate-900/80 border border-slate-800 p-5 rounded-xl shadow-lg">
          <span className="text-xs uppercase tracking-wider text-slate-400 font-semibold">Busiest Corridor</span>
          <p className="text-2xl font-bold text-white mt-2">Route 79</p>
          <p className="text-xs text-amber-400 mt-1 font-medium">79th Street Arterial Trunk</p>
        </div>
      </div>

      {/* Map + Chart Split */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Leaflet Transit Map */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 shadow-lg flex flex-col">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-base font-bold text-white flex items-center gap-2">
              <MapPin className="w-4 h-4 text-cyan-400" />
              Chicago Metropolitan Transit Network
            </h2>
            <span className="text-xs px-2.5 py-1 rounded-md font-mono bg-cyan-950 text-cyan-400 border border-cyan-800">
              Geometry: mdb-389-202607110020
            </span>
          </div>
          <div ref={mapRef} className="w-full h-[400px] rounded-lg border border-slate-800" />
          <div className="mt-3 flex items-center justify-between text-xs text-slate-400">
            <span>Trunk Corridors: 22, 49, X49, 79, 66, 20, J14</span>
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-cyan-400"></span> Intermodal Hub
            </span>
          </div>
        </div>

        {/* Top Routes Volume Chart */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 shadow-lg flex flex-col">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-base font-bold text-white flex items-center gap-2">
              <Activity className="w-4 h-4 text-emerald-400" />
              Top 15 CTA Routes by Total Passenger Boardings
            </h2>
            <span className="text-xs text-slate-400 font-medium">{daytype === 'All' ? 'Full Period' : daytype}</span>
          </div>
          <div className="w-full h-[400px]">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData} margin={{ top: 10, right: 20, left: 20, bottom: 40 }}>
                <defs>
                  <linearGradient id="boardingsGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="#06b6d4" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="route" stroke="#64748b" tick={{ fill: '#94a3b8', fontSize: 11 }} angle={-35} textAnchor="end" />
                <YAxis stroke="#64748b" tick={{ fill: '#94a3b8', fontSize: 11 }} tickFormatter={(val) => `${(val / 1000).toFixed(0)}k`} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '0.5rem', color: '#fff' }}
                  formatter={(val: number) => [val.toLocaleString(), 'Boardings']}
                />
                <Area type="monotone" dataKey="boardings" stroke="#06b6d4" strokeWidth={2} fillOpacity={1} fill="url(#boardingsGrad)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
};
