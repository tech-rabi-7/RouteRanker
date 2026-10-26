import React, { useEffect, useState, useRef } from 'react';
import { api, RouteSummary, RouteCompareResponse } from '../api/client';
import { GitCompare, MapPin, Layers, Check, AlertTriangle, ArrowRight } from 'lucide-react';
import L from 'leaflet';

export const RouteComparisonPage: React.FC = () => {
  const [allRoutes, setAllRoutes] = useState<RouteSummary[]>([]);
  const [selected, setSelected] = useState<string[]>(['49', 'X49']);
  const [compareData, setCompareData] = useState<RouteCompareResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const mapRef = useRef<HTMLDivElement>(null);
  const leafletMap = useRef<L.Map | null>(null);

  useEffect(() => {
    api.getRoutes().then(res => setAllRoutes(res.routes));
  }, []);

  useEffect(() => {
    if (selected.length < 2) return;
    setLoading(true);
    api.compareRoutes(selected)
      .then(setCompareData)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [selected]);

  // Leaflet Dual Map Trace
  useEffect(() => {
    if (!mapRef.current) return;
    if (leafletMap.current) {
      leafletMap.current.remove();
    }

    const map = L.map(mapRef.current).setView([41.8781, -87.68], 11);
    leafletMap.current = map;

    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; CARTO, CTA GTFS',
      maxZoom: 19,
    }).addTo(map);

    const colors = ['#00a1de', '#f43f5e', '#10b981', '#f59e0b', '#8b5cf6'];
    selected.forEach((r, idx) => {
      const color = colors[idx % colors.length];
      // Generate sample corridor path along Chicago grid
      const lon = -87.69 - (idx * 0.015);
      const coords: L.LatLngExpression[] = [
        [42.01, lon], [41.95, lon], [41.88, lon], [41.80, lon], [41.74, lon]
      ];
      L.polyline(coords, { color, weight: 4, opacity: 0.85 })
        .addTo(map)
        .bindPopup(`<b>Route ${r}</b><br/>Corridor Trace`);
    });

    return () => {
      map.remove();
      leafletMap.current = null;
    };
  }, [selected]);

  const toggleRoute = (route: string) => {
    if (selected.includes(route)) {
      if (selected.length > 2) {
        setSelected(selected.filter(r => r !== route));
      }
    } else {
      if (selected.length < 4) {
        setSelected([...selected, route]);
      }
    }
  };

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-extrabold text-white tracking-tight">Corridor Overlap & Route Comparison</h1>
        <p className="text-slate-400 mt-1">
          Side-by-side performance indicators, stop exclusivity analysis, and Jaccard corridor containment metrics.
        </p>
      </div>

      {/* Route Selector Badges */}
      <div className="bg-slate-900/80 border border-slate-800 p-4 rounded-xl shadow-md">
        <span className="text-xs font-semibold uppercase tracking-wider text-slate-400 block mb-2">
          Select 2 to 4 Routes to Compare:
        </span>
        <div className="flex flex-wrap gap-2">
          {['49', 'X49', '22', 'N22', '20', '66', '79', '8', '9', 'J14', '147'].map(r => {
            const isSelected = selected.includes(r);
            return (
              <button
                key={r}
                onClick={() => toggleRoute(r)}
                className={`px-3 py-1.5 text-xs font-mono font-bold rounded-lg border transition ${
                  isSelected
                    ? 'bg-cyan-500 text-white border-cyan-400 shadow-md'
                    : 'bg-slate-800/80 text-slate-400 border-slate-700 hover:text-white hover:bg-slate-700'
                }`}
              >
                Route {r}
              </button>
            );
          })}
        </div>
      </div>

      {/* Side-by-Side KPI Cards */}
      {compareData && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {compareData.routes.map((r, idx) => (
            <div key={r.route} className="bg-slate-900/80 border border-slate-800 p-5 rounded-xl shadow-lg relative overflow-hidden">
              <div className="flex items-center justify-between">
                <span className="text-sm font-bold font-mono text-cyan-400">Route {r.route}</span>
                <span className="text-xs px-2 py-0.5 rounded bg-slate-800 text-slate-400">
                  {r.served_stops_count} Total Stops
                </span>
              </div>
              <div className="mt-3 space-y-2 text-xs">
                <div className="flex justify-between">
                  <span className="text-slate-400">Total Boardings:</span>
                  <span className="font-semibold text-white">{r.total_boardings.toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Scheduled Trips:</span>
                  <span className="font-semibold text-white">{r.total_scheduled_trips.toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Boardings / Hour:</span>
                  <span className="font-semibold text-indigo-400 font-mono">{r.boardings_per_scheduled_service_hour}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Exclusive Stops:</span>
                  <span className="font-semibold text-emerald-400 font-mono">{r.exclusive_stops_count} stops</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Map & Overlap Metrics Table Split */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Leaflet Alignment Map */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 shadow-lg flex flex-col">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-base font-bold text-white flex items-center gap-2">
              <MapPin className="w-4 h-4 text-cyan-400" />
              Comparative Corridor Alignment
            </h2>
            <span className="text-xs px-2.5 py-1 rounded-md font-mono bg-cyan-950 text-cyan-400 border border-cyan-800">
              {compareData?.geometry_snapshot_version || 'mdb-389-202607110020'}
            </span>
          </div>
          <div ref={mapRef} className="w-full h-[360px] rounded-lg border border-slate-800" />
        </div>

        {/* Pairwise Overlap Table */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 shadow-lg flex flex-col justify-between">
          <div>
            <h2 className="text-base font-bold text-white mb-3 flex items-center gap-2">
              <GitCompare className="w-4 h-4 text-rose-400" />
              Pairwise Stop Overlap & Jaccard Containment
            </h2>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs text-slate-300">
                <thead className="bg-slate-800/80 uppercase text-slate-400">
                  <tr>
                    <th className="px-3 py-2.5">Pair</th>
                    <th className="px-3 py-2.5 text-right">Shared Stops</th>
                    <th className="px-3 py-2.5 text-right">Jaccard</th>
                    <th className="px-3 py-2.5 text-right">Smaller-Route Overlap</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800">
                  {compareData?.pairwise_overlaps.map((ov, idx) => (
                    <tr key={idx} className="hover:bg-slate-800/40">
                      <td className="px-3 py-3 font-mono font-semibold text-white">
                        {ov.route_a} ↔ {ov.route_b}
                      </td>
                      <td className="px-3 py-3 text-right font-mono text-cyan-400 font-bold">
                        {ov.shared_stops_count}
                      </td>
                      <td className="px-3 py-3 text-right font-mono text-slate-300">
                        {(ov.jaccard_similarity * 100).toFixed(1)}%
                      </td>
                      <td className="px-3 py-3 text-right font-mono font-bold text-amber-400">
                        {(ov.smaller_route_overlap * 100).toFixed(1)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          <p className="text-xs text-slate-400 mt-4 bg-slate-800/40 p-3 rounded border border-slate-800">
            * Smaller-route overlap = |S_A ∩ S_B| / min(|S_A|, |S_B|). Identifies express/local pairs and duplicate corridor redundancy.
          </p>
        </div>
      </div>
    </div>
  );
};
