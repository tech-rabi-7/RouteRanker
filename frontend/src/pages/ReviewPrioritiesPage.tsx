import React, { useEffect, useState } from 'react';
import { api, RankingItem } from '../api/client';
import { ArrowUpDown, AlertCircle, CheckCircle, Filter, Eye, Layers } from 'lucide-react';

export const ReviewPrioritiesPage: React.FC = () => {
  const [daytype, setDaytype] = useState<string>('W');
  const [sortBy, setSortBy] = useState<string>('service_productivity_metric_rank');
  const [order, setOrder] = useState<string>('asc');
  const [eligibleOnly, setEligibleOnly] = useState<boolean>(true);
  const [rankings, setRankings] = useState<RankingItem[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    api.getRankings(daytype, sortBy, order, eligibleOnly)
      .then(res => setRankings(res.rankings))
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [daytype, sortBy, order, eligibleOnly]);

  const toggleSort = (field: string) => {
    if (sortBy === field) {
      setOrder(order === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(field);
      setOrder('asc');
    }
  };

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-extrabold text-white tracking-tight">Review Priorities & Productivity</h1>
          <p className="text-slate-400 mt-1">
            Ratio-of-sums efficiency indicators, structural volume archetypes (KMeans), and transparent planning intervention triggers.
          </p>
        </div>

        {/* Controls */}
        <div className="flex flex-wrap items-center gap-3">
          {/* Daytype */}
          <div className="flex bg-slate-900 border border-slate-800 p-1 rounded-lg">
            {['W', 'A', 'U'].map(dt => (
              <button
                key={dt}
                onClick={() => setDaytype(dt)}
                className={`px-3 py-1.5 text-xs font-semibold rounded-md transition ${
                  daytype === dt ? 'bg-cyan-500 text-white shadow' : 'text-slate-400 hover:text-white'
                }`}
              >
                {dt === 'W' ? 'Weekday (W)' : dt === 'A' ? 'Saturday (A)' : 'Sunday (U)'}
              </button>
            ))}
          </div>

          {/* Eligibility Toggle */}
          <button
            onClick={() => setEligibleOnly(!eligibleOnly)}
            className={`px-3 py-1.5 text-xs font-semibold rounded-lg border transition ${
              eligibleOnly
                ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400'
                : 'bg-slate-900 border-slate-800 text-slate-400'
            }`}
          >
            {eligibleOnly ? 'Eligible Only (≥20 obs)' : 'All Routes'}
          </button>
        </div>
      </div>

      {/* Table Card */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-xl overflow-hidden shadow-xl">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm text-slate-300">
            <thead className="bg-slate-800/80 text-xs uppercase text-slate-400 border-b border-slate-700/50">
              <tr>
                <th className="px-5 py-3.5 cursor-pointer hover:text-white" onClick={() => toggleSort('service_productivity_metric_rank')}>
                  <div className="flex items-center gap-1.5">
                    <span>Rank</span>
                    <ArrowUpDown className="w-3.5 h-3.5" />
                  </div>
                </th>
                <th className="px-5 py-3.5 cursor-pointer hover:text-white" onClick={() => toggleSort('route')}>
                  Route
                </th>
                <th className="px-5 py-3.5 cursor-pointer hover:text-white text-right" onClick={() => toggleSort('observed_demand')}>
                  Observed Demand
                </th>
                <th className="px-5 py-3.5 cursor-pointer hover:text-white text-right" onClick={() => toggleSort('boardings_per_scheduled_trip')}>
                  Boardings / Trip
                </th>
                <th className="px-5 py-3.5 cursor-pointer hover:text-white text-right" onClick={() => toggleSort('boardings_per_scheduled_service_hour')}>
                  Boardings / Hour
                </th>
                <th className="px-5 py-3.5">Structural Archetype</th>
                <th className="px-5 py-3.5">Review Diagnostics</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800">
              {loading ? (
                <tr>
                  <td colSpan={7} className="px-6 py-12 text-center text-slate-400">
                    <div className="flex justify-center mb-2">
                      <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-cyan-400"></div>
                    </div>
                    Loading rankings table...
                  </td>
                </tr>
              ) : rankings.length === 0 ? (
                <tr>
                  <td colSpan={7} className="px-6 py-8 text-center text-slate-400">No route ranking data found.</td>
                </tr>
              ) : (
                rankings.map(r => (
                  <tr key={r.route} className="hover:bg-slate-800/40 transition">
                    <td className="px-5 py-4 font-mono font-bold text-cyan-400">
                      {r.service_productivity_metric_rank ? `#${r.service_productivity_metric_rank}` : '—'}
                    </td>
                    <td className="px-5 py-4">
                      <span className="font-bold text-white font-mono bg-slate-800 px-2.5 py-1 rounded border border-slate-700">
                        {r.route}
                      </span>
                    </td>
                    <td className="px-5 py-4 text-right font-medium text-slate-200">
                      {r.observed_demand.toLocaleString()}
                      <span className="text-xs text-slate-400 block">({r.n_obs} days)</span>
                    </td>
                    <td className="px-5 py-4 text-right font-mono font-semibold text-emerald-400">
                      {r.boardings_per_scheduled_trip.toFixed(2)}
                    </td>
                    <td className="px-5 py-4 text-right font-mono font-semibold text-indigo-400">
                      {r.boardings_per_scheduled_service_hour.toFixed(2)}
                    </td>
                    <td className="px-5 py-4">
                      <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-semibold bg-slate-800 text-slate-300 border border-slate-700">
                        <Layers className="w-3.5 h-3.5 text-cyan-400" />
                        Archetype #{r.cluster_id ?? 0}
                      </span>
                    </td>
                    <td className="px-5 py-4">
                      <div className="flex flex-wrap gap-1.5">
                        {r.planner_review_reasons.map((reason, idx) => (
                          <span
                            key={idx}
                            className={`text-xs px-2 py-0.5 rounded-full font-medium border ${
                              reason.includes('redundancy')
                                ? 'bg-amber-500/10 text-amber-400 border-amber-500/20'
                                : reason.includes('Top-tier')
                                ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                                : 'bg-rose-500/10 text-rose-300 border-rose-500/20'
                            }`}
                          >
                            {reason}
                          </span>
                        ))}
                      </div>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
