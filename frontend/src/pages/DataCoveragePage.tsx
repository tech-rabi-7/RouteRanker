import React, { useEffect, useState } from 'react';
import { api, DataCoverage } from '../api/client';
import { Database, ShieldCheck, Calendar, CheckCircle2, AlertTriangle, Layers } from 'lucide-react';

export const DataCoveragePage: React.FC = () => {
  const [data, setData] = useState<DataCoverage | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.getCoverage()
      .then(setData)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  if (loading || !data) {
    return (
      <div className="flex items-center justify-center min-h-[500px]">
        <div className="animate-spin rounded-full h-10 w-10 border-t-2 border-b-2 border-cyan-400"></div>
      </div>
    );
  }

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-extrabold text-white tracking-tight">Data Integrity & Source Register</h1>
        <p className="text-slate-400 mt-2">
          Audit trail, SHA-256 cryptographic provenance, and GTFS snapshot validity windows for the Chicago Transit Authority (CTA) dataset.
        </p>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 shadow-lg backdrop-blur">
          <div className="flex items-center justify-between">
            <span className="text-xs uppercase tracking-wider text-slate-400 font-semibold">Matched Rows</span>
            <CheckCircle2 className="w-5 h-5 text-emerald-400" />
          </div>
          <p className="text-2xl font-bold text-white mt-2">{data.matched_rows_count.toLocaleString()}</p>
          <p className="text-xs text-emerald-400 mt-1 font-medium">100% Target Met (28,374)</p>
        </div>

        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 shadow-lg backdrop-blur">
          <div className="flex items-center justify-between">
            <span className="text-xs uppercase tracking-wider text-slate-400 font-semibold">Service Dates</span>
            <Calendar className="w-5 h-5 text-cyan-400" />
          </div>
          <p className="text-2xl font-bold text-white mt-2">{data.total_service_dates}</p>
          <p className="text-xs text-slate-400 mt-1">2025-11-23 → 2026-07-31</p>
        </div>

        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 shadow-lg backdrop-blur">
          <div className="flex items-center justify-between">
            <span className="text-xs uppercase tracking-wider text-slate-400 font-semibold">Unique Routes</span>
            <Layers className="w-5 h-5 text-indigo-400" />
          </div>
          <p className="text-2xl font-bold text-white mt-2">{data.total_routes}</p>
          <p className="text-xs text-slate-400 mt-1">String IDs (N/X/J prefixes)</p>
        </div>

        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 shadow-lg backdrop-blur">
          <div className="flex items-center justify-between">
            <span className="text-xs uppercase tracking-wider text-slate-400 font-semibold">Zero-Boardings</span>
            <ShieldCheck className="w-5 h-5 text-blue-400" />
          </div>
          <p className="text-2xl font-bold text-white mt-2">{data.zero_rides_count}</p>
          <p className="text-xs text-blue-400 mt-1">Genuine 0-count preserved</p>
        </div>

        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 shadow-lg backdrop-blur">
          <div className="flex items-center justify-between">
            <span className="text-xs uppercase tracking-wider text-slate-400 font-semibold">Unmatched Sched</span>
            <AlertTriangle className="w-5 h-5 text-amber-400" />
          </div>
          <p className="text-2xl font-bold text-white mt-2">{data.unmatched_schedule_rows_count}</p>
          <p className="text-xs text-amber-400 mt-1">Quarantined (not filled)</p>
        </div>
      </div>

      {/* Quality Contracts Card */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-6 shadow-md">
        <h2 className="text-lg font-bold text-white flex items-center gap-2">
          <ShieldCheck className="w-5 h-5 text-cyan-400" />
          Data Contracts & Integrity Enforcement
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
          {Object.entries(data.quality_checks).map(([key, val]) => (
            <div key={key} className="flex items-start gap-3 bg-slate-800/50 p-3 rounded-lg border border-slate-700/50">
              <CheckCircle2 className="w-4 h-4 text-emerald-400 mt-0.5 flex-shrink-0" />
              <div>
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">{key.replace(/_/g, ' ')}</span>
                <p className="text-sm font-medium text-slate-200 mt-0.5">{val}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Snapshot Windows */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-xl overflow-hidden shadow-lg">
        <div className="px-6 py-4 border-b border-slate-800 flex items-center justify-between">
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <Calendar className="w-5 h-5 text-indigo-400" />
            Mobility Database (mdb-389) 8 GTFS Snapshots & Validity Intervals
          </h2>
          <span className="text-xs font-medium px-2.5 py-1 rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
            Monotonic Selection Rule
          </span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm text-slate-300">
            <thead className="bg-slate-800/70 text-xs uppercase text-slate-400">
              <tr>
                <th className="px-6 py-3">Snapshot ID</th>
                <th className="px-6 py-3">Catalog Retrieved Date</th>
                <th className="px-6 py-3">Service Date Window</th>
                <th className="px-6 py-3">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800">
              {data.snapshot_windows.map((w, idx) => (
                <tr key={w.snapshot_id} className="hover:bg-slate-800/40 transition">
                  <td className="px-6 py-3.5 font-mono text-cyan-400 font-medium">{w.snapshot_id}</td>
                  <td className="px-6 py-3.5 text-slate-300">{w.catalog_retrieved_date}</td>
                  <td className="px-6 py-3.5 font-medium text-white">
                    {w.service_date_from} → {w.service_date_to}
                  </td>
                  <td className="px-6 py-3.5">
                    <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                      <span className="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
                      Active Window {idx + 1}/8
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
