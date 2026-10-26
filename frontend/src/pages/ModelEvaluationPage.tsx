import React, { useEffect, useState } from 'react';
import { Award, ShieldAlert, Cpu, BarChart2, CheckCircle2, TrendingDown } from 'lucide-react';

export const ModelEvaluationPage: React.FC = () => {
  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-extrabold text-white tracking-tight">Machine Learning Evaluation & Benchmarks</h1>
        <p className="text-slate-400 mt-1">
          Chronological regression pipelines evaluated on real CTA daily ridership observations using WAPE, MAE, and RMSE.
        </p>
      </div>

      {/* Operational Release Delay Notice Banner */}
      <div className="bg-gradient-to-r from-cyan-950/40 via-slate-900 to-slate-900 border border-cyan-800/40 p-5 rounded-xl flex items-start gap-3.5 shadow-md">
        <ShieldAlert className="w-5 h-5 text-cyan-400 flex-shrink-0 mt-0.5" />
        <div>
          <span className="text-xs uppercase font-bold tracking-wider text-cyan-400">
            Retrospective Experiment vs. Operational Release-Delay Protocol
          </span>
          <p className="text-xs text-slate-300 mt-1 leading-relaxed">
            In standard retrospective mode, complete history including <code className="text-cyan-300">lag_1d</code> is available to measure maximum predictive capacity. For operational production readiness, RouteRanker supports an automated issue-time gate (<code className="text-cyan-300">--exclude-lag1</code> / <code className="text-cyan-300">--release-delay-days=N</code>) guaranteeing that features reflect real-world agency publication delays without look-ahead leakage.
          </p>
        </div>
      </div>

      {/* Experiment C: Controlled Aligned Comparison Spotlight */}
      <div className="bg-slate-900/90 border border-indigo-500/30 rounded-xl p-6 shadow-xl relative overflow-hidden">
        <div className="flex items-center justify-between mb-4 border-b border-slate-800 pb-3">
          <div className="flex items-center gap-2">
            <Award className="w-5 h-5 text-indigo-400" />
            <h2 className="text-lg font-bold text-white">Experiment C — Controlled Aligned Comparison</h2>
          </div>
          <span className="text-xs font-semibold px-2.5 py-1 rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 font-mono">
            MATCHED_SERVICE_SPLIT (7,014 Test Rows)
          </span>
        </div>
        <p className="text-xs text-slate-300 mb-4 leading-relaxed">
          To isolate the genuine marginal value of GTFS schedule variables (trips & vehicle service hours), both Demand-Only features and Demand+Service features were trained and evaluated on the <b>exact same date window</b>, eliminating sample size and date discrepancy bias.
        </p>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs text-slate-300">
            <thead className="bg-slate-800/70 uppercase text-slate-400">
              <tr>
                <th className="px-4 py-3">Feature Set</th>
                <th className="px-4 py-3">Split Window</th>
                <th className="px-4 py-3 text-right">Test Rows</th>
                <th className="px-4 py-3 text-right">Random Forest Test WAPE</th>
                <th className="px-4 py-3 text-right">XGBoost Test WAPE</th>
                <th className="px-4 py-3 text-right">Ridge Baseline</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800 font-mono">
              <tr className="hover:bg-slate-800/30">
                <td className="px-4 py-3.5 font-bold text-slate-300">Aligned Demand-Only (No GTFS)</td>
                <td className="px-4 py-3.5 text-slate-400">2025-11 → 2026-07</td>
                <td className="px-4 py-3.5 text-right text-white">7,014</td>
                <td className="px-4 py-3.5 text-right text-rose-400 font-bold">12.47%</td>
                <td className="px-4 py-3.5 text-right text-rose-400 font-bold">12.46%</td>
                <td className="px-4 py-3.5 text-right text-slate-400">12.59%</td>
              </tr>
              <tr className="bg-indigo-950/20 hover:bg-indigo-950/30">
                <td className="px-4 py-3.5 font-bold text-cyan-400">Demand + Matched GTFS Service</td>
                <td className="px-4 py-3.5 text-slate-400">2025-11 → 2026-07</td>
                <td className="px-4 py-3.5 text-right text-white">7,014</td>
                <td className="px-4 py-3.5 text-right text-emerald-400 font-bold">8.07%</td>
                <td className="px-4 py-3.5 text-right text-emerald-400 font-bold">8.05%</td>
                <td className="px-4 py-3.5 text-right text-slate-400">8.47%</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div className="mt-4 p-3 rounded-lg bg-emerald-950/20 border border-emerald-800/30 flex items-center justify-between text-xs text-emerald-300 font-medium">
          <span className="flex items-center gap-1.5">
            <CheckCircle2 className="w-4 h-4 text-emerald-400" />
            Empirical Schedule Feature Impact (WAPE Delta):
          </span>
          <span className="font-mono font-bold text-sm bg-emerald-900/50 px-2.5 py-0.5 rounded border border-emerald-700">
            -4.39% Absolute Error Reduction
          </span>
        </div>
      </div>

      {/* Experiment A vs Experiment B Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Experiment A Card */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 shadow-lg space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <h3 className="font-bold text-white text-base">Experiment A — Demand Only (Full Timeline)</h3>
            <span className="text-xs font-mono text-cyan-400">83,997 Rows</span>
          </div>
          <p className="text-xs text-slate-400">
            Trained on 63,143 rows (2024-08 to 2026-01), validated on 10,246 rows, and tested on 10,608 rows using pure calendar and lag features.
          </p>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs text-slate-300">
              <thead className="bg-slate-800/80 uppercase text-slate-400">
                <tr>
                  <th className="px-3 py-2">Model</th>
                  <th className="px-3 py-2 text-right">Val MAE</th>
                  <th className="px-3 py-2 text-right">Test RMSE</th>
                  <th className="px-3 py-2 text-right">Test WAPE</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800 font-mono">
                <tr>
                  <td className="px-3 py-2.5 text-slate-300">7-Day Lag Baseline</td>
                  <td className="px-3 py-2.5 text-right">412.5</td>
                  <td className="px-3 py-2.5 text-right">525.4</td>
                  <td className="px-3 py-2.5 text-right text-slate-400">14.65%</td>
                </tr>
                <tr>
                  <td className="px-3 py-2.5 text-slate-300">Ridge Pooled</td>
                  <td className="px-3 py-2.5 text-right">386.1</td>
                  <td className="px-3 py-2.5 text-right">488.0</td>
                  <td className="px-3 py-2.5 text-right text-slate-400">13.60%</td>
                </tr>
                <tr>
                  <td className="px-3 py-2.5 text-white font-bold">Random Forest</td>
                  <td className="px-3 py-2.5 text-right">382.6</td>
                  <td className="px-3 py-2.5 text-right">486.3</td>
                  <td className="px-3 py-2.5 text-right text-cyan-400 font-bold">13.51%</td>
                </tr>
                <tr className="bg-cyan-950/20 font-bold text-cyan-300">
                  <td className="px-3 py-2.5">XGBoost Regressor</td>
                  <td className="px-3 py-2.5 text-right">372.2</td>
                  <td className="px-3 py-2.5 text-right">477.1</td>
                  <td className="px-3 py-2.5 text-right text-emerald-400">13.19%</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* Experiment B Card */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 shadow-lg space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <h3 className="font-bold text-white text-base">Experiment B — Demand + Matched Service</h3>
            <span className="text-xs font-mono text-emerald-400">28,374 Rows</span>
          </div>
          <p className="text-xs text-slate-400">
            Trained on 14,501 rows (2025-11 to 2026-03), validated on 6,859 rows, and tested on 7,014 rows with GTFS trips and service hours.
          </p>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs text-slate-300">
              <thead className="bg-slate-800/80 uppercase text-slate-400">
                <tr>
                  <th className="px-3 py-2">Model</th>
                  <th className="px-3 py-2 text-right">Val MAE</th>
                  <th className="px-3 py-2 text-right">Test RMSE</th>
                  <th className="px-3 py-2 text-right">Test WAPE</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800 font-mono">
                <tr>
                  <td className="px-3 py-2.5 text-slate-300">7-Day Lag Baseline</td>
                  <td className="px-3 py-2.5 text-right">320.1</td>
                  <td className="px-3 py-2.5 text-right">418.5</td>
                  <td className="px-3 py-2.5 text-right text-slate-400">11.45%</td>
                </tr>
                <tr>
                  <td className="px-3 py-2.5 text-slate-300">Ridge Pooled</td>
                  <td className="px-3 py-2.5 text-right">234.4</td>
                  <td className="px-3 py-2.5 text-right">304.6</td>
                  <td className="px-3 py-2.5 text-right text-slate-400">8.47%</td>
                </tr>
                <tr>
                  <td className="px-3 py-2.5 text-white font-bold">Random Forest</td>
                  <td className="px-3 py-2.5 text-right">224.0</td>
                  <td className="px-3 py-2.5 text-right">293.3</td>
                  <td className="px-3 py-2.5 text-right text-cyan-400 font-bold">8.07%</td>
                </tr>
                <tr className="bg-emerald-950/20 font-bold text-emerald-300">
                  <td className="px-3 py-2.5">XGBoost Regressor</td>
                  <td className="px-3 py-2.5 text-right">221.7</td>
                  <td className="px-3 py-2.5 text-right">292.9</td>
                  <td className="px-3 py-2.5 text-right text-emerald-400">8.05%</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};
