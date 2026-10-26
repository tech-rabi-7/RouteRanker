import React, { useEffect, useState } from 'react';
import { api, RouteSummary, ScenarioResponse } from '../api/client';
import { Sliders, AlertOctagon, TrendingDown, TrendingUp, ArrowRight, Clock, Bus } from 'lucide-react';

export const ScenariosPage: React.FC = () => {
  const [routes, setRoutes] = useState<RouteSummary[]>([]);
  const [selectedRoute, setSelectedRoute] = useState<string>('20');
  const [daytype, setDaytype] = useState<string>('W');
  const [freqChangePct, setFreqChangePct] = useState<number>(20);
  const [durationAssumption, setDurationAssumption] = useState<number>(1.0);
  const [scenarioResult, setScenarioResult] = useState<ScenarioResponse | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    api.getRoutes().then(res => setRoutes(res.routes));
  }, []);

  // Run calculation when parameters change
  useEffect(() => {
    if (!selectedRoute) return;
    setLoading(true);
    api.runScenario({
      route: selectedRoute,
      daytype: daytype,
      frequency_change_pct: freqChangePct,
      duration_assumption: durationAssumption > 0 ? durationAssumption : undefined,
    })
      .then(setScenarioResult)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [selectedRoute, daytype, freqChangePct, durationAssumption]);

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-extrabold text-white tracking-tight">Interactive Scenario Calculator</h1>
        <p className="text-slate-400 mt-1">
          Evaluate trip frequency sensitivity holding forecast boardings constant to calculate conditional service productivity.
        </p>
      </div>

      {/* Calculator Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Input Parameters Panel */}
        <div className="bg-slate-900/80 border border-slate-800 p-6 rounded-xl shadow-lg space-y-6">
          <h2 className="text-base font-bold text-white flex items-center gap-2">
            <Sliders className="w-4 h-4 text-cyan-400" />
            Service Adjustment Controls
          </h2>

          {/* Route Selector */}
          <div>
            <label className="text-xs font-semibold uppercase text-slate-400 block mb-1.5">Route Identifier</label>
            <select
              value={selectedRoute}
              onChange={(e) => setSelectedRoute(e.target.value)}
              className="w-full bg-slate-800 border border-slate-700 text-white rounded-lg px-3.5 py-2.5 text-sm font-semibold focus:outline-none focus:border-cyan-400"
            >
              {routes.map(r => (
                <option key={r.route} value={r.route}>
                  Route {r.route}
                </option>
              ))}
            </select>
          </div>

          {/* Daytype */}
          <div>
            <label className="text-xs font-semibold uppercase text-slate-400 block mb-1.5">Service Daytype</label>
            <div className="grid grid-cols-3 gap-2">
              {['W', 'A', 'U'].map(dt => (
                <button
                  key={dt}
                  onClick={() => setDaytype(dt)}
                  className={`py-2 text-xs font-semibold rounded-lg transition ${
                    daytype === dt
                      ? 'bg-cyan-500 text-white shadow'
                      : 'bg-slate-800 text-slate-400 hover:text-white'
                  }`}
                >
                  {dt === 'W' ? 'Weekday' : dt === 'A' ? 'Saturday' : 'Sunday'}
                </button>
              ))}
            </div>
          </div>

          {/* Frequency Change Slider */}
          <div>
            <div className="flex justify-between items-center mb-1.5">
              <label className="text-xs font-semibold uppercase text-slate-400">Frequency / Trip Adjustment</label>
              <span className={`text-sm font-bold font-mono ${freqChangePct > 0 ? 'text-emerald-400' : freqChangePct < 0 ? 'text-rose-400' : 'text-slate-400'}`}>
                {freqChangePct > 0 ? `+${freqChangePct}%` : `${freqChangePct}%`}
              </span>
            </div>
            <input
              type="range"
              min="-50"
              max="100"
              step="5"
              value={freqChangePct}
              onChange={(e) => setFreqChangePct(Number(e.target.value))}
              className="w-full accent-cyan-500 bg-slate-800 h-2 rounded-lg cursor-pointer"
            />
            <div className="flex justify-between text-[11px] text-slate-400 mt-1 font-mono">
              <span>-50% (Cut)</span>
              <span>Baseline (0%)</span>
              <span>+100% (Double)</span>
            </div>
          </div>

          {/* Trip Duration Assumption */}
          <div>
            <label className="text-xs font-semibold uppercase text-slate-400 block mb-1.5">
              Assumed Duration / Trip (Hours)
            </label>
            <input
              type="number"
              step="0.1"
              min="0.2"
              max="3.0"
              value={durationAssumption}
              onChange={(e) => setDurationAssumption(Number(e.target.value))}
              className="w-full bg-slate-800 border border-slate-700 text-white rounded-lg px-3.5 py-2 text-sm font-mono focus:outline-none focus:border-cyan-400"
            />
            <span className="text-[11px] text-slate-400 mt-1 block">Default: Baseline average service hours / trips.</span>
          </div>
        </div>

        {/* Results Panel */}
        <div className="lg:col-span-2 space-y-6">
          {scenarioResult && (
            <div className="bg-slate-900/80 border border-slate-800 p-6 rounded-xl shadow-lg space-y-6">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800 pb-4">
                <div>
                  <h3 className="text-lg font-bold text-white">Deterministic Arithmetic Outcomes</h3>
                  <p className="text-xs text-slate-400 mt-0.5">
                    Boardings held strictly constant at{' '}
                    <span className="text-cyan-400 font-mono font-bold">
                      {scenarioResult.boardings_held_constant.toLocaleString()} rides
                    </span>
                  </p>
                </div>
                <span className="text-xs px-3 py-1 rounded-full font-mono bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
                  Route {selectedRoute} ({daytype})
                </span>
              </div>

              {/* Comparison Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Scheduled Service Card */}
                <div className="bg-slate-800/50 p-4 rounded-xl border border-slate-700/50 space-y-3">
                  <span className="text-xs font-semibold uppercase text-slate-400 block">Scheduled Trips & Hours</span>
                  <div className="flex items-center justify-between">
                    <div>
                      <span className="text-xs text-slate-400">Baseline Trips:</span>
                      <p className="text-xl font-bold text-white font-mono">{scenarioResult.baseline.scheduled_trips}</p>
                    </div>
                    <ArrowRight className="w-5 h-5 text-slate-400" />
                    <div className="text-right">
                      <span className="text-xs text-slate-400">Scenario Trips:</span>
                      <p className="text-xl font-bold text-cyan-400 font-mono">{scenarioResult.scenario.scheduled_trips}</p>
                    </div>
                  </div>
                  <div className="text-xs text-slate-300 border-t border-slate-700/50 pt-2 flex justify-between">
                    <span>Service Hours:</span>
                    <span className="font-mono text-white">
                      {scenarioResult.baseline.scheduled_service_hours}h → {scenarioResult.scenario.scheduled_service_hours}h
                      <span className={`ml-1 font-bold ${scenarioResult.difference.service_hours_delta >= 0 ? 'text-amber-400' : 'text-emerald-400'}`}>
                        ({scenarioResult.difference.service_hours_delta >= 0 ? `+${scenarioResult.difference.service_hours_delta}` : scenarioResult.difference.service_hours_delta}h)
                      </span>
                    </span>
                  </div>
                </div>

                {/* Conditional Productivity Card */}
                <div className="bg-slate-800/50 p-4 rounded-xl border border-slate-700/50 space-y-3">
                  <span className="text-xs font-semibold uppercase text-slate-400 block">Conditional Vehicle-Hour Productivity</span>
                  <div className="flex items-center justify-between">
                    <div>
                      <span className="text-xs text-slate-400">Baseline Ratio:</span>
                      <p className="text-xl font-bold text-white font-mono">
                        {scenarioResult.baseline.boardings_per_scheduled_service_hour}
                      </p>
                    </div>
                    <ArrowRight className="w-5 h-5 text-slate-400" />
                    <div className="text-right">
                      <span className="text-xs text-slate-400">Conditional Ratio:</span>
                      <p className="text-xl font-bold text-indigo-400 font-mono">
                        {scenarioResult.scenario.boardings_per_scheduled_service_hour}
                      </p>
                    </div>
                  </div>
                  <div className="text-xs text-slate-300 border-t border-slate-700/50 pt-2 flex justify-between">
                    <span>Boardings / Trip:</span>
                    <span className="font-mono text-white">
                      {scenarioResult.baseline.boardings_per_scheduled_trip} → {scenarioResult.scenario.boardings_per_scheduled_trip}
                    </span>
                  </div>
                </div>
              </div>

              {/* Missing Constraints Notice Card */}
              <div className="bg-amber-950/20 border border-amber-800/40 p-4 rounded-xl space-y-2">
                <div className="flex items-center gap-2 text-amber-400 font-bold text-xs uppercase tracking-wide">
                  <AlertOctagon className="w-4 h-4" />
                  Explicit Declaration of Unmodeled Operational Constraints
                </div>
                <p className="text-xs text-slate-300">
                  This simplified sensitivity model assumes static passenger demand. In real-world agency transit planning, the following physical constraints must be verified:
                </p>
                <ul className="text-xs text-slate-400 space-y-1 list-disc list-inside">
                  {scenarioResult.missing_operational_constraints.map((c, i) => (
                    <li key={i}>{c}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
