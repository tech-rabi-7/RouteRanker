import React, { useState } from 'react';
import {
  Database,
  MapPin,
  Bus,
  Award,
  GitCompare,
  Sliders,
  Cpu,
  Layers,
  ChevronRight,
  Activity,
} from 'lucide-react';

import { DataCoveragePage } from './pages/DataCoveragePage';
import { NetworkOverviewPage } from './pages/NetworkOverviewPage';
import { RouteDetailPage } from './pages/RouteDetailPage';
import { ReviewPrioritiesPage } from './pages/ReviewPrioritiesPage';
import { RouteComparisonPage } from './pages/RouteComparisonPage';
import { ScenariosPage } from './pages/ScenariosPage';
import { ModelEvaluationPage } from './pages/ModelEvaluationPage';

type TabId =
  | 'coverage'
  | 'network'
  | 'route-detail'
  | 'rankings'
  | 'compare'
  | 'scenarios'
  | 'models';

interface TabItem {
  id: TabId;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
  tag?: string;
}

const TABS: TabItem[] = [
  { id: 'coverage', label: 'Data & Coverage', icon: Database, tag: 'Provenance' },
  { id: 'network', label: 'Network Overview', icon: MapPin, tag: 'CTA GIS' },
  { id: 'route-detail', label: 'Route Detail', icon: Bus, tag: '125 Routes' },
  { id: 'rankings', label: 'Review Priorities', icon: Award, tag: 'KPI Ranks' },
  { id: 'compare', label: 'Route Comparison', icon: GitCompare, tag: 'Jaccard' },
  { id: 'scenarios', label: 'Scenarios', icon: Sliders, tag: 'Arithmetic' },
  { id: 'models', label: 'Model Evaluation', icon: Cpu, tag: 'WAPE' },
];

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabId>('coverage');

  return (
    <div className="flex min-h-screen bg-[#0b1120] text-slate-100">
      {/* Sidebar */}
      <aside className="w-64 bg-slate-900/90 border-r border-slate-800 flex flex-col justify-between flex-shrink-0">
        <div>
          {/* Logo & Platform Info */}
          <div className="p-6 border-b border-slate-800/80">
            <div className="flex items-center gap-2.5">
              <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-cyan-600 to-blue-500 flex items-center justify-center text-white shadow-lg shadow-cyan-500/20">
                <Bus className="w-5 h-5" />
              </div>
              <div>
                <span className="font-extrabold text-lg tracking-tight text-white block leading-none">
                  RouteRanker
                </span>
                <span className="text-[10px] uppercase font-bold tracking-wider text-cyan-400 mt-0.5 block">
                  v2.0 CTA Intelligence
                </span>
              </div>
            </div>
          </div>

          {/* Navigation Items */}
          <nav className="p-3 space-y-1">
            {TABS.map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`w-full flex items-center justify-between px-3.5 py-2.5 rounded-xl text-xs font-semibold transition group ${
                    isActive
                      ? 'bg-gradient-to-r from-cyan-500/20 to-blue-500/10 text-cyan-400 border border-cyan-500/30 shadow-md'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <Icon className={`w-4 h-4 transition ${isActive ? 'text-cyan-400' : 'text-slate-400 group-hover:text-slate-200'}`} />
                    <span>{tab.label}</span>
                  </div>
                  {tab.tag && (
                    <span
                      className={`text-[10px] px-2 py-0.5 rounded font-mono font-medium transition ${
                        isActive
                          ? 'bg-cyan-500/20 text-cyan-300'
                          : 'bg-slate-800 text-slate-400'
                      }`}
                    >
                      {tab.tag}
                    </span>
                  )}
                </button>
              );
            })}
          </nav>
        </div>

        {/* Footer / System Status */}
        <div className="p-4 border-t border-slate-800/80 bg-slate-950/40 text-xs text-slate-400 space-y-2">
          <div className="flex items-center justify-between">
            <span className="flex items-center gap-1.5 text-emerald-400 font-medium">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
              FastAPI Engine Active
            </span>
            <span className="font-mono text-[11px] text-slate-400">v2.0</span>
          </div>
          <p className="text-[11px] text-slate-400 leading-tight">
            CTA Bus Ridership (28k Matched Rows) &amp; mdb-389 GTFS.
          </p>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col overflow-y-auto">
        <header className="h-14 border-b border-slate-800/80 px-8 flex items-center justify-between bg-slate-900/40 backdrop-blur sticky top-0 z-30">
          <div className="flex items-center gap-2 text-xs text-slate-400">
            <span>RouteRanker Platform</span>
            <ChevronRight className="w-3.5 h-3.5 text-slate-400" />
            <span className="text-white font-medium capitalize">
              {TABS.find((t) => t.id === activeTab)?.label}
            </span>
          </div>
          <div className="flex items-center gap-3 text-xs">
            <span className="px-2.5 py-1 rounded bg-slate-800 text-slate-300 font-mono">
              PostgreSQL / DuckDB
            </span>
            <span className="px-2.5 py-1 rounded bg-cyan-950 text-cyan-400 font-mono border border-cyan-800">
              Target: Regression rides
            </span>
          </div>
        </header>

        <div className="p-8 max-w-7xl mx-auto w-full flex-1">
          {activeTab === 'coverage' && <DataCoveragePage />}
          {activeTab === 'network' && <NetworkOverviewPage />}
          {activeTab === 'route-detail' && <RouteDetailPage />}
          {activeTab === 'rankings' && <ReviewPrioritiesPage />}
          {activeTab === 'compare' && <RouteComparisonPage />}
          {activeTab === 'scenarios' && <ScenariosPage />}
          {activeTab === 'models' && <ModelEvaluationPage />}
        </div>
      </main>
    </div>
  );
};

export default App;
