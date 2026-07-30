'use client';

import React, { useState, useEffect } from 'react';
import { ShieldAlert, BarChart3 } from 'lucide-react';
import Dashboard from '../components/Dashboard';
import ModelMetrics, { type MetricsData } from '../components/ModelMetrics';

export default function Home() {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'metrics'>('dashboard');
  const [metrics, setMetrics] = useState<MetricsData | null>(null);

  useEffect(() => {
    // Fetch metrics from backend
    const apiBase = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    fetch(`${apiBase}/metrics`)
      .then(res => res.json())
      .then((data: MetricsData) => setMetrics(data))
      .catch(err => console.error('Failed to fetch metrics:', err));
  }, []);

  return (
    <div className="min-h-screen bg-black text-zinc-200 font-sans flex flex-col selection:bg-violet-900 selection:text-white">
      {/* Header */}
      <header className="border-b border-zinc-900 bg-black/40 backdrop-blur sticky top-0 z-30">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="p-2 rounded-xl bg-violet-600/10 border border-violet-500/20">
              <ShieldAlert className="w-6 h-6 text-violet-500" />
            </div>
            <div>
              <span className="font-extrabold text-lg text-zinc-100 tracking-tight block leading-none">VERITAS</span>
              <span className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest block mt-0.5">AI Fake News Detection</span>
            </div>
          </div>

          {/* Navigation Tabs */}
          <nav className="flex gap-1 bg-zinc-950 p-1 rounded-xl border border-zinc-850">
            <button
              onClick={() => setActiveTab('dashboard')}
              className={`px-4 py-1.5 rounded-lg text-xs font-bold transition flex items-center gap-1.5 ${
                activeTab === 'dashboard'
                  ? 'bg-zinc-900 text-zinc-100 border border-zinc-800/80'
                  : 'text-zinc-500 hover:text-zinc-300'
              }`}
            >
              <NewsheetIcon className="w-3.5 h-3.5" />
              Analysis Dashboard
            </button>
            <button
              onClick={() => setActiveTab('metrics')}
              className={`px-4 py-1.5 rounded-lg text-xs font-bold transition flex items-center gap-1.5 ${
                activeTab === 'metrics'
                  ? 'bg-zinc-900 text-zinc-100 border border-zinc-800/80'
                  : 'text-zinc-500 hover:text-zinc-300'
              }`}
            >
              <BarChart3 className="w-3.5 h-3.5" />
              Model Performance
            </button>
          </nav>

          {/* External Links */}
          <div className="hidden md:flex items-center gap-4">
            <a
              href="https://github.com"
              target="_blank"
              rel="noopener noreferrer"
              className="text-zinc-500 hover:text-zinc-300 transition"
            >
              <GithubIcon className="w-5 h-5" />
            </a>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-grow max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Banner header */}
        <div className="mb-10 text-center md:text-left space-y-2">
          <h1 className="text-4xl font-extrabold text-zinc-150 tracking-tight sm:text-5xl bg-clip-text text-transparent bg-gradient-to-r from-zinc-100 via-zinc-200 to-zinc-400">
            Explainable Fake News Prediction
          </h1>
          <p className="text-zinc-500 text-base max-w-2xl font-light">
            Leverage state-of-the-art Natural Language Processing models combined with text perturbation heuristics to understand which words influence predictions.
          </p>
        </div>

        {activeTab === 'dashboard' ? (
          <Dashboard />
        ) : (
          <ModelMetrics metrics={metrics} />
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-zinc-950 bg-black/80 py-6 text-center text-xs text-zinc-600">
        <div className="max-w-7xl mx-auto px-4">
          © {new Date().getFullYear()} Veritas AI. Production-ready Explainable AI Platform.
        </div>
      </footer>
    </div>
  );
}

// Simple fallback icon helper for Newspaper since Lucide can sometimes change exports
function NewsheetIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      {...props}
    >
      <path d="M4 22h16a2 2 0 0 0 2-2V4a2 2 0 0 0-2-2H8a2 2 0 0 0-2 2v16a2 2 0 0 1-2 2Zm0 0a2 2 0 0 1-2-2v-9c0-1.1.9-2 2-2h2" />
      <path d="M18 14h-8" />
      <path d="M15 18h-5" />
      <path d="M10 6h8v4h-8V6Z" />
    </svg>
  );
}

// Custom inline SVG for Github to prevent any package resolver issues
function GithubIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      {...props}
    >
      <path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4" />
      <path d="M9 18c-4.51 2-5-2-7-2" />
    </svg>
  );
}
