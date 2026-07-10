'use client';

import React from 'react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';

interface ConfusionMatrix {
  tn: number;
  fp: number;
  fn: number;
  tp: number;
}

interface TrainingHistory {
  epochs: number[];
  loss: number[];
  val_loss: number[];
  accuracy: number[];
  val_accuracy: number[];
}

export interface MetricsData {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  confusion_matrix?: ConfusionMatrix;
  history?: TrainingHistory;
}

interface ModelMetricsProps {
  metrics: MetricsData | null;
}

export default function ModelMetrics({ metrics }: ModelMetricsProps) {
  if (!metrics) {
    return (
      <div className="flex flex-col justify-center items-center py-20 border border-zinc-800 rounded-2xl bg-zinc-950/20">
        <div className="w-8 h-8 rounded-full border-2 border-violet-500 border-t-transparent animate-spin mb-4"></div>
        <div className="text-center text-zinc-400">Loading model performance metrics...</div>
      </div>
    );
  }

  // Format Recharts data
  const chartData = (metrics.history?.epochs ?? []).map((epoch, idx) => ({
    epoch: `Epoch ${epoch}`,
    loss: metrics.history?.loss[idx],
    valLoss: metrics.history?.val_loss[idx],
    accuracy: metrics.history?.accuracy[idx],
    valAccuracy: metrics.history?.val_accuracy[idx]
  }));

  const { tn, fp, fn, tp } = metrics.confusion_matrix ?? { tn: 0, fp: 0, fn: 0, tp: 0 };
  const total = tn + fp + fn + tp;

  return (
    <div className="space-y-8 animate-in fade-in duration-300">
      {/* Cards Grid */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {[
          { label: 'Accuracy', val: metrics.accuracy, color: 'text-violet-400 border-violet-950/50 bg-violet-950/10' },
          { label: 'Precision', val: metrics.precision, color: 'text-emerald-400 border-emerald-950/50 bg-emerald-950/10' },
          { label: 'Recall', val: metrics.recall, color: 'text-amber-400 border-amber-950/50 bg-amber-950/10' },
          { label: 'F1 Score', val: metrics.f1_score, color: 'text-cyan-400 border-cyan-950/50 bg-cyan-950/10' },
        ].map((c, i) => (
          <div key={i} className={`p-6 rounded-2xl border ${c.color} text-center backdrop-blur`}>
            <div className="text-zinc-500 text-sm font-medium mb-1">{c.label}</div>
            <div className="text-3xl font-extrabold font-mono">{(c.val * 100).toFixed(1)}%</div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Training Progress Chart */}
        <div className="lg:col-span-8 p-6 rounded-2xl border border-zinc-800 bg-zinc-900/40 backdrop-blur space-y-4">
          <h3 className="text-lg font-bold text-zinc-100">Training Progress</h3>
          {chartData.length > 0 ? (
            <div className="h-[280px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                  <XAxis dataKey="epoch" stroke="#71717a" fontSize={12} />
                  <YAxis stroke="#71717a" fontSize={12} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#18181b', borderColor: '#27272a', borderRadius: '8px' }}
                    labelStyle={{ color: '#a1a1aa', fontWeight: 'bold' }}
                  />
                  <Legend wrapperStyle={{ fontSize: '12px', color: '#a1a1aa' }} />
                  <Line type="monotone" dataKey="accuracy" name="Train Accuracy" stroke="#8b5cf6" strokeWidth={2} dot={{ r: 4 }} />
                  <Line type="monotone" dataKey="valAccuracy" name="Val Accuracy" stroke="#ec4899" strokeWidth={2} dot={{ r: 4 }} />
                  <Line type="monotone" dataKey="loss" name="Train Loss" stroke="#3b82f6" strokeWidth={2} strokeDasharray="5 5" dot={false} />
                  <Line type="monotone" dataKey="valLoss" name="Val Loss" stroke="#f59e0b" strokeWidth={2} strokeDasharray="5 5" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="h-[280px] flex items-center justify-center text-sm text-zinc-500">
              Training history is unavailable for this model.
            </div>
          )}
        </div>

        {/* Confusion Matrix */}
        <div className="lg:col-span-4 p-6 rounded-2xl border border-zinc-800 bg-zinc-900/40 backdrop-blur flex flex-col justify-between">
          <div>
            <h3 className="text-lg font-bold text-zinc-100 mb-4">Confusion Matrix</h3>
            
            <div className="grid grid-cols-2 gap-2 aspect-square max-w-[280px] mx-auto w-full">
              {/* TN */}
              <div className="flex flex-col justify-center items-center p-4 rounded-xl border border-zinc-800/80 bg-emerald-950/20 text-emerald-300 text-center">
                <span className="text-[10px] font-bold uppercase tracking-wider opacity-75">True Neg (Fake)</span>
                <span className="text-2xl font-black mt-1 font-mono">{tn}</span>
                <span className="text-[10px] mt-0.5 opacity-60">({total ? ((tn / total) * 100).toFixed(0) : 0}%)</span>
              </div>
              {/* FP */}
              <div className="flex flex-col justify-center items-center p-4 rounded-xl border border-zinc-800/80 bg-rose-950/20 text-rose-300 text-center">
                <span className="text-[10px] font-bold uppercase tracking-wider opacity-75">False Pos</span>
                <span className="text-2xl font-black mt-1 font-mono">{fp}</span>
                <span className="text-[10px] mt-0.5 opacity-60">({total ? ((fp / total) * 100).toFixed(0) : 0}%)</span>
              </div>
              {/* FN */}
              <div className="flex flex-col justify-center items-center p-4 rounded-xl border border-zinc-800/80 bg-rose-950/20 text-rose-300 text-center">
                <span className="text-[10px] font-bold uppercase tracking-wider opacity-75">False Neg</span>
                <span className="text-2xl font-black mt-1 font-mono">{fn}</span>
                <span className="text-[10px] mt-0.5 opacity-60">({total ? ((fn / total) * 100).toFixed(0) : 0}%)</span>
              </div>
              {/* TP */}
              <div className="flex flex-col justify-center items-center p-4 rounded-xl border border-zinc-800/80 bg-emerald-950/20 text-emerald-300 text-center">
                <span className="text-[10px] font-bold uppercase tracking-wider opacity-75">True Pos (Real)</span>
                <span className="text-2xl font-black mt-1 font-mono">{tp}</span>
                <span className="text-[10px] mt-0.5 opacity-60">({total ? ((tp / total) * 100).toFixed(0) : 0}%)</span>
              </div>
            </div>
          </div>
          
          <div className="mt-4 text-center text-xs text-zinc-500 font-medium">
            Total Evaluation Samples: {total} articles
          </div>
        </div>
      </div>
    </div>
  );
}
