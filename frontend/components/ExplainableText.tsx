'use client';

import React, { useState } from 'react';

interface AttributionToken {
  token: string;
  score: number;
}

interface ExplainableTextProps {
  attributions: AttributionToken[];
}

export default function ExplainableText({ attributions }: ExplainableTextProps) {
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  if (!attributions || attributions.length === 0) {
    return <p className="text-zinc-400 italic">No explainability details available. Submit an article to generate explanations.</p>;
  }

  // Find max absolute score for normalization
  const maxScore = Math.max(...attributions.map(t => Math.abs(t.score)), 0.001);

  return (
    <div className="leading-relaxed text-zinc-300 font-sans whitespace-pre-wrap max-h-[300px] overflow-y-auto p-4 rounded-xl border border-zinc-800 bg-zinc-950/50 backdrop-blur">
      {attributions.map((item, idx) => {
        const score = item.score;
        const absScore = Math.abs(score);
        
        // Skip styling for whitespace tokens or items with exactly zero score
        const isWhitespace = /^\s+$/.test(item.token);
        const hasScore = absScore > 0.0001;

        if (isWhitespace || !hasScore) {
          return <span key={idx}>{item.token}</span>;
        }

        // Normalize opacity (max opacity = 0.85)
        const opacity = Math.min((absScore / maxScore) * 0.85, 0.85);
        
        // Color: Positive is FAKE (Red), Negative is REAL (Green)
        const bgStyle = score > 0 
          ? { backgroundColor: `rgba(239, 68, 68, ${opacity})` } // red-500
          : { backgroundColor: `rgba(34, 197, 94, ${opacity})` }; // green-500

        const pct = (score * 100).toFixed(1);
        const tooltipText = score > 0 ? `+${pct}% (Fake)` : `${pct}% (Real)`;

        return (
          <span
            key={idx}
            style={bgStyle}
            className="relative inline px-0.5 rounded cursor-help transition-all duration-150 border-b border-transparent hover:border-zinc-300 hover:scale-[1.02] text-zinc-100 font-medium"
            onMouseEnter={() => setHoveredIdx(idx)}
            onMouseLeave={() => setHoveredIdx(null)}
          >
            {item.token}
            {hoveredIdx === idx && (
              <span className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-2.5 py-1 text-xs font-semibold text-white bg-zinc-900 border border-zinc-700 rounded-md shadow-xl z-20 whitespace-nowrap animate-in fade-in zoom-in-95 duration-100">
                {tooltipText}
              </span>
            )}
          </span>
        );
      })}
    </div>
  );
}
