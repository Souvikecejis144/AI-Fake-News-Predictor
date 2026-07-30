'use client';

import React, { useState } from 'react';
import { AlertTriangle, CheckCircle, FileText, Tag, RefreshCw, BarChart2, MessageSquare, Info } from 'lucide-react';
import ExplainableText from './ExplainableText';

interface AttributionToken {
  token: string;
  score: number;
}

interface AnalysisResults {
  prediction: 'REAL' | 'FAKE';
  confidence: number;
  modelUsed: string;
  attributions: AttributionToken[];
  summary: string;
  keywords: string[];
  topic: string;
  sentiment: {
    sentiment: 'POSITIVE' | 'NEGATIVE' | 'NEUTRAL';
    pos: number;
    neg: number;
    neu: number;
  };
}

export default function Dashboard() {
  const [text, setText] = useState('');
  const [modelType, setModelType] = useState<'ml' | 'transformer' | 'gemini'>('ml');
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState<AnalysisResults | null>(null);
  const [error, setError] = useState<string | null>(null);

  const sampleReal = "The city council announced a new public transit expansion project starting next Tuesday, officials confirmed. Researchers published a study in Science detailing a breakthrough in solar cell efficiency and durability. The Federal Reserve decided to keep interest rates steady after a two-day economic review meeting. Officials stated that public transit investments will improve municipal energy efficiency.";
  
  const sampleFake = "SHOCKING SECRETS: Government officials are hiding alien spacecraft in underground bunkers, exposed! MUST READ: Miracle cure for all diseases discovered by independent blogger, doctors hate him! Major corporations are using mind control signals through street lights, whistleblower claims. Read before it is deleted! SHARE THIS NOW!";

  const handleLoadSample = (sample: string) => {
    setText(sample);
    setError(null);
  };

  const handleClear = () => {
    setText('');
    setResults(null);
    setError(null);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!text.trim()) return;

    setIsLoading(true);
    setError(null);
    
    const apiBase = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

    try {
      // 1. Predict
      const predRes = await fetch(`${apiBase}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, model_type: modelType }),
      });
      if (!predRes.ok) throw new Error('Prediction API failed.');
      const predData = await predRes.json();

      // 2. Explain
      const explainRes = await fetch(`${apiBase}/explain`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, model_type: modelType }),
      });
      if (!explainRes.ok) throw new Error('Explainability API failed.');
      const explainData = await explainRes.json();

      // 3. Summary
      const summaryRes = await fetch(`${apiBase}/summary`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      const summaryData = summaryRes.ok ? await summaryRes.json() : { summary: 'Summary unavailable.' };

      // 4. Keywords
      const keywordsRes = await fetch(`${apiBase}/keywords`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      const keywordsData = keywordsRes.ok ? await keywordsRes.json() : { keywords: [] };

      // 5. Topics
      const topicsRes = await fetch(`${apiBase}/topics`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      const topicsData = topicsRes.ok ? await topicsRes.json() : { topic: 'General' };

      // 6. Sentiment
      const sentimentRes = await fetch(`${apiBase}/sentiment`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      const sentimentData = sentimentRes.ok 
        ? await sentimentRes.json() 
        : { sentiment: 'NEUTRAL', pos: 0.33, neg: 0.33, neu: 0.34 };

      // Combine results
      setResults({
        prediction: predData.prediction,
        confidence: predData.confidence,
        modelUsed: predData.model_used,
        attributions: explainData.attributions,
        summary: summaryData.summary,
        keywords: keywordsData.keywords,
        topic: topicsData.topic,
        sentiment: sentimentData,
      });
    } catch (err: unknown) {
      console.error(err);
      setError(
        err instanceof Error
          ? err.message
          : 'An error occurred while connecting to the backend. Please check if the FastAPI server is running.'
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="grid grid-cols-1 xl:grid-cols-12 gap-8 items-start animate-in fade-in duration-300">
      {/* Input Section */}
      <div className="xl:col-span-5 space-y-6">
        <div className="p-6 rounded-2xl border border-zinc-800 bg-zinc-900/40 backdrop-blur space-y-6">
          <div className="flex justify-between items-center">
            <h2 className="text-xl font-extrabold text-zinc-100 flex items-center gap-2">
              <FileText className="w-5 h-5 text-violet-500" />
              Source Article
            </h2>
            <div className="flex gap-2">
              <button
                type="button"
                onClick={() => handleLoadSample(sampleReal)}
                className="px-2.5 py-1 text-xs font-semibold rounded-lg bg-emerald-950/30 text-emerald-400 hover:bg-emerald-950/60 border border-emerald-900/50 transition"
              >
                Sample Real
              </button>
              <button
                type="button"
                onClick={() => handleLoadSample(sampleFake)}
                className="px-2.5 py-1 text-xs font-semibold rounded-lg bg-rose-950/30 text-rose-400 hover:bg-rose-950/60 border border-rose-900/50 transition"
              >
                Sample Fake
              </button>
            </div>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <label className="text-xs font-bold text-zinc-500 uppercase tracking-wider">Classification Model</label>
              <select
                value={modelType}
                onChange={(e) => setModelType(e.target.value as 'ml' | 'transformer' | 'gemini')}
                className="w-full p-3 rounded-xl border border-zinc-800 bg-zinc-950 text-zinc-200 focus:outline-none focus:border-violet-500 focus:ring-1 focus:ring-violet-500 text-sm transition"
              >
                <option value="ml">Local ML Model (TF-IDF + Logistic Regression)</option>
                <option value="transformer">Transformer Model (RoBERTa)</option>
                <option value="gemini">Google Gemini 1.5 Flash (LLM)</option>
              </select>
            </div>

            <div className="space-y-2">
              <label className="text-xs font-bold text-zinc-500 uppercase tracking-wider">News Content</label>
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Paste news article content or headline here to analyze (minimum 20 characters)..."
                rows={12}
                className="w-full p-4 rounded-xl border border-zinc-800 bg-zinc-950/80 text-zinc-200 placeholder-zinc-650 focus:outline-none focus:border-violet-500 focus:ring-1 focus:ring-violet-500 text-sm font-sans resize-none transition"
              />
            </div>

            {error && (
              <div className="p-4 rounded-xl border border-rose-900/50 bg-rose-950/20 text-rose-400 text-xs flex gap-2 items-start font-medium">
                <AlertTriangle className="w-4 h-4 shrink-0 text-rose-500" />
                <span>{error}</span>
              </div>
            )}

            <div className="flex gap-3">
              <button
                type="button"
                onClick={handleClear}
                disabled={isLoading || !text}
                className="w-1/3 py-3 px-4 rounded-xl border border-zinc-800 bg-zinc-900 text-zinc-400 hover:text-zinc-200 hover:bg-zinc-850 font-bold text-sm disabled:opacity-50 disabled:cursor-not-allowed transition"
              >
                Clear
              </button>
              <button
                type="submit"
                disabled={isLoading || text.trim().length < 20}
                className="flex-grow py-3 px-4 rounded-xl bg-violet-600 hover:bg-violet-500 text-zinc-100 font-bold text-sm flex items-center justify-center gap-2 shadow-lg hover:shadow-violet-950/20 disabled:opacity-50 disabled:cursor-not-allowed transition"
              >
                {isLoading ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    Analyzing Content...
                  </>
                ) : (
                  'Analyze Article'
                )}
              </button>
            </div>
          </form>
        </div>
      </div>

      {/* Results Section */}
      <div className="xl:col-span-7 space-y-6">
        {results ? (
          <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-300">
            {/* Primary Prediction Banner */}
            <div className={`p-6 rounded-2xl border flex flex-col md:flex-row justify-between items-start md:items-center gap-4 backdrop-blur ${
              results.prediction === 'REAL'
                ? 'border-emerald-950/50 bg-emerald-950/10 text-emerald-300'
                : 'border-rose-950/50 bg-rose-950/10 text-rose-300'
            }`}>
              <div className="space-y-1">
                <div className="text-xs font-bold opacity-60 uppercase tracking-widest">Prediction Results</div>
                <div className="flex items-center gap-2">
                  {results.prediction === 'REAL' ? (
                    <CheckCircle className="w-7 h-7 text-emerald-500" />
                  ) : (
                    <AlertTriangle className="w-7 h-7 text-rose-500" />
                  )}
                  <span className="text-3xl font-black font-sans leading-none">{results.prediction} NEWS</span>
                </div>
                <div className="text-xs font-medium text-zinc-400 mt-1 flex items-center gap-1.5">
                  <Info className="w-3.5 h-3.5" />
                  <span>Model: {results.modelUsed}</span>
                </div>
              </div>

              {/* Confidence ring */}
              <div className="flex items-center gap-4">
                <div className="relative w-16 h-16 flex items-center justify-center">
                  <svg className="w-full h-full transform -rotate-90">
                    <circle cx="32" cy="32" r="28" className="stroke-zinc-800" strokeWidth="6" fill="transparent" />
                    <circle
                      cx="32"
                      cy="32"
                      r="28"
                      className={results.prediction === 'REAL' ? 'stroke-emerald-500' : 'stroke-rose-500'}
                      strokeWidth="6"
                      fill="transparent"
                      strokeDasharray={175.9}
                      strokeDashoffset={175.9 - (175.9 * results.confidence)}
                      strokeLinecap="round"
                    />
                  </svg>
                  <span className="absolute text-sm font-extrabold font-mono text-zinc-150">
                    {(results.confidence * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="text-left leading-tight">
                  <div className="text-sm font-bold text-zinc-150">Confidence Score</div>
                  <div className="text-xs font-medium text-zinc-400">Likelihood of classification</div>
                </div>
              </div>
            </div>

            {/* Explainable AI Highlight Block */}
            <div className="p-6 rounded-2xl border border-zinc-800 bg-zinc-900/40 backdrop-blur space-y-4">
              <div>
                <h3 className="text-lg font-bold text-zinc-150 flex items-center gap-2">
                  <BarChart2 className="w-5 h-5 text-indigo-400" />
                  Explainable AI (Word Attribution)
                </h3>
                <p className="text-xs font-semibold text-zinc-500 mt-1">
                  Hover over highlighted words to see their attribution weight. Red highlights indicate evidence for <span className="text-rose-500">FAKE</span> content, and green highlights indicate evidence for <span className="text-emerald-500">REAL</span> content.
                </p>
              </div>
              <ExplainableText attributions={results.attributions} />
            </div>

            {/* Key Features Columns */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Left Column: Metadata */}
              <div className="space-y-6">
                {/* Topic & Keywords */}
                <div className="p-6 rounded-2xl border border-zinc-800 bg-zinc-900/40 backdrop-blur space-y-4">
                  <h3 className="text-sm font-bold text-zinc-150 uppercase tracking-widest flex items-center gap-2">
                    <Tag className="w-4 h-4 text-amber-500" />
                    Classification tags
                  </h3>
                  
                  <div className="space-y-4">
                    <div className="flex items-center gap-4 justify-between">
                      <span className="text-xs font-bold text-zinc-500 uppercase tracking-wider">Identified Topic</span>
                      <span className="px-3 py-1 rounded-full text-xs font-bold bg-zinc-850 text-zinc-200 border border-zinc-800 shadow">
                        {results.topic}
                      </span>
                    </div>

                    <div className="space-y-2">
                      <span className="text-xs font-bold text-zinc-500 uppercase tracking-wider block">Key Extraction Words</span>
                      <div className="flex flex-wrap gap-1.5">
                        {results.keywords.length > 0 ? (
                          results.keywords.map((kw, i) => (
                            <span key={i} className="px-2.5 py-0.5 rounded-lg text-xs font-medium bg-zinc-950/60 text-zinc-300 border border-zinc-850">
                              {kw}
                            </span>
                          ))
                        ) : (
                          <span className="text-zinc-500 text-xs italic">No keywords extracted.</span>
                        )}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Sentiment Analysis */}
                <div className="p-6 rounded-2xl border border-zinc-800 bg-zinc-900/40 backdrop-blur space-y-4">
                  <h3 className="text-sm font-bold text-zinc-150 uppercase tracking-widest flex items-center gap-2">
                    <MessageSquare className="w-4 h-4 text-cyan-500" />
                    Sentiment analysis
                  </h3>

                  <div className="space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-xs font-bold text-zinc-500 uppercase tracking-wider">Emotional Tone</span>
                      <span className={`text-xs font-bold ${
                        results.sentiment.sentiment === 'POSITIVE' ? 'text-emerald-400' :
                        results.sentiment.sentiment === 'NEGATIVE' ? 'text-rose-400' : 'text-zinc-400'
                      }`}>
                        {results.sentiment.sentiment}
                      </span>
                    </div>

                    {/* Bar Chart representation */}
                    <div className="space-y-1.5 pt-1">
                      <div className="h-2 w-full rounded-full bg-zinc-950 overflow-hidden flex">
                        <div style={{ width: `${results.sentiment.pos * 100}%` }} className="bg-emerald-500 h-full" />
                        <div style={{ width: `${results.sentiment.neu * 100}%` }} className="bg-zinc-700 h-full" />
                        <div style={{ width: `${results.sentiment.neg * 100}%` }} className="bg-rose-500 h-full" />
                      </div>
                      <div className="flex justify-between text-[10px] text-zinc-500 font-bold uppercase">
                        <span>Positive ({(results.sentiment.pos * 100).toFixed(0)}%)</span>
                        <span>Neutral ({(results.sentiment.neu * 100).toFixed(0)}%)</span>
                        <span>Negative ({(results.sentiment.neg * 100).toFixed(0)}%)</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Right Column: AI Summary */}
              <div className="p-6 rounded-2xl border border-zinc-800 bg-zinc-900/40 backdrop-blur flex flex-col justify-between">
                <div className="space-y-4">
                  <h3 className="text-sm font-bold text-zinc-150 uppercase tracking-widest flex items-center gap-2">
                    <FileText className="w-4 h-4 text-teal-500" />
                    Extractive AI Summary
                  </h3>
                  <p className="text-sm text-zinc-400 leading-relaxed font-sans font-light italic">
                    &quot;{results.summary}&quot;
                  </p>
                </div>
                <div className="pt-4 border-t border-zinc-850 mt-4 text-[10px] text-zinc-500 font-bold uppercase tracking-wider text-right">
                  Generated via Key-Sentence Attributions
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="h-full min-h-[400px] border border-dashed border-zinc-800 rounded-2xl flex flex-col items-center justify-center p-8 text-center bg-zinc-950/10">
            <FileText className="w-12 h-12 text-zinc-700 mb-4 stroke-1 animate-pulse" />
            <h3 className="text-lg font-bold text-zinc-400 mb-1">Awaiting Content Analysis</h3>
            <p className="text-zinc-500 text-sm max-w-md">
              Pasted articles will trigger standard and Explainable AI analysis including model predictions, sentiment logs, summaries, and topics.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
