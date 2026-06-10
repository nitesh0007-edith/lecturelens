"use client";

import { useState } from "react";
import { recommend } from "@/lib/api";
import type { Recommendation } from "@/lib/types";

interface Props {
  workspaceId: string;
}

export default function RecommendTab({ workspaceId }: Props) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<Recommendation[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function search() {
    const q = query.trim();
    if (!q || loading) return;
    setLoading(true);
    setError("");
    try {
      const res = await recommend(q, workspaceId, 8);
      setResults(res.recommendations);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to fetch recommendations");
    } finally {
      setLoading(false);
    }
  }

  const moduleColors: Record<string, string> = {
    IR: "bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300",
    DeepLearning: "bg-purple-100 text-purple-700 dark:bg-purple-900 dark:text-purple-300",
    MLAI: "bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300",
    TextasData: "bg-orange-100 text-orange-700 dark:bg-orange-900 dark:text-orange-300",
    BigData: "bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300",
    CyberSec: "bg-yellow-100 text-yellow-700 dark:bg-yellow-900 dark:text-yellow-300",
  };

  function moduleColor(mod: string) {
    return moduleColors[mod] ?? "bg-zinc-100 text-zinc-700 dark:bg-zinc-700 dark:text-zinc-300";
  }

  return (
    <div className="p-4 space-y-4">
      <div>
        <p className="text-sm text-zinc-500 dark:text-zinc-400 mb-3">
          Find the most relevant lectures for any topic across all modules.
        </p>
        <div className="flex gap-2">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && search()}
            placeholder="e.g. attention mechanisms, gradient descent…"
            className="flex-1 rounded-lg border border-zinc-300 dark:border-zinc-600 bg-zinc-50 dark:bg-zinc-800 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 dark:text-zinc-200"
          />
          <button
            onClick={search}
            disabled={loading || !query.trim()}
            className="px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-700 disabled:opacity-40 text-white text-sm font-medium transition-colors"
          >
            {loading ? "…" : "Find"}
          </button>
        </div>
        {error && <p className="mt-2 text-sm text-red-500">{error}</p>}
      </div>

      {results.length > 0 && (
        <div className="space-y-2">
          {results.map((r, i) => (
            <div
              key={i}
              className="rounded-lg border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-800 px-4 py-3"
            >
              <div className="flex items-center gap-2 flex-wrap">
                <span className={`text-xs font-semibold px-2 py-0.5 rounded ${moduleColor(r.module)}`}>
                  {r.module}
                </span>
                {r.week && (
                  <span className="text-xs text-zinc-500 dark:text-zinc-400">Week {r.week}</span>
                )}
                <span className="ml-auto text-xs text-zinc-400">
                  {r.chunk_hits} matching chunks
                </span>
              </div>
              {r.sample_titles.length > 0 && (
                <ul className="mt-2 space-y-0.5">
                  {r.sample_titles.map((t, j) => (
                    <li key={j} className="text-sm text-zinc-600 dark:text-zinc-400 truncate">
                      · {t}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
