"use client";

import { useState } from "react";
import type { Citation } from "@/lib/types";

interface Props {
  citations: Citation[];
}

export default function SourcesPanel({ citations }: Props) {
  const [expanded, setExpanded] = useState<number | null>(null);

  if (!citations.length) return null;

  return (
    <div className="mt-3 border border-zinc-200 dark:border-zinc-700 rounded-lg overflow-hidden text-sm">
      <div className="bg-zinc-50 dark:bg-zinc-800 px-3 py-2 font-medium text-zinc-600 dark:text-zinc-400 text-xs uppercase tracking-wide">
        Sources ({citations.length})
      </div>
      <div className="divide-y divide-zinc-100 dark:divide-zinc-800">
        {citations.map((c) => (
          <div key={c.index}>
            <button
              className="w-full text-left px-3 py-2 hover:bg-zinc-50 dark:hover:bg-zinc-800 transition-colors flex items-start gap-2"
              onClick={() => setExpanded(expanded === c.index ? null : c.index)}
            >
              <span className="shrink-0 inline-flex items-center justify-center w-5 h-5 rounded bg-indigo-100 dark:bg-indigo-900 text-indigo-700 dark:text-indigo-300 text-xs font-bold">
                {c.index}
              </span>
              <div className="min-w-0 flex-1">
                <span className="font-medium text-zinc-800 dark:text-zinc-200">
                  {c.module}
                  {c.week ? ` · Week ${c.week}` : ""}
                  {c.page_or_slide ? ` · slide ${c.page_or_slide}` : ""}
                </span>
                {c.title && (
                  <span className="ml-1 text-zinc-500 dark:text-zinc-400 truncate">
                    — {c.title}
                  </span>
                )}
              </div>
              <span className="shrink-0 text-zinc-400 text-xs">{expanded === c.index ? "▲" : "▼"}</span>
            </button>
            {expanded === c.index && (
              <div className="px-3 pb-3 pt-1 bg-zinc-50 dark:bg-zinc-800/50">
                <p className="text-zinc-600 dark:text-zinc-400 text-xs leading-relaxed line-clamp-6">
                  {c.text}
                </p>
                <p className="mt-1 text-zinc-400 dark:text-zinc-500 text-xs truncate">
                  {c.source_file}
                </p>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
