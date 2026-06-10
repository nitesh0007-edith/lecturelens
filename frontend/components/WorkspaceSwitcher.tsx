"use client";

import { useState } from "react";
import type { Workspace } from "@/lib/types";

interface Props {
  workspaces: Workspace[];
  active: string;
  onSwitch: (id: string) => void;
  onNew: () => void;
}

export default function WorkspaceSwitcher({ workspaces, active, onSwitch, onNew }: Props) {
  const [open, setOpen] = useState(false);
  const current = workspaces.find((w) => w.workspace_id === active);

  return (
    <div className="relative">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-2 rounded-lg border border-zinc-300 dark:border-zinc-600 bg-white dark:bg-zinc-800 px-3 py-1.5 text-sm text-zinc-700 dark:text-zinc-300 hover:bg-zinc-50 dark:hover:bg-zinc-700 transition-colors"
      >
        <span className="w-2 h-2 rounded-full bg-green-500 shrink-0" />
        <span className="max-w-[140px] truncate font-medium">{current?.name ?? active}</span>
        <svg className={`w-3 h-3 text-zinc-400 transition-transform ${open ? "rotate-180" : ""}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {open && (
        <div className="absolute left-0 top-full mt-1 w-56 rounded-xl border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-800 shadow-lg z-50 overflow-hidden">
          <div className="py-1">
            {workspaces.map((ws) => (
              <button
                key={ws.workspace_id}
                onClick={() => { onSwitch(ws.workspace_id); setOpen(false); }}
                className={`w-full text-left px-4 py-2 text-sm flex items-center gap-2 hover:bg-zinc-50 dark:hover:bg-zinc-700 transition-colors ${
                  ws.workspace_id === active ? "text-indigo-600 dark:text-indigo-400 font-medium" : "text-zinc-700 dark:text-zinc-300"
                }`}
              >
                {ws.workspace_id === active && <span className="w-1.5 h-1.5 rounded-full bg-indigo-500 shrink-0" />}
                <span className="truncate">{ws.name}</span>
                {ws.chunk_count > 0 && (
                  <span className="ml-auto text-xs text-zinc-400">{ws.chunk_count.toLocaleString()} chunks</span>
                )}
              </button>
            ))}
          </div>
          <div className="border-t border-zinc-100 dark:border-zinc-700 py-1">
            <button
              onClick={() => { onNew(); setOpen(false); }}
              className="w-full text-left px-4 py-2 text-sm text-indigo-600 dark:text-indigo-400 hover:bg-zinc-50 dark:hover:bg-zinc-700 transition-colors flex items-center gap-2"
            >
              <span className="text-lg leading-none">+</span> New workspace
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
