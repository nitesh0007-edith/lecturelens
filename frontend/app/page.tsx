"use client";

import { useState } from "react";
import ChatView from "@/components/ChatView";
import QuizTab from "@/components/QuizTab";
import RecommendTab from "@/components/RecommendTab";
import WorkspaceSwitcher from "@/components/WorkspaceSwitcher";
import UploadModal from "@/components/UploadModal";
import type { Workspace } from "@/lib/types";

const MODULES = ["", "IR", "DeepLearning", "MLAI", "TextasData", "BigData", "CyberSec", "IDSS", "IV", "ProgSD", "RPS"];
const WEEKS = ["", ...Array.from({ length: 12 }, (_, i) => String(i + 1))];

const DEMO_WORKSPACE: Workspace = {
  workspace_id: "uofg-msds-demo",
  name: "UofG MSc DS Demo",
  token: "",
  doc_count: 0,
  chunk_count: 13916,
};

type Tab = "chat" | "quiz" | "recommend";

export default function Home() {
  const [workspaces, setWorkspaces] = useState<Workspace[]>([DEMO_WORKSPACE]);
  const [activeWs, setActiveWs] = useState("uofg-msds-demo");
  const [tab, setTab] = useState<Tab>("chat");
  const [showUpload, setShowUpload] = useState(false);
  const [filterModule, setFilterModule] = useState("");
  const [filterWeek, setFilterWeek] = useState("");

  function handleCreated(ws: Workspace) {
    setWorkspaces((prev) => [...prev, ws]);
    setActiveWs(ws.workspace_id);
  }

  const filters: { module?: string; week?: number } = {};
  if (filterModule) filters.module = filterModule;
  if (filterWeek) filters.week = parseInt(filterWeek);

  const tabs: { id: Tab; label: string; icon: string }[] = [
    { id: "chat", label: "Ask", icon: "💬" },
    { id: "recommend", label: "Recommend", icon: "📚" },
    { id: "quiz", label: "Quiz", icon: "🧠" },
  ];

  return (
    <div className="flex flex-col h-screen bg-zinc-50 dark:bg-zinc-950">
      {/* Header */}
      <header className="shrink-0 border-b border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 px-4 py-3 flex items-center gap-3">
        <div className="flex items-center gap-2 mr-2">
          <div className="w-7 h-7 rounded-lg bg-indigo-600 flex items-center justify-center text-white text-xs font-bold select-none">
            LL
          </div>
          <span className="font-semibold text-zinc-900 dark:text-zinc-100 hidden sm:block">LectureLens</span>
        </div>

        <WorkspaceSwitcher
          workspaces={workspaces}
          active={activeWs}
          onSwitch={setActiveWs}
          onNew={() => setShowUpload(true)}
        />

        {/* Filter chips — only show in chat tab */}
        {tab === "chat" && (
          <div className="flex gap-2 ml-auto items-center">
            <select
              value={filterModule}
              onChange={(e) => setFilterModule(e.target.value)}
              className="text-xs rounded-lg border border-zinc-200 dark:border-zinc-700 bg-zinc-50 dark:bg-zinc-800 px-2 py-1 focus:outline-none dark:text-zinc-300"
            >
              <option value="">All modules</option>
              {MODULES.filter(Boolean).map((m) => <option key={m} value={m}>{m}</option>)}
            </select>
            <select
              value={filterWeek}
              onChange={(e) => setFilterWeek(e.target.value)}
              className="text-xs rounded-lg border border-zinc-200 dark:border-zinc-700 bg-zinc-50 dark:bg-zinc-800 px-2 py-1 focus:outline-none dark:text-zinc-300"
            >
              <option value="">All weeks</option>
              {WEEKS.filter(Boolean).map((w) => <option key={w} value={w}>Wk {w}</option>)}
            </select>
          </div>
        )}
      </header>

      {/* Tab bar */}
      <nav className="shrink-0 border-b border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 flex px-4">
        {tabs.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`flex items-center gap-1.5 px-4 py-2.5 text-sm font-medium border-b-2 transition-colors ${
              tab === t.id
                ? "border-indigo-600 text-indigo-600 dark:text-indigo-400"
                : "border-transparent text-zinc-500 dark:text-zinc-400 hover:text-zinc-700 dark:hover:text-zinc-300"
            }`}
          >
            <span>{t.icon}</span>
            <span>{t.label}</span>
          </button>
        ))}
      </nav>

      {/* Main content */}
      <main className="flex-1 overflow-hidden">
        <div className={`h-full ${tab === "chat" ? "block" : "hidden"}`}>
          <ChatView key={activeWs} workspaceId={activeWs} filters={filters} />
        </div>
        <div className={`h-full overflow-y-auto ${tab === "recommend" ? "block" : "hidden"}`}>
          <div className="max-w-2xl mx-auto">
            <RecommendTab workspaceId={activeWs} />
          </div>
        </div>
        <div className={`h-full overflow-y-auto ${tab === "quiz" ? "block" : "hidden"}`}>
          <div className="max-w-2xl mx-auto">
            <QuizTab workspaceId={activeWs} />
          </div>
        </div>
      </main>

      {showUpload && (
        <UploadModal onClose={() => setShowUpload(false)} onCreated={handleCreated} />
      )}
    </div>
  );
}
