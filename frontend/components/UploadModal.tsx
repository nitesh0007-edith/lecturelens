"use client";

import { useEffect, useRef, useState } from "react";
import { createWorkspace, uploadFile, getWorkspaceStatus } from "@/lib/api";
import type { Workspace, DocumentStatus } from "@/lib/types";

interface Props {
  onClose: () => void;
  onCreated: (ws: Workspace) => void;
}

export default function UploadModal({ onClose, onCreated }: Props) {
  const [step, setStep] = useState<"create" | "upload" | "ingesting">("create");
  const [name, setName] = useState("");
  const [workspace, setWorkspace] = useState<Workspace | null>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [docs, setDocs] = useState<DocumentStatus[]>([]);
  const [error, setError] = useState("");
  const [creating, setCreating] = useState(false);
  const [uploading, setUploading] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current); }, []);

  async function handleCreate() {
    if (!name.trim()) return;
    setCreating(true);
    setError("");
    try {
      const ws = await createWorkspace(name.trim());
      setWorkspace(ws);
      setStep("upload");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to create workspace");
    } finally {
      setCreating(false);
    }
  }

  async function handleUpload() {
    if (!workspace || !files.length) return;
    setUploading(true);
    setError("");
    try {
      for (const file of files) {
        await uploadFile(workspace.workspace_id, workspace.token, file);
      }
      setStep("ingesting");
      pollRef.current = setInterval(async () => {
        const status = await getWorkspaceStatus(workspace.workspace_id).catch(() => null);
        if (!status) return;
        setDocs(status.documents);
        const allDone = status.documents.every((d) => d.status === "done" || d.status === "error");
        if (allDone && pollRef.current) {
          clearInterval(pollRef.current);
          pollRef.current = null;
        }
      }, 2000);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }

  function handleDone() {
    if (workspace) onCreated(workspace);
    onClose();
  }

  const statusIcon: Record<string, string> = {
    pending: "⏳",
    parsing: "📄",
    indexing_sparse: "🔍",
    indexing_dense: "🧠",
    done: "✅",
    error: "❌",
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm p-4">
      <div className="w-full max-w-md bg-white dark:bg-zinc-900 rounded-2xl shadow-xl p-6 space-y-5">
        <div className="flex justify-between items-center">
          <h2 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100">
            {step === "create" ? "New Workspace" : step === "upload" ? "Upload Files" : "Ingesting…"}
          </h2>
          <button onClick={onClose} className="text-zinc-400 hover:text-zinc-600 dark:hover:text-zinc-200 text-xl leading-none">×</button>
        </div>

        {step === "create" && (
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-zinc-700 dark:text-zinc-300 mb-1">
                Workspace name
              </label>
              <input
                value={name}
                onChange={(e) => setName(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleCreate()}
                placeholder="e.g. my-course-notes"
                className="w-full rounded-lg border border-zinc-300 dark:border-zinc-600 bg-zinc-50 dark:bg-zinc-800 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 dark:text-zinc-200"
              />
            </div>
            {error && <p className="text-sm text-red-500">{error}</p>}
            <button
              onClick={handleCreate}
              disabled={creating || !name.trim()}
              className="w-full py-2 rounded-lg bg-indigo-600 hover:bg-indigo-700 disabled:opacity-40 text-white text-sm font-medium transition-colors"
            >
              {creating ? "Creating…" : "Create workspace"}
            </button>
          </div>
        )}

        {step === "upload" && workspace && (
          <div className="space-y-4">
            <div className="rounded-lg bg-zinc-50 dark:bg-zinc-800 px-4 py-3 text-sm space-y-1">
              <p className="text-zinc-500 dark:text-zinc-400">ID: <span className="font-mono text-zinc-700 dark:text-zinc-300">{workspace.workspace_id}</span></p>
              <p className="text-zinc-500 dark:text-zinc-400">Token: <span className="font-mono text-xs text-zinc-600 dark:text-zinc-400 break-all">{workspace.token}</span></p>
              <p className="text-xs text-amber-600 dark:text-amber-400">Save your token — you'll need it to upload more files.</p>
            </div>

            <div>
              <input
                ref={fileRef}
                type="file"
                multiple
                accept=".pdf,.pptx,.ipynb,.html,.htm"
                className="hidden"
                onChange={(e) => setFiles(Array.from(e.target.files ?? []))}
              />
              <button
                onClick={() => fileRef.current?.click()}
                className="w-full py-8 rounded-lg border-2 border-dashed border-zinc-300 dark:border-zinc-600 text-sm text-zinc-500 dark:text-zinc-400 hover:border-indigo-400 dark:hover:border-indigo-500 transition-colors"
              >
                {files.length
                  ? `${files.length} file${files.length > 1 ? "s" : ""} selected`
                  : "Click to select PDF, PPTX, IPYNB, or HTML files"}
              </button>
            </div>

            {files.length > 0 && (
              <ul className="space-y-1 text-xs text-zinc-600 dark:text-zinc-400">
                {files.map((f) => <li key={f.name} className="truncate">· {f.name}</li>)}
              </ul>
            )}

            {error && <p className="text-sm text-red-500">{error}</p>}
            <button
              onClick={handleUpload}
              disabled={uploading || !files.length}
              className="w-full py-2 rounded-lg bg-indigo-600 hover:bg-indigo-700 disabled:opacity-40 text-white text-sm font-medium transition-colors"
            >
              {uploading ? "Uploading…" : "Upload & ingest"}
            </button>
          </div>
        )}

        {step === "ingesting" && (
          <div className="space-y-4">
            <p className="text-sm text-zinc-500 dark:text-zinc-400">
              Processing your files. This may take a few minutes.
            </p>
            <div className="space-y-2">
              {docs.map((d) => (
                <div key={d.id} className="flex items-center gap-3 text-sm">
                  <span>{statusIcon[d.status] ?? "⏳"}</span>
                  <span className="flex-1 truncate text-zinc-700 dark:text-zinc-300">{d.filename}</span>
                  <span className="text-zinc-400 capitalize">{d.status}</span>
                  {d.chunks > 0 && <span className="text-zinc-400 text-xs">{d.chunks} chunks</span>}
                </div>
              ))}
            </div>
            {docs.length > 0 && docs.every((d) => d.status === "done" || d.status === "error") && (
              <button
                onClick={handleDone}
                className="w-full py-2 rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white text-sm font-medium transition-colors"
              >
                Start chatting →
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
