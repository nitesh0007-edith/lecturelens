import type {
  AskResponse,
  RecommendResponse,
  QuizResponse,
  ExamResponse,
  Workspace,
  UploadResponse,
  WorkspaceStatus,
  StreamToken,
} from "./types";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(text);
  }
  return res.json() as Promise<T>;
}

export async function ask(
  query: string,
  workspaceId: string,
  filters: { module?: string; week?: number; doc_type?: string } = {}
): Promise<AskResponse> {
  const res = await fetch(`${BASE}/api/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, workspace_id: workspaceId, ...filters }),
  });
  return json<AskResponse>(res);
}

export async function* askStream(
  query: string,
  workspaceId: string,
  filters: { module?: string; week?: number; doc_type?: string } = {}
): AsyncGenerator<StreamToken> {
  const res = await fetch(`${BASE}/api/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, workspace_id: workspaceId, stream: true, ...filters }),
  });
  if (!res.ok || !res.body) {
    throw new Error(await res.text().catch(() => "Stream failed"));
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const lines = buf.split("\n");
    buf = lines.pop() ?? "";
    for (const line of lines) {
      if (line.startsWith("data: ")) {
        try {
          // Backend sends {type, content} shape; normalise to StreamToken
          const raw = JSON.parse(line.slice(6));
          if (raw.type === "token") yield { token: raw.content };
          else if (raw.type === "citations") yield { citations: raw.citations };
          else if (raw.type === "done") yield { done: true };
          else yield raw as StreamToken; // passthrough for any other shape
        } catch {
          // ignore malformed SSE lines
        }
      }
    }
  }
}

export async function recommend(
  q: string,
  workspaceId: string,
  topN = 5,
  module?: string
): Promise<RecommendResponse> {
  const params = new URLSearchParams({ q, workspace_id: workspaceId, top_n: String(topN) });
  if (module) params.set("module", module);
  const res = await fetch(`${BASE}/api/recommend?${params}`);
  return json<RecommendResponse>(res);
}

export async function generateQuiz(
  topic: string,
  workspaceId: string,
  opts: { module?: string; week?: number; n?: number; quiz_type?: string } = {}
): Promise<QuizResponse> {
  const res = await fetch(`${BASE}/api/quiz`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ topic, workspace_id: workspaceId, ...opts }),
  });
  return json<QuizResponse>(res);
}

export async function generateExam(
  topic: string,
  workspaceId: string,
  opts: { module?: string; difficulty?: string; n?: number } = {}
): Promise<ExamResponse> {
  const res = await fetch(`${BASE}/api/exam`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ topic, workspace_id: workspaceId, ...opts }),
  });
  return json<ExamResponse>(res);
}

export async function createWorkspace(name: string): Promise<Workspace> {
  const res = await fetch(`${BASE}/api/workspaces/`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  return json<Workspace>(res);
}

export async function getWorkspace(workspaceId: string): Promise<Workspace> {
  const res = await fetch(`${BASE}/api/workspaces/${workspaceId}`);
  return json<Workspace>(res);
}

export async function uploadFile(
  workspaceId: string,
  token: string,
  file: File
): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/api/workspaces/${workspaceId}/upload`, {
    method: "POST",
    headers: { "x-workspace-token": token },
    body: form,
  });
  return json<UploadResponse>(res);
}

export async function getWorkspaceStatus(workspaceId: string): Promise<WorkspaceStatus> {
  const res = await fetch(`${BASE}/api/workspaces/${workspaceId}/status`);
  return json<WorkspaceStatus>(res);
}
