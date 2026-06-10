"use client";

import { useState } from "react";
import { generateQuiz, generateExam } from "@/lib/api";
import type { MCQQuestion, ShortAnswerQuestion, ExamQuestion } from "@/lib/types";

interface Props {
  workspaceId: string;
}

type Mode = "mcq" | "short_answer" | "exam";

function isMCQ(q: MCQQuestion | ShortAnswerQuestion): q is MCQQuestion {
  return "options" in q;
}

function MCQCard({ q, idx }: { q: MCQQuestion; idx: number }) {
  const [selected, setSelected] = useState<string | null>(null);
  const [revealed, setReveal] = useState(false);

  return (
    <div className="rounded-lg border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-800 p-4 space-y-3">
      <p className="text-sm font-medium text-zinc-800 dark:text-zinc-200">
        <span className="text-zinc-400 mr-1">Q{idx + 1}.</span> {q.question}
      </p>
      <div className="space-y-1.5">
        {q.options.map((opt) => {
          const isCorrect = opt.key === q.answer;
          const isSelected = opt.key === selected;
          let cls = "text-left w-full text-sm px-3 py-2 rounded-lg border transition-colors ";
          if (!revealed) {
            cls += isSelected
              ? "border-indigo-500 bg-indigo-50 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-300"
              : "border-zinc-200 dark:border-zinc-700 hover:bg-zinc-50 dark:hover:bg-zinc-700 text-zinc-700 dark:text-zinc-300";
          } else {
            if (isCorrect) cls += "border-green-500 bg-green-50 dark:bg-green-900/30 text-green-700 dark:text-green-300";
            else if (isSelected) cls += "border-red-400 bg-red-50 dark:bg-red-900/30 text-red-600 dark:text-red-400";
            else cls += "border-zinc-200 dark:border-zinc-700 text-zinc-500 dark:text-zinc-500";
          }
          return (
            <button key={opt.key} className={cls} onClick={() => !revealed && setSelected(opt.key)}>
              <span className="font-semibold mr-2">{opt.key}.</span>{opt.text}
            </button>
          );
        })}
      </div>
      {!revealed ? (
        <button
          onClick={() => setReveal(true)}
          disabled={!selected}
          className="text-xs px-3 py-1 rounded bg-indigo-600 text-white disabled:opacity-40"
        >
          Check answer
        </button>
      ) : (
        <div className="text-xs text-zinc-500 dark:text-zinc-400">
          {q.explanation && <p className="italic">{q.explanation}</p>}
        </div>
      )}
    </div>
  );
}

function ShortAnswerCard({ q, idx }: { q: ShortAnswerQuestion; idx: number }) {
  const [show, setShow] = useState(false);
  return (
    <div className="rounded-lg border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-800 p-4 space-y-2">
      <p className="text-sm font-medium text-zinc-800 dark:text-zinc-200">
        <span className="text-zinc-400 mr-1">Q{idx + 1}.</span> {q.question}
      </p>
      {!show ? (
        <button onClick={() => setShow(true)} className="text-xs px-3 py-1 rounded bg-zinc-100 dark:bg-zinc-700 text-zinc-600 dark:text-zinc-300 hover:bg-zinc-200 dark:hover:bg-zinc-600">
          Show answer
        </button>
      ) : (
        <p className="text-sm text-zinc-600 dark:text-zinc-400 bg-zinc-50 dark:bg-zinc-700/50 rounded p-2">{q.answer}</p>
      )}
    </div>
  );
}

function ExamCard({ q, idx }: { q: ExamQuestion; idx: number }) {
  const [show, setShow] = useState(false);
  return (
    <div className="rounded-lg border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-800 p-4 space-y-2">
      <div className="flex justify-between items-start gap-2">
        <p className="text-sm font-medium text-zinc-800 dark:text-zinc-200">
          <span className="text-zinc-400 mr-1">Q{idx + 1}.</span> {q.question}
        </p>
        {q.marks && (
          <span className="shrink-0 text-xs text-zinc-500 dark:text-zinc-400">[{q.marks} marks]</span>
        )}
      </div>
      {q.guidance && (
        !show ? (
          <button onClick={() => setShow(true)} className="text-xs px-3 py-1 rounded bg-zinc-100 dark:bg-zinc-700 text-zinc-600 dark:text-zinc-300 hover:bg-zinc-200 dark:hover:bg-zinc-600">
            Show guidance
          </button>
        ) : (
          <p className="text-sm text-zinc-500 dark:text-zinc-400 bg-zinc-50 dark:bg-zinc-700/50 rounded p-2 italic">{q.guidance}</p>
        )
      )}
    </div>
  );
}

const MODULES = ["IR", "DeepLearning", "MLAI", "TextasData", "BigData", "CyberSec", "IDSS", "IV", "ProgSD", "RPS"];

export default function QuizTab({ workspaceId }: Props) {
  const [topic, setTopic] = useState("");
  const [mode, setMode] = useState<Mode>("mcq");
  const [module, setModule] = useState("");
  const [difficulty, setDifficulty] = useState("medium");
  const [n, setN] = useState(5);
  const [questions, setQuestions] = useState<(MCQQuestion | ShortAnswerQuestion | ExamQuestion)[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function generate() {
    const t = topic.trim();
    if (!t || loading) return;
    setLoading(true);
    setError("");
    setQuestions([]);
    try {
      if (mode === "exam") {
        const res = await generateExam(t, workspaceId, { module: module || undefined, difficulty, n });
        setQuestions(res.questions);
      } else {
        const res = await generateQuiz(t, workspaceId, { module: module || undefined, n, quiz_type: mode });
        setQuestions(res.questions);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Generation failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="p-4 space-y-4">
      <div className="space-y-3">
        <div className="flex gap-1 p-1 bg-zinc-100 dark:bg-zinc-800 rounded-lg text-sm">
          {(["mcq", "short_answer", "exam"] as Mode[]).map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={`flex-1 py-1.5 rounded-md font-medium transition-colors ${
                mode === m
                  ? "bg-white dark:bg-zinc-700 text-zinc-900 dark:text-white shadow-sm"
                  : "text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300"
              }`}
            >
              {m === "mcq" ? "MCQ" : m === "short_answer" ? "Short Answer" : "Exam Style"}
            </button>
          ))}
        </div>

        <input
          value={topic}
          onChange={(e) => setTopic(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && generate()}
          placeholder="Topic, e.g. transformer architecture, SQL joins…"
          className="w-full rounded-lg border border-zinc-300 dark:border-zinc-600 bg-zinc-50 dark:bg-zinc-800 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 dark:text-zinc-200"
        />

        <div className="flex gap-2 flex-wrap">
          <select
            value={module}
            onChange={(e) => setModule(e.target.value)}
            className="rounded-lg border border-zinc-300 dark:border-zinc-600 bg-zinc-50 dark:bg-zinc-800 px-3 py-2 text-sm focus:outline-none dark:text-zinc-200"
          >
            <option value="">All modules</option>
            {MODULES.map((m) => <option key={m} value={m}>{m}</option>)}
          </select>

          {mode === "exam" && (
            <select
              value={difficulty}
              onChange={(e) => setDifficulty(e.target.value)}
              className="rounded-lg border border-zinc-300 dark:border-zinc-600 bg-zinc-50 dark:bg-zinc-800 px-3 py-2 text-sm focus:outline-none dark:text-zinc-200"
            >
              <option value="easy">Easy</option>
              <option value="medium">Medium</option>
              <option value="hard">Hard</option>
            </select>
          )}

          <select
            value={n}
            onChange={(e) => setN(Number(e.target.value))}
            className="rounded-lg border border-zinc-300 dark:border-zinc-600 bg-zinc-50 dark:bg-zinc-800 px-3 py-2 text-sm focus:outline-none dark:text-zinc-200"
          >
            {[3, 5, 8, 10].map((v) => <option key={v} value={v}>{v} questions</option>)}
          </select>

          <button
            onClick={generate}
            disabled={loading || !topic.trim()}
            className="px-5 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-700 disabled:opacity-40 text-white text-sm font-medium transition-colors"
          >
            {loading ? "Generating…" : "Generate"}
          </button>
        </div>
        {error && <p className="text-sm text-red-500">{error}</p>}
      </div>

      {questions.length > 0 && (
        <div className="space-y-3">
          {questions.map((q, i) =>
            mode === "exam" ? (
              <ExamCard key={i} q={q as ExamQuestion} idx={i} />
            ) : isMCQ(q as MCQQuestion | ShortAnswerQuestion) ? (
              <MCQCard key={i} q={q as MCQQuestion} idx={i} />
            ) : (
              <ShortAnswerCard key={i} q={q as ShortAnswerQuestion} idx={i} />
            )
          )}
        </div>
      )}
    </div>
  );
}
