"""
LLM-as-judge generation evaluation over a 40-question subset.

Metrics (scored by Gemini Flash):
  - Faithfulness: every claim supported by a cited chunk
  - Answer relevance: answer addresses the question

Also reports abstention rate on 10 deliberately unanswerable questions.

Usage:
    PYTHONPATH=backend python eval/eval_generation.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

QUESTIONS_PATH = Path(__file__).parent / "queries.jsonl"
RESULTS_PATH = Path(__file__).parent / "generation_results.md"
UNANSWERABLE_TAG = "unanswerable"


def load_questions(path: Path) -> list[dict]:
    questions = []
    with path.open() as f:
        for line in f:
            d = json.loads(line.strip())
            questions.append(d)
    return questions


def evaluate_with_llm_judge(question: str, answer: str, citations: list[dict]) -> dict:
    """Score faithfulness + relevance using Gemini Flash as judge."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
    from app.config import settings

    context_snippets = "\n".join(
        f"[{c['index']}] {c.get('text_snippet', '')}" for c in citations
    )

    faithfulness_prompt = f"""You are evaluating whether an AI answer is faithful to its cited sources.

Question: {question}
Answer: {answer}
Cited sources:
{context_snippets}

Score faithfulness from 0.0 to 1.0 (1.0 = every claim fully supported by citations, 0.0 = major unsupported claims).
Return JSON: {{"faithfulness": float, "reason": str}}"""

    relevance_prompt = f"""Score how well this answer addresses the question.

Question: {question}
Answer: {answer}

Score from 0.0 to 1.0 (1.0 = fully answers the question, 0.0 = completely misses it).
Return JSON: {{"relevance": float, "reason": str}}"""

    try:
        import google.generativeai as genai

        genai.configure(api_key=settings.gemini_api_key)
        model = genai.GenerativeModel(
            settings.gemini_model,
            generation_config=genai.GenerationConfig(
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )

        f_resp = model.generate_content(faithfulness_prompt)
        r_resp = model.generate_content(relevance_prompt)

        f_result = json.loads(f_resp.text)
        r_result = json.loads(r_resp.text)

        return {
            "faithfulness": f_result.get("faithfulness", 0.0),
            "faithfulness_reason": f_result.get("reason", ""),
            "relevance": r_result.get("relevance", 0.0),
            "relevance_reason": r_result.get("reason", ""),
        }
    except Exception as e:
        return {"faithfulness": -1.0, "relevance": -1.0, "error": str(e)}


def run_generation_eval(questions: list[dict], workspace_id: str = "uofg-msds-demo"):
    sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
    from app.generation.answer import generate_answer
    from app.retrieval.retriever import retrieve

    results = []
    abstentions = 0
    answerable_count = 0
    unanswerable_count = 0

    for i, q in enumerate(questions[:50]):
        qid = q.get("qid", str(i))
        query = q.get("query", "")
        is_unanswerable = q.get("tag") == UNANSWERABLE_TAG

        print(f"  [{i+1}/{len(questions[:50])}] {qid}: {query[:60]}...")

        chunks = retrieve(query, workspace_id)
        result = generate_answer(query, chunks)
        answer = result.get("answer", "")
        citations = result.get("citations", [])

        abstained = "couldn't find" in answer.lower() or "not enough information" in answer.lower()

        if is_unanswerable:
            unanswerable_count += 1
            if abstained:
                abstentions += 1
            scores = {"faithfulness": None, "relevance": None}
        else:
            answerable_count += 1
            if not abstained and citations:
                scores = evaluate_with_llm_judge(query, answer, citations)
            else:
                scores = {"faithfulness": 0.0, "relevance": 0.0}

        results.append({
            "qid": qid,
            "query": query,
            "unanswerable": is_unanswerable,
            "abstained": abstained,
            "answer_length": len(answer),
            **scores,
        })

        time.sleep(0.5)  # Rate limit

    abstention_rate = abstentions / unanswerable_count if unanswerable_count else 0.0

    answerable_results = [r for r in results if not r["unanswerable"] and r.get("faithfulness") is not None and r["faithfulness"] >= 0]
    avg_faithfulness = sum(r["faithfulness"] for r in answerable_results) / len(answerable_results) if answerable_results else 0.0
    avg_relevance = sum(r["relevance"] for r in answerable_results) / len(answerable_results) if answerable_results else 0.0

    summary = {
        "total_questions": len(results),
        "answerable": answerable_count,
        "unanswerable": unanswerable_count,
        "abstention_rate": round(abstention_rate, 3),
        "avg_faithfulness": round(avg_faithfulness, 3),
        "avg_relevance": round(avg_relevance, 3),
    }

    return results, summary


def write_results_md(results: list[dict], summary: dict, path: Path):
    lines = [
        "# LectureLens Generation Quality Evaluation",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        "|--------|-------|",
        f"| Total questions | {summary['total_questions']} |",
        f"| Answerable | {summary['answerable']} |",
        f"| Unanswerable | {summary['unanswerable']} |",
        f"| Abstention rate (on unanswerables) | {summary['abstention_rate']:.1%} |",
        f"| Avg faithfulness | {summary['avg_faithfulness']:.3f} |",
        f"| Avg answer relevance | {summary['avg_relevance']:.3f} |",
        "",
        "## Per-question results",
        "",
        "| QID | Unanswerable | Abstained | Faithfulness | Relevance |",
        "|-----|-------------|-----------|-------------|-----------|",
    ]
    for r in results:
        lines.append(
            f"| {r['qid']} | {r['unanswerable']} | {r['abstained']} "
            f"| {r.get('faithfulness', 'N/A')} | {r.get('relevance', 'N/A')} |"
        )
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    if not QUESTIONS_PATH.exists():
        print(f"queries file not found: {QUESTIONS_PATH}")
        sys.exit(1)

    questions = load_questions(QUESTIONS_PATH)
    print(f"Evaluating generation on {min(50, len(questions))} questions...")
    results, summary = run_generation_eval(questions)
    write_results_md(results, summary, RESULTS_PATH)
    print(f"\nSummary: {summary}")
    print(f"Results written to {RESULTS_PATH}")
