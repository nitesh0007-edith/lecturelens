"""
MCQ / short-answer quiz generation grounded in retrieved chunks.
Uses Groq JSON mode so output is always parseable.
"""

from __future__ import annotations

import json

from app.config import settings

QUIZ_SYSTEM_PROMPT = """You are an exam question generator for university-level Data Science courses.
Generate questions grounded ONLY in the provided lecture material.
Return ONLY valid JSON — no markdown, no extra text.
"""

MCQ_USER_PROMPT = """Generate {n} multiple-choice questions from the lecture material below.

For each question:
- "question": clear, specific question string
- "options": dict with keys "A", "B", "C", "D"
- "answer": correct option key ("A"/"B"/"C"/"D")
- "explanation": one-sentence explanation citing the source

Return as JSON array: [{{"question":..., "options":..., "answer":..., "explanation":...}}, ...]

Lecture material:
{context}
"""

SHORT_ANSWER_PROMPT = """Generate {n} short-answer questions from the lecture material below.

For each question:
- "question": question string
- "model_answer": 1-3 sentence answer grounded in the material
- "marks": suggested marks (1-5)

Return as JSON array: [{{"question":..., "model_answer":..., "marks":...}}, ...]

Lecture material:
{context}
"""


def _call_groq_json(system: str, user: str) -> list:
    try:
        from groq import Groq

        client = Groq(api_key=settings.groq_api_key)
        resp = client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
            max_tokens=2048,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content
        parsed = json.loads(raw)
        # Handle both {"questions": [...]} and bare [...]
        if isinstance(parsed, list):
            return parsed
        for v in parsed.values():
            if isinstance(v, list):
                return v
        return []
    except Exception:
        return []


def generate_mcqs(chunks: list[dict], n: int = 5) -> list[dict]:
    context = "\n\n".join(
        f"[{c.get('module', '')} Week {c.get('week', '?')} p.{c.get('page_or_slide', '?')}]\n{c.get('text', '')}"
        for c in chunks
    )
    prompt = MCQ_USER_PROMPT.format(n=n, context=context[:6000])
    return _call_groq_json(QUIZ_SYSTEM_PROMPT, prompt)


def generate_short_answers(chunks: list[dict], n: int = 5) -> list[dict]:
    context = "\n\n".join(
        f"[{c.get('module', '')} Week {c.get('week', '?')}]\n{c.get('text', '')}"
        for c in chunks
    )
    prompt = SHORT_ANSWER_PROMPT.format(n=n, context=context[:6000])
    return _call_groq_json(QUIZ_SYSTEM_PROMPT, prompt)
