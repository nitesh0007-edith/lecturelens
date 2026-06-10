"""
Exam-question generation styled on real past-paper chunks.
Injects past_paper chunks as style exemplars so the LLM mimics
the actual exam question format the professor uses.
"""

from __future__ import annotations

import json

from app.config import settings
from app.retrieval.retriever import retrieve


EXAM_SYSTEM_PROMPT = """You are generating exam questions for a university MSc Data Science course.
Match the style, difficulty, and structure of the provided past exam exemplars exactly.
Return ONLY valid JSON — no markdown, no extra text.
"""

EXAM_USER_PROMPT = """Generate {n} exam question(s) at difficulty level '{difficulty}' on the topic: "{topic}".

PAST EXAM EXEMPLARS (match this style):
{exemplars}

LECTURE MATERIAL (base answers on this):
{context}

Return as JSON array:
[{{"question": ..., "marks": ..., "difficulty": ..., "model_answer": ..., "hint": ...}}]
"""


def generate_exam_questions(
    topic: str,
    workspace_id: str,
    module: str | None = None,
    difficulty: str = "medium",
    n: int = 3,
) -> list[dict]:
    """
    Generate exam questions grounded in retrieved lecture chunks,
    styled on past-paper chunks from the same module.
    """
    # Retrieve topic chunks
    filters = {}
    if module:
        filters["module"] = module
    topic_chunks = retrieve(
        query=topic, workspace_id=workspace_id, filters=filters, rerank_top_k=6
    )

    # Retrieve past-paper exemplars
    past_paper_filters = {**filters, "doc_type": "past_paper"}
    exemplar_chunks = retrieve(
        query=topic, workspace_id=workspace_id, filters=past_paper_filters, rerank_top_k=4,
        use_rerank=False,
    )

    exemplars_text = "\n\n".join(
        f"---\n{c.get('text', '')}" for c in exemplar_chunks[:4]
    ) or "No past paper exemplars available."

    context_text = "\n\n".join(
        f"[{c.get('module', '')} Week {c.get('week', '?')}]\n{c.get('text', '')}"
        for c in topic_chunks
    )

    prompt = EXAM_USER_PROMPT.format(
        n=n,
        difficulty=difficulty,
        topic=topic,
        exemplars=exemplars_text[:3000],
        context=context_text[:4000],
    )

    try:
        from groq import Groq

        client = Groq(api_key=settings.groq_api_key)
        resp = client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": EXAM_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=2048,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
        for v in parsed.values():
            if isinstance(v, list):
                return v
        return []
    except Exception:
        return []
