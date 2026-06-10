"""
Structured JSON trace logging — Phase 7.

Every request gets a trace_id. Logs: query → rewritten query →
fused candidates (ids+scores) → reranked top-6 → answer →
cited chunk ids → latency per stage.

Output: JSON lines to stdout (grep-able) or Langfuse if configured.
Interview answer to "how do you know it works in production?".
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("lecturelens.trace")


@dataclass
class Trace:
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    query: str = ""
    workspace_id: str = ""
    stages: list[dict[str, Any]] = field(default_factory=list)
    _stage_start: float = field(default_factory=time.time, repr=False)

    def start_stage(self, name: str) -> None:
        self._stage_start = time.time()

    def end_stage(self, name: str, **data) -> None:
        elapsed_ms = round((time.time() - self._stage_start) * 1000, 1)
        self.stages.append({"stage": name, "latency_ms": elapsed_ms, **data})

    def emit(self, answer_length: int = 0, cached: bool = False) -> None:
        total_ms = sum(s.get("latency_ms", 0) for s in self.stages)
        record = {
            "trace_id": self.trace_id,
            "query": self.query,
            "workspace_id": self.workspace_id,
            "cached": cached,
            "answer_length": answer_length,
            "total_latency_ms": round(total_ms, 1),
            "stages": self.stages,
        }
        logger.info(json.dumps(record))


@contextmanager
def traced_stage(trace: Trace, name: str, **meta):
    """Context manager that times a stage and logs it."""
    trace.start_stage(name)
    try:
        yield trace
    finally:
        trace.end_stage(name, **meta)


def setup_trace_logging():
    """Call once at app startup to configure JSON log format."""
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
