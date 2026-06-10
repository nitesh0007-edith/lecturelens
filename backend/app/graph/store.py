"""
Concept graph: NetworkX + SQLite persistence.
Exposed as /api/graph endpoints for Cytoscape.js rendering.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import networkx as nx

from app.config import settings

_DB_PATH = settings.sqlite_db.parent / "graph.db"


def _get_conn() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH))
    conn.execute(
        """CREATE TABLE IF NOT EXISTS triples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workspace_id TEXT,
            subject TEXT,
            predicate TEXT,
            object TEXT,
            source_module TEXT,
            source_week INTEGER
        )"""
    )
    conn.commit()
    return conn


def store_triples(workspace_id: str, triples: list[dict]) -> int:
    conn = _get_conn()
    rows = [
        (
            workspace_id,
            t.get("subject", ""),
            t.get("predicate", ""),
            t.get("object", ""),
            t.get("source_module", ""),
            t.get("source_week"),
        )
        for t in triples
        if t.get("subject") and t.get("object")
    ]
    conn.executemany(
        "INSERT INTO triples (workspace_id, subject, predicate, object, source_module, source_week) VALUES (?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()
    return len(rows)


def get_graph_data(workspace_id: str, module: str | None = None) -> dict:
    """Return Cytoscape.js-compatible {nodes, edges} data."""
    conn = _get_conn()
    query = "SELECT subject, predicate, object, source_module FROM triples WHERE workspace_id=?"
    params: list = [workspace_id]
    if module:
        query += " AND source_module=?"
        params.append(module)

    rows = conn.execute(query, params).fetchall()
    conn.close()

    G = nx.DiGraph()
    for subject, predicate, obj, source_module in rows:
        G.add_node(subject, module=source_module)
        G.add_node(obj, module=source_module)
        G.add_edge(subject, obj, label=predicate)

    nodes = [{"data": {"id": n, "label": n, **G.nodes[n]}} for n in G.nodes]
    edges = [
        {"data": {"source": u, "target": v, "label": G.edges[u, v].get("label", "")}}
        for u, v in G.edges
    ]
    return {"nodes": nodes, "edges": edges}
