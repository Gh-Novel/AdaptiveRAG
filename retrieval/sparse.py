"""Sparse (keyword) retrieval via SQLite FTS5.

FTS5 maintains a disk-backed inverted index with native BM25 ranking and
Porter stemming. Replaces rank-bm25, which held the whole tokenized corpus
in RAM and scored every document on every query (O(n) — seconds at 280k
chunks). FTS5 answers the same queries in ~10 ms.
"""
from __future__ import annotations

import json
import re
import sqlite3

from config import PATHS, RETRIEVAL_CONFIG
from retrieval.dense import Hit

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def _connect() -> sqlite3.Connection:
    path = PATHS["fts_path"]
    if not path.exists():
        raise FileNotFoundError(
            f"FTS index not found at {path}. Run `python ingest.py` first."
        )
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _to_match_expr(query: str) -> str:
    """Sanitize free text into an FTS5 MATCH expression (OR of bare tokens)."""
    tokens = [t.lower() for t in _TOKEN_RE.findall(query)]
    return " OR ".join(tokens)


def sparse_search(query: str, k: int | None = None) -> list[Hit]:
    k = k or RETRIEVAL_CONFIG["sparse_k"]
    match = _to_match_expr(query)
    if not match:
        return []

    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT chunk_id, metadata, text, bm25(chunks_fts) AS s "
            "FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY s LIMIT ?",
            (match, k),
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        return []

    # SQLite bm25() returns <= 0 with more-negative = better; flip sign so
    # higher = better, then normalize to [0,1] like the old rank-bm25 path.
    raw = [-r[3] for r in rows]
    mx = max(raw) or 1.0

    hits: list[Hit] = []
    for rank, (row, score) in enumerate(zip(rows, raw)):
        chunk_id, meta_json, text, _ = row
        hits.append(
            Hit(
                chunk_id=chunk_id,
                text=text,
                metadata=json.loads(meta_json),
                score=float(score / mx),
                rank=rank,
            )
        )
    return hits
