"""Score each retrieved chunk for relevance + completeness.

Low-quality chunks (score < QUALITY_THRESHOLD) are expanded with their
immediate neighbours from the same document.
"""
from __future__ import annotations

import numpy as np

from ingestion.embedder import embed_query
from ingestion.indexer import fetch_embeddings, get_chroma_collection
from retrieval.dense import Hit

QUALITY_THRESHOLD = 0.6
_SENTENCE_END = set(".!?\":")


def _relevance(hit: Hit, query_vec: np.ndarray) -> float:
    embs = fetch_embeddings([hit.chunk_id])
    if hit.chunk_id not in embs:
        return float(hit.score)
    cvec = np.array(embs[hit.chunk_id], dtype=np.float32)
    return float(np.clip(np.dot(query_vec, cvec), 0.0, 1.0))


def _completeness(text: str) -> float:
    text = text.strip()
    if not text:
        return 0.0
    score = 1.0
    if text and not text[0].isupper():
        score -= 0.30
    if text and text[-1] not in _SENTENCE_END:
        score -= 0.20
    if len(text) < 200:
        score -= 0.20
    return max(0.0, score)


def score_chunk(hit: Hit, query_vec: np.ndarray) -> dict:
    rel = _relevance(hit, query_vec)
    comp = _completeness(hit.text)
    combined = rel * 0.6 + comp * 0.4
    return {
        "chunk_id": hit.chunk_id,
        "relevance": round(rel, 3),
        "completeness": round(comp, 3),
        "score": round(combined, 3),
        "needs_expansion": combined < QUALITY_THRESHOLD,
    }


def expand_chunk(hit: Hit) -> list[Hit]:
    """Return the ±1 neighbour chunks from the same document."""
    parts = hit.chunk_id.split("::")
    if len(parts) != 2:
        return []
    doc_id, chunk_part = parts
    try:
        num = int(chunk_part[1:])
    except ValueError:
        return []
    neighbor_ids = [
        f"{doc_id}::c{max(0, num - 1):04d}",
        f"{doc_id}::c{num + 1:04d}",
    ]
    coll = get_chroma_collection()
    try:
        res = coll.get(ids=neighbor_ids, include=["documents", "metadatas"])
    except Exception:
        return []
    out: list[Hit] = []
    for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"]):
        out.append(
            Hit(
                chunk_id=cid,
                text=doc,
                metadata=dict(meta),
                score=hit.score * 0.85,
                rank=hit.rank,
            )
        )
    return out


def score_and_expand(
    query: str, hits: list[Hit]
) -> tuple[list[Hit], list[dict]]:
    """Score every hit; expand low-quality ones. Return (expanded_hits, scores)."""
    if not hits:
        return hits, []
    qvec = np.array(embed_query(query), dtype=np.float32)
    scores: list[dict] = []
    expanded = list(hits)
    seen = {h.chunk_id for h in hits}
    for hit in hits:
        s = score_chunk(hit, qvec)
        scores.append(s)
        if s["needs_expansion"]:
            for n in expand_chunk(hit):
                if n.chunk_id not in seen:
                    expanded.append(n)
                    seen.add(n.chunk_id)
    return expanded, scores
