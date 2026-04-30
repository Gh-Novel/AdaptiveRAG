"""Reciprocal Rank Fusion of dense + sparse hits."""
from __future__ import annotations

from config import RETRIEVAL_CONFIG
from retrieval.dense import Hit, dense_search
from retrieval.sparse import sparse_search


def reciprocal_rank_fusion(
    rankings: list[list[Hit]],
    k_const: int | None = None,
    top_k: int | None = None,
) -> list[Hit]:
    k_const = k_const or RETRIEVAL_CONFIG["rrf_k"]
    fused: dict[str, dict] = {}
    for ranking in rankings:
        for r, hit in enumerate(ranking):
            entry = fused.setdefault(
                hit.chunk_id,
                {"hit": hit, "score": 0.0},
            )
            entry["score"] += 1.0 / (k_const + r + 1)
            # prefer higher-ranked instance for the canonical hit object
            if hit.rank < entry["hit"].rank:
                entry["hit"] = hit
    merged = sorted(fused.values(), key=lambda x: x["score"], reverse=True)
    out: list[Hit] = []
    for r, entry in enumerate(merged):
        h = entry["hit"]
        out.append(
            Hit(
                chunk_id=h.chunk_id,
                text=h.text,
                metadata=h.metadata,
                score=entry["score"],
                rank=r,
            )
        )
    if top_k:
        out = out[:top_k]
    return out


def hybrid_search(query: str, top_k: int | None = None) -> list[Hit]:
    dense = dense_search(query)
    sparse = sparse_search(query)
    return reciprocal_rank_fusion([dense, sparse], top_k=top_k)
