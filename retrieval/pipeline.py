"""End-to-end retrieval: hybrid search + cross-encoder reranking."""
from __future__ import annotations

from config import RETRIEVAL_CONFIG
from retrieval.dense import Hit
from retrieval.hybrid import hybrid_search
from retrieval.reranker import rerank


def hybrid_retrieve(query: str, top_n: int | None = None) -> list[Hit]:
    fused = hybrid_search(query, top_k=max(RETRIEVAL_CONFIG["dense_k"], RETRIEVAL_CONFIG["sparse_k"]))
    return rerank(query, fused, top_n=top_n)
