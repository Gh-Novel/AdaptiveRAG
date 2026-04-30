"""Sparse retrieval via BM25Okapi over the persisted token corpus."""
from __future__ import annotations

from functools import lru_cache

from rank_bm25 import BM25Okapi

from config import RETRIEVAL_CONFIG
from ingestion.indexer import load_bm25_corpus, tokenize
from retrieval.dense import Hit


@lru_cache(maxsize=1)
def _bm25_state():
    corpus = load_bm25_corpus()
    bm25 = BM25Okapi(corpus["tokenized"])
    return bm25, corpus


def sparse_search(query: str, k: int | None = None) -> list[Hit]:
    k = k or RETRIEVAL_CONFIG["sparse_k"]
    bm25, corpus = _bm25_state()
    tokens = tokenize(query)
    if not tokens:
        return []
    scores = bm25.get_scores(tokens)
    idx_sorted = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    max_score = float(scores[idx_sorted[0]]) if idx_sorted else 0.0
    hits: list[Hit] = []
    for r, i in enumerate(idx_sorted):
        s = float(scores[i])
        if s <= 0:
            continue
        norm = s / max_score if max_score > 0 else 0.0
        hits.append(
            Hit(
                chunk_id=corpus["ids"][i],
                text=corpus["docs"][i],
                metadata=dict(corpus["metas"][i]),
                score=norm,
                rank=r,
            )
        )
    return hits
