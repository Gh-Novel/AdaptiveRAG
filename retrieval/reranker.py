"""Cross-encoder reranker: deep relevance scoring on top of fused hits."""
from __future__ import annotations

from functools import lru_cache

from sentence_transformers import CrossEncoder

from config import RERANKER_CONFIG, RETRIEVAL_CONFIG
from retrieval.dense import Hit


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(RERANKER_CONFIG["model"], device=RERANKER_CONFIG["device"])


def rerank(query: str, hits: list[Hit], top_n: int | None = None) -> list[Hit]:
    if not hits:
        return []
    top_n = top_n or RETRIEVAL_CONFIG["rerank_top_n"]
    model = get_reranker()
    pairs = [(query, h.text) for h in hits]
    scores = model.predict(pairs, show_progress_bar=False)
    ranked = sorted(zip(hits, scores), key=lambda x: float(x[1]), reverse=True)[:top_n]
    out: list[Hit] = []
    for r, (h, s) in enumerate(ranked):
        out.append(
            Hit(
                chunk_id=h.chunk_id,
                text=h.text,
                metadata=h.metadata,
                score=float(s),
                rank=r,
            )
        )
    return out
