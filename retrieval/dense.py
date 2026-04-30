"""Dense (vector) retrieval over ChromaDB."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import RETRIEVAL_CONFIG
from ingestion.embedder import embed_query
from ingestion.indexer import get_chroma_collection


@dataclass
class Hit:
    chunk_id: str
    text: str
    metadata: dict
    score: float
    rank: int


def dense_search(query: str, k: int | None = None) -> list[Hit]:
    k = k or RETRIEVAL_CONFIG["dense_k"]
    coll = get_chroma_collection()
    qv = embed_query(query)
    res = coll.query(
        query_embeddings=[qv],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    ids = res["ids"][0]
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    hits: list[Hit] = []
    for r, (cid, doc, meta, dist) in enumerate(zip(ids, docs, metas, dists)):
        # cosine distance → similarity in [0,1]; chroma returns 1 - cosine_sim for cosine space
        score = max(0.0, 1.0 - float(dist))
        hits.append(Hit(chunk_id=cid, text=doc, metadata=dict(meta), score=score, rank=r))
    return hits
