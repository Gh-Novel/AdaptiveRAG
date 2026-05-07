"""Sentence-level hallucination detection.

For every sentence in the answer, checks whether at least one retrieved
chunk supports it via cosine similarity. Sentences with max similarity
below THRESHOLD are flagged as unsupported.
"""
from __future__ import annotations

import re

import numpy as np

from ingestion.embedder import embed_texts
from ingestion.indexer import fetch_embeddings
from retrieval.dense import Hit

THRESHOLD = 0.75
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[\(])")


def _sentences(text: str) -> list[str]:
    parts = _SENT_RE.split(text.strip())
    return [s.strip() for s in parts if len(s.strip()) > 25]


def detect_hallucinations(answer: str, hits: list[Hit]) -> list[dict]:
    """Return sentences not supported by any retrieved chunk.

    Each entry: {position, sentence, max_similarity, closest_chunk_id}
    """
    sentences = _sentences(answer)
    if not sentences or not hits:
        return []

    chunk_ids = [h.chunk_id for h in hits]
    chunk_embs = fetch_embeddings(chunk_ids)
    if not chunk_embs:
        return []

    ids_ordered = list(chunk_embs.keys())
    chunk_mat = np.array(
        [chunk_embs[cid] for cid in ids_ordered], dtype=np.float32
    )  # (n_chunks, 384)

    sent_vecs = np.array(embed_texts(sentences), dtype=np.float32)  # (n_sents, 384)
    # Cosine similarity: MiniLM embeddings are L2-normalised, so dot = cosine
    sims = sent_vecs @ chunk_mat.T  # (n_sents, n_chunks)

    unsupported = []
    for i, (sent, row) in enumerate(zip(sentences, sims)):
        max_sim = float(row.max())
        if max_sim < THRESHOLD:
            unsupported.append(
                {
                    "position": i,
                    "sentence": sent[:200],
                    "max_similarity": round(max_sim, 3),
                    "closest_chunk_id": ids_ordered[int(row.argmax())],
                }
            )
    return unsupported
