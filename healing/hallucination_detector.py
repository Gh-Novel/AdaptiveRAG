"""Sentence-level hallucination detection.

Every answer sentence must be supported by at least one retrieved passage.
Support is measured as max cosine similarity between the answer sentence and
the *sentences* of the retrieved chunks (plus the whole chunks as fallback).

Granularity matters: comparing a single sentence against a whole ~1400-char
chunk depresses similarity for legitimately grounded sentences (the chunk
embedding averages many topics), which produced false hallucination flags.
Sentence-to-sentence comparison separates cleanly: paraphrases land ~0.65-0.9,
unrelated content ~0.2-0.4.
"""
from __future__ import annotations

import re

import numpy as np

from ingestion.embedder import embed_texts
from retrieval.dense import Hit

THRESHOLD = 0.60
MIN_SENT_CHARS = 40       # answer fragments below this aren't checkable claims
MIN_SUPPORT_CHARS = 30    # chunk sentences below this carry no signal
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[\(“\"])")
_CITATION_RE = re.compile(r"\[\d+\]")


def _clean(text: str) -> str:
    """Strip citation markers and markdown so they don't dilute the embedding."""
    text = _CITATION_RE.sub("", text)
    text = text.replace("**", "").replace("__", "")
    return " ".join(text.split())


def _sentences(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        line = line.strip().lstrip("-•* ").strip()
        if not line:
            continue
        for s in _SENT_RE.split(line):
            s = _clean(s.strip())
            # Skip fragments and list lead-ins ("...the following tokens:")
            if len(s) >= MIN_SENT_CHARS and not s.endswith(":"):
                out.append(s)
    return out


def detect_hallucinations(answer: str, hits: list[Hit]) -> list[dict]:
    """Return answer sentences not supported by any retrieved passage.

    Each entry: {position, sentence, max_similarity, closest_chunk_id}
    """
    sentences = _sentences(answer)
    if not sentences or not hits:
        return []

    # Support corpus: every chunk sentence + the whole chunk as fallback.
    support_texts: list[str] = []
    support_ids: list[str] = []
    for h in hits:
        support_texts.append(_clean(h.text))
        support_ids.append(h.chunk_id)
        for s in _SENT_RE.split(h.text):
            s = _clean(s.strip())
            if len(s) >= MIN_SUPPORT_CHARS:
                support_texts.append(s)
                support_ids.append(h.chunk_id)

    # MiniLM embeddings are L2-normalised, so dot product = cosine.
    support_mat = np.array(embed_texts(support_texts), dtype=np.float32)
    sent_vecs = np.array(embed_texts(sentences), dtype=np.float32)
    sims = sent_vecs @ support_mat.T  # (n_sents, n_support)

    unsupported = []
    for i, (sent, row) in enumerate(zip(sentences, sims)):
        max_sim = float(row.max())
        if max_sim < THRESHOLD:
            unsupported.append(
                {
                    "position": i,
                    "sentence": sent[:200],
                    "max_similarity": round(max_sim, 3),
                    "closest_chunk_id": support_ids[int(row.argmax())],
                }
            )
    return unsupported
