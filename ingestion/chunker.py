"""Semantic-aware chunking.

Strategy: split each page into sentences, then greedily group sentences into
chunks targeting CHUNKING_CONFIG['target_chunk_chars']. Carry an overlap of
the last few sentences (~overlap_chars) to the next chunk so context isn't
sliced mid-thought. Headings hint chunk boundaries.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from config import CHUNKING_CONFIG
from ingestion.loader import LoadedDoc


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source_path: str
    title: str
    page_start: int
    page_end: int
    text: str


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")
_HEADING_HINT = re.compile(r"^(?:[0-9]+(?:\.[0-9]+)*\s+|abstract|introduction|conclusion|references|method|results|discussion|background)\b", re.IGNORECASE)


def _split_sentences(text: str) -> list[str]:
    parts: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if _HEADING_HINT.match(line):
            parts.append(line)
            continue
        for sent in _SENTENCE_SPLIT.split(line):
            sent = sent.strip()
            if sent:
                parts.append(sent)
    return parts


def chunk_document(doc: LoadedDoc, doc_id: str) -> list[Chunk]:
    target = CHUNKING_CONFIG["target_chunk_chars"]
    max_chars = CHUNKING_CONFIG["max_chunk_chars"]
    min_chars = CHUNKING_CONFIG["min_chunk_chars"]
    overlap = CHUNKING_CONFIG["overlap_chars"]

    units: list[tuple[int, str]] = []
    for page in doc.pages:
        for sent in _split_sentences(page.text):
            units.append((page.page_number, sent))

    chunks: list[Chunk] = []
    buf: list[tuple[int, str]] = []
    buf_len = 0

    def flush() -> None:
        nonlocal buf, buf_len
        if not buf:
            return
        text = " ".join(s for _, s in buf).strip()
        if len(text) < min_chars and chunks:
            chunks[-1].text = (chunks[-1].text + " " + text).strip()
            chunks[-1].page_end = buf[-1][0]
            buf, buf_len = [], 0
            return
        chunk = Chunk(
            chunk_id=f"{doc_id}::c{len(chunks):04d}",
            doc_id=doc_id,
            source_path=doc.source_path,
            title=doc.title,
            page_start=buf[0][0],
            page_end=buf[-1][0],
            text=text,
        )
        chunks.append(chunk)
        carry: list[tuple[int, str]] = []
        carry_len = 0
        for pn, s in reversed(buf):
            if carry_len + len(s) + 1 > overlap:
                break
            carry.insert(0, (pn, s))
            carry_len += len(s) + 1
        buf = carry
        buf_len = sum(len(s) + 1 for _, s in buf)

    for pn, sent in units:
        is_heading = bool(_HEADING_HINT.match(sent))
        if is_heading and buf_len >= min_chars:
            flush()
        buf.append((pn, sent))
        buf_len += len(sent) + 1
        if buf_len >= target:
            flush()
        elif buf_len >= max_chars:
            flush()
    flush()
    return chunks
