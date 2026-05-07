"""SHA-256-based document change detection.

Compare incoming documents against their stored checksums and classify
each as new / changed / unchanged. The checksum covers the full binary
content of the source file (PDF bytes) so even a single character edit
is detected.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from versioning.document_store import DocumentStore


@dataclass
class ChangeReport:
    new_docs: list[dict] = field(default_factory=list)
    changed_docs: list[dict] = field(default_factory=list)
    unchanged_docs: list[dict] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return bool(self.new_docs or self.changed_docs)

    def summary(self) -> str:
        parts = []
        if self.new_docs:
            parts.append(f"{len(self.new_docs)} new")
        if self.changed_docs:
            parts.append(f"{len(self.changed_docs)} changed")
        if self.unchanged_docs:
            parts.append(f"{len(self.unchanged_docs)} unchanged")
        return ", ".join(parts) or "no documents"


def sha256_file(path: str | Path) -> str:
    """Hash full binary content of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65_536), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def detect_changes(
    docs: list[dict],
    store: DocumentStore | None = None,
) -> ChangeReport:
    """
    Classify each doc as new / changed / unchanged by comparing SHA-256
    checksums against the DocumentStore.

    Parameters
    ----------
    docs:
        List of dicts with at least ``doc_id`` and ``source_path`` keys.
        Each dict is returned enriched with a ``checksum`` key.
    store:
        DocumentStore instance. A fresh one is created if not supplied.

    Returns
    -------
    ChangeReport with three lists (new / changed / unchanged), each item
    being the original doc dict augmented with ``"checksum"``.
    """
    if store is None:
        store = DocumentStore()

    report = ChangeReport()
    for doc in docs:
        path = Path(doc["source_path"])
        checksum = sha256_file(path) if path.exists() else sha256_text(str(doc))
        enriched = {**doc, "checksum": checksum}

        stored = store.get_checksum(doc["doc_id"])
        if stored is None:
            report.new_docs.append(enriched)
        elif stored != checksum:
            report.changed_docs.append(enriched)
        else:
            report.unchanged_docs.append(enriched)

    return report
