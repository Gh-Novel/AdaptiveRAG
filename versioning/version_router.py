"""Version-aware query router with audit logging.

Every query routed through this layer is:
  - dispatched to the correct versioned ChromaDB snapshot
  - logged to the query_log table (query text, version used, answer hash)

This enables the "replay any query against any historical snapshot" guarantee:
  router.query("what is DDIM?", version=1)   # before that paper was added
  router.query("what is DDIM?")              # against latest
"""
from __future__ import annotations

import hashlib

from versioning.document_store import DocumentStore
from versioning.index_manager import RAGVersionManager


class VersionRouter:
    """Intercepts RAG queries and routes them to a specific knowledge-base version."""

    def __init__(self) -> None:
        self.manager = RAGVersionManager()
        self.store = DocumentStore()

    # ── querying ─────────────────────────────────────────────────────

    def query(
        self,
        text: str,
        version: str | int = "latest",
        k: int = 12,
        log: bool = True,
    ) -> tuple[list, int]:
        """
        Run a dense search against the specified snapshot.

        Returns
        -------
        (hits: list[Hit], resolved_version: int)
        """
        resolved = (
            self.store.get_latest()
            if version == "latest"
            else int(version)
        )
        if resolved is None:
            raise RuntimeError("No versioned index exists yet. Run `python ingest.py` first.")

        hits = self.manager.query(text, version=resolved, k=k)

        if log:
            answer_hash = hashlib.sha256(
                (text + str(resolved)).encode()
            ).hexdigest()[:16]
            self.store.log_query(text, resolved, answer_hash)

        return hits, resolved

    # ── metadata helpers ─────────────────────────────────────────────

    def current_version(self) -> int | None:
        return self.store.get_latest()

    def list_versions(self) -> list[dict]:
        return self.store.get_history()

    def version_info(self, version: str | int = "latest") -> dict | None:
        v = (
            self.store.get_latest()
            if version == "latest"
            else int(version)
        )
        if v is None:
            return None
        return self.store.get_version_info(v)

    def docs_at_version(self, version: str | int = "latest") -> list[dict]:
        v = (
            self.store.get_latest()
            if version == "latest"
            else int(version)
        )
        if v is None:
            return []
        return self.store.docs_at_version(v)

    def get_query_log(self, limit: int = 50) -> list[dict]:
        return self.store.get_query_log(limit)

    def rollback(self, to_version: int) -> None:
        self.manager.rollback(to_version)

    def collection_exists(self, version: str | int = "latest") -> bool:
        return self.manager.collection_exists(version)
