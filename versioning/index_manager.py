"""ChromaDB snapshot manager.

Every call to add_documents():
  1. Detects which docs are new / changed / unchanged (SHA-256).
  2. Creates a new ChromaDB collection  ``kb_v{N}`` for the new snapshot.
  3. Copies unchanged chunks from the previous snapshot  (no re-embedding).
  4. Embeds and indexes only the changed/new chunks.
  5. Points ``latest`` at the new version in SQLite.

Rollback is a single SQLite write — no data is ever deleted.
"""
from __future__ import annotations

import os

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb
from chromadb.config import Settings

from config import PATHS
from versioning.change_detector import ChangeReport, detect_changes
from versioning.document_store import DocumentStore

_CHROMA_BATCH = 256


def _client() -> chromadb.PersistentClient:
    PATHS["chroma_dir"].mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(PATHS["chroma_dir"]),
        settings=Settings(anonymized_telemetry=False),
    )


def _collection_name(version: int) -> str:
    return f"kb_v{version}"


class RAGVersionManager:
    """Versioned knowledge-base layer on top of ChromaDB."""

    def __init__(self) -> None:
        self.store = DocumentStore()

    # ── public API ───────────────────────────────────────────────────

    def add_documents(
        self,
        chunks_by_doc: dict,
        batch_name: str = "",
        reason: str = "manual ingest",
    ) -> dict:
        """
        Index a batch of documents, only re-embedding what changed.

        Parameters
        ----------
        chunks_by_doc:
            ``{doc_id: {"chunks": [Chunk, ...], "source_path": str, "title": str}}``
        batch_name:
            Human-readable label for this batch (e.g. ``"initial_14_papers"``).
        reason:
            Short description of why this version was created.

        Returns
        -------
        dict with version, new, changed, unchanged counts.
        """
        from ingestion.embedder import embed_texts

        client = _client()

        # ── 1. change detection ──────────────────────────────────────
        doc_meta = [
            {"doc_id": did, "source_path": info["source_path"], "title": info["title"]}
            for did, info in chunks_by_doc.items()
        ]
        report: ChangeReport = detect_changes(doc_meta, self.store)

        if not report.has_changes:
            print(f"  No changes detected ({report.summary()}) — index unchanged.")
            return {
                "version": self.store.get_latest(),
                "new": 0,
                "changed": 0,
                "unchanged": len(report.unchanged_docs),
            }

        current_version = self.store.get_latest() or 0
        new_version = self.store.bump_version()
        new_coll_name = _collection_name(new_version)

        print(f"  Change summary: {report.summary()}")
        print(f"  Creating snapshot {new_coll_name}…")

        new_coll = client.get_or_create_collection(
            name=new_coll_name,
            metadata={"hnsw:space": "cosine"},
        )

        # ── 2. copy unchanged chunks from previous snapshot ──────────
        if current_version > 0 and report.unchanged_docs:
            prev_name = _collection_name(current_version)
            try:
                prev_coll = client.get_collection(prev_name)
                unchanged_ids = {d["doc_id"] for d in report.unchanged_docs}
                copied = self._copy_chunks(prev_coll, new_coll, unchanged_ids)
                print(f"  Copied {copied} chunks from {prev_name} (unchanged docs).")
            except Exception as exc:
                print(f"  Warning: could not copy from v{current_version}: {exc}")

        # ── 3. embed + index new/changed chunks ──────────────────────
        to_index_ids = {d["doc_id"] for d in report.new_docs + report.changed_docs}
        new_chunks = [
            c
            for did, info in chunks_by_doc.items()
            if did in to_index_ids
            for c in info["chunks"]
        ]

        if new_chunks:
            ids = [c.chunk_id for c in new_chunks]
            texts = [c.text for c in new_chunks]
            metas = [
                {
                    "doc_id": c.doc_id,
                    "source_path": c.source_path,
                    "title": c.title,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                }
                for c in new_chunks
            ]
            print(f"  Embedding {len(new_chunks)} chunks for {len(to_index_ids)} doc(s)…")
            embeddings = embed_texts(texts)
            for i in range(0, len(ids), _CHROMA_BATCH):
                new_coll.upsert(
                    ids=ids[i : i + _CHROMA_BATCH],
                    documents=texts[i : i + _CHROMA_BATCH],
                    metadatas=metas[i : i + _CHROMA_BATCH],
                    embeddings=embeddings[i : i + _CHROMA_BATCH],
                )

        # ── 4. persist metadata ──────────────────────────────────────
        for d in report.new_docs:
            self.store.add_doc(d["doc_id"], new_version, d["checksum"],
                               "active", d["source_path"], d["title"])
        for d in report.changed_docs:
            self.store.add_doc(d["doc_id"], new_version, d["checksum"],
                               "active", d["source_path"], d["title"])
        for d in report.unchanged_docs:
            self.store.add_doc(d["doc_id"], new_version, d["checksum"],
                               "active", d["source_path"], d["title"])

        self.store.log_version(
            version=new_version,
            batch_name=batch_name,
            docs_added=len(report.new_docs),
            docs_changed=len(report.changed_docs),
            docs_unchanged=len(report.unchanged_docs),
            reason=reason,
            collection_name=new_coll_name,
        )
        self.store.set_latest(new_version)

        print(
            f"  ✓ Created {new_coll_name} — "
            f"{len(report.new_docs)} new, {len(report.changed_docs)} changed, "
            f"{len(report.unchanged_docs)} unchanged"
        )
        return {
            "version": new_version,
            "new": len(report.new_docs),
            "changed": len(report.changed_docs),
            "unchanged": len(report.unchanged_docs),
            "collection": new_coll_name,
        }

    def rollback(self, to_version: int) -> None:
        """Point 'latest' at a previous snapshot (metadata-only, instant)."""
        known = {v["version"] for v in self.store.get_history()}
        if to_version not in known:
            raise ValueError(f"Version {to_version} not found. Known: {sorted(known)}")
        self.store.set_latest(to_version)
        print(f"  Rolled back to v{to_version}")

    def list_versions(self) -> list[dict]:
        return self.store.get_history()

    def get_collection(self, version: str | int = "latest") -> chromadb.Collection:
        client = _client()
        v = self.store.get_latest() if version == "latest" else int(version)
        if v is None:
            raise RuntimeError("No versioned snapshots exist yet. Run ingest first.")
        return client.get_collection(_collection_name(v))

    def collection_exists(self, version: str | int = "latest") -> bool:
        try:
            self.get_collection(version)
            return True
        except Exception:
            return False

    def query(
        self,
        text: str,
        version: str | int = "latest",
        k: int = 12,
    ) -> list:
        """Dense search against a specific snapshot. Returns list[Hit]."""
        from ingestion.embedder import embed_query
        from retrieval.dense import Hit

        coll = self.get_collection(version)
        qv = embed_query(text)
        res = coll.query(
            query_embeddings=[qv],
            n_results=min(k, coll.count()),
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for r, (cid, doc, meta, dist) in enumerate(
            zip(res["ids"][0], res["documents"][0],
                res["metadatas"][0], res["distances"][0])
        ):
            hits.append(
                Hit(
                    chunk_id=cid,
                    text=doc,
                    metadata=dict(meta),
                    score=max(0.0, 1.0 - float(dist)),
                    rank=r,
                )
            )
        return hits

    # ── internal helpers ─────────────────────────────────────────────

    def _copy_chunks(
        self,
        src: chromadb.Collection,
        dst: chromadb.Collection,
        doc_ids: set[str],
    ) -> int:
        """Copy all chunks whose doc_id is in *doc_ids* from src → dst."""
        if not doc_ids:
            return 0

        where = (
            {"doc_id": {"$in": list(doc_ids)}}
            if len(doc_ids) > 1
            else {"doc_id": list(doc_ids)[0]}
        )
        res = src.get(where=where, include=["documents", "metadatas", "embeddings"])
        if not res["ids"]:
            return 0

        ids, docs, metas, embs = (
            res["ids"], res["documents"], res["metadatas"], res["embeddings"]
        )
        for i in range(0, len(ids), _CHROMA_BATCH):
            dst.upsert(
                ids=ids[i : i + _CHROMA_BATCH],
                documents=docs[i : i + _CHROMA_BATCH],
                metadatas=metas[i : i + _CHROMA_BATCH],
                embeddings=embs[i : i + _CHROMA_BATCH],
            )
        return len(ids)
