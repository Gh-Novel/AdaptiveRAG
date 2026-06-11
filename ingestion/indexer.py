"""ChromaDB management + SQLite FTS5 sparse index.

Sparse retrieval moved from rank-bm25 (in-memory linear scan, O(n) per query)
to SQLite FTS5 (disk-backed inverted index with built-in BM25 ranking).
At 280k chunks rank-bm25 took seconds per query and ~3 GB RAM; FTS5 answers
in ~10 ms with near-zero memory.
"""
from __future__ import annotations

import json
import os
import sqlite3

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
import chromadb
from chromadb.config import Settings  # noqa: E402

from config import CHROMA_COLLECTION, PATHS
from ingestion.chunker import Chunk
from ingestion.embedder import embed_texts

_CHROMA_BATCH = 256


def _ensure_dirs() -> None:
    PATHS["chroma_dir"].mkdir(parents=True, exist_ok=True)
    PATHS["fts_path"].parent.mkdir(parents=True, exist_ok=True)


def _client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(
        path=str(PATHS["chroma_dir"]),
        settings=Settings(anonymized_telemetry=False),
    )


def get_chroma_collection():
    _ensure_dirs()
    client = _client()
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


# ── FTS5 sparse index ────────────────────────────────────────────────

def _fts_conn() -> sqlite3.Connection:
    _ensure_dirs()
    conn = sqlite3.connect(str(PATHS["fts_path"]))
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            chunk_id UNINDEXED,
            doc_id   UNINDEXED,
            metadata UNINDEXED,
            text,
            tokenize='porter unicode61'
        )
        """
    )
    return conn


def _fts_upsert(conn: sqlite3.Connection, chunks: list[Chunk]) -> None:
    """Replace all FTS rows for the documents covered by *chunks*."""
    doc_ids = sorted({c.doc_id for c in chunks})
    conn.executemany(
        "DELETE FROM chunks_fts WHERE doc_id = ?", [(d,) for d in doc_ids]
    )
    conn.executemany(
        "INSERT INTO chunks_fts (chunk_id, doc_id, metadata, text) VALUES (?,?,?,?)",
        [
            (
                c.chunk_id,
                c.doc_id,
                json.dumps(
                    {
                        "doc_id": c.doc_id,
                        "source_path": c.source_path,
                        "title": c.title,
                        "page_start": c.page_start,
                        "page_end": c.page_end,
                    }
                ),
                c.text,
            )
            for c in chunks
        ],
    )
    conn.commit()


def reset_index() -> None:
    _ensure_dirs()
    client = _client()
    try:
        client.delete_collection(CHROMA_COLLECTION)
    except Exception:
        pass
    for key in ("fts_path", "bm25_path", "manifest_path", "ingest_progress_path"):
        if PATHS[key].exists():
            PATHS[key].unlink()


def _chunk_payload(chunks: list[Chunk]):
    ids = [c.chunk_id for c in chunks]
    docs = [c.text for c in chunks]
    metas = [
        {
            "doc_id": c.doc_id,
            "source_path": c.source_path,
            "title": c.title,
            "page_start": c.page_start,
            "page_end": c.page_end,
        }
        for c in chunks
    ]
    return ids, docs, metas


def index_doc(chunks: list[Chunk], coll=None, fts=None) -> int:
    """Incrementally index ONE document's chunks (Chroma + FTS5).

    Pass coll/fts to reuse connections across documents in a long ingest run.
    Returns the number of chunks indexed.
    """
    if not chunks:
        return 0
    own_fts = fts is None
    coll = coll if coll is not None else get_chroma_collection()
    fts = fts if fts is not None else _fts_conn()

    ids, docs, metas = _chunk_payload(chunks)
    embeddings = embed_texts(docs)
    for i in range(0, len(ids), _CHROMA_BATCH):
        coll.upsert(
            ids=ids[i : i + _CHROMA_BATCH],
            documents=docs[i : i + _CHROMA_BATCH],
            metadatas=metas[i : i + _CHROMA_BATCH],
            embeddings=embeddings[i : i + _CHROMA_BATCH],
        )
    _fts_upsert(fts, chunks)
    if own_fts:
        fts.close()
    return len(ids)


def index_chunks(chunks: list[Chunk], reset: bool = False) -> dict:
    """Index a full batch of chunks (kept for compatibility with small corpora)."""
    _ensure_dirs()
    if reset:
        reset_index()

    coll = get_chroma_collection()
    fts = _fts_conn()
    try:
        print(f"  Embedding {len(chunks)} chunks...")
        index_doc(chunks, coll=coll, fts=fts)
    finally:
        fts.close()

    manifest = write_manifest(_group_count([c.doc_id for c in chunks]))
    return manifest


def write_manifest(chunks_per_doc: dict) -> dict:
    manifest = {
        "n_chunks": sum(chunks_per_doc.values()),
        "chunks_per_doc": chunks_per_doc,
    }
    with open(PATHS["manifest_path"], "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def index_chunks_versioned(
    chunks: list[Chunk],
    batch_name: str = "",
    reason: str = "ingest",
) -> dict:
    """Index chunks into a new versioned snapshot (kb_v{N}).

    Only re-embeds documents whose file checksum has changed since the
    last snapshot.  Unchanged document chunks are copied from the previous
    ChromaDB collection without re-embedding.

    Returns a dict with version, new, changed, unchanged counts.
    """
    from versioning.index_manager import RAGVersionManager

    manager = RAGVersionManager()
    chunks_by_doc: dict = {}
    for c in chunks:
        if c.doc_id not in chunks_by_doc:
            chunks_by_doc[c.doc_id] = {
                "chunks": [],
                "source_path": c.source_path,
                "title": c.title,
            }
        chunks_by_doc[c.doc_id]["chunks"].append(c)

    return manager.add_documents(
        chunks_by_doc,
        batch_name=batch_name or f"{len(chunks_by_doc)}_docs",
        reason=reason,
    )


def _group_count(items: list[str]) -> dict:
    out: dict = {}
    for x in items:
        out[x] = out.get(x, 0) + 1
    return out


def fetch_embeddings(chunk_ids: list[str]) -> dict[str, list[float]]:
    """Pull stored embeddings for a list of chunk ids (used for visualization)."""
    if not chunk_ids:
        return {}
    coll = get_chroma_collection()
    res = coll.get(ids=list(chunk_ids), include=["embeddings"])
    out: dict[str, list[float]] = {}
    for cid, vec in zip(res["ids"], res["embeddings"]):
        out[cid] = list(vec)
    return out
