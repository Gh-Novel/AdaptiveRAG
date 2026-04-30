"""ChromaDB management + BM25 corpus persistence."""
from __future__ import annotations

import json
import os
import pickle
import re

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
import chromadb
from chromadb.config import Settings  # noqa: E402

from config import CHROMA_COLLECTION, PATHS
from ingestion.chunker import Chunk
from ingestion.embedder import embed_texts


def _ensure_dirs() -> None:
    PATHS["chroma_dir"].mkdir(parents=True, exist_ok=True)
    PATHS["bm25_path"].parent.mkdir(parents=True, exist_ok=True)


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


def reset_index() -> None:
    _ensure_dirs()
    client = _client()
    try:
        client.delete_collection(CHROMA_COLLECTION)
    except Exception:
        pass
    if PATHS["bm25_path"].exists():
        PATHS["bm25_path"].unlink()
    if PATHS["manifest_path"].exists():
        PATHS["manifest_path"].unlink()


_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def index_chunks(chunks: list[Chunk], reset: bool = False) -> dict:
    _ensure_dirs()
    if reset:
        reset_index()

    coll = get_chroma_collection()

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

    print(f"  Embedding {len(docs)} chunks...")
    embeddings = embed_texts(docs)

    print(f"  Writing to ChromaDB ({CHROMA_COLLECTION})...")
    BATCH = 256
    for i in range(0, len(ids), BATCH):
        coll.upsert(
            ids=ids[i : i + BATCH],
            documents=docs[i : i + BATCH],
            metadatas=metas[i : i + BATCH],
            embeddings=embeddings[i : i + BATCH],
        )

    print("  Building BM25 corpus...")
    tokenized = [tokenize(d) for d in docs]
    with open(PATHS["bm25_path"], "wb") as f:
        pickle.dump(
            {"ids": ids, "tokenized": tokenized, "metas": metas, "docs": docs},
            f,
        )

    manifest = {
        "n_chunks": len(ids),
        "chunks_per_doc": _group_count([c.doc_id for c in chunks]),
    }
    with open(PATHS["manifest_path"], "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


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


def load_bm25_corpus() -> dict:
    if not PATHS["bm25_path"].exists():
        raise FileNotFoundError(
            f"BM25 corpus not found at {PATHS['bm25_path']}. Run ingestion first."
        )
    with open(PATHS["bm25_path"], "rb") as f:
        return pickle.load(f)
