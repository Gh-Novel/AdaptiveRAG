"""Ingest all PDFs from the papers/ directory into ChromaDB + BM25.

Writes two indexes every run:
  1. adaptive_rag  — flat ChromaDB collection used by the live RAG pipeline
  2. kb_v{N}       — versioned snapshot (skips unchanged docs via SHA-256)

The versioned snapshots power the Knowledge Base tab in the UI.
"""
from __future__ import annotations

import argparse
import time

from config import PATHS
from ingestion.chunker import chunk_document
from ingestion.indexer import index_chunks, index_chunks_versioned
from ingestion.loader import discover_pdfs, load_pdf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true",
                    help="Reset the flat adaptive_rag index before ingesting")
    ap.add_argument("--papers-dir", default=str(PATHS["papers_dir"]))
    ap.add_argument("--reason", default="",
                    help="Short description of why this batch was ingested")
    ap.add_argument("--no-version", action="store_true",
                    help="Skip creating a versioned snapshot (kb_vN)")
    args = ap.parse_args()

    pdfs = discover_pdfs(args.papers_dir)
    if not pdfs:
        print(f"No PDFs in {args.papers_dir}")
        return

    print(f"Found {len(pdfs)} PDFs in {args.papers_dir}")
    all_chunks = []
    t0 = time.time()
    for path in pdfs:
        doc_id = path.stem
        print(f"\n[{doc_id}]")
        doc = load_pdf(path)
        print(f"  Loaded: {len(doc.pages)} pages, title={doc.title!r}")
        chunks = chunk_document(doc, doc_id=doc_id)
        avg = sum(len(c.text) for c in chunks) // max(len(chunks), 1)
        print(f"  Chunked: {len(chunks)} chunks (avg {avg} chars)")
        all_chunks.extend(chunks)

    # ── 1. flat index (used by the live pipeline) ────────────────────
    print(f"\nIndexing {len(all_chunks)} chunks → adaptive_rag…")
    manifest = index_chunks(all_chunks, reset=args.reset)

    # ── 2. versioned snapshot (used by the KB tab) ───────────────────
    if not args.no_version:
        print("\nBuilding versioned snapshot…")
        reason = args.reason or f"ingest {len(pdfs)} PDF(s)"
        batch = f"{len(pdfs)}_docs"
        result = index_chunks_versioned(all_chunks, batch_name=batch, reason=reason)
        print(f"  Versioned result: {result}")

    dt = time.time() - t0
    print(f"\nDone in {dt:.1f}s. Manifest: {manifest}")


if __name__ == "__main__":
    main()
