"""Ingest all PDFs from the papers/ directory into ChromaDB + BM25."""
from __future__ import annotations

import argparse
import time

from config import PATHS
from ingestion.chunker import chunk_document
from ingestion.indexer import index_chunks
from ingestion.loader import discover_pdfs, load_pdf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="Reset the index first")
    ap.add_argument("--papers-dir", default=str(PATHS["papers_dir"]))
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
        print(f"  Chunked: {len(chunks)} chunks (avg {sum(len(c.text) for c in chunks)//max(len(chunks),1)} chars)")
        all_chunks.extend(chunks)

    print(f"\nIndexing {len(all_chunks)} chunks total...")
    manifest = index_chunks(all_chunks, reset=args.reset)
    dt = time.time() - t0
    print(f"\nDone in {dt:.1f}s. Manifest: {manifest}")


if __name__ == "__main__":
    main()
