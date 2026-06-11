"""Ingest PDFs into ChromaDB (dense) + SQLite FTS5 (sparse).

Processes one document at a time and records progress after each, so a
crashed or interrupted run resumes where it left off:

  python ingest.py                                  # 14-paper corpus
  python ingest.py --papers-dir papers_arxiv        # arXiv pilot/full corpus
  python ingest.py --reset                          # wipe and rebuild
  python ingest.py --limit 50                       # first 50 PDFs only

Versioned kb_vN snapshots are built only for small corpora (or with
--version) — at 100k+ chunks snapshot copying is the wrong tool.
"""
from __future__ import annotations

import argparse
import json
import time

from config import PATHS
from ingestion.chunker import chunk_document
from ingestion.indexer import (
    _fts_conn,
    get_chroma_collection,
    index_chunks_versioned,
    index_doc,
    reset_index,
    write_manifest,
)
from ingestion.loader import discover_pdfs, load_pdf

VERSIONING_AUTO_MAX_DOCS = 50  # snapshots auto-enabled only below this


def _load_progress() -> dict:
    p = PATHS["ingest_progress_path"]
    return json.loads(p.read_text()) if p.exists() else {}


def _save_progress(progress: dict) -> None:
    PATHS["ingest_progress_path"].write_text(json.dumps(progress, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="Wipe all indexes first")
    ap.add_argument("--papers-dir", default=str(PATHS["papers_dir"]))
    ap.add_argument("--limit", type=int, default=0, help="Only ingest first N PDFs")
    ap.add_argument("--reason", default="", help="Label for the versioned snapshot")
    ap.add_argument("--version", action="store_true",
                    help="Force kb_vN snapshot even for large corpora")
    ap.add_argument("--no-version", action="store_true",
                    help="Skip kb_vN snapshot even for small corpora")
    args = ap.parse_args()

    pdfs = discover_pdfs(args.papers_dir)
    if args.limit:
        pdfs = pdfs[: args.limit]
    if not pdfs:
        print(f"No PDFs in {args.papers_dir}")
        return

    if args.reset:
        print("Resetting indexes…")
        reset_index()

    progress = _load_progress()
    todo = [p for p in pdfs if p.stem not in progress]
    print(f"Found {len(pdfs)} PDFs in {args.papers_dir} "
          f"({len(pdfs) - len(todo)} already indexed, {len(todo)} to do)")

    versioning = (
        args.version
        or (len(pdfs) <= VERSIONING_AUTO_MAX_DOCS and not args.no_version)
    )
    all_chunks = [] if versioning else None

    coll = get_chroma_collection()
    fts = _fts_conn()
    t0 = time.time()
    n_done = 0
    t_parse = t_index = 0.0

    try:
        for i, path in enumerate(todo, start=1):
            doc_id = path.stem
            tp = time.time()
            try:
                doc = load_pdf(path)
                chunks = chunk_document(doc, doc_id=doc_id)
            except Exception as exc:
                print(f"[{i}/{len(todo)}] {doc_id}: PARSE FAILED — {exc}")
                progress[doc_id] = 0  # don't retry corrupt PDFs forever
                _save_progress(progress)
                continue
            t_parse += time.time() - tp

            ti = time.time()
            n = index_doc(chunks, coll=coll, fts=fts)
            t_index += time.time() - ti

            progress[doc_id] = n
            _save_progress(progress)
            n_done += 1
            if versioning and chunks:
                all_chunks.extend(chunks)

            if i % 10 == 0 or i == len(todo):
                rate = n_done / max(time.time() - t0, 1e-9)
                eta = (len(todo) - i) / max(rate, 1e-9)
                print(f"[{i}/{len(todo)}] {doc_id}: {n} chunks "
                      f"({rate:.1f} docs/s, ETA {eta/60:.1f} min)")
    finally:
        fts.close()

    manifest = write_manifest({k: v for k, v in progress.items() if v > 0})

    if versioning and all_chunks:
        print("\nBuilding versioned snapshot…")
        reason = args.reason or f"ingest {n_done} PDF(s)"
        result = index_chunks_versioned(
            all_chunks, batch_name=f"{n_done}_docs", reason=reason
        )
        print(f"  Versioned result: {result}")
    elif not versioning:
        print("\n(kb_vN snapshot skipped — corpus too large; use --version to force)")

    dt = time.time() - t0
    print(f"\nDone in {dt/60:.1f} min "
          f"(parse {t_parse/60:.1f} min, embed+index {t_index/60:.1f} min). "
          f"Total: {manifest['n_chunks']} chunks across "
          f"{len(manifest['chunks_per_doc'])} docs.")


if __name__ == "__main__":
    main()
