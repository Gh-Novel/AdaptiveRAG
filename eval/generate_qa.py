"""Generate a synthetic golden QA set for retrieval evaluation.

Method (standard synthetic-eval approach):
  1. Stratified-sample N chunks across distinct documents
  2. LLM writes one self-contained question per chunk
  3. The (question, chunk_id) pair becomes ground truth: a retriever that
     can't bring back the chunk a question was written FROM has failed.

Usage:
  python eval/generate_qa.py --n 100
  python eval/generate_qa.py --n 100 --out eval/golden_set.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.indexer import get_chroma_collection
from llm.client_factory import get_llm

_SYSTEM = (
    "You write evaluation questions for a retrieval benchmark over AI research "
    "papers. Given a passage, write ONE specific, self-contained question that "
    "this passage clearly answers. The question must be answerable from the "
    "passage alone, must not reference 'the passage/paper/text', and should use "
    "natural phrasing a researcher would type into a search box. "
    "ALWAYS write the question in English, regardless of the passage language."
)

_PROMPT = """Passage:
\"\"\"{text}\"\"\"

Return strict JSON only: {{"question": "<the question>"}}"""

MIN_CHUNK_CHARS = 500  # skip stub chunks — questions from them are degenerate


def sample_chunks(n: int, seed: int = 42) -> list[dict]:
    """Stratified sample: at most one chunk per document until n is reached."""
    coll = get_chroma_collection()
    total = coll.count()
    res = coll.get(include=["documents", "metadatas"], limit=total)

    by_doc: dict[str, list[tuple[str, str, dict]]] = {}
    for cid, text, meta in zip(res["ids"], res["documents"], res["metadatas"]):
        if len(text) < MIN_CHUNK_CHARS:
            continue
        by_doc.setdefault(meta["doc_id"], []).append((cid, text, meta))

    rng = random.Random(seed)
    docs = sorted(by_doc)
    rng.shuffle(docs)
    for d in docs:
        rng.shuffle(by_doc[d])  # shuffle once; round-robin then never repeats

    picked: list[dict] = []
    round_i = 0
    while len(picked) < n:
        progressed = False
        for d in docs:
            if len(picked) >= n:
                break
            pool = by_doc[d]
            if round_i < len(pool):
                cid, text, meta = pool[round_i]
                picked.append(
                    {"chunk_id": cid, "doc_id": meta["doc_id"],
                     "title": meta.get("title", ""), "text": text}
                )
                progressed = True
        if not progressed:
            break
        round_i += 1
    return picked[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="eval/golden_set.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    llm = get_llm()
    chunks = sample_chunks(args.n, seed=args.seed)
    print(f"Sampled {len(chunks)} chunks from "
          f"{len({c['doc_id'] for c in chunks})} distinct docs")

    golden: list[dict] = []
    t0 = time.time()
    for i, c in enumerate(chunks, 1):
        out = llm.generate_json(
            prompt=_PROMPT.format(text=c["text"][:2400]),
            system=_SYSTEM,
            temperature=0.3,
        )
        q = (out or {}).get("question", "").strip() if isinstance(out, dict) else ""
        if not q or len(q) < 12:
            print(f"  [{i}] skipped (bad generation)")
            continue
        golden.append(
            {"question": q, "chunk_id": c["chunk_id"],
             "doc_id": c["doc_id"], "title": c["title"]}
        )
        if i % 10 == 0:
            rate = i / (time.time() - t0)
            print(f"  [{i}/{len(chunks)}] {rate:.1f} q/s — latest: {q[:80]}")
            Path(args.out).write_text(json.dumps(golden, indent=1))

    Path(args.out).write_text(json.dumps(golden, indent=1))
    print(f"\nWrote {len(golden)} questions to {args.out} "
          f"in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
