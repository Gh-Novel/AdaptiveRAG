"""Measure retrieval quality against the golden QA set.

For every question we know which chunk it was generated from. We score:
  chunk recall@k — ground-truth chunk in top-k results
  doc recall@k   — ground-truth document in top-k (overlapping chunks from
                   the same doc usually contain the same answer, so this is
                   the metric closest to "did the user get the answer")
  MRR            — mean reciprocal rank of the ground-truth chunk

Three retrieval modes are compared, which shows WHY the hybrid pipeline
exists: dense-only, sparse-only, and hybrid+rerank.

Usage:
  python eval/run_eval.py
  python eval/run_eval.py --golden eval/golden_set.json --k 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.dense import dense_search
from retrieval.hybrid import hybrid_search
from retrieval.pipeline import hybrid_retrieve
from retrieval.sparse import sparse_search


def _score(golden: list[dict], retrieve, k: int) -> dict:
    chunk_hits = doc_hits = 0
    rr_sum = 0.0
    latencies = []
    for item in golden:
        t0 = time.time()
        hits = retrieve(item["question"])[:k]
        latencies.append(time.time() - t0)
        ids = [h.chunk_id for h in hits]
        docs = [h.metadata.get("doc_id") for h in hits]
        if item["chunk_id"] in ids:
            chunk_hits += 1
            rr_sum += 1.0 / (ids.index(item["chunk_id"]) + 1)
        if item["doc_id"] in docs:
            doc_hits += 1
    n = len(golden)
    lat_sorted = sorted(latencies)
    return {
        "chunk_recall": chunk_hits / n,
        "doc_recall": doc_hits / n,
        "mrr": rr_sum / n,
        "p50_ms": lat_sorted[n // 2] * 1000,
        "p95_ms": lat_sorted[int(n * 0.95)] * 1000,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="eval/golden_set.json")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default="eval/results.json")
    args = ap.parse_args()

    golden = json.loads(Path(args.golden).read_text())
    print(f"Evaluating {len(golden)} questions, k={args.k}\n")

    modes = {
        "dense only": lambda q: dense_search(q, k=args.k),
        "sparse only (FTS5)": lambda q: sparse_search(q, k=args.k),
        "hybrid RRF (no rerank)": lambda q: hybrid_search(q, top_k=args.k),
        "hybrid + rerank (full)": lambda q: hybrid_retrieve(q, top_n=args.k),
    }

    results = {}
    header = f"{'mode':<24} {'chunk@'+str(args.k):>8} {'doc@'+str(args.k):>8} {'MRR':>6} {'p50':>8} {'p95':>8}"
    print(header)
    print("-" * len(header))
    for name, fn in modes.items():
        r = _score(golden, fn, args.k)
        results[name] = r
        print(f"{name:<24} {r['chunk_recall']:>7.1%} {r['doc_recall']:>7.1%} "
              f"{r['mrr']:>6.3f} {r['p50_ms']:>6.0f}ms {r['p95_ms']:>6.0f}ms")

    Path(args.out).write_text(json.dumps(
        {"n_questions": len(golden), "k": args.k, "results": results}, indent=2
    ))
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
