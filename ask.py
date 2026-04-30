"""CLI: python ask.py 'your question here'"""
from __future__ import annotations

import argparse
import json

from agent.loop import run_agent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="+")
    ap.add_argument("--trace", action="store_true")
    args = ap.parse_args()
    q = " ".join(args.question)
    res = run_agent(q)
    print("\n=== ROUTE ===")
    print(res.route)
    print("\n=== ANSWER ===")
    print(res.answer)
    print("\n=== CITATIONS ===")
    for c in res.citations:
        print(f"  [{c['n']}] {c['title']} (p.{c['page_start']}-{c['page_end']}) score={c['score']:.3f}")
    print(f"\nConfidence: {res.confidence:.2f}  iterations: {res.iterations}")
    if args.trace:
        print("\n=== TRACE ===")
        for s in res.trace:
            print(f"  • {s.kind}: {json.dumps(s.detail, default=str)[:200]}")


if __name__ == "__main__":
    main()
