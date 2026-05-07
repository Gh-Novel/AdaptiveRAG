"""Multi-angle query rewriting.

Generates 3 alternative phrasings via the LLM, runs hybrid retrieval on
all of them, merges and deduplicates the results. This increases the
probability of hitting relevant chunks that a vague original query missed.
"""
from __future__ import annotations

from retrieval.dense import Hit
from retrieval.pipeline import hybrid_retrieve

REWRITE_SYSTEM = "You are a search-query optimisation expert for a research paper assistant."

REWRITE_PROMPT = """Generate 3 alternative phrasings of the user query.
Each phrasing should approach the topic from a different angle so that
together they maximise retrieval coverage across academic papers.

Original query: {query}

Return strict JSON only:
{{"rewrites": ["<phrasing 1>", "<phrasing 2>", "<phrasing 3>"]}}
"""


def rewrite_query(query: str, llm=None) -> list[str]:
    if llm is None:
        from llm.client_factory import get_llm
        llm = get_llm()
    out = llm.generate_json(
        prompt=REWRITE_PROMPT.format(query=query),
        system=REWRITE_SYSTEM,
        temperature=0.3,
    )
    rewrites = out.get("rewrites") if isinstance(out, dict) else None
    if not rewrites or not isinstance(rewrites, list):
        return [query]
    return [str(r).strip() for r in rewrites[:3] if str(r).strip()]


def multi_query_retrieve(query: str, llm=None, top_n: int = 5) -> list[Hit]:
    """Retrieve with original + 3 rewrites, then merge and deduplicate by score."""
    rewrites = rewrite_query(query, llm=llm)
    all_queries = [query] + rewrites
    best: dict[str, Hit] = {}
    for q in all_queries:
        for h in hybrid_retrieve(q, top_n=top_n):
            if h.chunk_id not in best or h.score > best[h.chunk_id].score:
                best[h.chunk_id] = h
    return sorted(best.values(), key=lambda h: h.score, reverse=True)[: top_n * 2]
