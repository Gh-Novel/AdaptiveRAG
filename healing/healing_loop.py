"""Self-Healing RAG orchestrator.

Runs up to MAX_HEAL_ATTEMPTS rounds of:
  diagnose → fix (hallucination / chunk quality / knowledge gap) → regenerate
Returns a HealingResult with the improved answer, citations, health score (0-100),
and a full trace of every action taken.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from healing.chunk_quality_scorer import expand_chunk, score_and_expand
from healing.hallucination_detector import detect_hallucinations
from healing.knowledge_gap_detector import detect_knowledge_gap, web_search
from healing.query_rewriter import multi_query_retrieve
from retrieval.dense import Hit
from retrieval.pipeline import hybrid_retrieve

MAX_HEAL_ATTEMPTS = 3

_ANSWER_SYSTEM = (
    "You are a careful research assistant. Use ONLY the provided passages to "
    "answer the question. Cite inline with [N]. If passages are insufficient, say so."
)
_ANSWER_PROMPT = (
    "Question: {question}\n\nPassages:\n{context}\n\n"
    "Write a concise, well-grounded answer with inline [N] citations."
)


@dataclass
class HealingResult:
    answer: str
    citations: list[dict]
    health_score: float
    attempts_used: int
    healing_trace: list[dict] = field(default_factory=list)


# ── internal helpers ────────────────────────────────────────────────

def _format_context(hits: list[Hit]) -> tuple[str, list[dict]]:
    lines, citations = [], []
    for i, h in enumerate(hits, start=1):
        meta = h.metadata
        title = meta.get("title") or meta.get("source_path", "?")
        tag = " [WEB]" if meta.get("source") == "web" else ""
        head = f"[{i}] {title}{tag} (p.{meta.get('page_start')}-{meta.get('page_end')})"
        lines.append(f"{head}\n{h.text}")
        citations.append(
            {
                "n": i,
                "chunk_id": h.chunk_id,
                "title": title + tag,
                "source_path": meta.get("source_path"),
                "page_start": meta.get("page_start"),
                "page_end": meta.get("page_end"),
                "score": float(h.score),
                "from_web": meta.get("source") == "web",
            }
        )
    return "\n\n".join(lines), citations


def _dedupe(hits: list[Hit], limit: int = 8) -> list[Hit]:
    seen: set[str] = set()
    out: list[Hit] = []
    for h in sorted(hits, key=lambda x: x.score, reverse=True):
        if h.chunk_id not in seen:
            seen.add(h.chunk_id)
            out.append(h)
            if len(out) >= limit:
                break
    return out


def _health_score(
    answer: str,
    unsupported: list[dict],
    chunk_scores: list[dict],
    gap: bool,
) -> float:
    n_sents = max(1, len(re.split(r"(?<=[.!?])\s+", answer.strip())))
    grounding = 1.0 - len(unsupported) / n_sents
    avg_quality = (
        sum(s["score"] for s in chunk_scores) / len(chunk_scores)
        if chunk_scores
        else 0.7
    )
    score = (grounding * 0.50 + avg_quality * 0.30 + 0.20) * 100
    if gap:
        score -= 20
    return round(float(np.clip(score, 0, 100)), 1)


# ── public API ──────────────────────────────────────────────────────

def self_heal(
    query: str,
    answer: str,
    hits: list[Hit],
    citations: list[dict],
    llm=None,
) -> HealingResult:
    if llm is None:
        from llm.client_factory import get_llm
        llm = get_llm()

    current_answer = answer
    current_hits = list(hits)
    current_citations = list(citations)
    healing_trace: list[dict] = []
    attempts_used = 0

    for attempt in range(MAX_HEAL_ATTEMPTS):
        log: dict = {
            "attempt": attempt + 1,
            "issues": [],
            "actions": [],
            "healthy": False,
        }

        # ── 1. Diagnose ────────────────────────────────────────────
        unsupported = detect_hallucinations(current_answer, current_hits)
        _, chunk_scores = score_and_expand(query, current_hits)
        low_quality = [s for s in chunk_scores if s["needs_expansion"]]
        gap = detect_knowledge_gap(current_answer, current_hits)

        if unsupported:
            log["issues"].append("hallucination")
        if low_quality:
            log["issues"].append("low_chunk_quality")
        if gap:
            log["issues"].append("knowledge_gap")

        if not log["issues"]:
            log["healthy"] = True
            healing_trace.append(log)
            break

        attempts_used += 1

        # ── 2a. Fix hallucinations ─────────────────────────────────
        if "hallucination" in log["issues"]:
            flagged_sents = [u["sentence"] for u in unsupported[:3]]
            log["actions"].append(
                {
                    "type": "hallucination_fix",
                    "flagged_count": len(unsupported),
                    "flagged_sentences": flagged_sents,
                    "detail": f"Re-retrieving for {len(flagged_sents)} unsupported sentence(s)",
                }
            )
            seen = {h.chunk_id for h in current_hits}
            for sent in flagged_sents:
                for nh in hybrid_retrieve(sent, top_n=3):
                    if nh.chunk_id not in seen:
                        current_hits.append(nh)
                        seen.add(nh.chunk_id)

        # ── 2b. Fix low chunk quality ──────────────────────────────
        if "low_chunk_quality" in log["issues"]:
            log["actions"].append(
                {
                    "type": "chunk_expansion",
                    "expanded_count": len(low_quality),
                    "detail": f"Expanding {len(low_quality)} low-quality chunk(s) with neighbours",
                }
            )
            seen = {h.chunk_id for h in current_hits}
            for lq in low_quality:
                orig = next(
                    (h for h in current_hits if h.chunk_id == lq["chunk_id"]), None
                )
                if orig:
                    for n in expand_chunk(orig):
                        if n.chunk_id not in seen:
                            current_hits.append(n)
                            seen.add(n.chunk_id)

        # ── 2c. Fix knowledge gap ──────────────────────────────────
        if "knowledge_gap" in log["issues"]:
            web_hits = web_search(query)
            if web_hits:
                log["actions"].append(
                    {
                        "type": "web_search",
                        "results_count": len(web_hits),
                        "detail": f"Knowledge gap → fetched {len(web_hits)} web result(s) via Tavily",
                    }
                )
                current_hits.extend(web_hits)
            else:
                # No Tavily key — expand coverage with multi-query rewriting
                rw_hits = multi_query_retrieve(query, llm=llm, top_n=4)
                seen = {h.chunk_id for h in current_hits}
                added = sum(
                    1
                    for h in rw_hits
                    if h.chunk_id not in seen
                    and not current_hits.append(h)  # side-effect add
                )
                log["actions"].append(
                    {
                        "type": "query_rewrite",
                        "added_count": added,
                        "detail": f"No web API — query rewriting added {added} new chunk(s)",
                    }
                )

        # ── 3. Regenerate ──────────────────────────────────────────
        unique = _dedupe(current_hits, limit=8)
        context_block, current_citations = _format_context(unique)
        current_answer = llm.generate(
            prompt=_ANSWER_PROMPT.format(question=query, context=context_block),
            system=_ANSWER_SYSTEM,
            temperature=0.1,
        )
        log["actions"].append(
            {
                "type": "regenerate",
                "passages_used": len(unique),
                "detail": f"Answer regenerated using {len(unique)} passages",
            }
        )
        healing_trace.append(log)

    # ── Final health score ─────────────────────────────────────────
    final_unsupported = detect_hallucinations(current_answer, current_hits)
    _, final_scores = score_and_expand(query, current_hits)
    final_gap = detect_knowledge_gap(current_answer, current_hits)
    health = _health_score(current_answer, final_unsupported, final_scores, final_gap)

    return HealingResult(
        answer=current_answer,
        citations=current_citations,
        health_score=health,
        attempts_used=attempts_used,
        healing_trace=healing_trace,
    )
