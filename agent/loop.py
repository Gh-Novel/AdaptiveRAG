"""The agentic RAG loop.

Pipeline:
  1. Self-RAG router: RETRIEVE / ANSWER_DIRECTLY / CLARIFY
  2. (RETRIEVE branch) plan -> tools -> answer -> self-critique -> refine
  3. Self-Healing layer: hallucination check, chunk expansion, gap detection
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import AGENT_CONFIG
from agent.critic import critique, refine_query
from agent.planner import plan
from agent.router import route
from agent.tools import ToolResult, vector_search
from llm.client_factory import get_llm
from retrieval.dense import Hit


@dataclass
class TraceStep:
    kind: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    answer: str
    citations: list[dict]
    confidence: float
    trace: list[TraceStep]
    iterations: int
    route: str
    healing_trace: list[dict] = field(default_factory=list)
    health_score: float = 100.0


ANSWER_SYSTEM = (
    "You are a careful research assistant. Use ONLY the provided passages to "
    "answer the question. Cite sources inline with [N] where N is the passage "
    "number. If the passages are insufficient, say so explicitly."
)

ANSWER_PROMPT = """Question: {question}

Passages:
{context}

Write a concise, well-grounded answer. Use inline citations like [1], [2] that
match the passage numbers above. If multiple passages support a claim, cite
them all. If the passages do not contain enough information, say so plainly.
"""


def _format_context_block(hits: list[Hit]) -> tuple[str, list[dict]]:
    lines = []
    citations = []
    for i, h in enumerate(hits, start=1):
        meta = h.metadata
        title = meta.get("title") or meta.get("source_path", "?")
        pages = f"p.{meta.get('page_start')}-{meta.get('page_end')}"
        head = f"[{i}] {title} ({pages})"
        lines.append(f"{head}\n{h.text}")
        citations.append(
            {
                "n": i,
                "chunk_id": h.chunk_id,
                "title": title,
                "source_path": meta.get("source_path"),
                "page_start": meta.get("page_start"),
                "page_end": meta.get("page_end"),
                "score": float(h.score),
            }
        )
    return "\n\n".join(lines), citations


def _dedupe_hits(hits: list[Hit], limit: int) -> list[Hit]:
    seen: set[str] = set()
    out: list[Hit] = []
    for h in hits:
        if h.chunk_id in seen:
            continue
        seen.add(h.chunk_id)
        out.append(h)
        if len(out) >= limit:
            break
    return out


def run_agent(
    query: str,
    llm=None,
    enable_healing: bool = True,
) -> AgentResult:
    llm = llm or get_llm()
    trace: list[TraceStep] = []

    # 1. Router
    decision = route(query, llm=llm)
    trace.append(TraceStep("router", decision))

    if decision["action"] == "ANSWER_DIRECTLY":
        ans = llm.generate(
            prompt=query,
            system="You are a helpful research assistant. Be concise.",
            temperature=0.2,
        )
        return AgentResult(
            answer=ans, citations=[], confidence=1.0,
            trace=trace, iterations=0, route="ANSWER_DIRECTLY",
        )

    if decision["action"] == "CLARIFY":
        ans = llm.generate(
            prompt=(
                "The user asked: " + query + "\n\n"
                "It is too ambiguous to answer well. Ask one short clarifying "
                "question to narrow it down."
            ),
            system="You are a helpful research assistant.",
            temperature=0.2,
        )
        return AgentResult(
            answer=ans, citations=[], confidence=0.0,
            trace=trace, iterations=0, route="CLARIFY",
        )

    # 2. RETRIEVE branch — agentic loop
    current_query = query
    last_critique: dict[str, Any] = {}
    accumulated: list[Hit] = []
    answer = ""
    citations: list[dict] = []
    unique_hits: list[Hit] = []
    final_iteration = 0

    for iteration in range(AGENT_CONFIG["max_iterations"]):
        prior_summary = ""
        if accumulated:
            titles = sorted({h.metadata.get("title", "?") for h in accumulated})
            prior_summary = "Already gathered passages from: " + ", ".join(titles)

        steps = plan(current_query, prior_summary=prior_summary, llm=llm)
        trace.append(TraceStep("plan", {"iteration": iteration, "steps": steps}))

        for step in steps:
            tool_res: ToolResult = vector_search(step["query"])
            accumulated.extend(tool_res.hits)
            trace.append(
                TraceStep(
                    "tool",
                    {
                        "tool": "vector_search",
                        "query": step["query"],
                        "n_hits": len(tool_res.hits),
                        "top_titles": [h.metadata.get("title") for h in tool_res.hits[:3]],
                    },
                )
            )

        unique_hits = _dedupe_hits(accumulated, limit=8)
        context_block, citations = _format_context_block(unique_hits)

        answer = llm.generate(
            prompt=ANSWER_PROMPT.format(question=query, context=context_block),
            system=ANSWER_SYSTEM,
            temperature=0.1,
        )
        trace.append(TraceStep("answer", {"iteration": iteration, "n_passages": len(unique_hits)}))

        crit = critique(query, answer, context_block, llm=llm)
        last_critique = crit
        final_iteration = iteration + 1
        trace.append(TraceStep("critique", {"iteration": iteration, **crit}))

        if crit["confidence"] >= AGENT_CONFIG["confidence_threshold"] and crit["grounded"]:
            break

        current_query = refine_query(query, crit.get("missing", ""), llm=llm)
        trace.append(TraceStep("refine", {"new_query": current_query}))

    # 3. Self-Healing layer
    healing_trace: list[dict] = []
    health_score = 100.0
    if enable_healing and unique_hits:
        from healing.healing_loop import self_heal
        healed = self_heal(query, answer, unique_hits, citations, llm=llm)
        answer = healed.answer
        citations = healed.citations
        healing_trace = healed.healing_trace
        health_score = healed.health_score
        trace.append(
            TraceStep(
                "healing",
                {
                    "attempts": healed.attempts_used,
                    "health_score": health_score,
                    "issues_found": [
                        a["issues"]
                        for a in healing_trace
                        if not a.get("healthy", False)
                    ],
                },
            )
        )

    return AgentResult(
        answer=answer,
        citations=citations,
        confidence=last_critique.get("confidence", 0.0),
        trace=trace,
        iterations=final_iteration,
        route="RETRIEVE",
        healing_trace=healing_trace,
        health_score=health_score,
    )
