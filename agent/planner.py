"""Multi-step query planner. Break a question into focused sub-queries."""
from __future__ import annotations

from config import AGENT_CONFIG
from llm.ollama_client import OllamaClient

PLANNER_SYSTEM = (
    "You are a research planner. Given a user question, decompose it into a small "
    "number of focused sub-queries. Each sub-query targets one piece of information "
    "needed to answer the original question. Avoid redundant or overly broad steps."
)

PLANNER_PROMPT = """Decompose the user question into 1-{max_steps} focused retrieval sub-queries.
Use fewer steps when the question is simple; only use multiple steps for genuinely
multi-part or comparative questions.

Each sub-query should be a self-contained search query (10-20 words) phrased to
match passages in academic papers.

Respond with strict JSON only:
{{"steps": [
  {{"query": "<search query>", "rationale": "<what this sub-query is looking for>"}}
]}}

User question: {query}

Context already gathered (may be empty):
{context_summary}
"""


def plan(query: str, prior_summary: str = "", llm: OllamaClient | None = None) -> list[dict]:
    llm = llm or OllamaClient()
    out = llm.generate_json(
        prompt=PLANNER_PROMPT.format(
            query=query,
            max_steps=AGENT_CONFIG["max_plan_steps"],
            context_summary=prior_summary or "(none)",
        ),
        system=PLANNER_SYSTEM,
        temperature=0.1,
    )
    steps = out.get("steps") if isinstance(out, dict) else None
    if not steps or not isinstance(steps, list):
        return [{"query": query, "rationale": "fallback: use the original question"}]
    cleaned: list[dict] = []
    for s in steps[: AGENT_CONFIG["max_plan_steps"]]:
        if isinstance(s, dict) and s.get("query"):
            cleaned.append(
                {"query": str(s["query"]).strip(), "rationale": str(s.get("rationale", "")).strip()}
            )
    return cleaned or [{"query": query, "rationale": "fallback"}]
