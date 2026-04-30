"""Self-RAG router. Decide whether to retrieve, answer directly, or clarify."""
from __future__ import annotations

from llm.ollama_client import OllamaClient

ROUTER_SYSTEM = (
    "You are a routing classifier for an AI research assistant whose knowledge base "
    "contains papers on Transformers, BERT, GPT-3, diffusion (DDPM/DDIM), RAG, "
    "Self-RAG, HyDE, ViT, CLIP, ReAct, Chain-of-Thought, and an LLM survey. "
    "Decide how to handle a user query."
)

ROUTER_PROMPT = """Classify the query into one of three actions:

- "RETRIEVE": the user is asking about substantive content (concepts, methods, comparisons,
  details from papers). The knowledge base is likely needed. Default to this when unsure.
- "ANSWER_DIRECTLY": pure conversational/meta queries (greetings, "what can you do",
  "thanks") that need NO knowledge lookup.
- "CLARIFY": the query is too ambiguous or under-specified to act on (e.g. "tell me more"
  with no prior context, "what about that paper" with no referent).

Respond with strict JSON only:
{{"action": "RETRIEVE" | "ANSWER_DIRECTLY" | "CLARIFY", "reason": "<one short sentence>"}}

Query: {query}
"""


def route(query: str, llm: OllamaClient | None = None) -> dict:
    llm = llm or OllamaClient()
    out = llm.generate_json(
        prompt=ROUTER_PROMPT.format(query=query),
        system=ROUTER_SYSTEM,
        temperature=0.0,
    )
    action = str(out.get("action", "RETRIEVE")).upper()
    if action not in {"RETRIEVE", "ANSWER_DIRECTLY", "CLARIFY"}:
        action = "RETRIEVE"
    return {"action": action, "reason": out.get("reason", "")}
