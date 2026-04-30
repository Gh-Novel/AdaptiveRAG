"""Self-critique: judge answer for grounding, completeness, confidence."""
from __future__ import annotations

from llm.ollama_client import OllamaClient

CRITIC_SYSTEM = (
    "You are a strict reviewer. Judge whether an AI-generated answer is grounded "
    "in the provided context and whether it fully answers the user's question."
)

CRITIC_PROMPT = """Question: {question}

Answer to review:
{answer}

Context that was provided to the answerer:
{context}

Score the answer. Return strict JSON only:
{{
  "grounded": true | false,            // Is every factual claim supported by the context?
  "complete": true | false,            // Does it fully address the question?
  "confidence": 0.0-1.0,               // Overall confidence in the answer
  "missing": "<what info is missing or weakly supported, or empty string>"
}}
"""


def critique(question: str, answer: str, context: str, llm: OllamaClient | None = None) -> dict:
    llm = llm or OllamaClient()
    out = llm.generate_json(
        prompt=CRITIC_PROMPT.format(question=question, answer=answer, context=context),
        system=CRITIC_SYSTEM,
        temperature=0.0,
    )
    return {
        "grounded": bool(out.get("grounded", False)),
        "complete": bool(out.get("complete", False)),
        "confidence": float(out.get("confidence", 0.0) or 0.0),
        "missing": str(out.get("missing", "") or ""),
    }


REFINE_SYSTEM = (
    "You rewrite a search query so it retrieves the missing information."
)

REFINE_PROMPT = """Original question: {question}

A previous attempt was missing the following information:
{missing}

Rewrite the query to specifically target the missing information. Output the
rewritten search query as a single line of text, no quotes, no explanation.
"""


def refine_query(question: str, missing: str, llm: OllamaClient | None = None) -> str:
    llm = llm or OllamaClient()
    out = llm.generate(
        prompt=REFINE_PROMPT.format(question=question, missing=missing or "more detail"),
        system=REFINE_SYSTEM,
        temperature=0.1,
    )
    return out.strip().splitlines()[0] if out.strip() else question
