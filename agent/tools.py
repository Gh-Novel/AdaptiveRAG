"""Agent tools.

The vector_search tool drives the hybrid retriever. image_reason performs
multimodal RAG: caption the image, retrieve text by caption+query, then ask
Qwen3-VL to ground its answer in both image and text.
"""
from __future__ import annotations

from dataclasses import dataclass

from llm.ollama_client import OllamaClient
from retrieval.dense import Hit
from retrieval.pipeline import hybrid_retrieve


@dataclass
class ToolResult:
    tool: str
    query: str
    hits: list[Hit]
    notes: str = ""


def vector_search(query: str, top_n: int | None = None) -> ToolResult:
    hits = hybrid_retrieve(query, top_n=top_n)
    return ToolResult(tool="vector_search", query=query, hits=hits)


CAPTION_SYSTEM = "You describe images in concise, factual language."
CAPTION_PROMPT = (
    "Describe this image in 1-3 sentences. Mention the type of figure (chart, diagram, "
    "screenshot, photo, equation, etc.), key labels, and the main visual content."
)


def caption_image(image_path: str, llm: OllamaClient | None = None) -> str:
    llm = llm or OllamaClient()
    return llm.generate(
        prompt=CAPTION_PROMPT,
        system=CAPTION_SYSTEM,
        images=[image_path],
        temperature=0.0,
    )


MM_SYSTEM = (
    "You are a careful research assistant. Answer using ONLY the provided image and "
    "the cited text passages. If the answer is not supported, say so."
)
MM_PROMPT = """Image (provided separately) + question.

Question: {question}

Relevant passages:
{context}

Answer concisely. When citing a passage, use [N] where N is the passage number.
"""


def image_retrieve_and_reason(
    image_path: str, query: str, llm: OllamaClient | None = None
) -> dict:
    llm = llm or OllamaClient()
    caption = caption_image(image_path, llm=llm)
    fused_query = f"{caption} {query}".strip()
    hits = hybrid_retrieve(fused_query)
    context_block = _format_context(hits)
    answer = llm.generate(
        prompt=MM_PROMPT.format(question=query, context=context_block),
        system=MM_SYSTEM,
        images=[image_path],
        temperature=0.1,
    )
    return {"caption": caption, "answer": answer, "hits": hits}


def _format_context(hits: list[Hit]) -> str:
    lines = []
    for i, h in enumerate(hits, start=1):
        meta = h.metadata
        head = f"[{i}] {meta.get('title', meta.get('source_path', '?'))} "
        head += f"(p.{meta.get('page_start')}-{meta.get('page_end')})"
        lines.append(f"{head}\n{h.text}")
    return "\n\n".join(lines)
