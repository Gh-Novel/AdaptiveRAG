"""Detect when the knowledge base lacks sufficient coverage.

Two signals:
  1. Answer contains phrases like "I don't have information..."
  2. All retrieved chunks have similarity < GAP_SIM_THRESHOLD

When a gap is detected, optionally call Tavily web search
(requires TAVILY_API_KEY env variable).
"""
from __future__ import annotations

import os

import requests

from retrieval.dense import Hit

GAP_SIM_THRESHOLD = 0.50

_GAP_PHRASES = [
    "i don't have",
    "i do not have",
    "not in context",
    "no information about",
    "cannot find",
    "not available in",
    "not found in",
    "i'm not able to find",
    "there is no information",
    "the provided passages do not",
    "the context does not contain",
    "i cannot answer",
    "not mentioned in",
    "not discussed in",
    "insufficient information",
    "no relevant",
]


def detect_knowledge_gap(answer: str, hits: list[Hit]) -> bool:
    al = answer.lower()
    if any(p in al for p in _GAP_PHRASES):
        return True
    if hits and max(h.score for h in hits) < GAP_SIM_THRESHOLD:
        return True
    return False


def web_search(query: str, max_results: int = 3) -> list[Hit]:
    """Call Tavily API if TAVILY_API_KEY is set; otherwise return []."""
    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        return []
    try:
        r = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "max_results": max_results,
                "include_answer": False,
            },
            timeout=10,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
    except Exception:
        return []
    return [
        Hit(
            chunk_id=f"web::{i}",
            text=res.get("content") or res.get("snippet", ""),
            metadata={
                "title": res.get("title", "Web result"),
                "source_path": res.get("url", "web"),
                "page_start": 0,
                "page_end": 0,
                "source": "web",
            },
            score=float(res.get("score", 0.7)),
            rank=i,
        )
        for i, res in enumerate(results)
        if res.get("content") or res.get("snippet")
    ]
