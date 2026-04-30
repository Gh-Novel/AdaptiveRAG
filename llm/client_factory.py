"""Return the right LLM client based on environment.

Local (Ollama running)  →  OllamaClient
Hosted (GROQ_API_KEY set)  →  GroqClient
"""
from __future__ import annotations

import os


def get_llm():
    if os.environ.get("GROQ_API_KEY"):
        from llm.groq_client import GroqClient
        return GroqClient()
    from llm.ollama_client import OllamaClient
    return OllamaClient()
