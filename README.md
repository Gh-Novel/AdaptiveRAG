---
title: AdaptiveRAG
sdk: docker
pinned: true
license: mit
short_description: Agentic + Self-RAG + Modular RAG with visual pipeline UI
---

# AdaptiveRAG — Agentic + Self-RAG + Modular RAG

Live demo of a production-grade RAG pipeline — every stage is visible in the UI.

**Tech stack:** ChromaDB · sentence-transformers · BM25 · Reciprocal Rank Fusion · BGE cross-encoder · LLaMA 3.1 via Groq

**Knowledge base:** 14 foundational AI papers (Transformers, BERT, GPT-3, DDPM, RAG, Self-RAG, HyDE, ViT, CLIP, ReAct, Chain-of-Thought, LLM Survey)

## What it shows

| Stage | What you see |
|---|---|
| Question encoding | 384-dim embedding vector + bar chart of first 32 dims |
| Self-RAG router | RETRIEVE / ANSWER_DIRECTLY / CLARIFY decision + reason |
| Planner | Sub-query decomposition with rationales |
| Dense retrieval | Cosine similarity scores vs ChromaDB |
| Sparse retrieval | BM25 keyword match scores |
| RRF fusion | Combined ranking chart |
| Cross-encoder rerank | BGE relevance scores |
| Vector space | 2D PCA projection of query + hits |
| Self-critique | Grounded / Complete / Confidence score |

## Run locally

```bash
git clone https://github.com/Gh-Novel/AdaptiveRAG
cd AdaptiveRAG
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# needs Ollama running with qwen3-vl:8b-instruct-q8_0-optimized
streamlit run app.py
```
