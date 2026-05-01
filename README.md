---
title: AdaptiveRAG
sdk: docker
pinned: true
license: mit
short_description: Agentic + Self-RAG + Modular RAG with visual pipeline UI
---

<div align="center">

# 📚 AdaptiveRAG

### Production-grade RAG combining Modular · Self-RAG · Agentic patterns

[![HF Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/NoobNovel/AdaptiveRAG)
[![GitHub](https://img.shields.io/badge/GitHub-Gh--Novel%2FAdaptiveRAG-black?logo=github)](https://github.com/Gh-Novel/AdaptiveRAG)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?logo=streamlit)](https://streamlit.io)

**Every stage of the pipeline is visualized live — from raw text to grounded answer with citations.**

[🚀 Try Live Demo](https://huggingface.co/spaces/NoobNovel/AdaptiveRAG) · [💻 Run Locally](#run-locally)

</div>

---

## 🎬 Demo

<!-- Replace the URL below with your actual demo video link -->
> 📹 **[Watch full pipeline demo →](https://your-video-link-here)**

*Shows: question encoding → Self-RAG routing → hybrid retrieval → 2D vector space → self-critique*

---

## 🧠 What makes this different

Most RAG demos do: `embed query → cosine search → stuff into prompt`. This does:

```
User question
   ↓  embed (MiniLM-L6 → 384-dim vector)
   ↓  Self-RAG router  →  RETRIEVE / ANSWER_DIRECTLY / CLARIFY
   ↓  Planner          →  break into focused sub-queries
   ↓  Dense retrieval  →  ChromaDB cosine similarity (k=12)
   ↓  Sparse retrieval →  BM25 keyword matching (k=12)
   ↓  RRF fusion       →  Reciprocal Rank Fusion merge
   ↓  Cross-encoder    →  BGE reranker deep relevance scoring (top 5)
   ↓  LLM answer       →  Qwen3-VL (local) / LLaMA 3.1 via Groq (hosted)
   ↓  Self-critique     →  grounded? complete? confidence score
   ↓  Refine & retry   →  if confidence < 0.85
   →  Answer + citations + trace
```

---

## 🔬 Underhood Pipeline View

Every step renders its inputs and outputs **as it runs**:

| Step | What you see |
|------|-------------|
| **1 · Question encoding** | Embedding model · 384 dimensions · L2 norm · latency · first-32-dim bar chart · raw `vector[0:8]` values |
| **2 · Self-RAG router** | Color-coded decision pill (`RETRIEVE` / `ANSWER_DIRECTLY` / `CLARIFY`) + LLM reasoning |
| **3 · Planner** | Sub-query cards with rationale for each step |
| **4 · Dense retrieval** | Cosine similarity bar chart + chunk cards with scores |
| **4 · Sparse retrieval** | BM25 normalized score chart + chunk cards |
| **4 · RRF fusion** | Merged ranking chart showing how both lists combine |
| **4 · Cross-encoder rerank** | BGE relevance score chart (final top-5) |
| **4 · Vector space** | 2D PCA scatter — query vs all hits, colored by source (dense / sparse / both) |
| **5 · Context assembly** | Exact passages handed to the LLM, with metadata |
| **6 · Self-critique** | Grounded ✅ · Complete ✅ · Confidence score vs threshold |

---

## 🗂️ Knowledge Base

14 foundational AI papers pre-indexed as **1,934 semantic chunks**:

| Category | Papers |
|----------|--------|
| Transformers | Attention Is All You Need · BERT · GPT-3 |
| Diffusion | DDPM · DDIM |
| RAG | RAG Original · RAG Survey · Self-RAG · HyDE |
| Vision | ViT · CLIP |
| Agents | ReAct · Chain-of-Thought |
| LLMs | LLM Survey |

---

## ⚙️ Tech Stack

| Component | Tool | Why |
|-----------|------|-----|
| Vector DB | ChromaDB (local) | No API cost, persistent |
| Dense embeddings | `all-MiniLM-L6-v2` | Fast, 384-dim, normalized |
| Sparse retrieval | `rank-bm25` (BM25Okapi) | Keyword precision |
| Fusion | Reciprocal Rank Fusion | Combines rankings without score normalization |
| Reranker | `BAAI/bge-reranker-base` | Cross-encoder, deep relevance scoring |
| LLM (local) | Qwen3-VL 8B via Ollama | Vision-language, runs on Apple Silicon |
| LLM (hosted) | LLaMA 3.1 8B via Groq | Free API, fast inference |
| UI | Streamlit | Fast to build, easy to demo |

---

## 🚀 Run Locally

```bash
git clone https://github.com/Gh-Novel/AdaptiveRAG
cd AdaptiveRAG

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# start Ollama with the vision-language model
ollama serve
ollama pull qwen3-vl:8b-instruct-q8_0-optimized

streamlit run app.py
```

Or use the CLI:

```bash
.venv/bin/python ask.py "How does Self-RAG decide when to retrieve?"
```

---

## ☁️ Hosted on Hugging Face

The live demo runs on HF Spaces (CPU free tier) with **Groq API** handling LLM calls.

- Embedding + retrieval + reranking run locally inside the container (MiniLM + BGE)
- `GROQ_API_KEY` secret drives routing, planning, answering, and self-critique
- Pre-built index (1,934 chunks, ~59 MB) is committed via git-lfs — no ingestion on startup

**[🤗 Open Live Demo](https://huggingface.co/spaces/NoobNovel/AdaptiveRAG)**
