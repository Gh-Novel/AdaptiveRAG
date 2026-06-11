---
title: AdaptiveRAG
sdk: docker
pinned: true
license: mit
short_description: Agentic + Self-RAG + Modular RAG with visual pipeline UI
---

# AdaptiveRAG — Agentic + Self-RAG + Modular RAG

![Python](https://img.shields.io/badge/Python-3.12-blue)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-orange)
![SQLite FTS5](https://img.shields.io/badge/Sparse-SQLite_FTS5-green)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A production-grade RAG system where **every internal stage is visible live in the UI** — embedding vectors, routing decisions, dense vs. sparse hits, rank fusion, cross-encoder scores, self-critique, and self-healing. Built to demonstrate how a real retrieval pipeline works under the hood, not just that it works.

**▶ Live demo:** https://huggingface.co/spaces/NoobNovel/AdaptiveRAG

https://github.com/user-attachments/assets/5ae152da-00bd-4812-9743-2aaab1ffc5d6

---

## Architecture

```
                              ┌─────────────────────┐
 question ──► embed (MiniLM) ─► Self-RAG router     │ RETRIEVE / ANSWER_DIRECTLY / CLARIFY
                              └──────────┬──────────┘
                                         ▼
                              ┌─────────────────────┐
                              │ Planner (LLM)       │ decomposes into ≤3 sub-queries
                              └──────────┬──────────┘
                                         ▼
                   ┌─────────────────────┴─────────────────────┐
                   ▼                                           ▼
        Dense retrieval (ChromaDB,                Sparse retrieval (SQLite FTS5,
        cosine, HNSW, 384-dim)                    native BM25 + Porter stemming)
                   └─────────────────────┬─────────────────────┘
                                         ▼
                          Reciprocal Rank Fusion (k=60)
                                         ▼
                          Cross-encoder rerank (BGE-reranker-base)
                                         ▼
                          Answer generation with [N] citations
                                         ▼
                          Self-critique → refine query → retry (≤3 iterations)
                                         ▼
                  ⚕️ Self-Healing: hallucination detection · chunk expansion
                     · query rewriting · knowledge-gap web search
                                         ▼
                              answer + citations + health score
```

## What makes it more than a tutorial RAG

| Layer | What it does | Why it matters |
|---|---|---|
| **Self-RAG router** | LLM decides RETRIEVE / ANSWER_DIRECTLY / CLARIFY *before* touching the index | No wasted retrieval on greetings or chit-chat |
| **Agentic loop** | plan → retrieve → answer → self-critique → refine → retry | Answers below the 0.85 confidence threshold get a refined second pass |
| **Hybrid retrieval** | Dense (semantic) ∥ Sparse (keyword) merged with RRF | Dense misses exact terms, sparse misses synonyms — fusion catches both |
| **Cross-encoder rerank** | BGE reranker scores (query, chunk) pairs jointly | Far more accurate than bi-encoder cosine; only run on the fused candidate set |
| **⚕️ Self-Healing** | Per-sentence grounding check (cosine < 0.75 → flagged), low-quality chunk expansion via document neighbours, multi-angle query rewriting, knowledge-gap detection with optional Tavily web search | The answer is diagnosed and regenerated up to 3× — with a 0–100 health score shown in the UI |
| **KB Versioning** | Every ingest = snapshot (`kb_v1`, `kb_v2`…) with SHA-256 change detection; only changed docs re-embedded; rollback = one metadata write; every query logged with the version that answered it | Production concern most demos skip: replay any historical query, roll back bad data in milliseconds |
| **Multimodal Q&A** | Image → caption → hybrid retrieve → vision-LLM reasons over image + passages | Figure-level questions on papers (local Qwen3-VL) |
| **Underhood UI** | Every stage renders its actual inputs/outputs as it runs, incl. 2D PCA projection of the query + hit embeddings | You can *watch* RRF change the ranking |

## Engineered for scale (2,000+ papers)

The pipeline was load-tested beyond toy size and the bottlenecks were fixed, not hidden:

- **Sparse search: rank-bm25 → SQLite FTS5.** The original in-memory BM25 scored every chunk in Python — O(n) per query, seconds at ~300k chunks, ~3 GB RAM. Replaced with a disk-backed FTS5 inverted index (native BM25 + Porter stemming): **2.7 ms per query**, near-zero memory, and the latency stays flat as the corpus grows.
- **Embedding on Apple Silicon MPS** with auto device detection — CPU on the hosted Space, GPU-accelerated locally.
- **Resumable ingestion.** Per-document checkpointing (`ingest_progress.json`): kill the run at paper 1,400 of 2,000 and it resumes at 1,401. Corrupt PDFs are logged and skipped, never fatal.
- **Bulk corpus tooling.** `download_arxiv.py` pulls papers by category from the arXiv API with full metadata (title/authors/abstract) — titles scale via a metadata sidecar instead of a hand-written dict.
- **Dual LLM backend.** One env var (`GROQ_API_KEY`) switches the whole system between local Ollama (Qwen3-VL) and the Groq API (LLaMA 3.1) — same client interface, zero code changes.

## Tech stack

| Component | Choice |
|---|---|
| Dense embeddings | `all-MiniLM-L6-v2` (384-dim, normalized) |
| Vector store | ChromaDB (HNSW, cosine) |
| Sparse index | SQLite FTS5 (BM25, Porter stemming) |
| Fusion | Reciprocal Rank Fusion (k=60) |
| Reranker | `BAAI/bge-reranker-base` cross-encoder |
| LLM (hosted) | LLaMA 3.1 8B via Groq |
| LLM (local) | Qwen3-VL 8B via Ollama (multimodal) |
| Versioning | SQLite + SHA-256 change detection |
| UI | Streamlit |
| PDF parsing | PyMuPDF with semantic chunking (~1,400 chars, 200 overlap, heading-aware) |

## Knowledge base

14 foundational AI papers (Transformers, BERT, GPT-3, DDPM/DDIM, RAG, Self-RAG, HyDE, ViT, CLIP, ReAct, Chain-of-Thought, LLM surveys) — expandable to thousands via the arXiv bulk downloader.

## Run locally

```bash
git clone https://github.com/Gh-Novel/AdaptiveRAG
cd AdaptiveRAG
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Option A — local LLM: needs Ollama running (multimodal Qwen3-VL)
ollama serve

# Option B — hosted LLM: set a free Groq key instead
export GROQ_API_KEY=...

streamlit run app.py
```

```bash
# Optional: grow the corpus
python download_arxiv.py --total 200        # bulk-download arXiv papers + metadata
python ingest.py --papers-dir papers_arxiv  # resumable, incremental indexing
```
