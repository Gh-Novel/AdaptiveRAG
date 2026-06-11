# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

AdaptiveRAG is a portfolio/demo project: a production-grade RAG pipeline (Modular + Self-RAG + Agentic patterns) over 14 foundational AI papers, with a Streamlit UI that exposes every internal pipeline stage live (embeddings, routing, retrieval, fusion, reranking, critique, healing). It runs locally against Ollama and is deployed on a Hugging Face Docker Space using the Groq API.

## Commands

```bash
source .venv/bin/activate                  # Python 3.12 venv

python ingest.py                           # ingest papers/ → Chroma + FTS5 (+ kb_vN if ≤50 docs)
python ingest.py --reset                   # wipe all indexes first
python ingest.py --papers-dir papers_arxiv # ingest the arXiv corpus (resumable per-doc)
python download_arxiv.py --total 200       # bulk-download arXiv PDFs + metadata.json

streamlit run app.py                       # the UI (port 8501 locally; Dockerfile uses 7860)
python ask.py "your question" --trace      # CLI agent run with full trace

python -c "import ast; ast.parse(open('app.py').read())"   # quick syntax check (no test suite exists)
```

There is no test suite, linter config, or build step. Verification is done by running `ingest.py` / `ask.py` / the Streamlit app.

### Git remotes — push to BOTH

```bash
git push origin main && git push hf main
```

- `origin` → github.com/Gh-Novel/AdaptiveRAG
- `hf` → huggingface.co/spaces/NoobNovel/AdaptiveRAG (deploys the Space on push)
- `storage/chroma/chroma.sqlite3` and `storage/bm25.pkl` are tracked via **Git LFS** (see .gitattributes). The prebuilt index is committed so the HF Space works without ingesting. After the FTS5 swap, `storage/fts.db` must also be committed (LFS) and `bm25.pkl` is legacy.
- `papers/` is gitignored (re-download with `download_papers.sh`).
- README.md frontmatter (`sdk: docker`, etc.) is required by the HF Space — don't remove it.

## Architecture

### Dual LLM backend (the key environment switch)

`config.py` sets `HOSTED = bool(os.environ.get("GROQ_API_KEY"))`. Everything flows from that:

- **GROQ_API_KEY set** (HF Space): `llm/groq_client.py` — `llama-3.1-8b-instant`
- **Not set** (local): `llm/ollama_client.py` — `qwen3-vl:8b-instruct-q8_0-optimized` at localhost:11434

`llm/client_factory.py:get_llm()` picks the client; both expose the same interface: `generate()`, `generate_json()`, `health()`. Anything needing an LLM should accept an `llm=None` param and fall back to `get_llm()`.

### Data flow

```
ingestion:  loader.py (PyMuPDF + title overrides + metadata.json sidecar from
            download_arxiv.py) → chunker.py (semantic, ~1400 chars, 200 overlap)
            → embedder.py (MiniLM-L6, 384-dim, normalized; device auto → MPS on Apple Silicon)
            → indexer.py writes THREE artifacts:
              1. ChromaDB collection "adaptive_rag" (storage/chroma/) — flat index used by the live pipeline
              2. storage/fts.db — SQLite FTS5 inverted index (BM25 + Porter stemming)
              3. kb_vN versioned Chroma collections (via versioning/) — auto only for ≤50-doc corpora
            ingest.py is per-document incremental and resumable (storage/ingest_progress.json)

retrieval:  dense.py (Chroma cosine) ∥ sparse.py (SQLite FTS5 BM25 — replaced rank-bm25's
            O(n) in-memory scan, ~10ms at 280k chunks) → hybrid.py (RRF, k=60)
            → reranker.py (BAAI/bge-reranker-base cross-encoder) → pipeline.py:hybrid_retrieve()

agent:      loop.py:run_agent() = router (RETRIEVE/ANSWER_DIRECTLY/CLARIFY)
            → planner (≤3 sub-queries) → tools.py:vector_search per sub-query
            → answer with [N] citations → critic → refine & retry (≤3 iterations,
            confidence threshold 0.85) → healing layer

healing:    healing_loop.py:self_heal() — ≤3 rounds of diagnose→fix→regenerate.
            Detectors: hallucination (answer-sentence vs chunk cosine < 0.75),
            chunk quality (expands doc_id::cNNNN neighbours), knowledge gap
            (gap phrases or max score < 0.50 → Tavily web search if TAVILY_API_KEY,
            else multi-query rewrite). Produces health_score 0-100.

versioning: document_store.py (SQLite storage/versions.db) + change_detector.py (SHA-256)
            + index_manager.py (kb_vN snapshots; rollback = one pointer update;
            old collections never deleted) + version_router.py (query any version, audit log).
```

### Shared currency: the `Hit` dataclass

`retrieval/dense.py:Hit` (chunk_id, text, metadata, score, rank) is passed through every layer — retrieval, agent, healing, versioning, UI. Chunk IDs follow `{doc_id}::c{NNNN}`; `healing/chunk_quality_scorer.py:expand_chunk()` parses this format to fetch ±1 neighbours, so don't change the ID scheme.

### app.py (Streamlit UI)

Three tabs: **Underhood pipeline** (`visual_pipeline()` re-implements the agent loop inline so each stage renders as it runs — it intentionally duplicates `agent/loop.py` logic rather than calling it), **Image Q&A** (multimodal; only meaningful with Ollama/Qwen3-VL locally), **Knowledge Base** (version history, rollback, cross-version queries). A backend guard at the top stops the app with instructions if neither Groq nor Ollama is reachable.

## Known constraints

- **Reranker model**: `cross-encoder/ms-marco-MiniLM-L-6-v2` returns NaN under torch 2.11 on Apple Silicon — that's why `BAAI/bge-reranker-base` is used. Don't "simplify" back.
- ChromaDB telemetry and Streamlit torch-watcher noise are deliberately silenced at the top of `app.py` and via `Settings(anonymized_telemetry=False)` in `ingestion/indexer.py`.
- Models (embedder, reranker, BM25) are cached with `@lru_cache` singletons; ChromaDB clients are created per-call.
- The HF Space runs on free CPU (2 vCPU / 16 GB); keep heavy models out of the hosted path.
