"""Central configuration for AdaptiveRAG."""
import os
from pathlib import Path

ROOT = Path(__file__).parent.resolve()

# Detect hosting environment
HOSTED = bool(os.environ.get("GROQ_API_KEY"))

LLM_CONFIG = {
    "provider": "groq" if HOSTED else "ollama",
    "model": "llama-3.1-8b-instant" if HOSTED else "qwen3-vl:8b-instruct-q8_0-optimized",
    "base_url": "https://api.groq.com/openai/v1" if HOSTED else "http://localhost:11434",
    "temperature": 0.1,
    "timeout": 60 if HOSTED else 180,
    # Do NOT force num_ctx locally: the optimized Qwen3-VL build loads with
    # ctx 24576, and requesting a different value makes Ollama reload a second
    # 12 GB instance → memory thrash on 24 GB → requests stall until timeout.
    "num_ctx": None,
}

EMBEDDING_CONFIG = {
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "device": "auto",  # auto → MPS on Apple Silicon, else CPU
    "batch_size": 64,
}

RERANKER_CONFIG = {
    "model": "BAAI/bge-reranker-base",
    "device": "cpu",
}

CHUNKING_CONFIG = {
    "target_chunk_chars": 1400,
    "max_chunk_chars": 2200,
    "min_chunk_chars": 350,
    "overlap_chars": 200,
}

RETRIEVAL_CONFIG = {
    "dense_k": 12,
    "sparse_k": 12,
    "rrf_k": 60,
    "rerank_top_n": 5,
}

AGENT_CONFIG = {
    "max_iterations": 3,
    "confidence_threshold": 0.85,
    "max_plan_steps": 3,
}

PATHS = {
    "papers_dir": ROOT / "papers",
    "chroma_dir": ROOT / "storage" / "chroma",
    "bm25_path": ROOT / "storage" / "bm25.pkl",  # legacy (replaced by FTS5)
    "fts_path": ROOT / "storage" / "fts.db",
    "manifest_path": ROOT / "storage" / "manifest.json",
    "ingest_progress_path": ROOT / "storage" / "ingest_progress.json",
    "versions_db": ROOT / "storage" / "versions.db",
}

CHROMA_COLLECTION = "adaptive_rag"
