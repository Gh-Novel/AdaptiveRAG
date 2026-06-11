"""Dense embeddings via sentence-transformers."""
from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from config import EMBEDDING_CONFIG


def _resolve_device() -> str:
    dev = EMBEDDING_CONFIG.get("device", "auto")
    if dev != "auto":
        return dev
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(
        EMBEDDING_CONFIG["model"],
        device=_resolve_device(),
    )


def embed_texts(texts: list[str], show_progress: bool = False) -> list[list[float]]:
    model = get_embedder()
    vecs = model.encode(
        texts,
        batch_size=EMBEDDING_CONFIG["batch_size"],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
    )
    return vecs.tolist()


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
