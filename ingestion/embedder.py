"""Dense embeddings via sentence-transformers."""
from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from config import EMBEDDING_CONFIG


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(
        EMBEDDING_CONFIG["model"],
        device=EMBEDDING_CONFIG["device"],
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedder()
    vecs = model.encode(
        texts,
        batch_size=EMBEDDING_CONFIG["batch_size"],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vecs.tolist()


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
