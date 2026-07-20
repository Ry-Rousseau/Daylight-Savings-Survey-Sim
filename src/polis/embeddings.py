"""Local embedding model (Layer 1 — Engine).

Embeddings run locally on the GPU (RTX 4070 Ti SUPER), separate from the paid
OpenRouter chat endpoint which is chat-only. Model id is a field and logged for
provenance (R1/R6 spirit). BGE wants unit-normalized vectors for cosine, so we
normalize at encode time and treat relevance as a plain dot product downstream.
"""
from __future__ import annotations

import numpy as np

DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # 384-dim, fast, strong retrieval


class EmbeddingModel:
    """Lazy-loaded sentence-transformers wrapper; loads on first ``encode``."""

    def __init__(self, model_id: str = DEFAULT_EMBED_MODEL, device: str | None = None):
        self.model_id = model_id
        self._device = device
        self._model = None

    def _ensure(self):
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._device = device
            self._model = SentenceTransformer(self.model_id, device=device)
        return self._model

    @property
    def device(self) -> str | None:
        return self._device

    def encode(self, texts: str | list[str]) -> np.ndarray:
        """Return unit-normalized float32 embeddings. 1-D for a single string."""
        single = isinstance(texts, str)
        model = self._ensure()
        vecs = model.encode(
            [texts] if single else list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        return vecs[0] if single else vecs
