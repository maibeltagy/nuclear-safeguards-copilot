"""SentenceTransformer embedding wrapper."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from Configuration.settings import EmbeddingSettings, settings


class Embedder:
    """
    Encode text into dense vectors using Sentence Transformers.

    Lab 8 uses the same model family with normalized embeddings so
    cosine similarity equals dot product.
    """

    def __init__(self, config: EmbeddingSettings | None = None) -> None:
        self.config = config or settings.embedding
        self.model = SentenceTransformer(self.config.model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts and return a 2D numpy array."""
        if not texts:
            return np.zeros((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)

        vectors = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=len(texts) > 64,
        )
        return vectors.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Embed a single query string."""
        return self.encode([query])[0]

    @property
    def dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())
