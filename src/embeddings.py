"""Local embedding module using SentenceTransformers (384 dimensions).

Uses sentence-transformers/all-MiniLM-L6-v2 with L2 normalization
to compute cosine similarity via dot product.
"""

from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer


class LocalEmbedder:
    """Wrapper for sentence-transformers embedding model."""

    _instance: Optional["LocalEmbedder"] = None

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimensions = getattr(self.model, "get_embedding_dimension", self.model.get_sentence_embedding_dimension)()

    def embed_query(self, query: str) -> List[float]:
        """Generate normalized embedding vector for a single query."""
        if not query or not query.strip():
            return [0.0] * self.dimensions
        vec = self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vec.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate normalized embedding vectors for a list of document chunks."""
        if not texts:
            return []
        vecs = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vecs.tolist()


def get_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> LocalEmbedder:
    """Singleton getter to avoid reloading the model weights repeatedly."""
    if LocalEmbedder._instance is None:
        LocalEmbedder._instance = LocalEmbedder(model_name)
    return LocalEmbedder._instance
