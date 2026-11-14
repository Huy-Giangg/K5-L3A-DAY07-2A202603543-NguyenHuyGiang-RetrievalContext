"""Three distinct retrieval methods for Day 7 Agentic RAG:

1. DenseRetriever: Semantic search via LocalEmbedder (384-dim normalized dot product)
2. LexicalRetriever: Exact lexical matching via BM25 (Okapi)
3. HybridRetriever: Reciprocal Rank Fusion (RRF) combining Dense and BM25
"""

import re
import unicodedata
from typing import Any, Dict, List, Optional
import numpy as np
from rank_bm25 import BM25Okapi
from src.embeddings import LocalEmbedder, get_embedder


def tokenize(text: str) -> List[str]:
    """Normalize and tokenize text into lowercase word tokens."""
    return re.findall(r"\w+", unicodedata.normalize("NFC", text).casefold())


def match_filters(metadata: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> bool:
    """Check if chunk metadata satisfies all specified filter key-values."""
    if not filters:
        return True
    for key, val in filters.items():
        if metadata.get(key) != val:
            return False
    return True


class DenseRetriever:
    """Method 1: Dense Semantic Retrieval using SentenceTransformers embeddings."""

    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        doc_vectors: Optional[np.ndarray] = None,
        embedder: Optional[LocalEmbedder] = None,
    ) -> None:
        self.chunks = chunks
        self.embedder = embedder or get_embedder()
        if doc_vectors is not None:
            self.doc_vectors = doc_vectors
        else:
            texts = [c["text"] for c in chunks]
            self.doc_vectors = np.array(self.embedder.embed_documents(texts), dtype=np.float32)

    def search(
        self,
        query: str,
        *,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            return []

        q_vec = np.array(self.embedder.embed_query(query), dtype=np.float32)
        # Cosine similarity is dot product of normalized vectors
        scores = np.dot(self.doc_vectors, q_vec)

        # Filter candidates if filters are provided
        candidates = []
        for idx, (chunk, score) in enumerate(zip(self.chunks, scores)):
            if match_filters(chunk.get("metadata", {}), filters):
                candidates.append({
                    **chunk,
                    "score": float(score),
                    "strategy": "dense",
                })

        candidates.sort(key=lambda c: (-c["score"], c["id"]))
        return candidates[:top_k]


class LexicalRetriever:
    """Method 2: Lexical Retrieval using BM25Okapi."""

    def __init__(self, chunks: List[Dict[str, Any]]) -> None:
        self.chunks = chunks
        self.corpus_tokens = [tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(
        self,
        query: str,
        *,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        candidates = []
        for idx, (chunk, score) in enumerate(zip(self.chunks, scores)):
            if score > 0 and match_filters(chunk.get("metadata", {}), filters):
                candidates.append({
                    **chunk,
                    "score": float(score),
                    "strategy": "bm25",
                })

        candidates.sort(key=lambda c: (-c["score"], c["id"]))
        return candidates[:top_k]


class HybridRetriever:
    """Method 3: Hybrid Retrieval combining Dense + BM25 with Reciprocal Rank Fusion (RRF).

    RRF formula: score(d) = sum(1 / (k + rank_i(d)))
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        lexical_retriever: LexicalRetriever,
        rrf_k: int = 60,
    ) -> None:
        self.dense = dense_retriever
        self.lexical = lexical_retriever
        self.rrf_k = rrf_k
        self.chunks_by_id = {c["id"]: c for c in dense_retriever.chunks}

    def search(
        self,
        query: str,
        *,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        pool_k = max(top_k * 3, 20)
        dense_results = self.dense.search(query, filters=filters, top_k=pool_k)
        bm25_results = self.lexical.search(query, filters=filters, top_k=pool_k)

        rrf_scores: Dict[str, float] = {}
        # Rank is 1-based index
        for rank, item in enumerate(dense_results, start=1):
            cid = item["id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))

        for rank, item in enumerate(bm25_results, start=1):
            cid = item["id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))

        merged = []
        for cid, score in rrf_scores.items():
            chunk = self.chunks_by_id[cid]
            merged.append({
                **chunk,
                "score": score,
                "strategy": "hybrid_rrf",
            })

        merged.sort(key=lambda c: (-c["score"], c["id"]))
        return merged[:top_k]
