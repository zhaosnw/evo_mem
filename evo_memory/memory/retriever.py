"""Retriever module for memory search.

Implements the Search operation: R_t = R(M_t, x_t)
where R retrieves relevant memory entries based on similarity.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from dataclasses import dataclass

from .base import Memory, MemoryEntry


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(vec1)
    b = np.array(vec2)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


@dataclass
class RetrievalResult:
    """Result of a retrieval operation."""
    entry: MemoryEntry
    score: float
    rank: int


class Retriever(ABC):
    """
    Abstract base class for memory retrieval.

    The retriever implements: R_t = R(M_t, x_t)
    """

    @abstractmethod
    def encode(self, text: str) -> List[float]:
        """Encode text to embedding vector."""
        pass

    @abstractmethod
    def retrieve(
        self,
        query: str,
        memory: Memory,
        top_k: int = 4,
        threshold: float = 0.0,
    ) -> List[RetrievalResult]:
        """
        Retrieve top-k relevant memory entries.

        Args:
            query: Query text (x_t)
            memory: Memory store (M_t)
            top_k: Number of entries to retrieve
            threshold: Minimum similarity threshold

        Returns:
            List of retrieval results sorted by relevance
        """
        pass

    def update_embeddings(self, memory: Memory) -> None:
        """Update embeddings for all memory entries."""
        for entry in memory:
            if entry.embedding is None:
                text = entry.to_text(include_trajectory=True, include_feedback=True)
                entry.embedding = self.encode(text)


class EmbeddingRetriever(Retriever):
    """
    Embedding-based retriever using sentence transformers.

    Uses BAAI/bge-base-en-v1.5 as the default encoder (as in the paper).
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        device: Optional[str] = None,
        cache_embeddings: bool = True,
    ):
        """
        Initialize embedding retriever.

        Args:
            model_name: Name of the sentence transformer model
            device: Device to use (cuda, cpu, or None for auto)
            cache_embeddings: Whether to cache computed embeddings
        """
        self.model_name = model_name
        self.device = device
        self.cache_embeddings = cache_embeddings
        self._model = None
        self._embedding_cache: Dict[str, List[float]] = {}

    @property
    def model(self):
        """Lazy load the embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device=self.device)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for EmbeddingRetriever. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model

    def encode(self, text: str) -> List[float]:
        """Encode text to embedding vector."""
        # Check cache
        if self.cache_embeddings and text in self._embedding_cache:
            return self._embedding_cache[text]

        # Encode
        embedding = self.model.encode(text, convert_to_numpy=True)
        embedding_list = embedding.tolist()

        # Cache
        if self.cache_embeddings:
            self._embedding_cache[text] = embedding_list

        return embedding_list

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Encode multiple texts efficiently."""
        # Filter out cached texts
        to_encode = []
        to_encode_idx = []
        results = [None] * len(texts)

        for i, text in enumerate(texts):
            if self.cache_embeddings and text in self._embedding_cache:
                results[i] = self._embedding_cache[text]
            else:
                to_encode.append(text)
                to_encode_idx.append(i)

        # Encode uncached texts
        if to_encode:
            embeddings = self.model.encode(to_encode, convert_to_numpy=True)
            for idx, (text, emb) in zip(to_encode_idx, zip(to_encode, embeddings)):
                emb_list = emb.tolist()
                results[idx] = emb_list
                if self.cache_embeddings:
                    self._embedding_cache[text] = emb_list

        return results

    def retrieve(
        self,
        query: str,
        memory: Memory,
        top_k: int = 4,
        threshold: float = 0.0,
    ) -> List[RetrievalResult]:
        """
        Retrieve top-k relevant memory entries.

        Implements: R_t = Top-k_{m_i ∈ M_t} φ(x_t, m_i)
        where φ is cosine similarity.
        """
        if len(memory) == 0:
            return []

        # Encode query
        query_embedding = self.encode(query)

        # Compute similarities
        results = []
        for entry in memory:
            # Ensure entry has embedding
            if entry.embedding is None:
                text = entry.to_text(include_trajectory=True, include_feedback=True)
                entry.embedding = self.encode(text)

            # Compute similarity
            score = cosine_similarity(query_embedding, entry.embedding)

            if score >= threshold:
                results.append((entry, score))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)

        # Return top-k
        return [
            RetrievalResult(entry=entry, score=score, rank=i + 1)
            for i, (entry, score) in enumerate(results[:top_k])
        ]

    def compute_task_similarity(self, memory: Memory) -> float:
        """
        Compute within-dataset task similarity.

        As described in the paper: measures average cosine distance
        between each task embedding and dataset cluster center.
        """
        if len(memory) < 2:
            return 1.0

        # Ensure all entries have embeddings
        self.update_embeddings(memory)

        # Compute cluster center
        embeddings = [e.embedding for e in memory if e.embedding is not None]
        if not embeddings:
            return 1.0

        center = np.mean(embeddings, axis=0)

        # Compute average similarity to center
        similarities = [
            cosine_similarity(emb, center.tolist())
            for emb in embeddings
        ]

        return float(np.mean(similarities))

    def clear_cache(self) -> None:
        """Clear embedding cache."""
        self._embedding_cache.clear()


class RandomRetriever(Retriever):
    """Random retriever for baseline comparison."""

    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim

    def encode(self, text: str) -> List[float]:
        """Generate random embedding."""
        return np.random.randn(self.embedding_dim).tolist()

    def retrieve(
        self,
        query: str,
        memory: Memory,
        top_k: int = 4,
        threshold: float = 0.0,
    ) -> List[RetrievalResult]:
        """Retrieve random entries."""
        import random

        entries = list(memory)
        if len(entries) <= top_k:
            selected = entries
        else:
            selected = random.sample(entries, top_k)

        return [
            RetrievalResult(entry=entry, score=0.5, rank=i + 1)
            for i, entry in enumerate(selected)
        ]


class RecencyRetriever(Retriever):
    """Retrieve most recent entries (for ExpRecent baseline)."""

    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim

    def encode(self, text: str) -> List[float]:
        """Generate placeholder embedding."""
        return [0.0] * self.embedding_dim

    def retrieve(
        self,
        query: str,
        memory: Memory,
        top_k: int = 4,
        threshold: float = 0.0,
    ) -> List[RetrievalResult]:
        """Retrieve most recent entries."""
        recent = memory.get_recent(top_k)

        return [
            RetrievalResult(entry=entry, score=1.0 - (i * 0.1), rank=i + 1)
            for i, entry in enumerate(recent)
        ]
