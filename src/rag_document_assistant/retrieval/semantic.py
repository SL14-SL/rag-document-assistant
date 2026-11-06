"""Local semantic retrieval using normalized sentence embeddings."""

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from rag_document_assistant.ingestion.pdf_loader import DocumentSection
from rag_document_assistant.retrieval.bm25 import SearchResult

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class TextEncoder(Protocol):
    def encode(self, texts: list[str]) -> NDArray[np.float32]: ...


class LocalSentenceEncoder:
    """Load an embedding model on the CPU; optionally forbid network access."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        local_files_only: bool = False,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            model_name,
            device="cpu",
            local_files_only=local_files_only,
        )

    def encode(self, texts: list[str]) -> NDArray[np.float32]:
        # Reject oversized inputs instead of silently losing document content.
        for text in texts:
            token_ids = self.model.tokenizer(
                text,
                truncation=False,
                add_special_tokens=True,
            )["input_ids"]
            if len(token_ids) > self.model.max_seq_length:
                raise ValueError(
                    "Text exceeds the embedding model input limit; "
                    "split it into smaller chunks first"
                )

        return np.asarray(
            self.model.encode(
                texts,
                batch_size=16,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )


def normalize_vectors(values: NDArray[np.float32]) -> NDArray[np.float32]:
    """Validate and normalize a two-dimensional embedding matrix."""
    vectors = np.asarray(values, dtype=np.float32)

    if vectors.ndim != 2 or vectors.shape[1] == 0:
        raise ValueError("Expected a non-empty two-dimensional embedding matrix")
    if not np.isfinite(vectors).all():
        raise ValueError("Embeddings must contain only finite values")

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Embedding vectors must not be zero")

    return vectors / norms


class SemanticIndex:
    """Exact cosine-similarity search over the small local corpus."""

    def __init__(
        self,
        sections: list[DocumentSection],
        *,
        encoder: TextEncoder,
    ) -> None:
        if not sections:
            raise ValueError("Cannot index an empty corpus")

        citation_ids = [section.citation_id for section in sections]
        if len(citation_ids) != len(set(citation_ids)):
            raise ValueError("Citation IDs must be unique")

        self.sections = tuple(sections)
        self.encoder = encoder

        texts = [
            f"{section.title}\n{section.heading}\n{section.text}"
            for section in sections
        ]
        self.embeddings = normalize_vectors(encoder.encode(texts))

        if self.embeddings.shape[0] != len(sections):
            raise ValueError("Expected one embedding per section")

    @classmethod
    def from_embeddings(
        cls,
        sections: list[DocumentSection],
        embeddings: NDArray[np.float32],
        *,
        encoder: TextEncoder,
    ) -> "SemanticIndex":
        """Restore an index without encoding document text again."""
        if not sections:
            raise ValueError("Cannot restore an empty corpus")

        citation_ids = [section.citation_id for section in sections]
        if len(citation_ids) != len(set(citation_ids)):
            raise ValueError("Citation IDs must be unique")

        vectors = normalize_vectors(embeddings)
        if vectors.shape[0] != len(sections):
            raise ValueError("Expected one embedding per section")

        instance = cls.__new__(cls)
        instance.sections = tuple(sections)
        instance.encoder = encoder
        instance.embeddings = vectors
        return instance

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not query.strip():
            return []

        query_embedding = normalize_vectors(self.encoder.encode([query]))
        if query_embedding.shape != (1, self.embeddings.shape[1]):
            raise ValueError("Query and document embedding dimensions must match")

        scores = self.embeddings @ query_embedding[0]
        ranked = sorted(
            range(len(self.sections)),
            key=lambda i: (-float(scores[i]), self.sections[i].citation_id),
        )

        return [
            SearchResult(section=self.sections[i], score=float(scores[i]))
            for i in ranked[:top_k]
        ]