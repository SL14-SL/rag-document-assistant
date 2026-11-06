"""Local BM25 retrieval over document sections."""

import math
import re
from collections import Counter
from dataclasses import dataclass

from rag_document_assistant.ingestion.pdf_loader import DocumentSection

STOP_WORDS = frozenset(
    """
    a an and are as at be been by can could do does for from
    how i in is it may me my of on or our should that the their
    this to was we what when where which who will with would
    you your
    """.split()
)


def tokenize(text: str) -> list[str]:
    """Normalize English search terms without external dependencies."""
    return [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in STOP_WORDS
    ]


@dataclass(frozen=True)
class SearchResult:
    section: DocumentSection
    score: float


class BM25Index:
    """An in-memory index intended as a reproducible retrieval baseline."""

    def __init__(
        self,
        sections: list[DocumentSection],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not sections:
            raise ValueError("Cannot index an empty corpus")
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("Expected k1 > 0 and 0 <= b <= 1")

        citation_ids = [section.citation_id for section in sections]
        if len(citation_ids) != len(set(citation_ids)):
            raise ValueError("Citation IDs must be unique")

        self.sections = tuple(sections)
        self.k1 = k1
        self.b = b

        # Include headings and document titles to provide search context.
        self.term_counts = [
            Counter(tokenize(f"{section.title} {section.heading} {section.text}"))
            for section in self.sections
        ]
        self.lengths = [sum(counts.values()) for counts in self.term_counts]
        self.average_length = sum(self.lengths) / len(self.lengths)

        if self.average_length == 0:
            raise ValueError("Corpus contains no searchable terms")

        self.document_frequency: Counter[str] = Counter()
        for counts in self.term_counts:
            self.document_frequency.update(counts.keys())

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        """Return positive-scoring matches, ordered by score then citation ID."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        terms = set(tokenize(query))
        if not terms:
            return []

        results = []
        corpus_size = len(self.sections)

        for section, counts, length in zip(
            self.sections, self.term_counts, self.lengths, strict=True
        ):
            score = 0.0
            length_factor = 1 - self.b + self.b * length / self.average_length

            for term in sorted(terms):
                frequency = counts[term]
                if frequency == 0:
                    continue

                document_frequency = self.document_frequency[term]
                inverse_frequency = math.log(
                    1
                    + (corpus_size - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                score += (
                    inverse_frequency
                    * frequency
                    * (self.k1 + 1)
                    / (frequency + self.k1 * length_factor)
                )

            if score > 0:
                results.append(SearchResult(section=section, score=score))

        results.sort(key=lambda result: (-result.score, result.section.citation_id))
        return results[:top_k]