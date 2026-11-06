import numpy as np
import pytest

from rag_document_assistant.ingestion.pdf_loader import DocumentSection
from rag_document_assistant.retrieval.semantic import (
    SemanticIndex,
    normalize_vectors,
)


class FakeEncoder:
    def encode(self, texts):
        vectors = []
        for text in texts:
            if "Hotel" in text:
                vectors.append([3.0, 0.0])
            elif "Password" in text:
                vectors.append([0.0, 2.0])
            else:
                vectors.append([1.0, 0.0])
        return np.asarray(vectors, dtype=np.float32)


def make_section(document_id, text):
    return DocumentSection(
        document_id=document_id,
        title=text,
        version="1.0",
        status="active",
        source_path=f"documents_pdf/{document_id}.pdf",
        section_id="S1",
        heading="Rules",
        page_start=1,
        page_end=1,
        text=text,
    )


def test_search_uses_cosine_similarity_and_preserves_source():
    hotel = make_section("NL-01", "Hotel policy")
    password = make_section("NL-02", "Password policy")
    index = SemanticIndex([password, hotel], encoder=FakeEncoder())

    results = index.search("Accommodation", top_k=1)

    assert results[0].section == hotel
    assert results[0].section.citation_id == "NL-01:S1"
    assert results[0].score == pytest.approx(1.0)


def test_equal_scores_use_citation_id_order():
    first = make_section("NL-01", "Hotel policy")
    second = make_section("NL-02", "Hotel policy")
    index = SemanticIndex([second, first], encoder=FakeEncoder())

    assert [
        result.section.document_id
        for result in index.search("Accommodation")
    ] == ["NL-01", "NL-02"]


def test_blank_query_returns_no_results():
    index = SemanticIndex(
        [make_section("NL-01", "Hotel policy")],
        encoder=FakeEncoder(),
    )

    assert index.search("   ") == []


def test_invalid_top_k_is_rejected():
    index = SemanticIndex(
        [make_section("NL-01", "Hotel policy")],
        encoder=FakeEncoder(),
    )

    with pytest.raises(ValueError, match="top_k"):
        index.search("Accommodation", top_k=0)


def test_empty_corpus_is_rejected():
    with pytest.raises(ValueError, match="empty corpus"):
        SemanticIndex([], encoder=FakeEncoder())


def test_duplicate_citations_are_rejected():
    section = make_section("NL-01", "Hotel policy")

    with pytest.raises(ValueError, match="unique"):
        SemanticIndex([section, section], encoder=FakeEncoder())


@pytest.mark.parametrize(
    "values",
    [
        [[0.0, 0.0]],
        [[float("nan"), 1.0]],
        [1.0, 2.0],
    ],
)
def test_invalid_embeddings_are_rejected(values):
    with pytest.raises(ValueError):
        normalize_vectors(np.asarray(values, dtype=np.float32))


def test_vectors_are_normalized():
    result = normalize_vectors(np.array([[3.0, 4.0]], dtype=np.float32))

    np.testing.assert_allclose(result, [[0.6, 0.8]])