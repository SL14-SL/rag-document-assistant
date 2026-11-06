import pytest

from rag_document_assistant.ingestion.pdf_loader import DocumentSection
from rag_document_assistant.retrieval.bm25 import BM25Index, tokenize


def make_section(document_id: str, text: str) -> DocumentSection:
    return DocumentSection(
        document_id=document_id,
        title="Example Policy",
        version="1.0",
        status="active",
        source_path=f"documents_pdf/{document_id}.pdf",
        section_id="S1",
        heading="Rules",
        page_start=2,
        page_end=2,
        text=text,
    )


def test_tokenization_normalizes_case_and_removes_stop_words():
    assert tokenize("What is the HOTEL limit: EUR 120?") == [
        "hotel",
        "limit",
        "eur",
        "120",
    ]


def test_search_ranks_matching_passage_and_preserves_source():
    hotel = make_section("NL-01", "Hotel bookings require manager approval.")
    password = make_section("NL-02", "Passwords require fourteen characters.")
    index = BM25Index([password, hotel])

    results = index.search("hotel booking", top_k=1)

    assert len(results) == 1
    assert results[0].section == hotel
    assert results[0].section.citation_id == "NL-01:S1"
    assert results[0].section.page_start == 2
    assert results[0].score > 0


@pytest.mark.parametrize("query", ["", "the and is", "zebras"])
def test_search_returns_no_matches_for_empty_or_unknown_terms(query):
    index = BM25Index([make_section("NL-01", "Hotel booking rules.")])

    assert index.search(query) == []


def test_equal_scores_have_deterministic_order():
    first = make_section("NL-01", "Hotel booking rules.")
    second = make_section("NL-02", "Hotel booking rules.")
    index = BM25Index([second, first])

    results = index.search("hotel")

    assert [result.section.document_id for result in results] == [
        "NL-01",
        "NL-02",
    ]


def test_top_k_limits_results():
    sections = [
        make_section(f"NL-{number:02}", "Hotel booking rules.")
        for number in range(1, 5)
    ]

    assert len(BM25Index(sections).search("hotel", top_k=2)) == 2


def test_empty_corpus_is_rejected():
    with pytest.raises(ValueError, match="empty corpus"):
        BM25Index([])


def test_duplicate_citation_ids_are_rejected():
    section = make_section("NL-01", "Hotel booking rules.")

    with pytest.raises(ValueError, match="unique"):
        BM25Index([section, section])


def test_invalid_top_k_is_rejected():
    index = BM25Index([make_section("NL-01", "Hotel booking rules.")])

    with pytest.raises(ValueError, match="top_k"):
        index.search("hotel", top_k=0)