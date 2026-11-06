import pytest

from rag_document_assistant.evaluation.retrieval import (
    evaluate_retrieval,
    score_ranking,
)
from rag_document_assistant.ingestion.pdf_loader import DocumentSection
from rag_document_assistant.retrieval.bm25 import BM25Index


def make_index() -> BM25Index:
    section = DocumentSection(
        document_id="NL-01",
        title="Hotel Policy",
        version="1.0",
        status="active",
        source_path="documents_pdf/01.pdf",
        section_id="S1",
        heading="Hotel",
        page_start=1,
        page_end=1,
        text="The hotel limit is EUR 120.",
    )
    return BM25Index([section])


def make_question(
    *,
    question_id="Q001",
    answerability="answerable",
    document_id="NL-01",
):
    return {
        "id": question_id,
        "question": "What is the hotel limit?",
        "answerability": answerability,
        "evidence": [{"document_id": document_id, "section": "S1"}],
    }


def test_partial_evidence_and_reciprocal_rank():
    scores = score_ranking(
        ["OTHER:S1", "NL-01:S1", "NL-02:S1"],
        {"NL-01:S1", "NL-02:S1"},
        k=2,
    )

    assert scores == {
        "recall": 0.5,
        "complete_evidence": 0.0,
        "reciprocal_rank": 0.5,
    }


def test_complete_evidence():
    scores = score_ranking(
        ["NL-01:S1", "NL-02:S1"],
        {"NL-01:S1", "NL-02:S1"},
        k=3,
    )

    assert scores["recall"] == 1.0
    assert scores["complete_evidence"] == 1.0
    assert scores["reciprocal_rank"] == 1.0


def test_no_relevant_results():
    assert score_ranking([], {"NL-01:S1"}, k=3) == {
        "recall": 0.0,
        "complete_evidence": 0.0,
        "reciprocal_rank": 0.0,
    }


def test_duplicate_results_do_not_inflate_recall():
    scores = score_ranking(
        ["NL-01:S1", "NL-01:S1"],
        {"NL-01:S1", "NL-02:S1"},
        k=3,
    )

    assert scores["recall"] == 0.5


def test_unanswerable_questions_are_excluded_from_retrieval_average():
    report = evaluate_retrieval(
        make_index(),
        [
            make_question(),
            make_question(question_id="Q002", answerability="unanswerable"),
        ],
    )

    assert report["evaluated_count"] == 1
    assert report["skipped_count"] == 1
    assert report["metrics"]["3"]["recall"] == 1.0


def test_unavailable_evidence_is_reported_explicitly():
    report = evaluate_retrieval(
        make_index(),
        [make_question(document_id="NL-99")],
    )

    assert report["evaluated_count"] == 0
    assert report["metrics"]["3"]["recall"] is None
    assert report["questions"][0]["unavailable"] == ["NL-99:S1"]


def test_conflict_questions_are_included():
    report = evaluate_retrieval(
        make_index(),
        [make_question(answerability="conflicting")],
    )

    assert report["evaluated_count"] == 1


def test_duplicate_question_ids_are_rejected():
    with pytest.raises(ValueError, match="Duplicate question"):
        evaluate_retrieval(make_index(), [make_question(), make_question()])


def test_empty_reference_evidence_is_rejected():
    question = make_question()
    question["evidence"] = []

    with pytest.raises(ValueError, match="missing reference"):
        evaluate_retrieval(make_index(), [question])


@pytest.mark.parametrize("ks", [(), (0,), (-1, 3)])
def test_invalid_cutoffs_are_rejected(ks):
    with pytest.raises(ValueError, match="positive integers"):
        evaluate_retrieval(make_index(), [], ks=ks)