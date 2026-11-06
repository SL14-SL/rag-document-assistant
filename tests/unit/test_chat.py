import json
from unittest.mock import Mock, patch

import pytest

from rag_document_assistant.generation.chat import answer_question, save_report
from rag_document_assistant.ingestion.pdf_loader import DocumentSection
from rag_document_assistant.retrieval.bm25 import SearchResult


def make_index():
    section = DocumentSection(
        document_id="NL-09",
        title="Travel Expense Policy",
        version="2.1",
        status="active",
        source_path="documents_pdf/09.pdf",
        section_id="S1",
        heading="Hotel costs",
        page_start=1,
        page_end=1,
        text="The hotel limit is EUR 120.",
    )
    index = Mock()
    index.search.return_value = [SearchResult(section=section, score=0.8)]
    return index


def model_response():
    return {
        "model": "test-model",
        "message": {"content": "The limit is EUR 120 [NL-09:S1]."},
        "done": True,
        "done_reason": "stop",
        "eval_count": 12,
    }


def test_multiple_questions_reuse_the_supplied_index():
    index = make_index()

    with patch(
        "rag_document_assistant.generation.chat.generate_answer",
        return_value=model_response(),
    ) as generate:
        first = answer_question(index, "What is the limit?", model="test-model")
        second = answer_question(index, "Do I need approval?", model="test-model")

    assert index.search.call_count == 2
    assert generate.call_count == 2
    assert first["citation_check"]["unknown_ids"] == ()
    assert second["sources"][0]["citation_id"] == "NL-09:S1"

    # Each request contains only its own question, not prior conversation turns.
    assert len(second["messages"]) == 2
    payload = json.loads(second["messages"][1]["content"])
    assert payload["question"] == "Do I need approval?"


def test_unknown_source_is_reported():
    response = model_response()
    response["message"]["content"] = "See [NL-99:S1]."

    with patch(
        "rag_document_assistant.generation.chat.generate_answer",
        return_value=response,
    ):
        report = answer_question(make_index(), "Question?", model="test-model")

    assert report["citation_check"]["unknown_ids"] == ("NL-99:S1",)


def test_blank_question_is_rejected_before_search():
    index = make_index()

    with pytest.raises(ValueError, match="Question"):
        answer_question(index, "   ", model="test-model")

    index.search.assert_not_called()


def test_no_sources_prevents_model_call():
    index = make_index()
    index.search.return_value = []

    with (
        patch(
            "rag_document_assistant.generation.chat.generate_answer"
        ) as generate,
        pytest.raises(ValueError, match="No source passages"),
    ):
        answer_question(index, "Question?", model="test-model")

    generate.assert_not_called()


def test_reports_do_not_overwrite_each_other(tmp_path):
    first = save_report({"question": "First"}, tmp_path)
    second = save_report({"question": "Second"}, tmp_path)

    assert first != second
    assert json.loads(first.read_text(encoding="utf-8"))["question"] == "First"
    assert json.loads(second.read_text(encoding="utf-8"))["question"] == "Second"