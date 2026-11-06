import json
from io import BytesIO
from unittest.mock import Mock, patch
from urllib.error import URLError

import pytest

from rag_document_assistant.generation.answer import (
    build_messages,
    check_citations,
    generate_answer,
)
from rag_document_assistant.ingestion.pdf_loader import DocumentSection
from rag_document_assistant.retrieval.bm25 import SearchResult


def make_result():
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
    return SearchResult(section=section, score=0.8)


def test_prompt_contains_question_and_original_source():
    messages = build_messages("What is the hotel limit?", [make_result()])
    payload = json.loads(messages[1]["content"])

    assert messages[0]["role"] == "system"
    assert payload["question"] == "What is the hotel limit?"
    assert payload["source_passages"][0]["source_id"] == "NL-09:S1"
    assert payload["source_passages"][0]["text"] == "The hotel limit is EUR 120."


def test_prompt_requires_sources():
    with pytest.raises(ValueError, match="source passage"):
        build_messages("Question?", [])


def test_prompt_rejects_blank_question():
    with pytest.raises(ValueError, match="Question"):
        build_messages(" ", [make_result()])


def test_known_citation():
    check = check_citations("The limit is EUR 120 [NL-09:S1].", {"NL-09:S1"})

    assert check.has_citations
    assert check.cited_ids == ("NL-09:S1",)
    assert check.unknown_ids == ()


def test_unknown_citation():
    check = check_citations("See [NL-99:S1].", {"NL-09:S1"})

    assert check.unknown_ids == ("NL-99:S1",)


def test_answer_without_citation():
    check = check_citations("The documents do not provide an answer.", {"NL-09:S1"})

    assert not check.has_citations
    assert check.unknown_ids == ()


def test_citations_are_deduplicated():
    check = check_citations("[NL-09:S1] and NL-09:S1", {"NL-09:S1"})

    assert check.cited_ids == ("NL-09:S1",)


def test_local_request_and_response():
    response = {
        "message": {"content": "EUR 120 [NL-09:S1]."},
        "done": True,
        "done_reason": "stop",
    }

    with patch(
        "rag_document_assistant.generation.answer.urlopen",
        return_value=BytesIO(json.dumps(response).encode()),
    ) as mocked:
        result = generate_answer([{"role": "user", "content": "Question"}])

    request = mocked.call_args.args[0]
    payload = json.loads(request.data)

    assert request.full_url == "http://127.0.0.1:11434/api/chat"
    assert payload["stream"] is False
    assert payload["options"]["temperature"] == 0
    assert result["message"]["content"] == "EUR 120 [NL-09:S1]."


def test_connection_error_is_explained():
    with (
        patch(
            "rag_document_assistant.generation.answer.urlopen",
            side_effect=URLError("Connection refused"),
        ),
        pytest.raises(RuntimeError, match="Local Ollama request failed"),
    ):
        generate_answer([])


@pytest.mark.parametrize(
    "response",
    [
        {"message": {"content": ""}, "done": True},
        {"message": {"content": "Partial answer"}, "done": False},
        {"done": True},
    ],
)
def test_invalid_responses_are_rejected(response):
    with (
        patch(
            "rag_document_assistant.generation.answer.urlopen",
            return_value=BytesIO(json.dumps(response).encode()),
        ),
        pytest.raises(RuntimeError),
    ):
        generate_answer([])

def test_answer_report_records_current_prompt_version():
    from rag_document_assistant.generation.answer import PROMPT_VERSION
    from rag_document_assistant.generation.chat import answer_question

    index = Mock()
    index.search.return_value = [make_result()]

    response = {
        "model": "test-model",
        "message": {"content": "EUR 120 [NL-09:S1]."},
        "done": True,
        "done_reason": "stop",
    }

    with patch(
        "rag_document_assistant.generation.chat.generate_answer",
        return_value=response,
    ):
        report = answer_question(index, "What is the limit?", model="test-model")

    assert report["prompt_version"] == PROMPT_VERSION