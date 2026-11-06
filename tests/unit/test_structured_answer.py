import json
from io import BytesIO
from unittest.mock import patch

import pytest

from rag_document_assistant.generation.answer import generate_answer
from rag_document_assistant.generation.structured import (
    StructuredAnswer,
    render_answer,
    validate_answer,
)


def make_content(
    *,
    status="answered",
    source_ids=None,
    text="The limit is EUR 120.",
):
    return json.dumps(
        {
            "status": status,
            "statements": [
                {
                    "text": text,
                    "source_ids": source_ids if source_ids is not None else ["NL-09:S1"],
                }
            ],
        }
    )


def test_valid_answer_is_rendered_with_citations():
    answer = validate_answer(make_content(), {"NL-09:S1"})

    assert render_answer(answer) == "The limit is EUR 120. [NL-09:S1]"


def test_unknown_source_is_rejected():
    with pytest.raises(ValueError, match="Unknown source"):
        validate_answer(
            make_content(source_ids=["NL-99:S1"]),
            {"NL-09:S1"},
        )


def test_duplicate_sources_are_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        validate_answer(
            make_content(source_ids=["NL-09:S1", "NL-09:S1"]),
            {"NL-09:S1"},
        )


def test_blank_statement_is_rejected():
    with pytest.raises(ValueError):
        validate_answer(make_content(text="   "), {"NL-09:S1"})


def test_statement_without_sources_is_rejected():
    with pytest.raises(ValueError):
        validate_answer(make_content(source_ids=[]), {"NL-09:S1"})


def test_answered_status_requires_statements():
    content = json.dumps({"status": "answered", "statements": []})

    with pytest.raises(ValueError, match="require statements"):
        validate_answer(content, {"NL-09:S1"})


def test_insufficient_status_can_have_no_statements():
    content = json.dumps({"status": "insufficient", "statements": []})
    answer = validate_answer(content, {"NL-09:S1"})

    assert "do not establish a complete answer" in render_answer(answer)


def test_invalid_status_is_rejected():
    with pytest.raises(ValueError):
        validate_answer(make_content(status="certain"), {"NL-09:S1"})


def test_schema_and_output_limit_are_sent_to_ollama():
    response = {
        "message": {"content": make_content()},
        "done": True,
        "done_reason": "stop",
    }
    schema = StructuredAnswer.model_json_schema()

    with patch(
        "rag_document_assistant.generation.answer.urlopen",
        return_value=BytesIO(json.dumps(response).encode()),
    ) as mocked:
        generate_answer(
            [{"role": "user", "content": "Question"}],
            response_schema=schema,
            max_output_tokens=512,
        )

    payload = json.loads(mocked.call_args.args[0].data)
    assert payload["format"] == schema
    assert payload["options"]["num_predict"] == 512