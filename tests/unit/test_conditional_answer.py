import json

import pytest

from rag_document_assistant.generation.conditional import (
    build_messages,
    validate_answer,
)


def test_original_question_and_sources_are_preserved():
    payload = {
        "question": "Does P1 guarantee resolution in 30 minutes?",
        "source_passages": [
            {"source_id": "NL-16:S2", "text": "During support hours..."}
        ],
    }
    original = {
        "messages": [
            {"role": "system", "content": "Previous instructions"},
            {"role": "user", "content": json.dumps(payload)},
        ],
        "required_claims": ["SECRET_REFERENCE"],
    }

    messages = build_messages(original)
    actual = json.loads(messages[1]["content"])

    assert actual["question"] == payload["question"]
    assert actual["source_passages"] == payload["source_passages"]
    assert "SECRET_REFERENCE" not in json.dumps(messages)


def test_answer_and_conditions_are_parsed():
    content = json.dumps(
        {
            "answer": "No, this is an initial response target [NL-16:S2].",
            "conditions": [
                "The target applies during support hours [NL-16:S2]."
            ],
        }
    )

    answer = validate_answer(content)

    assert len(answer.conditions) == 1


def test_blank_condition_is_rejected():
    content = json.dumps({"answer": "No.", "conditions": ["   "]})

    with pytest.raises(ValueError, match="Conditions"):
        validate_answer(content)


def test_blank_answer_is_rejected():
    content = json.dumps({"answer": "   ", "conditions": []})

    with pytest.raises(ValueError, match="Answer"):
        validate_answer(content)