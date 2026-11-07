import json

import pytest

from rag_document_assistant.evaluation.critic import (
    build_critic_messages,
    validate_review,
)

SOURCE = "Requests must be submitted at least 15 working days before the start date."


def make_content(source_id="NL-07:S1", quote=SOURCE):
    return json.dumps(
        {
            "issues": [
                {
                    "kind": "missing_condition",
                    "description": "The advance submission deadline is missing.",
                    "source_id": source_id,
                    "evidence_quote": quote,
                }
            ]
        }
    )


def test_reference_criteria_are_not_forwarded():
    report = {
        "question": "What must I arrange?",
        "answer": "Obtain approval.",
        "sources": [{"citation_id": "NL-07:S1", "text": SOURCE}],
        "required_claims": ["SECRET_REFERENCE_CRITERION"],
        "manual_review": {"notes": "SECRET_MANUAL_REVIEW"},
    }

    messages = build_critic_messages(report)
    payload = json.loads(messages[1]["content"])

    assert payload["candidate_answer"] == "Obtain approval."
    assert payload["source_passages"] == {"NL-07:S1": SOURCE}
    assert "SECRET_REFERENCE_CRITERION" not in json.dumps(messages)
    assert "SECRET_MANUAL_REVIEW" not in json.dumps(messages)


def test_review_with_matching_quote_is_accepted():
    review = validate_review(make_content(), {"NL-07:S1": SOURCE})

    assert review.issues[0].kind == "missing_condition"


def test_unknown_source_is_rejected():
    with pytest.raises(ValueError, match="Unknown source"):
        validate_review(make_content(source_id="NL-99:S1"), {"NL-07:S1": SOURCE})


def test_invented_evidence_is_rejected():
    with pytest.raises(ValueError, match="does not match"):
        validate_review(
            make_content(quote="Requests must be submitted one day before travel."),
            {"NL-07:S1": SOURCE},
        )


def test_empty_review_is_allowed():
    review = validate_review('{"issues": []}', {"NL-07:S1": SOURCE})

    assert review.issues == []