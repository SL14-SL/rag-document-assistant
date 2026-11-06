import json

import pytest

from rag_document_assistant.evaluation.answers import (
    prepare_review,
    select_questions,
)


def make_question(question_id="Q003", split="development"):
    return {
        "id": question_id,
        "split": split,
        "question": "May I work from Spain?",
        "answerability": "answerable",
        "evidence": [{"document_id": "NL-07", "section": "S1", "page": 1}],
        "required_claims": ["Workation is limited to the EU"],
    }


def write_questions(tmp_path, questions):
    path = tmp_path / "questions.jsonl"
    path.write_text(
        "".join(json.dumps(question) + "\n" for question in questions),
        encoding="utf-8",
    )
    return path


def test_question_selection_preserves_requested_order(tmp_path):
    path = write_questions(
        tmp_path,
        [make_question("Q003"), make_question("Q027")],
    )

    selected = select_questions(path, ["Q027", "Q003"])

    assert [question["id"] for question in selected] == ["Q027", "Q003"]


def test_validation_questions_are_rejected(tmp_path):
    path = write_questions(
        tmp_path,
        [make_question(split="validation")],
    )

    with pytest.raises(ValueError, match="development"):
        select_questions(path, ["Q003"])


def test_unknown_question_id_is_rejected(tmp_path):
    path = write_questions(tmp_path, [make_question()])

    with pytest.raises(ValueError, match="Unknown"):
        select_questions(path, ["Q999"])


def test_duplicate_requested_ids_are_rejected(tmp_path):
    path = write_questions(tmp_path, [make_question()])

    with pytest.raises(ValueError, match="unique"):
        select_questions(path, ["Q003", "Q003"])


def test_review_contains_criteria_without_prejudging_quality():
    report = {
        "answer": "Workation is permitted within the EU [NL-07:S1].",
        "citation_check": {
            "cited_ids": ["NL-07:S1"],
            "unknown_ids": [],
            "has_citations": True,
        },
    }

    review = prepare_review(make_question(), report)

    assert review["answer"] == report["answer"]
    assert review["required_claims"][0]["met"] is None
    assert review["manual_review"]["correct"] is None
    assert review["manual_review"]["claims_supported_by_sources"] is None