"""Verify evaluation split selection and separation."""

import json

import pytest

from rag_document_assistant.evaluation.answers import select_questions


def test_select_validation_questions_preserves_requested_order(tmp_path):
    path = tmp_path / "validation.jsonl"
    questions = [
        {"id": "Q005", "split": "validation"},
        {"id": "Q010", "split": "validation"},
    ]
    path.write_text(
        "\n".join(json.dumps(question) for question in questions) + "\n",
        encoding="utf-8",
    )

    selected = select_questions(
        path,
        ["Q010", "Q005"],
        split="validation",
    )

    assert [question["id"] for question in selected] == ["Q010", "Q005"]


def test_development_default_rejects_validation_data(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text(
        json.dumps({"id": "Q005", "split": "validation"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Only development"):
        select_questions(path, ["Q005"])


def test_validation_rejects_mixed_splits(tmp_path):
    path = tmp_path / "questions.jsonl"
    questions = [
        {"id": "Q005", "split": "validation"},
        {"id": "Q001", "split": "development"},
    ]
    path.write_text(
        "\n".join(json.dumps(question) for question in questions) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Only validation"):
        select_questions(path, ["Q005"], split="validation")