"""Evaluate retrieval against annotated evidence section IDs."""

from collections import Counter
from statistics import mean
from typing import Any

from rag_document_assistant.retrieval.bm25 import BM25Index


def score_ranking(
    retrieved: list[str],
    expected: set[str],
    *,
    k: int,
) -> dict[str, float]:
    """Score one ranked list against its required evidence."""
    if not expected:
        raise ValueError("Expected evidence must not be empty")
    if k < 1:
        raise ValueError("k must be at least 1")

    top = retrieved[:k]
    found = set(top) & expected

    reciprocal_rank = next(
        (
            1.0 / rank
            for rank, citation_id in enumerate(top, 1)
            if citation_id in expected
        ),
        0.0,
    )

    return {
        "recall": len(found) / len(expected),
        "complete_evidence": float(found == expected),
        "reciprocal_rank": reciprocal_rank,
    }


def evaluate_retrieval(
    index: BM25Index,
    questions: list[dict[str, Any]],
    *,
    ks: tuple[int, ...] = (3, 5),
) -> dict[str, Any]:
    """Evaluate answerable questions with evidence available in this index."""
    if not ks or any(k < 1 for k in ks):
        raise ValueError("ks must contain positive integers")

    ks = tuple(sorted(set(ks)))
    available = {section.citation_id for section in index.sections}
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for question in questions:
        question_id = question["id"]
        if question_id in seen_ids:
            raise ValueError(f"Duplicate question ID: {question_id}")
        seen_ids.add(question_id)

        row: dict[str, Any] = {
            "id": question_id,
            "question": question["question"],
            "answerability": question["answerability"],
        }

        answerability = question["answerability"]
        if answerability not in {"answerable", "conflicting", "unanswerable"}:
            raise ValueError(f"{question_id}: unknown answerability")

        if answerability == "unanswerable":
            row.update(
                status="skipped",
                reason="unanswerable_requires_answer_evaluation",
            )
            rows.append(row)
            continue

        expected = {
            f'{ref["document_id"]}:{ref["section"]}'
            for ref in question["evidence"]
        }
        if not expected:
            raise ValueError(f"{question_id}: missing reference evidence")

        row["expected"] = sorted(expected)
        unavailable = expected - available

        if unavailable:
            row.update(
                status="skipped",
                reason="reference_evidence_outside_index",
                unavailable=sorted(unavailable),
            )
            rows.append(row)
            continue

        results = index.search(question["question"], top_k=max(ks))
        retrieved = [result.section.citation_id for result in results]

        row.update(
            status="evaluated",
            retrieved=retrieved,
            metrics={
                str(k): score_ranking(retrieved, expected, k=k)
                for k in ks
            },
            missing_at_max_k=sorted(expected - set(retrieved)),
        )
        rows.append(row)

    evaluated = [row for row in rows if row["status"] == "evaluated"]

    metrics = {
        str(k): {
            name: mean(row["metrics"][str(k)][name] for row in evaluated)
            if evaluated
            else None
            for name in ("recall", "complete_evidence", "reciprocal_rank")
        }
        for k in ks
    }

    return {
        "question_count": len(rows),
        "evaluated_count": len(evaluated),
        "skipped_count": len(rows) - len(evaluated),
        "skip_reasons": dict(
            Counter(row["reason"] for row in rows if row["status"] == "skipped")
        ),
        "aggregation": "Macro average over evaluated questions",
        "metrics": metrics,
        "questions": rows,
    }