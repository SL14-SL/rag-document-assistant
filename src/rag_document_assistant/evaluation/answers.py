"""Generate answers to development questions and prepare manual review."""

import argparse
import hashlib
import json
from pathlib import Path

from rag_document_assistant.generation.answer import (
    DEFAULT_GENERATION_MODEL,
    PROMPT_VERSION,
)
from rag_document_assistant.generation.chat import answer_question
from rag_document_assistant.retrieval.semantic import (
    DEFAULT_MODEL,
    LocalSentenceEncoder,
)
from rag_document_assistant.retrieval.storage import load_index


def select_questions(path: Path, ids: list[str]) -> list[dict]:
    """Read development questions only and preserve requested order."""
    questions = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if any(question["split"] != "development" for question in questions):
        raise ValueError("Only development questions are allowed")

    by_id = {question["id"]: question for question in questions}
    if len(by_id) != len(questions):
        raise ValueError("Duplicate question IDs in development data")
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Provide unique question IDs")

    unknown = set(ids) - set(by_id)
    if unknown:
        raise ValueError(f"Unknown development IDs: {', '.join(sorted(unknown))}")

    return [by_id[question_id] for question_id in ids]


def prepare_review(question: dict, answer_report: dict) -> dict:
    """Attach reference criteria after generation, for human review only."""
    return {
        "id": question["id"],
        "question": question["question"],
        "answerability": question["answerability"],
        "answer": answer_report["answer"],
        "expected_evidence": question["evidence"],
        "required_claims": [
            {"criterion": claim, "met": None}
            for claim in question["required_claims"]
        ],
        "citation_check": answer_report["citation_check"],
        "manual_review": {
            "correct": None,
            "complete": None,
            "claims_supported_by_sources": None,
            "appropriate_uncertainty_or_abstention": None,
            "notes": "",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--model", default=DEFAULT_GENERATION_MODEL)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("artifacts/index/semantic.npz"),
    )
    args = parser.parse_args()

    root = args.root.resolve()
    question_path = root / "evaluation" / "development.jsonl"
    questions = select_questions(question_path, args.ids)

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory is not empty; choose a new run directory")

    print("Loading the search model and stored index...", flush=True)
    encoder = LocalSentenceEncoder(local_files_only=args.offline)
    index = load_index(
        args.index,
        root=root,
        model_name=DEFAULT_MODEL,
        encoder=encoder,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    configuration = {
        "model": args.model,
        "embedding_model": DEFAULT_MODEL,
        "prompt_version": PROMPT_VERSION,
        "question_ids": args.ids,
        "top_k": 5,
        "question_sha256": hashlib.sha256(question_path.read_bytes()).hexdigest(),
        "index_sha256": hashlib.sha256(args.index.read_bytes()).hexdigest(),
        "review_status": "pending_manual_review",
    }
    (args.output_dir / "run.json").write_text(
        json.dumps(configuration, indent=2) + "\n",
        encoding="utf-8",
    )

    completed = 0
    failures = []

    for number, question in enumerate(questions, 1):
        question_id = question["id"]
        print(
            f"\n[{number}/{len(questions)}] {question_id}: "
            f"{question['question']}",
            flush=True,
        )

        try:
            # Only the question text is passed to the answer pipeline.
            report = answer_question(
                index,
                question["question"],
                model=args.model,
            )
        except (RuntimeError, ValueError) as exc:
            failures.append({"id": question_id, "error": str(exc)})
            print(f"FAILED: {exc}", flush=True)
            continue

        report["question_id"] = question_id
        review = prepare_review(question, report)

        for suffix, content in (("answer", report), ("review", review)):
            path = args.output_dir / f"{question_id}-{suffix}.json"
            path.write_text(
                json.dumps(content, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

        completed += 1
        check = report["citation_check"]
        seconds = report["timing"]["generation_request_seconds"]
        print(
            f"Saved | generation: {seconds:.2f}s | "
            f"source IDs detected: {check['has_citations']} | "
            f"unknown IDs: {len(check['unknown_ids'])}",
            flush=True,
        )

    summary = {
        "requested": len(questions),
        "completed": completed,
        "failures": failures,
        "quality_assessment": "Pending manual review; no quality score calculated",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nCompleted: {completed}/{len(questions)}")
    print(f"Results: {args.output_dir}")
    print("Review the *-review.json files against the supplied source passages.")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()