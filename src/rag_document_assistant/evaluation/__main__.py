"""Evaluate BM25 using development questions only."""

import argparse
import hashlib
import json
from pathlib import Path

from rag_document_assistant.evaluation.retrieval import evaluate_retrieval
from rag_document_assistant.ingestion.pdf_loader import load_corpus
from rag_document_assistant.retrieval.bm25 import BM25Index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evaluation/bm25-development.json"),
    )
    args = parser.parse_args()

    root = args.root.resolve()
    question_path = root / "evaluation" / "development.jsonl"
    question_bytes = question_path.read_bytes()
    questions = [
        json.loads(line)
        for line in question_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]

    if not questions:
        raise ValueError("Development question set is empty")
    if any(question["split"] != "development" for question in questions):
        raise ValueError("Expected development questions only")

    index = BM25Index(load_corpus(root))
    report = evaluate_retrieval(index, questions)

    report["configuration"] = {
        "retriever": "bm25",
        "k1": index.k1,
        "b": index.b,
        "include_archived": False,
        "search_fields": ["title", "heading", "text"],
        "question_file": "evaluation/development.jsonl",
        "question_sha256": hashlib.sha256(question_bytes).hexdigest(),
        "manifest_sha256": hashlib.sha256(
            (root / "manifest.json").read_bytes()
        ).hexdigest(),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        f"Questions: {report['question_count']} | "
        f"Evaluated: {report['evaluated_count']} | "
        f"Skipped: {report['skipped_count']}"
    )

    for k, metrics in report["metrics"].items():
        if metrics["recall"] is None:
            print(f"@{k}: no eligible questions")
            continue

        print(
            f"@{k}: Recall={metrics['recall']:.3f} | "
            f"Complete evidence={metrics['complete_evidence']:.3f} | "
            f"MRR={metrics['reciprocal_rank']:.3f}"
        )

    for reason, count in report["skip_reasons"].items():
        print(f"Skipped: {reason} = {count}")

    print("\nIncomplete evidence at the largest k:")
    for row in report["questions"]:
        if row["status"] == "evaluated" and row["missing_at_max_k"]:
            missing = ", ".join(row["missing_at_max_k"])
            print(f"  {row['id']}: missing {missing}")

    print(f"\nReport: {args.output}")


if __name__ == "__main__":
    main()