"""Evaluate local retrieval using development questions only."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

from rag_document_assistant.evaluation.retrieval import (
    RetrievalIndex,
    evaluate_retrieval,
)
from rag_document_assistant.ingestion.pdf_loader import load_corpus
from rag_document_assistant.retrieval.bm25 import BM25Index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--method",
        choices=["bm25", "semantic"],
        default="bm25",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Load the embedding model from local files only",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.output or Path(
        f"artifacts/evaluation/{args.method}-development.json"
    )

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

    sections = load_corpus(root)
    configuration = {
        "retriever": args.method,
        "include_archived": False,
        "search_fields": ["title", "heading", "text"],
        "question_file": "evaluation/development.jsonl",
        "question_sha256": hashlib.sha256(question_bytes).hexdigest(),
        "manifest_sha256": hashlib.sha256(
            (root / "manifest.json").read_bytes()
        ).hexdigest(),
    }

    index: RetrievalIndex
    start = perf_counter()

    if args.method == "semantic":
        from rag_document_assistant.retrieval.semantic import (
            DEFAULT_MODEL,
            LocalSentenceEncoder,
            SemanticIndex,
        )

        encoder = LocalSentenceEncoder(local_files_only=args.offline)
        index = SemanticIndex(sections, encoder=encoder)
        configuration.update(
            model=DEFAULT_MODEL,
            device="cpu",
            local_files_only=args.offline,
            similarity="cosine",
            max_sequence_length=encoder.model.max_seq_length,
        )
    else:
        index = BM25Index(sections)
        configuration.update(k1=index.k1, b=index.b)

    setup_seconds = perf_counter() - start

    start = perf_counter()
    report = evaluate_retrieval(index, questions)
    evaluation_seconds = perf_counter() - start

    report["configuration"] = configuration
    report["timing"] = {
        "index_setup_seconds": setup_seconds,
        "evaluation_seconds": evaluation_seconds,
        "note": (
            "Setup includes model loading for semantic retrieval. "
            "Evaluation includes query encoding and metric calculation. "
            "These are single-run observations, not latency benchmarks."
        ),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Retriever: {args.method}")
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
    incomplete = [
        row
        for row in report["questions"]
        if row["status"] == "evaluated" and row["missing_at_max_k"]
    ]

    if not incomplete:
        print("  None")
    else:
        for row in incomplete:
            missing = ", ".join(row["missing_at_max_k"])
            print(f"  {row['id']}: missing {missing}")

    print(f"\nIndex setup: {setup_seconds:.2f} seconds")
    print(f"Evaluation: {evaluation_seconds:.2f} seconds")
    print(f"Report: {output}")


if __name__ == "__main__":
    main()