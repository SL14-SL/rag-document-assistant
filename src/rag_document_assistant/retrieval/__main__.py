"""Search local PDFs using BM25 or semantic retrieval."""

import argparse
from pathlib import Path

from rag_document_assistant.ingestion.pdf_loader import load_corpus
from rag_document_assistant.retrieval.bm25 import BM25Index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="An English question or search query")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--include-archived", action="store_true")
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
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    if not args.query.strip():
        parser.error("query must not be empty")

    sections = load_corpus(args.root, include_archived=args.include_archived)

    if args.method == "semantic":
        from rag_document_assistant.retrieval.semantic import (
            LocalSentenceEncoder,
            SemanticIndex,
        )

        encoder = LocalSentenceEncoder(local_files_only=args.offline)
        index = SemanticIndex(sections, encoder=encoder)
    else:
        index = BM25Index(sections)

    results = index.search(args.query, top_k=args.top_k)

    if not results:
        print("No matching passages found.")
        return

    score_label = "Cosine similarity" if args.method == "semantic" else "BM25 score"

    for rank, result in enumerate(results, 1):
        section = result.section
        pages = (
            str(section.page_start)
            if section.page_start == section.page_end
            else f"{section.page_start}-{section.page_end}"
        )

        print(f"\n{rank}. [{section.citation_id}] {section.title}")
        print(
            f"   Page: {pages} | Version: {section.version}"
            f" | Status: {section.status} | {score_label}: {result.score:.3f}"
        )
        print(f"   Source: {section.source_path}")
        print(f"   Section: {section.heading}\n")
        print(section.text)


if __name__ == "__main__":
    main()