"""Validate customer PDF coverage, persisted embeddings and retrieval."""

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import numpy as np
from pypdf import PdfReader

from rag_document_assistant.retrieval.semantic import (
    DEFAULT_MODEL,
    LocalSentenceEncoder,
)
from rag_document_assistant.retrieval.storage import load_index

BENCHMARK_QUERIES = [
    "What are the requirements for approving purchases?",
    "How often must production equipment be inspected?",
    "What happens when a supplier fails a quality inspection?",
    "How are employee training records maintained?",
    "What are the requirements for reporting IT incidents?",
]


def require(condition: bool, message: str) -> None:
    """Fail explicitly if a validation criterion is not met."""
    if not condition:
        raise ValueError(message)


def validate_index(
    root: Path,
    *,
    expected_documents: int,
    expected_pages: int,
) -> dict:
    """Validate a persisted customer index against its source PDFs."""
    root = root.resolve()

    entries = json.loads(
        (root / "manifest.json").read_text(encoding="utf-8")
    )

    active = {
        entry["document_id"]: entry
        for entry in entries
        if entry["status"] == "active"
    }

    require(
        len(active) == expected_documents,
        "Unexpected number of active documents",
    )

    pdf_pages = {}

    for document_id, entry in active.items():
        path = (root / entry["pdf_path"]).resolve()

        require(
            path.is_relative_to(root),
            f"Source path leaves corpus: {document_id}",
        )

        reader = PdfReader(path)

        pdf_pages[document_id] = len(reader.pages)

        for page_number, page in enumerate(reader.pages, start=1):
            require(
                bool((page.extract_text() or "").strip()),
                f"Empty PDF page: {document_id}, page {page_number}",
            )

    require(
        sum(pdf_pages.values()) == expected_pages,
        "Unexpected number of PDF pages",
    )

    print(
        f"PASS: {len(active)} active documents, "
        f"{sum(pdf_pages.values())} non-empty pages"
    )

    started = perf_counter()

    encoder = LocalSentenceEncoder(local_files_only=True)

    index_path = root / "artifacts/index/semantic.npz"

    index = load_index(
        index_path,
        root=root,
        model_name=DEFAULT_MODEL,
        encoder=encoder,
    )

    setup_seconds = perf_counter() - started

    print("PASS: persisted index and corpus fingerprint")

    tokenizer = encoder.model.tokenizer
    max_tokens = encoder.model.max_seq_length

    token_lengths = []
    content_token_lengths = []
    citation_ids = set()
    covered_pages = set()
    chunks_per_document = Counter()
    chunks_per_page = Counter()

    for section in index.sections:
        document_id = section.document_id
        citation_id = section.citation_id

        require(
            document_id in active,
            f"Unexpected indexed document: {document_id}",
        )

        require(
            citation_id not in citation_ids,
            f"Duplicate citation ID: {citation_id}",
        )

        citation_ids.add(citation_id)

        require(
            bool(section.text.strip()),
            f"Empty chunk: {citation_id}",
        )

        require(
            section.status == "active",
            f"Inactive indexed section: {citation_id}",
        )

        require(
            1 <= section.page_start
            <= section.page_end
            <= pdf_pages[document_id],
            f"Invalid page reference: {citation_id}",
        )

        source = (root / section.source_path).resolve()

        expected_source = (
            root / active[document_id]["pdf_path"]
        ).resolve()

        require(
            source.is_relative_to(root) and source.is_file(),
            f"Invalid source file: {citation_id}",
        )

        require(
            source == expected_source,
            f"Source does not match manifest: {citation_id}",
        )

        embedding_text = (
            f"{section.title}\n"
            f"{section.heading}\n"
            f"{section.text}"
        )

        token_ids = tokenizer(
            embedding_text,
            truncation=False,
            add_special_tokens=True,
        )["input_ids"]

        token_count = len(token_ids)

        require(
            token_count <= max_tokens,
            f"Embedding input exceeds token limit: {citation_id}",
        )

        content_tokens = tokenizer(
            section.text,
            truncation=False,
            add_special_tokens=False,
        )["input_ids"]

        token_lengths.append(token_count)
        content_token_lengths.append(len(content_tokens))

        chunks_per_document[document_id] += 1

        for page in range(
            section.page_start,
            section.page_end + 1,
        ):
            covered_pages.add((document_id, page))
            chunks_per_page[(document_id, page)] += 1

    expected_coverage = {
        (document_id, page)
        for document_id, count in pdf_pages.items()
        for page in range(1, count + 1)
    }

    require(
        covered_pages == expected_coverage,
        f"Missing indexed pages: {expected_coverage - covered_pages}",
    )

    print("PASS: all PDF pages represented in index")
    print("PASS: unique citation IDs and valid source references")
    print("PASS: no empty or oversized embedding inputs")

    embeddings = index.embeddings

    expected_dimension = encoder.model.get_embedding_dimension()

    require(
        embeddings.shape == (
            len(index.sections),
            expected_dimension,
        ),
        "Unexpected embedding matrix shape",
    )

    require(
        bool(np.isfinite(embeddings).all()),
        "Non-finite embedding values",
    )

    require(
        bool(
            np.allclose(
                np.linalg.norm(embeddings, axis=1),
                1.0,
                atol=1e-5,
            )
        ),
        "Embedding vectors are not normalized",
    )

    print("PASS: embedding dimensions and normalization")

    # Warm up the encoder before measuring retrieval latency.
    index.search(BENCHMARK_QUERIES[0], top_k=5)

    search_times = []

    for query in BENCHMARK_QUERIES:
        for _ in range(3):
            started = perf_counter()

            results = index.search(query, top_k=5)

            search_times.append(
                (perf_counter() - started) * 1000
            )

            require(
                len(results) == min(5, len(index.sections)),
                "Unexpected number of retrieval results",
            )

    print("PASS: retrieval benchmark")

    return {
        "documents": len(active),
        "pages": sum(pdf_pages.values()),
        "indexed_pages": len(covered_pages),
        "chunks": len(index.sections),
        "pages_with_multiple_chunks": sum(
            count > 1
            for count in chunks_per_page.values()
        ),
        "embedding_dimension": embeddings.shape[1],
        "embedding_dtype": str(embeddings.dtype),
        "embedding_model": DEFAULT_MODEL,
        "embedding_token_limit": max_tokens,
        "embedding_input_tokens": {
            "minimum": min(token_lengths),
            "mean": round(mean(token_lengths), 2),
            "maximum": max(token_lengths),
        },
        "content_tokens": {
            "minimum": min(content_token_lengths),
            "mean": round(mean(content_token_lengths), 2),
            "maximum": max(content_token_lengths),
        },
        "oversized_embedding_inputs": 0,
        "empty_chunks": 0,
        "duplicate_citation_ids": 0,
        "missing_pages": 0,
        "index_size_bytes": index_path.stat().st_size,
        "setup_seconds": round(setup_seconds, 3),
        "retrieval_latency_ms": {
            "measurements": len(search_times),
            "minimum": round(min(search_times), 2),
            "median": round(median(search_times), 2),
            "mean": round(mean(search_times), 2),
            "maximum": round(max(search_times), 2),
        },
        "chunks_per_document": dict(
            sorted(chunks_per_document.items())
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--expected-documents",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--expected-pages",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    report = validate_index(
        args.root,
        expected_documents=args.expected_documents,
        expected_pages=args.expected_pages,
    )

    serialized = json.dumps(report, indent=2)

    print()
    print(serialized)

    if args.output is not None:
        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        args.output.write_text(
            serialized + "\n",
            encoding="utf-8",
        )

        print(f"\nReport saved: {args.output}")

    print("\nCustomer index validation passed.")


if __name__ == "__main__":
    main()