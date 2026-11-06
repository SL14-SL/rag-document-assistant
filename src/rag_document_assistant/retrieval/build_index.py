"""Build and persist the active-document semantic index."""

import argparse
from pathlib import Path
from time import perf_counter

from rag_document_assistant.ingestion.pdf_loader import load_corpus
from rag_document_assistant.retrieval.semantic import (
    DEFAULT_MODEL,
    LocalSentenceEncoder,
    SemanticIndex,
)
from rag_document_assistant.retrieval.storage import (
    corpus_fingerprint,
    save_index,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/index/semantic.npz"),
    )
    args = parser.parse_args()

    started = perf_counter()
    fingerprint = corpus_fingerprint(args.root)
    sections = load_corpus(args.root)
    encoder = LocalSentenceEncoder(local_files_only=args.offline)
    index = SemanticIndex(sections, encoder=encoder)

    if corpus_fingerprint(args.root) != fingerprint:
        raise RuntimeError("Corpus changed during index creation; retry")

    save_index(
        args.output,
        index,
        fingerprint=fingerprint,
        model_name=DEFAULT_MODEL,
    )

    print(
        f"Saved {len(index.sections)} sections "
        f"from {len({s.document_id for s in index.sections})} documents."
    )
    print(f"Index: {args.output}")
    print(f"Build time: {perf_counter() - started:.2f} seconds")


if __name__ == "__main__":
    main()