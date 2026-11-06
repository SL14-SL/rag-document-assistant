"""Answer a question using local PDF retrieval and local Ollama generation."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from rag_document_assistant.generation.answer import (
    PROMPT_VERSION,
    build_messages,
    check_citations,
    generate_answer,
)
from rag_document_assistant.retrieval.semantic import (
    DEFAULT_MODEL,
    LocalSentenceEncoder,
)
from rag_document_assistant.retrieval.storage import load_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--model", default="llama3:latest")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/answers/latest.json"),
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("artifacts/index/semantic.npz"),
    )
    args = parser.parse_args()

    if not args.question.strip():
        parser.error("question must not be empty")

    started = perf_counter()

    encoder = LocalSentenceEncoder(local_files_only=args.offline)
    index = load_index(
        args.index,
        root=args.root,
        model_name=DEFAULT_MODEL,
        encoder=encoder,
    )
    results = index.search(args.question, top_k=5)

    retrieval_seconds = perf_counter() - started
    messages = build_messages(args.question, results)

    print("Generating an answer with local Ollama...", flush=True)
    generation_started = perf_counter()
    response = generate_answer(messages, model=args.model)
    generation_seconds = perf_counter() - generation_started

    answer = response["message"]["content"]
    allowed_ids = {result.section.citation_id for result in results}
    citation_check = check_citations(answer, allowed_ids)

    report = {
        "question": args.question,
        "answer": answer,
        "model": response.get("model", args.model),
        "embedding_model": DEFAULT_MODEL,
        "prompt_version": PROMPT_VERSION,
        "messages": messages,
        "sources": [
            {
                **asdict(result.section),
                "citation_id": result.section.citation_id,
                "retrieval_score": result.score,
            }
            for result in results
        ],
        "citation_check": asdict(citation_check),
        "done_reason": response.get("done_reason"),
        "timing": {
            "retrieval_setup_and_search_seconds": retrieval_seconds,
            "generation_request_seconds": generation_seconds,
        },
        "ollama_metrics": {
            key: response[key]
            for key in (
                "total_duration",
                "load_duration",
                "prompt_eval_count",
                "prompt_eval_duration",
                "eval_count",
                "eval_duration",
            )
            if key in response
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\nAnswer:\n")
    print(answer)

    print("\nSource ID check:")
    if citation_check.unknown_ids:
        print(
            "WARNING: Answer names sources that were not supplied: "
            + ", ".join(citation_check.unknown_ids)
        )
    elif not citation_check.has_citations:
        print("No source IDs detected; review the answer manually.")
    else:
        print("All detected source IDs belong to the supplied passages.")

    if response.get("done_reason") == "length":
        print("WARNING: The answer reached the output token limit and may be cut off.")

    print("\nCited sources:")
    for result in results:
        section = result.section
        if section.citation_id not in citation_check.cited_ids:
            continue

        pages = (
            str(section.page_start)
            if section.page_start == section.page_end
            else f"{section.page_start}-{section.page_end}"
        )
        print(
            f"  [{section.citation_id}] {section.title}, "
            f"page {pages}, version {section.version}"
        )

    print(f"\nRetrieval including setup: {retrieval_seconds:.2f} seconds")
    print(f"Generation request: {generation_seconds:.2f} seconds")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()