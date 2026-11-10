"""Interactive local document Q&A with a persistent in-memory search index."""

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from rag_document_assistant.generation.answer import (
    DEFAULT_GENERATION_MODEL,
    PROMPT_VERSION,
    build_messages,
    check_citations,
    generate_answer,
)
from rag_document_assistant.generation.openai_backend import (
    generate_openai_answer,
)
from rag_document_assistant.generation.runtime import collect_runtime_metadata
from rag_document_assistant.generation.sources import format_sources
from rag_document_assistant.retrieval.semantic import (
    DEFAULT_MODEL,
    LocalSentenceEncoder,
    SemanticIndex,
)
from rag_document_assistant.retrieval.storage import load_index


def answer_question(
    index: SemanticIndex,
    question: str,
    *,
    model: str,
    provider: str = "ollama",
) -> dict:
    """Answer one independent question using an already loaded index."""
    if provider not in {"ollama", "openai"}:
        raise ValueError(f"Unsupported provider: {provider}")
    if not question.strip():
        raise ValueError("Question must not be empty")

    started = perf_counter()
    results = index.search(question, top_k=5)
    search_seconds = perf_counter() - started

    if not results:
        raise ValueError("No source passages found")

    messages = build_messages(question, results)

    runtime_metadata = (
        collect_runtime_metadata(model)
        if provider == "ollama"
        else None
    )

    started = perf_counter()
    if provider == "ollama":
        response = generate_answer(messages, model=model)
    else:
        response = generate_openai_answer(messages, model=model)
    generation_seconds = perf_counter() - started

    answer = response["message"]["content"]
    allowed_ids = {result.section.citation_id for result in results}
    citation_check = check_citations(answer, allowed_ids)

    return {
        "question": question,
        "answer": answer,
        "model": response.get("model", model),
        "embedding_model": DEFAULT_MODEL,
        "prompt_version": PROMPT_VERSION,
        "messages": messages,
        "generation_request": response.get("generation_request"),
        "runtime_metadata": runtime_metadata,
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
            "search_seconds": search_seconds,
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
        "provider": provider,
        "api_response": response.get("api_response"),
    }


def save_report(report: dict, directory: Path) -> Path:
    """Save each answer separately rather than overwriting previous results."""
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    path = directory / f"{timestamp}-{uuid4().hex[:8]}.json"
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def print_report(report: dict) -> None:
    print("\nAnswer:\n")
    print(report["answer"])

    check = report["citation_check"]
    if check["unknown_ids"]:
        print(
            "\nWARNING: Unknown source IDs: "
            + ", ".join(check["unknown_ids"])
        )
    elif not check["has_citations"]:
        print("\nNo source IDs detected; review the answer manually.")
    else:
        print("\nAll detected source IDs belong to the supplied passages.")

    if report["done_reason"] == "length":
        print("WARNING: The answer reached the output token limit.")

    cited = set(check["cited_ids"])
    print("\nCited sources:")
    for source in report["sources"]:
        if source["citation_id"] not in cited:
            continue

        pages = (
            str(source["page_start"])
            if source["page_start"] == source["page_end"]
            else f"{source['page_start']}-{source['page_end']}"
        )
        print(
            f"  [{source['citation_id']}] {source['title']}, "
            f"page {pages}, version {source['version']}"
        )

    timing = report["timing"]
    print(f"\nSearch: {timing['search_seconds']:.3f} seconds")
    print(f"Generation request: {timing['generation_request_seconds']:.2f} seconds")
    print("\nOriginal passages supplied to the model:\n")
    print(
        format_sources(
            report["sources"],
            cited_ids=set(report["citation_check"]["cited_ids"]),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--model", default=DEFAULT_GENERATION_MODEL)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("artifacts/index/semantic.npz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/answers/chat"),
    )
    args = parser.parse_args()

    print("Loading the local search model and index...", flush=True)
    started = perf_counter()

    encoder = LocalSentenceEncoder(local_files_only=args.offline)
    index = load_index(
        args.index,
        root=args.root,
        model_name=DEFAULT_MODEL,
        encoder=encoder,
    )
    setup_seconds = perf_counter() - started

    print(f"Ready. Search initialization: {setup_seconds:.2f} seconds")
    print("Ask complete questions in English. Each question is independent.")
    print("Type /exit to quit.\n")

    while True:
        try:
            question = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.")
            break

        if question.lower() == "/exit":
            print("Session ended.")
            break
        if not question:
            continue

        print("Searching and generating an answer...", flush=True)

        try:
            report = answer_question(index, question, model=args.model)
        except (RuntimeError, ValueError) as exc:
            print(f"\nRequest failed: {exc}\n")
            continue
        except KeyboardInterrupt:
            print("\nSession interrupted.")
            break

        report["session_setup_seconds"] = setup_seconds
        print_report(report)

        try:
            path = save_report(report, args.output_dir)
            print(f"Report: {path}\n")
        except OSError as exc:
            print(f"Could not save the report: {exc}\n")


if __name__ == "__main__":
    main()