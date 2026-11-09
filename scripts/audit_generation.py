"""Read-only audit of saved generation inputs and runtime measurements."""

import argparse
import json
from pathlib import Path
from statistics import median

from rag_document_assistant.generation.answer import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()

    paths = sorted(args.directory.glob("Q*-answer.json"))
    if not paths:
        raise SystemExit("No Q*-answer.json files found")

    prompt_tokens = []
    generation_times = []
    models = set()
    versions = set()
    findings = []

    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        question_id = report.get("question_id", path.stem)
        models.add(report.get("model", "unknown"))
        versions.add(report.get("prompt_version", "unknown"))

        messages = report.get("messages", [])
        issues = []

        if len(messages) != 2:
            issues.append("Expected exactly two messages")
        else:
            if messages[0].get("role") != "system":
                issues.append("First message is not a system message")
            if messages[0].get("content") != SYSTEM_PROMPT:
                issues.append("Stored system prompt differs from current code")
            if messages[1].get("role") != "user":
                issues.append("Second message is not a user message")

            try:
                payload = json.loads(messages[1]["content"])
                expected_sources = [
                    {
                        "source_id": source["citation_id"],
                        "title": source["title"],
                        "heading": source["heading"],
                        "text": source["text"],
                    }
                    for source in report["sources"]
                ]

                if payload.get("question") != report["question"]:
                    issues.append("Question differs between report and prompt")
                if payload.get("source_passages") != expected_sources:
                    issues.append("Sources differ between report and prompt")
            except (KeyError, TypeError, json.JSONDecodeError):
                issues.append("Cannot inspect user-message payload")

        if report.get("prompt_version") != PROMPT_VERSION:
            issues.append("Stored prompt version differs from current code")

        source_ids = [source["citation_id"] for source in report["sources"]]
        if len(source_ids) != len(set(source_ids)):
            issues.append("Duplicate source IDs in supplied context")

        if report.get("done_reason") == "length":
            issues.append("Output token limit reached")

        metrics = report.get("ollama_metrics", {})
        tokens = metrics.get("prompt_eval_count")
        if isinstance(tokens, int):
            prompt_tokens.append(tokens)

        seconds = report.get("timing", {}).get("generation_request_seconds")
        if isinstance(seconds, (int, float)):
            generation_times.append(seconds)

        prompt_seconds = metrics.get("prompt_eval_duration", 0) / 1e9
        output_seconds = metrics.get("eval_duration", 0) / 1e9
        output_tokens = metrics.get("eval_count")

        print(
            f"{question_id}: sources={len(source_ids)}, "
            f"prompt_tokens={tokens}, output_tokens={output_tokens}, "
            f"prompt_time={prompt_seconds:.2f}s, "
            f"output_time={output_seconds:.2f}s, "
            f"stop={report.get('done_reason')}"
        )

        for issue in issues:
            findings.append((question_id, issue))

    print("\nSUMMARY")
    print("Reports:", len(paths))
    print("Models:", sorted(models))
    print("Prompt versions:", sorted(versions))

    if prompt_tokens:
        print(
            "Prompt tokens: "
            f"min={min(prompt_tokens)}, "
            f"median={median(prompt_tokens):.1f}, "
            f"max={max(prompt_tokens)}"
        )

    if generation_times:
        print(f"Median request time: {median(generation_times):.2f}s")

    print("\nINPUT CONSISTENCY FINDINGS")
    if not findings:
        print("No inconsistencies detected by these checks.")
    else:
        for question_id, issue in findings:
            print(f"{question_id}: {issue}")

    print(
        "\nLimits: this checks saved inputs, not factual correctness. "
        "Reported token counts do not independently prove that no input "
        "was truncated. Runtime options and model digest were not recorded "
        "in the original answer reports."
    )


if __name__ == "__main__":
    main()