"""Compare saved local generation requests with OpenAI Responses."""

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from openai import OpenAI

from rag_document_assistant.generation.answer import check_citations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ids", nargs="+", required=True)
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        parser.error("Set OPENAI_API_KEY in the terminal environment")
    if len(args.ids) != len(set(args.ids)):
        parser.error("Question IDs must be unique")
    if args.output_dir.exists():
        parser.error("Output directory already exists; choose a new directory")

    # Validate every input before making any paid requests.
    inputs = []
    for question_id in args.ids:
        if not question_id.startswith("Q") or not question_id[1:].isdigit():
            parser.error(f"Invalid question ID: {question_id}")

        path = args.input_dir / f"{question_id}-answer.json"
        original_bytes = path.read_bytes()
        original = json.loads(original_bytes)
        request = original["generation_request"]

        if original["prompt_version"] != "grounded-v1.2":
            parser.error(f"{question_id}: expected grounded-v1.2")
        if request["options"]["num_predict"] != 256:
            parser.error(f"{question_id}: expected a 256-token local limit")

        inputs.append((question_id, original, request, original_bytes))

    args.output_dir.mkdir(parents=True)
    client = OpenAI(
        base_url="https://api.openai.com/v1",
        timeout=180,
        max_retries=0,
    )

    for number, (question_id, original, request, original_bytes) in enumerate(
        inputs, 1
    ):
        payload = {
            "model": "gpt-5.4-2026-03-05",
            "input": request["messages"],
            "temperature": 0,
            "reasoning": {"effort": "none"},
            "max_output_tokens": 256,
            "store": False,
        }

        print(f"\n[{number}/{len(inputs)}] {question_id}", flush=True)
        started = perf_counter()
        response = client.responses.create(**payload)
        seconds = perf_counter() - started
        answer = response.output_text

        allowed_ids = {
            source["citation_id"] for source in original["sources"]
        }
        check = check_citations(answer, allowed_ids)

        report = {
            "question_id": question_id,
            "question": original["question"],
            "original_report_sha256": hashlib.sha256(
                original_bytes
            ).hexdigest(),
            "prompt_version": original["prompt_version"],
            "local_model": original.get("model"),
            "local_answer": original.get("answer"),
            "answer": answer,
            "sources": original["sources"],
            "generation_request": payload,
            "model": response.model,
            "citation_check": asdict(check),
            "generation_seconds": seconds,
            "response": response.model_dump(mode="json"),
        }

        output = args.output_dir / f"{question_id}-comparison.json"
        output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        print(answer)
        print("Status:", response.status)
        print(f"Request: {seconds:.2f} seconds")
        print("Report:", output)

        if response.status != "completed" or not answer.strip():
            raise SystemExit(
                "Response was not complete. Inspect the saved report "
                "before running more requests."
            )


if __name__ == "__main__":
    main()