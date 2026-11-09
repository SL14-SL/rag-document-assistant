"""Replay a saved generation request with a fixed seed and runtime metadata."""

import argparse
import hashlib
import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

BASE_URL = "http://127.0.0.1:11434"


def request_json(
    endpoint: str,
    payload: dict | None = None,
    *,
    timeout: float = 30,
) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        BASE_URL + endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read())

    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object")
    if result.get("error"):
        raise RuntimeError(result["error"])
    return result


def runtime_snapshot(model: str) -> dict:
    models = request_json("/api/tags")["models"]
    matches = [
        item
        for item in models
        if model in {item.get("name"), item.get("model")}
    ]
    if len(matches) != 1:
        raise ValueError(f"Cannot uniquely identify installed model: {model}")

    return {
        "ollama": request_json("/api/version"),
        "model": matches[0],
        "platform": platform.platform(),
        "python": platform.python_version(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.output_dir.exists():
        parser.error("Output directory already exists; choose a new name")

    source_bytes = args.input.read_bytes()
    original = json.loads(source_bytes)
    model = original["model"]
    messages = original["messages"]

    if not isinstance(messages, list) or not messages:
        raise ValueError("Saved report contains no messages")

    before = runtime_snapshot(model)
    model_details = request_json("/api/show", {"model": model})

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "keep_alive": "10m",
        "options": {
            "temperature": 0,
            "seed": args.seed,
            "num_ctx": 4096,
            "num_predict": 256,
        },
    }
    request_bytes = json.dumps(payload).encode("utf-8")

    args.output_dir.mkdir(parents=True)
    metadata = {
        "created_at": datetime.now(UTC).isoformat(),
        "source_file": str(args.input),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
        "prompt_version": original.get("prompt_version"),
        "request": payload,
        "runtime_before": before,
        "model_details": model_details,
    }
    (args.output_dir / "request.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    answers = []
    complete = []

    for number in range(1, args.repeats + 1):
        print(f"\nRepeat {number}/{args.repeats}...", flush=True)

        # Send exactly the same request bytes on every repetition.
        request = Request(
            BASE_URL + "/api/chat",
            data=request_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=1800) as response:
                result = json.loads(response.read())
        except Exception as exc:
            (args.output_dir / f"repeat-{number}-error.json").write_text(
                json.dumps({"error": str(exc)}, indent=2) + "\n",
                encoding="utf-8",
            )
            raise

        (args.output_dir / f"repeat-{number}.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        if result.get("error"):
            raise RuntimeError(result["error"])

        answer = result.get("message", {}).get("content", "")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Model returned no answer")

        finished = (
            result.get("done") is True
            and result.get("done_reason") != "length"
        )
        answers.append(answer)
        complete.append(finished)

        print(answer)
        print("Stop reason:", result.get("done_reason"))

    after = runtime_snapshot(model)
    summary = {
        "repetitions": len(answers),
        "unique_answer_texts": len(set(answers)),
        "all_answers_complete": all(complete),
        "same_ollama_version": before["ollama"] == after["ollama"],
        "same_model_digest": (
            before["model"]["digest"] == after["model"]["digest"]
        ),
        "runtime_after": after,
        "note": (
            "A fixed seed and matching metadata improve reproducibility, "
            "but do not establish factual correctness or capture every "
            "internal runtime condition."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nSUMMARY")
    print("Unique answer texts:", summary["unique_answer_texts"])
    print("All answers complete:", summary["all_answers_complete"])
    print("Same Ollama version:", summary["same_ollama_version"])
    print("Same model digest:", summary["same_model_digest"])
    print("Results:", args.output_dir)


if __name__ == "__main__":
    main()