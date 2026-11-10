"""Record locally observed Ollama runtime and model metadata."""

import json
from datetime import UTC, datetime
from urllib.error import URLError
from urllib.request import urlopen

from rag_document_assistant.generation.ollama_config import get_ollama_base_url


def _get_json(endpoint: str) -> dict:
    url = f"{get_ollama_base_url()}{endpoint}"
    with urlopen(url, timeout=5) as response:
        payload = json.loads(response.read())

    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object")

    return payload


def collect_runtime_metadata(model: str) -> dict:
    """Inspect runtime metadata without preventing answer generation on failure."""
    metadata = {
        "observed_at": datetime.now(UTC).isoformat(),
        "requested_model": model,
        "ollama_version": None,
        "model_digest": None,
        "errors": [],
    }

    try:
        version = _get_json("/api/version")
        value = version.get("version")
        if not isinstance(value, str) or not value:
            raise ValueError("Ollama version is missing")
        metadata["ollama_version"] = value
    except (URLError, OSError, ValueError) as exc:
        metadata["errors"].append(f"Version lookup failed: {exc}")

    try:
        tags = _get_json("/api/tags")
        models = tags.get("models")
        if not isinstance(models, list):
            raise ValueError("Model list is missing")

        # Ollama uses :latest when no tag is specified.
        name = model if ":" in model.rsplit("/", 1)[-1] else f"{model}:latest"

        matching = [
            item
            for item in models
            if isinstance(item, dict)
            and name in (item.get("name"), item.get("model"))
        ]
        if len(matching) != 1:
            raise ValueError(f"Expected one installed model matching {name}")

        digest = matching[0].get("digest")
        if not isinstance(digest, str) or not digest:
            raise ValueError("Model digest is missing")

        metadata["model_digest"] = digest
    except (URLError, OSError, ValueError) as exc:
        metadata["errors"].append(f"Model lookup failed: {exc}")

    return metadata