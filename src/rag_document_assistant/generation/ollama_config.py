"""Shared Ollama connection configuration."""

import os


def get_ollama_base_url() -> str:
    """Resolve the endpoint when called, allowing environment overrides."""
    return (
        os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        .strip()
        .rstrip("/")
    )