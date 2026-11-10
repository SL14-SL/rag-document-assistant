"""Check configuration for host and container execution."""

from rag_document_assistant.generation.ollama_config import get_ollama_base_url


def test_default_uses_local_ollama(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

    assert get_ollama_base_url() == "http://127.0.0.1:11434"


def test_container_address_can_override_default(monkeypatch):
    monkeypatch.setenv(
        "OLLAMA_BASE_URL",
        "http://host.docker.internal:11434",
    )

    assert get_ollama_base_url() == "http://host.docker.internal:11434"


def test_trailing_slash_is_removed(monkeypatch):
    monkeypatch.setenv(
        "OLLAMA_BASE_URL",
        "http://host.docker.internal:11434/",
    )

    assert get_ollama_base_url() == "http://host.docker.internal:11434"