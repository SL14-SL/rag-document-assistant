"""Test runtime metadata collection without a running Ollama server."""

from rag_document_assistant.generation import runtime


def test_collect_runtime_metadata_matches_requested_model(monkeypatch):
    responses = {
        "/api/version": {"version": "test-version"},
        "/api/tags": {
            "models": [
                {"name": "other:7b", "digest": "other-digest"},
                {"name": "qwen2.5:14b", "digest": "expected-digest"},
            ]
        },
    }
    monkeypatch.setattr(runtime, "_get_json", responses.__getitem__)

    result = runtime.collect_runtime_metadata("qwen2.5:14b")

    assert result["ollama_version"] == "test-version"
    assert result["model_digest"] == "expected-digest"
    assert result["errors"] == []


def test_missing_model_does_not_use_another_models_digest(monkeypatch):
    responses = {
        "/api/version": {"version": "test-version"},
        "/api/tags": {
            "models": [{"name": "other:7b", "digest": "other-digest"}]
        },
    }
    monkeypatch.setattr(runtime, "_get_json", responses.__getitem__)

    result = runtime.collect_runtime_metadata("qwen2.5:14b")

    assert result["model_digest"] is None
    assert result["ollama_version"] == "test-version"
    assert len(result["errors"]) == 1


def test_unavailable_runtime_records_errors(monkeypatch):
    def unavailable(endpoint):
        raise OSError("Connection refused")

    monkeypatch.setattr(runtime, "_get_json", unavailable)

    result = runtime.collect_runtime_metadata("qwen2.5:14b")

    assert result["ollama_version"] is None
    assert result["model_digest"] is None
    assert len(result["errors"]) == 2