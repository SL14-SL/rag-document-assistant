"""Test the API backend without paid network requests."""

from types import SimpleNamespace

import pytest

from rag_document_assistant.generation import openai_backend


def install_fake_client(monkeypatch, *, status="completed", text="Answer"):
    captured = {}

    class FakeResponse:
        model = openai_backend.OPENAI_MODEL
        output_text = text

        def __init__(self):
            self.status = status

        def model_dump(self, **kwargs):
            return {"status": self.status, "usage": {}}

    class FakeClient:
        def __init__(self, **kwargs):
            self.responses = SimpleNamespace(create=self.create)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def create(self, **kwargs):
            captured.update(kwargs)
            return FakeResponse()

    monkeypatch.setattr(openai_backend, "OpenAI", FakeClient)
    return captured


def test_api_receives_original_messages_and_explicit_options(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    captured = install_fake_client(monkeypatch)
    messages = [
        {"role": "system", "content": "Use supplied sources only."},
        {"role": "user", "content": "Question and source passages"},
    ]

    result = openai_backend.generate_openai_answer(messages)

    assert captured["input"] == messages
    assert captured["store"] is False
    assert captured["max_output_tokens"] == 256
    assert captured["reasoning"] == {"effort": "none"}
    assert result["message"]["content"] == "Answer"


def test_missing_key_fails_before_client_creation(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def unexpected_client(**kwargs):
        pytest.fail("Client must not be created without an API key")

    monkeypatch.setattr(openai_backend, "OpenAI", unexpected_client)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        openai_backend.generate_openai_answer([])


def test_incomplete_response_is_not_accepted(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    install_fake_client(monkeypatch, status="incomplete", text="Partial")

    with pytest.raises(RuntimeError, match="complete answer"):
        openai_backend.generate_openai_answer([])