"""Check shared generation configuration and source metadata."""

import json
from types import SimpleNamespace

from rag_document_assistant.generation.answer import (
    DEFAULT_GENERATION_MODEL,
    DEFAULT_SEED,
    build_messages,
)


def test_default_generation_configuration():
    assert DEFAULT_GENERATION_MODEL == "qwen2.5:14b"
    assert DEFAULT_SEED == 42


def test_messages_preserve_document_metadata_and_full_text():
    text = (
        "During support hours, the initial response target is 30 minutes. "
        "This is not a resolution guarantee."
    )
    section = SimpleNamespace(
        citation_id="NL-16:S2",
        title="Support Priorities",
        version="1.0",
        status="active",
        heading="Response targets",
        text=text,
    )
    result = SimpleNamespace(section=section)

    messages = build_messages("What is the response target?", [result])
    payload = json.loads(messages[1]["content"])

    assert payload["question"] == "What is the response target?"
    assert payload["source_passages"] == [
        {
            "source_id": "NL-16:S2",
            "title": "Support Priorities",
            "version": "1.0",
            "status": "active",
            "heading": "Response targets",
            "text": text,
        }
    ]