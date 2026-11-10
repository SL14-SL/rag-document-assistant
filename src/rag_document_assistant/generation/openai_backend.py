"""Generate answers through OpenAI using the shared source prompt."""

import os

from openai import OpenAI, OpenAIError

OPENAI_MODEL = "gpt-5.4-2026-03-05"


def generate_openai_answer(
    messages: list[dict[str, str]],
    *,
    model: str = OPENAI_MODEL,
) -> dict:
    """Return an answer in the common generation response format."""
    key = os.environ.get("OPENAI_API_KEY", "")
    if (
        not key
        or not key.isascii()
        or any(character.isspace() for character in key)
    ):
        raise ValueError("Set a valid OPENAI_API_KEY before using OpenAI.")

    payload = {
        "model": model,
        "input": messages,
        "temperature": 0,
        "reasoning": {"effort": "none"},
        "max_output_tokens": 256,
        "store": False,
    }

    try:
        with OpenAI(
            api_key=key,
            base_url="https://api.openai.com/v1",
            timeout=180,
            max_retries=0,
        ) as client:
            response = client.responses.create(**payload)
    except OpenAIError as exc:
        # Avoid displaying raw request or credential details in the UI.
        raise RuntimeError(
            "OpenAI request failed. Check connectivity, API key, "
            "model access and available API credit."
        ) from exc

    if response.status != "completed" or not response.output_text.strip():
        raise RuntimeError(
            "OpenAI did not return a complete answer. "
            "No answer has been accepted."
        )

    return {
        "model": response.model,
        "message": {"content": response.output_text},
        "done": True,
        "done_reason": "stop",
        "generation_request": payload,
        "api_response": response.model_dump(mode="json"),
    }