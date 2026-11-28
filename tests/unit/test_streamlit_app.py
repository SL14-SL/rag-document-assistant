from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[2] / "app.py"


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    monkeypatch.delenv("RAG_CORPUS_ROOT", raising=False)
    monkeypatch.delenv("RAG_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


@pytest.fixture
def customer_corpus(tmp_path, monkeypatch):
    (tmp_path / "manifest.json").write_text(
        "[]",
        encoding="utf-8",
    )

    monkeypatch.setenv(
        "RAG_CORPUS_ROOT",
        str(tmp_path),
    )

    return tmp_path


def run_app():
    app = AppTest.from_file(
        str(APP_PATH),
        default_timeout=30,
    ).run()

    assert not app.exception

    return app


def test_demo_displays_provider_selection():
    app = run_app()

    assert len(app.sidebar.selectbox) == 1

    assert app.sidebar.selectbox[0].options == [
        "Local Ollama",
        "OpenAI API",
    ]


def test_demo_displays_portfolio_notice():
    app = run_app()

    assert any(
        "Portfolio demo" in element.value
        for element in app.info
    )


def test_customer_defaults_to_ollama(customer_corpus):
    app = run_app()

    assert len(app.sidebar.selectbox) == 0

    assert any(
        "Answer provider: Local Ollama" in element.value
        for element in app.sidebar.markdown
    )


def test_customer_does_not_display_portfolio_notice(
    customer_corpus,
):
    app = run_app()

    assert not any(
        "Portfolio demo" in element.value
        for element in app.info
    )

    assert any(
        "retrieved document passages" in element.value
        for element in app.info
    )


def test_customer_uses_generic_question_placeholder(
    customer_corpus,
):
    app = run_app()

    assert app.text_area[0].proto.placeholder == (
        "Enter a question about your documents."
    )


def test_openai_only_configuration(
    customer_corpus,
    monkeypatch,
):
    monkeypatch.setenv("RAG_PROVIDER", "openai")

    app = run_app()

    assert len(app.sidebar.selectbox) == 0

    assert any(
        "Answer provider: OpenAI API" in element.value
        for element in app.sidebar.markdown
    )


def test_openai_displays_data_transfer_warning(
    customer_corpus,
    monkeypatch,
):
    monkeypatch.setenv("RAG_PROVIDER", "openai")

    app = run_app()

    assert any(
        "sent to OpenAI" in element.value
        for element in app.sidebar.warning
    )


def test_ollama_does_not_display_openai_warning(
    customer_corpus,
    monkeypatch,
):
    monkeypatch.setenv("RAG_PROVIDER", "ollama")

    app = run_app()

    assert not any(
        "sent to OpenAI" in element.value
        for element in app.sidebar.warning
    )


def test_invalid_provider_displays_error(monkeypatch):
    monkeypatch.setenv("RAG_PROVIDER", "invalid")

    app = run_app()

    assert any(
        "RAG_PROVIDER" in element.value
        for element in app.error
    )


def test_missing_corpus_displays_error(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "RAG_CORPUS_ROOT",
        str(tmp_path / "missing"),
    )

    app = run_app()

    assert any(
        "Corpus directory does not exist" in element.value
        for element in app.error
    )


def test_missing_manifest_displays_error(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv(
        "RAG_CORPUS_ROOT",
        str(tmp_path),
    )

    app = run_app()

    assert any(
        "Corpus manifest does not exist" in element.value
        for element in app.error
    )


def test_empty_question_displays_warning(customer_corpus):
    app = run_app()

    ask_button = next(
        button
        for button in app.button
        if button.label == "Ask"
    )

    ask_button.click().run()

    assert not app.exception

    assert any(
        "Enter a question first" in element.value
        for element in app.warning
    )