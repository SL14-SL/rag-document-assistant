import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from rag_document_assistant.generation import chat
from rag_document_assistant.ingestion.pdf_loader import DocumentSection
from rag_document_assistant.retrieval import semantic
from rag_document_assistant.retrieval.semantic import (
    DEFAULT_MODEL,
    SemanticIndex,
)
from rag_document_assistant.retrieval.storage import (
    corpus_fingerprint,
    save_index,
)

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


QUESTION = "How long do customers have to return unused products?"

SOURCE_TEXT = (
    "Customers may return unused products within "
    "45 calendar days of delivery."
)

ANSWER = (
    "Customers may return unused products within "
    "45 calendar days [DOC-001:S1]."
)


class FakeEncoder:
    """Deterministic encoder without downloading or loading an ML model."""

    def __init__(self, **kwargs):
        pass

    def encode(self, texts):
        return np.asarray(
            [[1.0, 0.0] for _ in texts],
            dtype=np.float32,
        )


@pytest.fixture
def indexed_customer_corpus(customer_corpus):
    root = customer_corpus

    pdf_directory = root / "pdfs"
    pdf_directory.mkdir()

    pdf = pdf_directory / "returns_policy.pdf"
    pdf.write_bytes(b"%PDF-1.4\nSynthetic PDF fixture\n")

    entry = {
        "document_id": "DOC-001",
        "title": "Returns policy",
        "version": "unknown",
        "status": "active",
        "format": "generic",
        "pdf_path": "pdfs/returns_policy.pdf",
        "pdf_sha256": hashlib.sha256(
            pdf.read_bytes()
        ).hexdigest(),
    }

    (root / "manifest.json").write_text(
        json.dumps([entry]),
        encoding="utf-8",
    )

    section = DocumentSection(
        document_id="DOC-001",
        title="Returns policy",
        version="unknown",
        status="active",
        source_path="pdfs/returns_policy.pdf",
        section_id="S1",
        heading="Page 1",
        page_start=1,
        page_end=1,
        text=SOURCE_TEXT,
    )

    index = SemanticIndex(
        [section],
        encoder=FakeEncoder(),
    )

    save_index(
        root / "artifacts/index/semantic.npz",
        index,
        fingerprint=corpus_fingerprint(root),
        model_name=DEFAULT_MODEL,
    )

    return root


@pytest.fixture
def mocked_providers(monkeypatch):
    monkeypatch.setattr(
        semantic,
        "LocalSentenceEncoder",
        FakeEncoder,
    )

    def response():
        return {
            "model": "test-model",
            "message": {"content": ANSWER},
            "done": True,
            "done_reason": "stop",
        }

    ollama = Mock(return_value=response())
    openai = Mock(return_value=response())
    runtime = Mock(return_value={})

    monkeypatch.setattr(
        chat,
        "generate_answer",
        ollama,
    )

    monkeypatch.setattr(
        chat,
        "generate_openai_answer",
        openai,
    )

    monkeypatch.setattr(
        chat,
        "collect_runtime_metadata",
        runtime,
    )

    return {
        "ollama": ollama,
        "openai": openai,
        "runtime": runtime,
    }


def submit_question(app, question=QUESTION):
    app.text_area[0].set_value(question)

    ask_button = next(
        button
        for button in app.button
        if button.label == "Ask"
    )

    ask_button.click().run()

    assert not app.exception

    return app


def saved_reports(root):
    return sorted(
        (root / "artifacts/answers/streamlit").glob("*.json")
    )


@pytest.mark.parametrize("provider", ["ollama", "openai"])
def test_customer_question_uses_configured_provider(
    indexed_customer_corpus,
    mocked_providers,
    monkeypatch,
    provider,
):
    monkeypatch.setenv("RAG_PROVIDER", provider)

    app = submit_question(run_app())

    assert not app.error

    mocked_providers[provider].assert_called_once()

    other_provider = (
        "openai" if provider == "ollama" else "ollama"
    )

    mocked_providers[other_provider].assert_not_called()

    if provider == "ollama":
        mocked_providers["runtime"].assert_called_once()
    else:
        mocked_providers["runtime"].assert_not_called()

    report = app.session_state["report"]

    assert report["provider"] == provider
    assert report["question"] == QUESTION
    assert report["answer"] == ANSWER

    assert report["citation_check"]["cited_ids"] == (
        "DOC-001:S1",
    )

    assert report["citation_check"]["unknown_ids"] == ()

    assert report["sources"][0]["citation_id"] == (
        "DOC-001:S1"
    )

    assert report["sources"][0]["page_start"] == 1

    assert any(
        ANSWER in element.value
        for element in app.markdown
    )

    assert any(
        "DOC-001:S1" in element.label
        and "cited in answer" in element.label
        for element in app.expander
    )


def test_customer_report_is_saved_in_customer_directory(
    indexed_customer_corpus,
    mocked_providers,
):
    app = submit_question(run_app())

    reports = saved_reports(indexed_customer_corpus)

    assert len(reports) == 1

    report = json.loads(
        reports[0].read_text(encoding="utf-8")
    )

    assert report["question"] == QUESTION
    assert report["answer"] == ANSWER
    assert report["provider"] == "ollama"

    assert report["sources"][0]["source_path"] == (
        "pdfs/returns_policy.pdf"
    )

    assert report["citation_check"]["cited_ids"] == [
        "DOC-001:S1"
    ]

    assert app.session_state["report_path"] == (
        reports[0]
        .relative_to(indexed_customer_corpus)
        .as_posix()
    )


def test_changed_pdf_prevents_generation_with_cached_index(
    indexed_customer_corpus,
    mocked_providers,
):
    app = submit_question(run_app())

    assert not app.error

    mocked_providers["ollama"].assert_called_once()

    pdf = (
        indexed_customer_corpus
        / "pdfs/returns_policy.pdf"
    )

    pdf.write_bytes(b"%PDF-1.4\nUpdated document\n")

    app = submit_question(app)

    assert any(
        "PDF checksum mismatch" in element.value
        for element in app.error
    )

    assert "report" not in app.session_state

    mocked_providers["ollama"].assert_called_once()
    mocked_providers["openai"].assert_not_called()

    assert len(
        saved_reports(indexed_customer_corpus)
    ) == 1


def test_changed_manifest_prevents_generation_with_cached_index(
    indexed_customer_corpus,
    mocked_providers,
):
    app = submit_question(run_app())

    assert not app.error

    manifest = (
        indexed_customer_corpus / "manifest.json"
    )

    entries = json.loads(
        manifest.read_text(encoding="utf-8")
    )

    entries[0]["title"] = "Updated returns policy"

    manifest.write_text(
        json.dumps(entries),
        encoding="utf-8",
    )

    app = submit_question(app)

    assert any(
        "Document manifest changed" in element.value
        for element in app.error
    )

    assert "report" not in app.session_state

    mocked_providers["ollama"].assert_called_once()
    mocked_providers["openai"].assert_not_called()

    assert len(
        saved_reports(indexed_customer_corpus)
    ) == 1


def test_missing_index_prevents_generation(
    indexed_customer_corpus,
    mocked_providers,
):
    index_path = (
        indexed_customer_corpus
        / "artifacts/index/semantic.npz"
    )

    index_path.unlink()

    app = submit_question(run_app())

    assert app.error

    assert "report" not in app.session_state

    mocked_providers["ollama"].assert_not_called()
    mocked_providers["openai"].assert_not_called()

    assert saved_reports(indexed_customer_corpus) == []


def test_new_question_clears_previous_answer(
    indexed_customer_corpus,
    mocked_providers,
):
    app = submit_question(run_app())

    assert "report" in app.session_state

    reset_button = next(
        button
        for button in app.button
        if button.label == "New question"
    )

    reset_button.click().run()

    assert not app.exception
    assert app.text_area[0].value == ""

    assert "report" not in app.session_state
    assert "report_path" not in app.session_state

    mocked_providers["ollama"].assert_called_once()

    assert len(
        saved_reports(indexed_customer_corpus)
    ) == 1