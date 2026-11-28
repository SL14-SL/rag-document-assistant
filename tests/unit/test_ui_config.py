import pytest

from rag_document_assistant.ui.config import load_app_config


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.delenv("RAG_CORPUS_ROOT", raising=False)
    monkeypatch.delenv("RAG_PROVIDER", raising=False)

    (tmp_path / "manifest.json").write_text(
        "[]",
        encoding="utf-8",
    )

    customer = tmp_path / "customer_data"
    customer.mkdir()

    (customer / "manifest.json").write_text(
        "[]",
        encoding="utf-8",
    )

    return tmp_path


def test_default_configuration_uses_demo(project):
    config = load_app_config(project)

    assert config.is_demo is True
    assert config.corpus_root == project
    assert config.provider == "both"
    assert config.allowed_providers == ("ollama", "openai")


def test_demo_index_path(project):
    config = load_app_config(project)

    assert config.index_path == (
        project / "artifacts/index/semantic.npz"
    )


def test_customer_configuration(project, monkeypatch):
    monkeypatch.setenv(
        "RAG_CORPUS_ROOT",
        "customer_data",
    )

    config = load_app_config(project)

    assert config.is_demo is False
    assert config.corpus_root == project / "customer_data"
    assert config.index_path == (
        project / "customer_data/artifacts/index/semantic.npz"
    )


def test_customer_defaults_to_local_provider(project, monkeypatch):
    monkeypatch.setenv(
        "RAG_CORPUS_ROOT",
        "customer_data",
    )

    config = load_app_config(project)

    assert config.provider == "ollama"
    assert config.allowed_providers == ("ollama",)


def test_absolute_corpus_path(project, monkeypatch):
    customer = project / "customer_data"

    monkeypatch.setenv(
        "RAG_CORPUS_ROOT",
        str(customer),
    )

    config = load_app_config(project)

    assert config.corpus_root == customer


@pytest.mark.parametrize(
    "provider, expected",
    [
        ("ollama", ("ollama",)),
        ("openai", ("openai",)),
        ("both", ("ollama", "openai")),
    ],
)
def test_explicit_provider_configuration(
    project,
    monkeypatch,
    provider,
    expected,
):
    monkeypatch.setenv("RAG_PROVIDER", provider)

    config = load_app_config(project)

    assert config.allowed_providers == expected


def test_provider_is_normalized(project, monkeypatch):
    monkeypatch.setenv("RAG_PROVIDER", " OpenAI ")

    config = load_app_config(project)

    assert config.provider == "openai"


def test_invalid_provider_is_rejected(project, monkeypatch):
    monkeypatch.setenv("RAG_PROVIDER", "unsupported")

    with pytest.raises(ValueError, match="RAG_PROVIDER"):
        load_app_config(project)


def test_missing_corpus_is_rejected(project, monkeypatch):
    monkeypatch.setenv(
        "RAG_CORPUS_ROOT",
        "missing",
    )

    with pytest.raises(
        ValueError,
        match="Corpus directory does not exist",
    ):
        load_app_config(project)


def test_missing_manifest_is_rejected(project, monkeypatch):
    (project / "customer_data/manifest.json").unlink()

    monkeypatch.setenv(
        "RAG_CORPUS_ROOT",
        "customer_data",
    )

    with pytest.raises(
        ValueError,
        match="Corpus manifest does not exist",
    ):
        load_app_config(project)


def test_customer_reports_are_separated(project, monkeypatch):
    monkeypatch.setenv(
        "RAG_CORPUS_ROOT",
        "customer_data",
    )

    config = load_app_config(project)

    assert config.report_directory == (
        project / "customer_data/artifacts/answers/streamlit"
    )