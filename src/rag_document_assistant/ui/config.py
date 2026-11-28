"""Runtime configuration for demo and customer deployments."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    corpus_root: Path
    index_path: Path
    report_directory: Path
    provider: str
    is_demo: bool

    @property
    def allowed_providers(self) -> tuple[str, ...]:
        if self.provider == "both":
            return ("ollama", "openai")

        return (self.provider,)


def load_app_config(project_root: Path) -> AppConfig:
    """Resolve document paths and allowed generation providers."""
    project_root = project_root.resolve()

    configured_root = os.getenv("RAG_CORPUS_ROOT", "").strip()

    if configured_root:
        corpus_root = (
            project_root / configured_root
        ).resolve()
    else:
        corpus_root = project_root

    is_demo = corpus_root == project_root

    default_provider = "both" if is_demo else "ollama"

    provider = os.getenv(
        "RAG_PROVIDER",
        default_provider,
    ).strip().lower()

    if provider not in {"ollama", "openai", "both"}:
        raise ValueError(
            "RAG_PROVIDER must be one of: ollama, openai, both"
        )

    if not corpus_root.is_dir():
        raise ValueError(
            f"Corpus directory does not exist: {corpus_root}"
        )

    if not (corpus_root / "manifest.json").is_file():
        raise ValueError(
            f"Corpus manifest does not exist: {corpus_root}"
        )

    return AppConfig(
        corpus_root=corpus_root,
        index_path=(
            corpus_root / "artifacts/index/semantic.npz"
        ),
        report_directory=(
            corpus_root / "artifacts/answers/streamlit"
        ),
        provider=provider,
        is_demo=is_demo,
    )