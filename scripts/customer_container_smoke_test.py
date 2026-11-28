"""Verify customer deployment without calling a generation provider."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen

from rag_document_assistant.retrieval.semantic import (
    DEFAULT_MODEL,
    LocalSentenceEncoder,
)
from rag_document_assistant.retrieval.storage import load_index
from rag_document_assistant.ui.config import load_app_config

QUESTIONS = {
    "How long do customers have to return unused products?": (
        "DOC-001:S1", 1
    ),
    "How quickly must workplace accidents be reported?": (
        "DOC-003:S2", 2
    ),
    "When is standard shipping free of charge?": (
        "DOC-002:S1", 1
    ),
    "How long does refund processing take?": (
        "DOC-001:S2", 2
    ),
    "Who is allowed to operate forklifts?": (
        "DOC-003:S1", 1
    ),
}


def main() -> None:
    config = load_app_config(Path("/app"))
    root = config.corpus_root

    if os.geteuid() == 0:
        raise RuntimeError("Customer container must not run as root")

    print("PASS: non-root user")

    if config.is_demo:
        raise RuntimeError("Expected customer configuration")

    if config.allowed_providers != ("ollama",):
        raise RuntimeError("Expected local-only provider configuration")

    print("PASS: customer configuration and provider restriction")

    with urlopen(
        "http://127.0.0.1:8501/_stcore/health",
        timeout=5,
    ) as response:
        if response.status != 200:
            raise RuntimeError("Streamlit health check failed")

    print("PASS: Streamlit health endpoint")

    protected_paths = [
        root / "pdfs",
        root / "manifest.json",
        config.index_path,
    ]

    for path in protected_paths:
        if not os.access(path, os.R_OK):
            raise RuntimeError(f"Path is not readable: {path}")

        if os.access(path, os.W_OK):
            raise RuntimeError(f"Path must be read-only: {path}")

    print("PASS: read-only customer documents and index")

    encoder = LocalSentenceEncoder(local_files_only=True)

    index = load_index(
        config.index_path,
        root=root,
        model_name=DEFAULT_MODEL,
        encoder=encoder,
    )

    if len(index.sections) != 6:
        raise RuntimeError("Expected six sections in the test index")

    print("PASS: persisted customer index")

    for question, (expected_id, expected_page) in QUESTIONS.items():
        result = index.search(question, top_k=3)[0]
        section = result.section

        if section.citation_id != expected_id:
            raise RuntimeError(
                f"Unexpected source for {question}: "
                f"{section.citation_id}"
            )

        if section.page_start != expected_page:
            raise RuntimeError("Unexpected source page")

        source = (root / section.source_path).resolve()

        if not source.is_relative_to(root):
            raise RuntimeError("Source path leaves corpus directory")

        if not source.read_bytes().startswith(b"%PDF-"):
            raise RuntimeError("Invalid source PDF")

        print(f"PASS: {question}")

    report_directory = config.report_directory
    report_directory.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(dir=report_directory) as temporary:
        probe = Path(temporary) / "write-check.txt"
        probe.write_text("customer smoke test", encoding="utf-8")

        if probe.read_text(encoding="utf-8") != "customer smoke test":
            raise RuntimeError("Report directory read/write check failed")

    print("PASS: report directory writable")

    print(
        "\nCustomer container smoke test passed. "
        "No generation requests were made."
    )


if __name__ == "__main__":
    main()