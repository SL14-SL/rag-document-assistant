"""Check the container without calling a generation provider."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen

from rag_document_assistant.retrieval.semantic import (
    DEFAULT_MODEL,
    LocalSentenceEncoder,
)
from rag_document_assistant.retrieval.storage import load_index


def main() -> None:
    root = Path.cwd()

    if os.geteuid() == 0:
        raise RuntimeError("The application must run as a non-root user.")
    print("PASS: non-root user")

    with urlopen("http://127.0.0.1:8501/_stcore/health", timeout=5) as response:
        if response.status != 200:
            raise RuntimeError("Streamlit health check failed.")
    print("PASS: Streamlit health endpoint")

    encoder = LocalSentenceEncoder(local_files_only=True)
    index = load_index(
        root / "artifacts/index/semantic.npz",
        root=root,
        model_name=DEFAULT_MODEL,
        encoder=encoder,
    )
    results = index.search("What is the hotel limit in Germany?", top_k=5)
    source_ids = {result.section.citation_id for result in results}

    if "NL-09:S1" not in source_ids:
        raise RuntimeError(f"Expected hotel policy passage; received {source_ids}")
    print("PASS: bundled encoder and semantic index")

    pdf = root / "documents_pdf/09.pdf"
    if not pdf.is_file() or not pdf.read_bytes().startswith(b"%PDF-"):
        raise RuntimeError("The source PDF is missing or invalid.")
    print("PASS: source PDF available")

    report_directory = root / "artifacts/answers"
    with TemporaryDirectory(dir=report_directory) as temporary:
        probe = Path(temporary) / "write-check.txt"
        probe.write_text("container smoke test", encoding="utf-8")
        if probe.read_text(encoding="utf-8") != "container smoke test":
            raise RuntimeError("Report directory read/write check failed.")
    print("PASS: report directory writable")

    print("\nContainer smoke test passed. No generation requests were made.")


if __name__ == "__main__":
    main()