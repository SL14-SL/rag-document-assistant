"""Generate a manifest for an independent customer PDF corpus."""

import argparse
import hashlib
import json
from pathlib import Path

from rag_document_assistant.ingestion.pdf_loader import DocumentLoadError


def build_manifest(root: Path) -> list[dict[str, str]]:
    """Discover customer PDFs and build deterministic manifest entries."""
    root = root.resolve()
    pdf_directory = root / "pdfs"

    if not pdf_directory.is_dir():
        raise DocumentLoadError(
            f"PDF directory does not exist: {pdf_directory}"
        )

    if not pdf_directory.resolve().is_relative_to(root):
        raise DocumentLoadError(
            "PDF directory leaves corpus directory"
        )

    paths = sorted(
        (
            path
            for path in pdf_directory.iterdir()
            if path.is_file() and path.suffix.lower() == ".pdf"
        ),
        key=lambda path: (path.name.casefold(), path.name),
    )

    if not paths:
        raise DocumentLoadError(
            f"No PDF files found in {pdf_directory}"
        )

    entries = []

    for number, path in enumerate(paths, start=1):
        resolved_path = path.resolve()

        if not resolved_path.is_relative_to(root):
            raise DocumentLoadError(
                f"PDF path leaves corpus directory: {path.name}"
            )

        title = " ".join(
            path.stem.replace("_", " ").replace("-", " ").split()
        )

        entries.append(
            {
                "document_id": f"DOC-{number:03d}",
                "title": title or path.stem,
                "status": "active",
                "version": "unknown",
                "format": "generic",
                "pdf_path": path.relative_to(root).as_posix(),
                "pdf_sha256": hashlib.sha256(
                    resolved_path.read_bytes()
                ).hexdigest(),
            }
        )

    return entries


def write_manifest(
    root: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a customer manifest without silently replacing existing data."""
    root = root.resolve()
    output = root / "manifest.json"

    if output.exists() and not overwrite:
        raise FileExistsError(
            f"Manifest already exists: {output}. "
            "Use --overwrite to regenerate it."
        )

    entries = build_manifest(root)

    content = json.dumps(
        entries,
        indent=2,
        ensure_ascii=False,
    ) + "\n"

    with output.open(
        "w" if overwrite else "x",
        encoding="utf-8",
    ) as stream:
        stream.write(content)

    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Corpus directory containing a pdfs/ subdirectory",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate an existing manifest",
    )

    args = parser.parse_args()

    output = write_manifest(
        args.root,
        overwrite=args.overwrite,
    )

    print(f"Manifest written: {output}")


if __name__ == "__main__":
    main()