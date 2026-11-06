"""Persist a local semantic index without pickle."""

import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

from rag_document_assistant.ingestion.pdf_loader import DocumentSection
from rag_document_assistant.retrieval.semantic import (
    SemanticIndex,
    TextEncoder,
)

INDEX_FORMAT_VERSION = 1


def corpus_fingerprint(root: Path) -> str:
    """Fingerprint the manifest and verify its active PDF files."""
    root = root.resolve()
    manifest_bytes = (root / "manifest.json").read_bytes()
    entries = json.loads(manifest_bytes)

    if not isinstance(entries, list) or not entries:
        raise ValueError("Manifest must be a non-empty list")

    seen = set()
    for entry in entries:
        document_id = entry["document_id"]
        if document_id in seen:
            raise ValueError(f"Duplicate document ID: {document_id}")
        seen.add(document_id)

        if entry["status"] not in {"active", "archived"}:
            raise ValueError(f"{document_id}: unknown status")
        if entry["status"] != "active":
            continue

        path = (root / entry["pdf_path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("PDF path leaves project directory")

        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry["pdf_sha256"]:
            raise ValueError(f"{document_id}: PDF checksum mismatch")

    return hashlib.sha256(manifest_bytes).hexdigest()


def save_index(
    path: Path,
    index: SemanticIndex,
    *,
    fingerprint: str,
    model_name: str,
) -> None:
    """Write one archive atomically to avoid partially written indexes."""
    metadata = {
        "format_version": INDEX_FORMAT_VERSION,
        "corpus_fingerprint": fingerprint,
        "model_name": model_name,
        "search_fields": ["title", "heading", "text"],
        "include_archived": False,
        "sections": [asdict(section) for section in index.sections],
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=".index-",
        suffix=".npz",
    )
    temporary_path = Path(temporary_name)

    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(
                stream,
                embeddings=index.embeddings,
                metadata=np.asarray(json.dumps(metadata)),
            )
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_index(
    path: Path,
    *,
    root: Path,
    model_name: str,
    encoder: TextEncoder,
) -> SemanticIndex:
    """Load stored vectors and reject stale corpus or model metadata."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Index not found: {path}. Run the build_index command first."
        )

    fingerprint = corpus_fingerprint(root)

    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        embeddings = archive["embeddings"].copy()

    if metadata["format_version"] != INDEX_FORMAT_VERSION:
        raise ValueError("Index format changed; rebuild the index")
    if metadata["corpus_fingerprint"] != fingerprint:
        raise ValueError("Document manifest changed; rebuild the index")
    if metadata["model_name"] != model_name:
        raise ValueError("Embedding model changed; rebuild the index")
    if metadata["search_fields"] != ["title", "heading", "text"]:
        raise ValueError("Indexed search fields changed; rebuild the index")
    if metadata["include_archived"] is not False:
        raise ValueError("Expected an active-document index")

    sections = [
        DocumentSection(**section)
        for section in metadata["sections"]
    ]
    if any(section.status != "active" for section in sections):
        raise ValueError("Stored index contains non-active sections")

    return SemanticIndex.from_embeddings(
        sections,
        embeddings,
        encoder=encoder,
    )