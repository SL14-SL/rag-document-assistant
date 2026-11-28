import hashlib
import json

import pytest

from rag_document_assistant.ingestion.manifest import (
    build_manifest,
    write_manifest,
)
from rag_document_assistant.ingestion.pdf_loader import (
    DocumentLoadError,
)


@pytest.fixture
def corpus(tmp_path):
    pdf_directory = tmp_path / "pdfs"
    pdf_directory.mkdir()

    (pdf_directory / "travel_policy.pdf").write_bytes(
        b"travel policy"
    )
    (pdf_directory / "employee_handbook.PDF").write_bytes(
        b"employee handbook"
    )
    (pdf_directory / "notes.txt").write_text(
        "Not a PDF",
        encoding="utf-8",
    )

    return tmp_path


def test_build_manifest_discovers_only_pdfs(corpus):
    entries = build_manifest(corpus)

    assert len(entries) == 2

    assert [entry["pdf_path"] for entry in entries] == [
        "pdfs/employee_handbook.PDF",
        "pdfs/travel_policy.pdf",
    ]


def test_manifest_contains_required_metadata(corpus):
    entry = build_manifest(corpus)[0]

    assert entry == {
        "document_id": "DOC-001",
        "title": "employee handbook",
        "status": "active",
        "version": "unknown",
        "format": "generic",
        "pdf_path": "pdfs/employee_handbook.PDF",
        "pdf_sha256": hashlib.sha256(
            b"employee handbook"
        ).hexdigest(),
    }


def test_manifest_ids_are_unique(corpus):
    entries = build_manifest(corpus)

    assert [entry["document_id"] for entry in entries] == [
        "DOC-001",
        "DOC-002",
    ]


def test_manifest_generation_is_deterministic(corpus):
    assert build_manifest(corpus) == build_manifest(corpus)


def test_missing_pdf_directory_is_rejected(tmp_path):
    with pytest.raises(
        DocumentLoadError,
        match="PDF directory does not exist",
    ):
        build_manifest(tmp_path)


def test_empty_pdf_directory_is_rejected(tmp_path):
    (tmp_path / "pdfs").mkdir()

    with pytest.raises(
        DocumentLoadError,
        match="No PDF files found",
    ):
        build_manifest(tmp_path)


def test_external_pdf_path_is_rejected(corpus, tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-external.pdf"
    outside.write_bytes(b"external document")

    try:
        (corpus / "pdfs" / "external.pdf").symlink_to(outside)

        with pytest.raises(
            DocumentLoadError,
            match="PDF path leaves corpus directory",
        ):
            build_manifest(corpus)
    finally:
        outside.unlink(missing_ok=True)


def test_write_manifest(corpus):
    output = write_manifest(corpus)

    assert output == corpus / "manifest.json"

    entries = json.loads(
        output.read_text(encoding="utf-8")
    )

    assert entries == build_manifest(corpus)


def test_existing_manifest_is_not_overwritten(corpus):
    output = write_manifest(corpus)
    original = output.read_bytes()

    with pytest.raises(
        FileExistsError,
        match="Manifest already exists",
    ):
        write_manifest(corpus)

    assert output.read_bytes() == original


def test_manifest_can_be_regenerated(corpus):
    output = write_manifest(corpus)

    (corpus / "pdfs" / "new_policy.pdf").write_bytes(
        b"new policy"
    )

    write_manifest(corpus, overwrite=True)

    entries = json.loads(
        output.read_text(encoding="utf-8")
    )

    assert len(entries) == 3


def test_changed_pdf_produces_different_checksum(corpus):
    original = build_manifest(corpus)

    (corpus / "pdfs" / "employee_handbook.PDF").write_bytes(
        b"updated employee handbook"
    )

    updated = build_manifest(corpus)

    assert original[0]["pdf_sha256"] != updated[0]["pdf_sha256"]