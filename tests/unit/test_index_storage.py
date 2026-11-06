import hashlib
import json

import numpy as np
import pytest

from rag_document_assistant.ingestion.pdf_loader import DocumentSection
from rag_document_assistant.retrieval.semantic import SemanticIndex
from rag_document_assistant.retrieval.storage import (
    corpus_fingerprint,
    load_index,
    save_index,
)


class RecordingEncoder:
    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)


def prepare_corpus(tmp_path):
    pdf = tmp_path / "01.pdf"
    pdf.write_bytes(b"fixture bytes; parsing is tested by the PDF loader")
    entry = {
        "document_id": "NL-01",
        "status": "active",
        "pdf_path": "01.pdf",
        "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([entry]), encoding="utf-8")
    return pdf, manifest


def make_index():
    section = DocumentSection(
        document_id="NL-01",
        title="Hotel policy",
        version="1.0",
        status="active",
        source_path="01.pdf",
        section_id="S1",
        heading="Rules",
        page_start=1,
        page_end=1,
        text="Hotel bookings require approval.",
    )
    return SemanticIndex([section], encoder=RecordingEncoder())


def test_round_trip_does_not_reencode_documents(tmp_path):
    prepare_corpus(tmp_path)
    original = make_index()
    path = tmp_path / "index.npz"
    save_index(
        path,
        original,
        fingerprint=corpus_fingerprint(tmp_path),
        model_name="test-model",
    )

    encoder = RecordingEncoder()
    restored = load_index(
        path,
        root=tmp_path,
        model_name="test-model",
        encoder=encoder,
    )

    assert encoder.calls == []
    assert restored.sections == original.sections
    np.testing.assert_allclose(restored.embeddings, original.embeddings)

    results = restored.search("hotel")
    assert encoder.calls == [["hotel"]]
    assert results[0].section.citation_id == "NL-01:S1"


def test_changed_pdf_is_rejected(tmp_path):
    pdf, _ = prepare_corpus(tmp_path)
    path = tmp_path / "index.npz"
    save_index(
        path,
        make_index(),
        fingerprint=corpus_fingerprint(tmp_path),
        model_name="test-model",
    )
    pdf.write_bytes(b"changed")

    with pytest.raises(ValueError, match="checksum mismatch"):
        load_index(
            path,
            root=tmp_path,
            model_name="test-model",
            encoder=RecordingEncoder(),
        )


def test_changed_manifest_is_rejected(tmp_path):
    _, manifest = prepare_corpus(tmp_path)
    path = tmp_path / "index.npz"
    save_index(
        path,
        make_index(),
        fingerprint=corpus_fingerprint(tmp_path),
        model_name="test-model",
    )
    manifest.write_text(
        manifest.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="manifest changed"):
        load_index(
            path,
            root=tmp_path,
            model_name="test-model",
            encoder=RecordingEncoder(),
        )


def test_changed_model_is_rejected(tmp_path):
    prepare_corpus(tmp_path)
    path = tmp_path / "index.npz"
    save_index(
        path,
        make_index(),
        fingerprint=corpus_fingerprint(tmp_path),
        model_name="old-model",
    )

    with pytest.raises(ValueError, match="model changed"):
        load_index(
            path,
            root=tmp_path,
            model_name="new-model",
            encoder=RecordingEncoder(),
        )


def test_missing_index_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_index"):
        load_index(
            tmp_path / "missing.npz",
            root=tmp_path,
            model_name="test-model",
            encoder=RecordingEncoder(),
        )