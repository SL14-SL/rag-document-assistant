import hashlib
import re
from types import SimpleNamespace

import pytest
from pypdf import PdfWriter

from rag_document_assistant.ingestion import generic_pdf_loader
from rag_document_assistant.ingestion.generic_pdf_loader import (
    load_generic_document,
    normalize_text,
    split_page,
)
from rag_document_assistant.ingestion.pdf_loader import (
    DocumentLoadError,
)


class FakeTokenizer:
    """Deterministic word-based tokenizer for unit tests."""

    def __call__(
        self,
        text,
        *,
        truncation=False,
        add_special_tokens=True,
        return_offsets_mapping=False,
    ):
        matches = list(re.finditer(r"\S+", text))

        input_ids = list(range(len(matches)))

        if add_special_tokens:
            input_ids = [-1, *input_ids, -2]

        result = {"input_ids": input_ids}

        if return_offsets_mapping:
            result["offset_mapping"] = [
                match.span()
                for match in matches
            ]

        return result


@pytest.fixture
def tokenizer():
    return FakeTokenizer()


@pytest.fixture
def corpus(tmp_path):
    pdf_directory = tmp_path / "pdfs"
    pdf_directory.mkdir()

    path = pdf_directory / "employee_handbook.pdf"
    path.write_bytes(b"example PDF fixture")

    entry = {
        "document_id": "DOC-001",
        "title": "Employee Handbook",
        "version": "unknown",
        "status": "active",
        "format": "generic",
        "pdf_path": "pdfs/employee_handbook.pdf",
        "pdf_sha256": hashlib.sha256(
            path.read_bytes()
        ).hexdigest(),
    }

    return tmp_path, entry


def mock_pdf(monkeypatch, pages, *, encrypted=False):
    reader = SimpleNamespace(
        is_encrypted=encrypted,
        pages=[
            SimpleNamespace(
                extract_text=lambda text=text: text
            )
            for text in pages
        ],
    )

    monkeypatch.setattr(
        generic_pdf_loader,
        "PdfReader",
        lambda *_args, **_kwargs: reader,
    )


def chunk(text, tokenizer, **kwargs):
    return split_page(
        text,
        tokenizer=tokenizer,
        title="Example",
        heading="Page 1",
        max_seq_length=kwargs.pop("max_seq_length", 256),
        **kwargs,
    )


def test_normalize_text():
    assert normalize_text(
        "First   sentence.\n\nSecond sentence.\t"
    ) == "First sentence. Second sentence."


def test_short_page_creates_one_chunk(tokenizer):
    assert chunk(
        "Employees require approval.",
        tokenizer,
    ) == ["Employees require approval."]


def test_empty_page_returns_no_chunks(tokenizer):
    assert chunk(" \n ", tokenizer) == []


def test_long_page_is_split(tokenizer):
    text = " ".join(f"word{i}" for i in range(25))

    chunks = chunk(
        text,
        tokenizer,
        chunk_tokens=10,
        overlap_tokens=2,
    )

    assert len(chunks) == 3
    assert all(len(value.split()) <= 10 for value in chunks)


def test_chunks_preserve_overlap(tokenizer):
    text = " ".join(f"word{i}" for i in range(25))

    chunks = chunk(
        text,
        tokenizer,
        chunk_tokens=10,
        overlap_tokens=2,
    )

    for previous, current in zip(
        chunks,
        chunks[1:],
        strict=False,
    ):
        assert previous.split()[-2:] == current.split()[:2]


def test_chunks_preserve_all_words(tokenizer):
    words = [f"word{i}" for i in range(25)]

    chunks = chunk(
        " ".join(words),
        tokenizer,
        chunk_tokens=10,
        overlap_tokens=2,
    )

    covered = {
        word
        for passage in chunks
        for word in passage.split()
    }

    assert covered == set(words)


def test_metadata_is_included_in_token_budget(tokenizer):
    chunks = chunk(
        " ".join(f"word{i}" for i in range(30)),
        tokenizer,
        max_seq_length=12,
        chunk_tokens=10,
        overlap_tokens=2,
    )

    for passage in chunks:
        encoded = tokenizer(
            f"Example\nPage 1\n{passage}",
            add_special_tokens=True,
        )

        assert len(encoded["input_ids"]) <= 12


@pytest.mark.parametrize(
    "chunk_tokens, overlap_tokens",
    [
        (0, 0),
        (-1, 0),
        (10, -1),
        (10, 10),
        (10, 11),
    ],
)
def test_invalid_chunk_configuration_is_rejected(
    tokenizer,
    chunk_tokens,
    overlap_tokens,
):
    with pytest.raises(ValueError):
        chunk(
            "Example text",
            tokenizer,
            chunk_tokens=chunk_tokens,
            overlap_tokens=overlap_tokens,
        )


def test_invalid_model_limit_is_rejected(tokenizer):
    with pytest.raises(ValueError, match="max_seq_length"):
        chunk(
            "Example text",
            tokenizer,
            max_seq_length=0,
        )


def test_insufficient_token_budget_is_rejected(tokenizer):
    with pytest.raises(
        DocumentLoadError,
        match="metadata leaves no room",
    ):
        chunk(
            "Example text",
            tokenizer,
            max_seq_length=5,
        )


def test_generic_pdf_does_not_require_nordlicht_metadata(
    corpus,
    tokenizer,
    monkeypatch,
):
    root, entry = corpus

    mock_pdf(
        monkeypatch,
        ["Employees must submit travel expenses within 30 days."],
    )

    sections = load_generic_document(
        root,
        entry,
        tokenizer=tokenizer,
        max_seq_length=256,
    )

    assert len(sections) == 1

    section = sections[0]

    assert section.citation_id == "DOC-001:S1"
    assert section.title == "Employee Handbook"
    assert section.version == "unknown"
    assert section.status == "active"
    assert section.source_path == "pdfs/employee_handbook.pdf"
    assert section.page_start == section.page_end == 1
    assert "30 days" in section.text


def test_multiple_pages_preserve_page_references(
    corpus,
    tokenizer,
    monkeypatch,
):
    root, entry = corpus

    mock_pdf(
        monkeypatch,
        [
            "First page information.",
            "Second page information.",
            "Third page information.",
        ],
    )

    sections = load_generic_document(
        root,
        entry,
        tokenizer=tokenizer,
        max_seq_length=256,
    )

    assert [section.page_start for section in sections] == [
        1, 2, 3,
    ]

    assert all(
        section.page_start == section.page_end
        for section in sections
    )

    assert [section.citation_id for section in sections] == [
        "DOC-001:S1",
        "DOC-001:S2",
        "DOC-001:S3",
    ]


def test_multiple_chunks_preserve_unique_ids(
    corpus,
    tokenizer,
    monkeypatch,
):
    root, entry = corpus

    mock_pdf(
        monkeypatch,
        [
            " ".join(f"first{i}" for i in range(25)),
            " ".join(f"second{i}" for i in range(25)),
        ],
    )

    sections = load_generic_document(
        root,
        entry,
        tokenizer=tokenizer,
        max_seq_length=256,
        chunk_tokens=10,
        overlap_tokens=2,
    )

    assert len(sections) == 6
    assert len({s.citation_id for s in sections}) == 6

    assert [s.page_start for s in sections] == [
        1, 1, 1, 2, 2, 2,
    ]


def test_pdf_checksum_mismatch_is_rejected(corpus, tokenizer):
    root, entry = corpus

    (root / entry["pdf_path"]).write_bytes(b"changed")

    with pytest.raises(
        DocumentLoadError,
        match="checksum mismatch",
    ):
        load_generic_document(
            root,
            entry,
            tokenizer=tokenizer,
            max_seq_length=256,
        )


def test_missing_pdf_is_rejected(corpus, tokenizer):
    root, entry = corpus

    (root / entry["pdf_path"]).unlink()

    with pytest.raises(
        DocumentLoadError,
        match="cannot read PDF",
    ):
        load_generic_document(
            root,
            entry,
            tokenizer=tokenizer,
            max_seq_length=256,
        )


def test_external_pdf_path_is_rejected(corpus, tokenizer):
    root, entry = corpus

    entry["pdf_path"] = "../external.pdf"

    with pytest.raises(
        DocumentLoadError,
        match="source path leaves corpus directory",
    ):
        load_generic_document(
            root,
            entry,
            tokenizer=tokenizer,
            max_seq_length=256,
        )


def test_encrypted_pdf_is_rejected(
    corpus,
    tokenizer,
    monkeypatch,
):
    root, entry = corpus

    mock_pdf(monkeypatch, [], encrypted=True)

    with pytest.raises(
        DocumentLoadError,
        match="encrypted PDFs",
    ):
        load_generic_document(
            root,
            entry,
            tokenizer=tokenizer,
            max_seq_length=256,
        )


def test_empty_pdf_is_rejected(
    corpus,
    tokenizer,
    monkeypatch,
):
    root, entry = corpus

    mock_pdf(monkeypatch, [])

    with pytest.raises(
        DocumentLoadError,
        match="PDF contains no pages",
    ):
        load_generic_document(
            root,
            entry,
            tokenizer=tokenizer,
            max_seq_length=256,
        )


def test_unreadable_page_is_rejected(
    corpus,
    tokenizer,
    monkeypatch,
):
    root, entry = corpus

    mock_pdf(
        monkeypatch,
        ["Readable first page.", ""],
    )

    with pytest.raises(
        DocumentLoadError,
        match="page 2: no extractable text",
    ):
        load_generic_document(
            root,
            entry,
            tokenizer=tokenizer,
            max_seq_length=256,
        )


def test_corrupted_pdf_is_rejected(corpus, tokenizer):
    root, entry = corpus

    with pytest.raises(
        DocumentLoadError,
        match="cannot read PDF",
    ):
        load_generic_document(
            root,
            entry,
            tokenizer=tokenizer,
            max_seq_length=256,
        )


def test_real_blank_pdf_is_rejected(corpus, tokenizer):
    root, entry = corpus
    path = root / entry["pdf_path"]

    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.write(path)

    entry["pdf_sha256"] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

    with pytest.raises(
        DocumentLoadError,
        match="no extractable text",
    ):
        load_generic_document(
            root,
            entry,
            tokenizer=tokenizer,
            max_seq_length=256,
        )

def test_load_corpus_dispatches_to_generic_loader(
    corpus,
    tokenizer,
    monkeypatch,
):
    import json

    from rag_document_assistant.ingestion.pdf_loader import (
        load_corpus,
    )

    root, entry = corpus

    (root / "manifest.json").write_text(
        json.dumps([entry]),
        encoding="utf-8",
    )

    mock_pdf(
        monkeypatch,
        ["Employees must submit expenses within 30 days."],
    )

    encoder = SimpleNamespace(
        model=SimpleNamespace(
            tokenizer=tokenizer,
            max_seq_length=256,
        )
    )

    sections = load_corpus(
        root,
        encoder=encoder,
    )

    assert len(sections) == 1
    assert sections[0].citation_id == "DOC-001:S1"
    assert sections[0].page_start == 1
    assert "30 days" in sections[0].text


def test_load_corpus_rejects_unknown_document_format(corpus):
    import json

    from rag_document_assistant.ingestion.pdf_loader import (
        load_corpus,
    )

    root, entry = corpus

    entry["format"] = "unsupported"

    (root / "manifest.json").write_text(
        json.dumps([entry]),
        encoding="utf-8",
    )

    with pytest.raises(
        DocumentLoadError,
        match="unsupported document format",
    ):
        load_corpus(root)


def test_generic_archived_documents_are_excluded(
    corpus,
    tokenizer,
    monkeypatch,
):
    import json

    from rag_document_assistant.ingestion.pdf_loader import (
        load_corpus,
    )

    root, entry = corpus

    entry["status"] = "archived"

    (root / "manifest.json").write_text(
        json.dumps([entry]),
        encoding="utf-8",
    )

    mock_pdf(
        monkeypatch,
        ["Historical policy information."],
    )

    encoder = SimpleNamespace(
        model=SimpleNamespace(
            tokenizer=tokenizer,
            max_seq_length=256,
        )
    )

    assert load_corpus(
        root,
        encoder=encoder,
    ) == []

    sections = load_corpus(
        root,
        encoder=encoder,
        include_archived=True,
    )

    assert len(sections) == 1
    assert sections[0].status == "archived"