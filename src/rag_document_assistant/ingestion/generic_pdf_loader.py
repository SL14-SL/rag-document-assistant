"""Load generic text-based PDFs with token-aware, page-level chunking."""

import hashlib
from pathlib import Path
from typing import Protocol

from pypdf import PdfReader

from rag_document_assistant.ingestion.pdf_loader import (
    DocumentLoadError,
    DocumentSection,
)

DEFAULT_CHUNK_TOKENS = 200
DEFAULT_OVERLAP_TOKENS = 30


class ChunkTokenizer(Protocol):
    def __call__(self, text: str, **kwargs) -> dict: ...


def normalize_text(text: str) -> str:
    """Normalize whitespace without rewriting document content."""
    return " ".join(text.split())


def split_page(
    text: str,
    *,
    tokenizer: ChunkTokenizer,
    title: str,
    heading: str,
    max_seq_length: int,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[str]:
    """Split page text while respecting the complete embedding input limit."""
    if chunk_tokens < 1:
        raise ValueError("chunk_tokens must be positive")

    if not 0 <= overlap_tokens < chunk_tokens:
        raise ValueError(
            "overlap_tokens must be smaller than chunk_tokens"
        )

    if max_seq_length < 1:
        raise ValueError("max_seq_length must be positive")

    text = normalize_text(text)

    if not text:
        return []

    encoded = tokenizer(
        text,
        truncation=False,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )

    offsets = encoded["offset_mapping"]

    if not offsets:
        raise DocumentLoadError("Page contains no indexable tokens")

    chunks = []
    start = 0

    while start < len(offsets):
        end = min(start + chunk_tokens, len(offsets))

        while end > start:
            char_start = offsets[start][0]
            char_end = offsets[end - 1][1]

            passage = text[char_start:char_end].strip()

            embedding_text = f"{title}\n{heading}\n{passage}"

            token_ids = tokenizer(
                embedding_text,
                truncation=False,
                add_special_tokens=True,
            )["input_ids"]

            if passage and len(token_ids) <= max_seq_length:
                break

            end -= 1

        if end <= start:
            raise DocumentLoadError(
                "Document metadata leaves no room for an indexable passage"
            )

        chunks.append(passage)

        if end == len(offsets):
            break

        start = max(start + 1, end - overlap_tokens)

    return chunks


def load_generic_document(
    root: Path,
    entry: dict,
    *,
    tokenizer: ChunkTokenizer,
    max_seq_length: int,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[DocumentSection]:
    """Load one manifest-listed generic PDF."""
    root = root.resolve()

    document_id = entry["document_id"]
    title = entry["title"]
    status = entry["status"]
    version = entry["version"]

    path = (root / entry["pdf_path"]).resolve()

    if not path.is_relative_to(root):
        raise DocumentLoadError(
            f"{document_id}: source path leaves corpus directory"
        )

    try:
        pdf_bytes = path.read_bytes()
    except OSError as exc:
        raise DocumentLoadError(
            f"{document_id}: cannot read PDF: {exc}"
        ) from exc

    if hashlib.sha256(pdf_bytes).hexdigest() != entry["pdf_sha256"]:
        raise DocumentLoadError(
            f"{document_id}: PDF checksum mismatch"
        )

    try:
        reader = PdfReader(path)

        if reader.is_encrypted:
            raise DocumentLoadError(
                f"{document_id}: encrypted PDFs are not supported"
            )

        pages = [
            page.extract_text() or ""
            for page in reader.pages
        ]

    except DocumentLoadError:
        raise
    except Exception as exc:
        raise DocumentLoadError(
            f"{document_id}: cannot read PDF: {exc}"
        ) from exc

    if not pages:
        raise DocumentLoadError(
            f"{document_id}: PDF contains no pages"
        )

    sections = []

    for page_number, page_text in enumerate(pages, start=1):
        if not page_text.strip():
            raise DocumentLoadError(
                f"{document_id}, page {page_number}: "
                "no extractable text; OCR may be needed"
            )

        heading = f"Page {page_number}"

        chunks = split_page(
            page_text,
            tokenizer=tokenizer,
            title=title,
            heading=heading,
            max_seq_length=max_seq_length,
            chunk_tokens=chunk_tokens,
            overlap_tokens=overlap_tokens,
        )

        for chunk in chunks:
            sections.append(
                DocumentSection(
                    document_id=document_id,
                    title=title,
                    version=version,
                    status=status,
                    source_path=path.relative_to(root).as_posix(),
                    section_id=f"S{len(sections) + 1}",
                    heading=heading,
                    page_start=page_number,
                    page_end=page_number,
                    text=chunk,
                )
            )

    if not sections:
        raise DocumentLoadError(
            f"{document_id}: no indexable passages found"
        )

    return sections