"""PDF ingestion for the controlled Nordlicht document format, without OCR."""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

SECTION = re.compile(r"^(S\d+):\s+(.+)$")


class DocumentLoadError(ValueError):
    """A document cannot be loaded without losing required source information."""


@dataclass(frozen=True)
class DocumentSection:
    document_id: str
    title: str
    version: str
    status: str
    source_path: str
    section_id: str
    heading: str
    page_start: int
    page_end: int
    text: str

    @property
    def citation_id(self) -> str:
        return f"{self.document_id}:{self.section_id}"


def clean_page(text: str, document_id: str) -> list[str]:
    """Remove known template furniture, retaining body lines and section headings."""
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line == "NORDLICHT SOFTWARE  /  INTERNAL KNOWLEDGE DEMO":
            continue
        if line.startswith(f"{document_id} | ") and line.endswith("Synthetic portfolio data"):
            continue
        if re.fullmatch(r"Page \d+", line):
            continue
        lines.append(line)
    return lines


def parse_sections(
    pages: list[str], *, document_id: str, title: str,
    version: str, status: str, source_path: str,
) -> list[DocumentSection]:
    """Parse explicit S-number headings and preserve actual one-based page ranges."""
    sections = []
    seen = set()
    current = None
    heading = ""
    body: list[str] = []
    start = end = 0

    def finish() -> None:
        if current is None:
            return
        if not body:
            raise DocumentLoadError(f"{document_id}:{current}: empty section")
        sections.append(DocumentSection(
            document_id=document_id, title=title, version=version, status=status,
            source_path=source_path, section_id=current, heading=heading,
            page_start=start, page_end=end, text=" ".join(body),
        ))

    for page_number, text in enumerate(pages, 1):
        if not text.strip():
            raise DocumentLoadError(
                f"{document_id}, page {page_number}: no extractable text; OCR may be needed"
            )
        for line in clean_page(text, document_id):
            match = SECTION.fullmatch(line)
            if match:
                finish()
                current, heading = match.groups()
                if current in seen:
                    raise DocumentLoadError(f"{document_id}: duplicate section {current}")
                seen.add(current)
                body = []
                start = end = page_number
            elif current is not None:
                body.append(line)
                end = page_number
    finish()
    if not sections:
        raise DocumentLoadError(f"{document_id}: no S-number section headings found")
    return sections


def load_corpus(root: Path, *, include_archived: bool = False) -> list[DocumentSection]:
    """Read only manifest-listed PDFs; Markdown and evaluation data are never input."""
    root = root.resolve()
    entries = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        raise DocumentLoadError("Manifest must be a non-empty list")
    sections = []
    seen = set()
    for entry in entries:
        document_id = entry["document_id"]
        if document_id in seen:
            raise DocumentLoadError(f"Duplicate document ID: {document_id}")
        seen.add(document_id)
        status = entry["status"]
        if status not in {"active", "archived"}:
            raise DocumentLoadError(f"{document_id}: unknown status {status}")
        if status == "archived" and not include_archived:
            continue
        path = (root / entry["pdf_path"]).resolve()
        if not path.is_relative_to(root):
            raise DocumentLoadError(f"{document_id}: source path leaves project directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["pdf_sha256"]:
            raise DocumentLoadError(f"{document_id}: PDF checksum mismatch")
        try:
            reader = PdfReader(path)
            if reader.is_encrypted:
                raise DocumentLoadError(f"{document_id}: encrypted PDFs are not supported")
            pages = [page.extract_text() or "" for page in reader.pages]
        except DocumentLoadError:
            raise
        except Exception as exc:
            raise DocumentLoadError(f"{document_id}: cannot read PDF: {exc}") from exc
        if not pages:
            raise DocumentLoadError(f"{document_id}: PDF contains no pages")
        lines = clean_page(pages[0], document_id)
        if not lines:
            raise DocumentLoadError(f"{document_id}: first page has no readable content")
        for label, expected in (
            ("Document ID", document_id), ("Version", entry["version"]), ("Status", status),
        ):
            if f"{label}: {expected}" not in lines:
                raise DocumentLoadError(f"{document_id}: PDF metadata mismatch for {label}")
        sections.extend(parse_sections(
            pages, document_id=document_id, title=lines[0], version=entry["version"],
            status=status, source_path=path.relative_to(root).as_posix(),
        ))
    return sections
