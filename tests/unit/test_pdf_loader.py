import hashlib
import json
from pathlib import Path

import pytest
from pypdf import PdfWriter

from rag_document_assistant.ingestion.pdf_loader import (
    DocumentLoadError,
    load_corpus,
    parse_sections,
)


def parse(pages):
    return parse_sections(
        pages, document_id="NL-99", title="Example", version="1.0",
        status="active", source_path="documents_pdf/99.pdf",
    )


def test_multi_page_section_retains_page_range_and_removes_furniture():
    sections = parse([
        "Example\nS1: First\nFirst paragraph.\nNL-99 | ACTIVE | Synthetic portfolio data\nPage 1",
        "NORDLICHT SOFTWARE  /  INTERNAL KNOWLEDGE DEMO\nContinued.\nS2: Second\nNext.\nPage 2",
    ])
    assert sections[0].text == "First paragraph. Continued."
    assert (sections[0].page_start, sections[0].page_end) == (1, 2)
    assert sections[1].citation_id == "NL-99:S2"
    assert sections[1].page_start == 2


@pytest.mark.parametrize("pages, message", [
    (["Title only"], "no S-number"),
    (["S1: First\nS2: Second\nText"], "empty section"),
    (["S1: First\nText\nS1: Again\nText"], "duplicate section"),
    ([""], "no extractable text"),
])
def test_invalid_content_is_rejected(pages, message):
    with pytest.raises(DocumentLoadError, match=message):
        parse(pages)


def test_blank_pdf_is_rejected(tmp_path):
    pdf = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.write(pdf)
    entry = {"document_id": "NL-99", "status": "active", "version": "1.0",
             "pdf_path": "blank.pdf", "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest()}
    (tmp_path / "manifest.json").write_text(json.dumps([entry]))
    with pytest.raises(DocumentLoadError, match="no readable content"):
        load_corpus(tmp_path)


def test_pdf_hash_mismatch_is_rejected(tmp_path):
    (tmp_path / "changed.pdf").write_bytes(b"changed")
    entry = {"document_id": "NL-99", "status": "active", "version": "1.0",
             "pdf_path": "changed.pdf", "pdf_sha256": "wrong"}
    (tmp_path / "manifest.json").write_text(json.dumps([entry]))
    with pytest.raises(DocumentLoadError, match="checksum mismatch"):
        load_corpus(tmp_path)


def test_real_dataset_sources_and_archived_filter():
    root = Path.cwd()
    if not (root / "documents_pdf").is_dir():
        pytest.skip("Run from the project root containing the Nordlicht PDFs")
    active = load_corpus(root)
    all_sections = load_corpus(root, include_archived=True)
    assert len(active) == 48
    assert len(all_sections) == 50
    assert {s.status for s in active} == {"active"}
    assert len({s.document_id for s in active}) == 24
    hotel = next(s for s in active if s.citation_id == "NL-09:S1")
    assert "EUR 120" in hotel.text
    assert "EUR 100" not in hotel.text
    assert hotel.page_start == hotel.page_end == 1
    by_id = {s.citation_id: s for s in all_sections}
    for path in (root / "evaluation").glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            for ref in json.loads(line)["evidence"]:
                section = by_id[f'{ref["document_id"]}:{ref["section"]}']
                assert section.page_start <= ref["page"] <= section.page_end
