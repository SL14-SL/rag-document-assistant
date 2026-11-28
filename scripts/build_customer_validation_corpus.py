"""Generate the synthetic Westbridge PDF corpus from versioned fixtures."""

import argparse
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory

from pypdf import PdfReader
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas

FIXTURE = Path(__file__).resolve().parent / "fixtures/customer_validation_documents.json"


def build_corpus(root: Path, fixture: Path = FIXTURE) -> None:
    """Build exactly 20 ten-page PDFs, refusing to overwrite existing files."""
    data = json.loads(fixture.read_text(encoding="utf-8"))
    documents = data["documents"]
    names = [document["filename"] for document in documents]
    if len(names) != 20 or len(set(names)) != 20:
        raise ValueError("Expected 20 uniquely named documents")
    if any(not re.fullmatch(r"[a-z][a-z0-9_]*\.pdf", name) for name in names):
        raise ValueError("Invalid PDF filename")
    if any(len(document["pages"]) != 10 for document in documents):
        raise ValueError("Expected ten pages per document")

    directory = root / "pdfs"
    directory.mkdir(parents=True, exist_ok=True)
    if any(path.suffix.lower() == ".pdf" for path in directory.iterdir()):
        raise FileExistsError(f"PDF directory must be empty: {directory}")

    with TemporaryDirectory(dir=root) as temporary:
        staging = Path(temporary)
        for document in documents:
            path = staging / document["filename"]
            pdf = canvas.Canvas(str(path), pagesize=A4, invariant=1)
            pdf.setTitle(document["title"])
            pdf.setAuthor(data["company"])
            width, height = A4
            for number, page in enumerate(document["pages"], start=1):
                pdf.setFont("Helvetica-Bold", 11)
                pdf.drawString(48, height - 48, data["company"])
                pdf.setFont("Helvetica", 10)
                pdf.drawString(48, height - 66, document["title"])
                pdf.setFont("Helvetica-Bold", 16)
                pdf.drawString(48, height - 104, page["heading"])
                y = height - 136
                pdf.setFont("Helvetica", 11)
                for paragraph in page["paragraphs"]:
                    if not paragraph.strip():
                        raise ValueError(f"Empty paragraph: {path.name}, page {number}")
                    for line in simpleSplit(paragraph, "Helvetica", 11, width - 96):
                        if y < 65:
                            raise ValueError(f"Text overflow: {path.name}, page {number}")
                        pdf.drawString(48, y, line)
                        y -= 15
                    y -= 10
                pdf.setFont("Helvetica", 9)
                pdf.drawString(48, 36, "Synthetic validation document | Version 1.0")
                pdf.drawRightString(width - 48, 36, f"Page {number} / 10")
                pdf.showPage()
            pdf.save()
            reader = PdfReader(path)
            if len(reader.pages) != 10:
                raise RuntimeError(f"Incorrect page count: {path.name}")
            for source, rendered in zip(document["pages"], reader.pages, strict=True):
                actual = " ".join((rendered.extract_text() or "").split())
                for paragraph in source["paragraphs"]:
                    if " ".join(paragraph.split()) not in actual:
                        raise RuntimeError(f"Text extraction mismatch: {path.name}")
        for name in names:
            (staging / name).rename(directory / name)
            print(f"Created: {directory / name}")
    print("Validated: 20 PDFs, 200 pages, complete extractable body text.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    args = parser.parse_args()
    build_corpus(args.root, args.fixture)


if __name__ == "__main__":
    main()
