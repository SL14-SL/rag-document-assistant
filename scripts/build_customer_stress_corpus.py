"""Generate a dense synthetic PDF corpus for technical load testing."""

import argparse
from pathlib import Path

from pypdf import PdfReader
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas

MIN_WORDS_PER_PAGE = 330
FONT_NAME = "Helvetica"
FONT_SIZE = 10
LINE_HEIGHT = 13
MARGIN = 45

PARAGRAPHS = [
    (
        "The responsible department reviews operational records "
        "for reference {reference}. Employees verify that the "
        "required information is complete before submitting "
        "the records to their supervisor. Missing entries are "
        "documented separately and assigned to the appropriate "
        "process owner for clarification."
    ),
    (
        "During the scheduled review of process {reference}, "
        "the assigned coordinator examines available documentation, "
        "checks the recorded observations and identifies unresolved "
        "issues. Relevant findings are communicated to the "
        "department responsible for subsequent corrective activities."
    ),
    (
        "Operational case {reference} includes a documented "
        "assessment of completed activities and outstanding tasks. "
        "The assessment records the current processing status, "
        "available supporting evidence and any dependencies that "
        "may affect completion of the remaining work."
    ),
    (
        "The department maintains supporting records for "
        "reference {reference} in its designated documentation "
        "system. Records contain descriptive information about "
        "the activity, the responsible employees and the outcome "
        "of the applicable internal verification procedures."
    ),
    (
        "For activity {reference}, the assigned process owner "
        "coordinates communication between participating teams. "
        "Relevant observations are collected, reviewed and "
        "incorporated into the operational summary. Unresolved "
        "questions remain visible until the responsible team "
        "provides sufficient clarification."
    ),
    (
        "The internal review associated with reference {reference} "
        "considers the available operational context and the "
        "completeness of the supporting records. The reviewer "
        "documents any limitations affecting the assessment "
        "and identifies additional information required before "
        "the review can be concluded."
    ),
    (
        "The processing history for case {reference} provides "
        "a descriptive overview of the activities performed. "
        "Employees record relevant changes, preserve supporting "
        "documentation and ensure that subsequent reviewers can "
        "understand the context of previously recorded decisions."
    ),
    (
        "The operational summary for reference {reference} "
        "contains observations from participating departments, "
        "descriptions of completed activities and information "
        "about remaining dependencies. The summary supports "
        "internal coordination and provides context for "
        "subsequent process reviews."
    ),
    (
        "The documentation package for reference {reference} "
        "is checked against the applicable internal requirements. "
        "The assigned reviewer identifies missing supporting records "
        "and confirms that relevant observations are traceable. "
        "Any unresolved discrepancies are recorded for subsequent "
        "investigation by the responsible department."
    ),
    (
        "The coordination meeting for reference {reference} "
        "provides an opportunity to review outstanding activities. "
        "Participating employees report progress, explain operational "
        "dependencies and identify information needed from other teams. "
        "The resulting action items are recorded in the departmental "
        "tracking system."
    ),
    (
        "The assessment of reference {reference} includes a review "
        "of the available process documentation and recorded decisions. "
        "The responsible employee identifies inconsistencies between "
        "the supporting records and the reported outcome. Additional "
        "clarification is requested whenever the available evidence "
        "is incomplete."
    ),
    (
        "The follow-up review for reference {reference} examines "
        "whether previously documented activities have been completed. "
        "The coordinator compares the current status with earlier "
        "observations and records relevant developments. Outstanding "
        "items remain assigned to the responsible process owner "
        "until their resolution has been documented."
    ),
]


def expand_page(
    original_text: str,
    *,
    document_number: int,
    page_number: int,
) -> str:
    """Extend source text without repeating templates on a page."""
    paragraphs = [
        " ".join(original_text.split())
    ]

    paragraph_number = 0
    used_templates = set()

    while len(" ".join(paragraphs).split()) < MIN_WORDS_PER_PAGE:
        if paragraph_number >= len(PARAGRAPHS):
            raise ValueError(
                "Not enough unique paragraph templates "
                f"for document {document_number}, page {page_number}"
            )

        reference = (
            f"WB-{document_number:02d}-"
            f"{page_number:02d}-"
            f"{paragraph_number + 1:02d}"
        )

        template_index = (
            document_number
            + page_number
            + paragraph_number
        ) % len(PARAGRAPHS)

        if template_index in used_templates:
            raise ValueError(
                "Repeated paragraph template "
                f"in document {document_number}, page {page_number}"
            )

        used_templates.add(template_index)

        paragraphs.append(
            PARAGRAPHS[template_index].format(
                reference=reference
            )
        )

        paragraph_number += 1

    return "\n\n".join(paragraphs)


def write_page(
    pdf: canvas.Canvas,
    *,
    title: str,
    page_number: int,
    body: str,
) -> None:
    """Render one page and reject layout overflow."""
    width, height = A4

    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(
        MARGIN,
        height - MARGIN,
        title,
    )

    pdf.setFont(FONT_NAME, 9)
    pdf.drawString(
        MARGIN,
        height - MARGIN - 20,
        f"Synthetic technical load test | Page {page_number}",
    )

    text = pdf.beginText(
        MARGIN,
        height - MARGIN - 45,
    )

    text.setFont(FONT_NAME, FONT_SIZE)
    text.setLeading(LINE_HEIGHT)

    for paragraph in body.split("\n\n"):
        lines = simpleSplit(
            paragraph,
            FONT_NAME,
            FONT_SIZE,
            width - 2 * MARGIN,
        )

        for line in lines:
            if text.getY() < MARGIN:
                raise ValueError(
                    f"Page overflow: {title}, page {page_number}"
                )

            text.textLine(line)

        text.textLine("")

    pdf.drawText(text)
    pdf.showPage()


def build_stress_corpus(
    source_root: Path,
    output_root: Path,
) -> None:
    """Generate 20 dense PDFs with 200 pages in total."""
    source_directory = source_root / "pdfs"
    output_directory = output_root / "pdfs"

    sources = sorted(source_directory.glob("*.pdf"))

    if len(sources) != 20:
        raise ValueError("Expected exactly 20 source PDFs")

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    if any(output_directory.iterdir()):
        raise ValueError(
            "Output PDF directory must be empty"
        )

    total_pages = 0
    total_words = 0
    minimum_words = float("inf")
    maximum_words = 0

    for document_number, source in enumerate(
        sources,
        start=1,
    ):
        reader = PdfReader(source)

        if len(reader.pages) != 10:
            raise ValueError(
                f"Expected ten pages: {source}"
            )

        target = output_directory / source.name

        pdf = canvas.Canvas(
            str(target),
            pagesize=A4,
            invariant=1,
        )

        title = source.stem.replace("_", " ").title()

        for page_number, page in enumerate(
            reader.pages,
            start=1,
        ):
            original_text = page.extract_text() or ""

            if not original_text.strip():
                raise ValueError(
                    f"Empty source page: {source}, {page_number}"
                )

            body = expand_page(
                original_text,
                document_number=document_number,
                page_number=page_number,
            )

            write_page(
                pdf,
                title=title,
                page_number=page_number,
                body=body,
            )

        pdf.save()

        generated = PdfReader(target)

        if len(generated.pages) != 10:
            raise ValueError(
                f"Unexpected generated page count: {target}"
            )

        document_words = 0

        for page in generated.pages:
            words = len(
                (page.extract_text() or "").split()
            )

            if words < MIN_WORDS_PER_PAGE:
                raise ValueError(
                    f"Generated page contains too little text: {target}"
                )

            minimum_words = min(minimum_words, words)
            maximum_words = max(maximum_words, words)

            document_words += words
            total_words += words
            total_pages += 1

        print(
            f"Created: {target.name} "
            f"| Pages: 10 "
            f"| Words: {document_words}"
        )

    if total_pages != 200:
        raise ValueError("Expected exactly 200 generated pages")

    print()
    print("Documents:", len(sources))
    print("Pages:", total_pages)
    print("Total words:", total_words)
    print("Minimum words per page:", minimum_words)
    print("Maximum words per page:", maximum_words)
    print(
        "Average words per page:",
        round(total_words / total_pages, 1),
    )

    print("\nStress corpus generation passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--root",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    build_stress_corpus(
        args.source_root,
        args.root,
    )


if __name__ == "__main__":
    main()