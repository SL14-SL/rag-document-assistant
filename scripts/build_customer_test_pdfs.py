"""Create a small synthetic PDF corpus for customer deployment tests."""

import argparse
from pathlib import Path
from textwrap import wrap

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

DOCUMENTS = {
    "shipping_policy.pdf": [
        (
            "Shipping conditions",
            "Standard shipping costs EUR 8.50 per order. "
            "Orders exceeding EUR 150 qualify for free standard shipping. "
            "Express shipping costs EUR 24.90 regardless of order value.",
        ),
        (
            "Delivery times",
            "Standard domestic deliveries take three to five business days. "
            "Express orders placed before 12:00 are delivered "
            "on the next business day.",
        ),
    ],
    "returns_policy.pdf": [
        (
            "Return conditions",
            "Customers may return unused products within 45 calendar days "
            "of delivery. The original invoice must be included. "
            "Customized products cannot be returned.",
        ),
        (
            "Refund processing",
            "Approved refunds are processed within seven business days "
            "after the returned goods have been inspected. "
            "Refunds are issued using the original payment method.",
        ),
    ],
    "warehouse_safety.pdf": [
        (
            "Warehouse safety",
            "Safety shoes and high-visibility clothing are mandatory "
            "in all warehouse areas. Forklifts may only be operated "
            "by authorized employees with valid certification. "
            "Pedestrians must use designated walkways. "
            "Emergency exits must remain unobstructed.",
        ),
        (
            "Incident reporting",
            "Workplace accidents must be reported to the warehouse "
            "supervisor within two hours. A written incident report "
            "must be submitted within 24 hours.",
        ),
    ],
}


def build_test_pdfs(root: Path) -> None:
    """Create synthetic documents without using customer information."""
    directory = root / "pdfs"
    directory.mkdir(parents=True, exist_ok=True)

    for filename in DOCUMENTS:
        path = directory / filename

        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing PDF: {path}"
            )

    for filename, pages in DOCUMENTS.items():
        path = directory / filename

        pdf = canvas.Canvas(
            str(path),
            pagesize=A4,
            invariant=1,
        )

        for title, body in pages:
            pdf.setFont("Helvetica-Bold", 16)
            pdf.drawString(50, 790, title)

            text = pdf.beginText(50, 750)
            text.setFont("Helvetica", 11)
            text.setLeading(16)

            for line in wrap(body, width=85):
                text.textLine(line)

            pdf.drawText(text)
            pdf.showPage()

        pdf.save()

        print(f"Created: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--root",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    build_test_pdfs(args.root)


if __name__ == "__main__":
    main()