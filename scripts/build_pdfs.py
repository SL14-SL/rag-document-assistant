"""Build text-based PDFs from the English Markdown sources. Requires reportlab."""

import json
import re
from hashlib import sha256
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer

ROOT = Path(__file__).resolve().parents[1]


def build():
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    out = ROOT / "documents_pdf"
    out.mkdir(exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="DocTitle",
            fontName="Helvetica-Bold",
            fontSize=23,
            leading=28,
            textColor=colors.HexColor("#17334a"),
            spaceAfter=16,
        )
    )
    styles.add(
        ParagraphStyle(
            name="DocMeta",
            fontName="Helvetica",
            fontSize=9,
            leading=14,
            textColor=colors.HexColor("#506273"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="DocBody",
            fontName="Helvetica",
            fontSize=11,
            leading=17,
            spaceAfter=12,
            alignment=TA_LEFT,
        )
    )
    styles.add(
        ParagraphStyle(
            name="DocHeading",
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=18,
            spaceBefore=14,
            spaceAfter=7,
            textColor=colors.HexColor("#17334a"),
        )
    )
    for item in manifest:
        source = (ROOT / item["path"]).read_text(encoding="utf-8")
        preamble, *sections = re.split(r"^## ", source, flags=re.MULTILINE)
        lines = preamble.strip().splitlines()
        title = lines[0].removeprefix("# ").replace("\u2014", "-")
        metadata = [line for line in lines[1:] if ": " in line]
        story = [Paragraph(escape(title), styles["DocTitle"])]
        story.extend(Paragraph(escape(line), styles["DocMeta"]) for line in metadata)
        story += [
            Spacer(1, 9 * mm),
            Paragraph("Synthetic example document. Not a real company policy.", styles["DocMeta"]),
            Spacer(1, 3 * mm),
        ]
        for section in sections:
            heading, body = section.split("\n", 1)
            paragraphs = [
                p.strip().replace("\n", " ") for p in body.strip().split("\n\n") if p.strip()
            ]
            first = [
                Paragraph(escape(heading), styles["DocHeading"]),
                Paragraph(escape(paragraphs[0]), styles["DocBody"]),
            ]
            story.append(KeepTogether(first))
            story.extend(Paragraph(escape(p), styles["DocBody"]) for p in paragraphs[1:])
        target = out / (Path(item["path"]).stem + ".pdf")

        def footer(canvas, doc, doc_id=item["document_id"], status=item["status"]):
            width, height = doc.pagesize
            canvas.saveState()
            canvas.setStrokeColor(colors.HexColor("#dce4eb"))
            canvas.line(22 * mm, 21 * mm, width - 22 * mm, 21 * mm)
            canvas.setFont("Helvetica", 8)
            canvas.setFillColor(colors.HexColor("#506273"))
            canvas.drawString(
                22 * mm, height - 15 * mm, "NORDLICHT SOFTWARE  /  INTERNAL KNOWLEDGE DEMO"
            )
            canvas.drawString(
                22 * mm, 15 * mm, f"{doc_id} | {status.upper()} | Synthetic portfolio data"
            )
            canvas.drawRightString(width - 22 * mm, 15 * mm, f"Page {doc.page}")
            canvas.restoreState()

        pdf = SimpleDocTemplate(
            str(target),
            pagesize=(210 * mm, 297 * mm),
            rightMargin=22 * mm,
            leftMargin=22 * mm,
            topMargin=28 * mm,
            bottomMargin=28 * mm,
            title=title,
            author="Nordlicht Software (fictional)",
            invariant=1,
        )
        pdf.build(story, onFirstPage=footer, onLaterPages=footer)
        item["pdf_path"] = str(target.relative_to(ROOT))
        item["pdf_sha256"] = sha256(target.read_bytes()).hexdigest()
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Built {len(manifest)} PDFs and updated their hashes in manifest.json.")


if __name__ == "__main__":
    build()
