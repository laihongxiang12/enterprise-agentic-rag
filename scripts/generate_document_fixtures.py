"""Regenerate the original fictional PDF fixtures used by loader tests."""

from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen.canvas import Canvas

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "documents"


def _create_pdf(path: Path, *, title: str, lines: list[str], font: str = "Helvetica") -> None:
    canvas = Canvas(str(path))
    canvas.setTitle(title)
    canvas.setFont(font, 15)
    y = 800
    for line in lines:
        canvas.drawString(60, y, line)
        y -= 28
    canvas.showPage()
    canvas.save()


def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    _create_pdf(
        FIXTURE_DIR / "it_oncall_zh.pdf",
        title="Northstar IT 值班制度",
        lines=["Northstar IT 值班制度", "一级告警须在十分钟内确认。", "禁止在工单中记录访问密钥。"],
        font="STSong-Light",
    )
    _create_pdf(
        FIXTURE_DIR / "expense_policy_en.pdf",
        title="Northstar Expense Policy",
        lines=["Northstar Expense Policy", "Receipts are required for expenses over USD 25."],
    )
    source = PdfReader(FIXTURE_DIR / "expense_policy_en.pdf")
    writer = PdfWriter()
    for page in source.pages:
        writer.add_page(page)
    writer.encrypt("fixture-password")
    with (FIXTURE_DIR / "encrypted_policy.pdf").open("wb") as stream:
        writer.write(stream)


if __name__ == "__main__":
    main()
