"""Original synthetic documents created solely for this Multimodal RAG project."""
from io import BytesIO
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.pagesizes import A4
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf"
QA = ROOT / "tmp" / "pdfs"
STYLES = getSampleStyleSheet()
STYLES.add(ParagraphStyle("DocTitle", fontName="Helvetica-Bold", fontSize=22, leading=27, spaceAfter=12))
STYLES.add(ParagraphStyle("IntroText", fontName="Helvetica", fontSize=10.5, leading=15, spaceAfter=10))
STYLES.add(ParagraphStyle("SmallNote", fontSize=8, leading=11, textColor=colors.HexColor("#475569"), spaceAfter=10))

DOCS = [
    {
        "file": "01_aster_product_catalog.pdf",
        "title": "Aster product catalog",
        "subtitle": "Reference catalog | Edition 1 | Synthetic demonstration",
        "intro": "Aster Devices is a fictional company used only in this Multimodal RAG project. "
                 "The Pulse line contains three portable audio products. All prices are in USD; "
                 "listed stock reflects a single demonstration snapshot. The Pulse Pro is the midrange model.",
        "table_title": "Product specifications",
        "table": [["Model", "Price (USD)", "Stock (units)", "Warranty (months)"],
                  ["Pulse Mini", "99", "120", "12"], ["Pulse Pro", "149", "80", "12"],
                  ["Pulse Max", "219", "40", "24"]],
        "chart_title": "Playback duration by model",
        "chart_labels": ["Pulse Mini", "Pulse Pro", "Pulse Max"],
        "chart_values": [10, 16, 24], "chart_unit": "Hours per charge",
        "note": "Playback duration is shown in the figure only, providing an image-based retrieval question. "
                "These are invented specifications, not claims about real products.",
    },
    {
        "file": "02_aster_sales_report.pdf",
        "title": "Aster annual sales report",
        "subtitle": "Reporting year 2025 | Synthetic demonstration",
        "intro": "This fictional annual sales report summarizes completed orders, recognized revenue and returns. "
                 "Revenue excludes tax and shipping. The regional figure describes the distribution of annual "
                 "orders. The report is created exclusively for testing this project's document retrieval.",
        "table_title": "Quarterly sales performance",
        "table": [["Quarter", "Orders", "Revenue (USD)", "Returned orders"],
                  ["Q1", "120", "18,000", "6"], ["Q2", "150", "24,000", "5"],
                  ["Q3", "180", "30,000", "4"], ["Q4", "210", "36,000", "3"],
                  ["Annual total", "660", "108,000", "18"]],
        "chart_title": "Share of annual orders by region",
        "chart_labels": ["North", "South", "West", "East"],
        "chart_values": [42, 28, 18, 12], "chart_unit": "Share of orders (%)",
        "note": "Quarter totals and regional shares are separate views of the same fictional business. "
                "The figure's percentages sum to 100. No real customer or transaction data is included.",
    },
    {
        "file": "03_aster_support_handbook.pdf",
        "title": "Aster support handbook",
        "subtitle": "Customer service reference | Edition 1 | Synthetic demonstration",
        "intro": "Customers may request a return within 30 calendar days of delivery. Approved refunds are "
                 "processed within 5 business days. Standard support is available Monday to Friday, "
                 "09:00-18:00 UTC. Warranty periods are listed in the Aster product catalog.",
        "table_title": "Service response targets",
        "table": [["Priority", "Example issue", "First response", "Resolution target"],
                  ["P1", "Service unavailable", "1 business hour", "8 business hours"],
                  ["P2", "Feature impaired", "4 business hours", "24 business hours"],
                  ["P3", "General question", "24 business hours", "72 business hours"]],
        "chart_title": "Support tickets by category",
        "chart_labels": ["Setup", "Delivery", "Billing", "Other"],
        "chart_values": [42, 30, 18, 10], "chart_unit": "Tickets in sample",
        "note": "Targets are fictional business-hour objectives, not a contractual SLA. The 100-ticket sample "
                "is provided to test chart questions and comparisons across documents.",
    },
]


def chart(doc):
    fig, ax = plt.subplots(figsize=(7.0, 2.6), dpi=160)
    bars = ax.bar(doc["chart_labels"], doc["chart_values"], color=["#2563eb", "#0f766e", "#7c3aed", "#64748b"][:len(doc["chart_values"])], width=0.6)
    ax.set_ylabel(doc["chart_unit"], fontsize=10)
    ax.set_ylim(0, max(doc["chart_values"]) * 1.28)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.15)
    ax.set_axisbelow(True)
    ax.bar_label(bars, padding=4, fontsize=11)
    fig.tight_layout()
    buffer = BytesIO()
    fig.savefig(buffer, format="png", facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    return buffer


def footer(canvas, document):
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#64748b"))
    canvas.drawString(42, 25, "MULTIMODAL RAG | Original synthetic corpus | No real-world claims")
    canvas.drawRightString(A4[0]-42, 25, str(document.page))


def build():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    QA.mkdir(parents=True, exist_ok=True)
    for doc in DOCS:
        path = OUTPUT / doc["file"]
        story = [Paragraph(doc["title"], STYLES["DocTitle"]),
                 Paragraph(doc["subtitle"], STYLES["SmallNote"]),
                 Paragraph(doc["intro"], STYLES["IntroText"]),
                 Spacer(1, 8), Paragraph(doc["table_title"], STYLES["Heading2"])]
        table = Table(doc["table"], colWidths=[100, 135, 130, 146], hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#e2e8f0")),
            ("TEXTCOLOR", (0,0), (-1,-1), colors.HexColor("#0f172a")),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,-1), 9),
            ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
            ("TOPPADDING", (0,0), (-1,-1), 9),
            ("BOTTOMPADDING", (0,0), (-1,-1), 9),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ]))
        story += [table, Spacer(1, 12), Paragraph(doc["chart_title"], STYLES["Heading2"]),
                  Image(chart(doc), width=490, height=182), Spacer(1, 8),
                  Paragraph(doc["note"], STYLES["SmallNote"])]
        pdf = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=42, rightMargin=42,
                                topMargin=40, bottomMargin=40, title=doc["title"], author="Multimodal RAG Demo")
        pdf.build(story, onFirstPage=footer, onLaterPages=footer)
        with pymupdf.open(path) as rendered:
            assert len(rendered) == 1
            assert rendered[0].get_images()
            rendered[0].get_pixmap(matrix=pymupdf.Matrix(1.4, 1.4)).save(QA / f"{path.stem}.png")
        print(path.name, flush=True)


if __name__ == "__main__":
    build()
