"""
src/report_builder.py

Scans output/report_sections/*.md in alphabetical filename order and
compiles them into output/progress_report.pdf.

Section file format (plain markdown subset):
  # Text          → H1 heading
  ## Text         → H2 heading
  ### Text        → H3 heading
  ![alt](path)    → embed image from path (skipped if file missing)
  [TABLE](path)   → embed CSV file as table (skipped if file missing)
  blank line      → vertical spacer
  other text      → body paragraph
"""

import os
import re
import csv

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, PageBreak, HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors

SECTIONS_DIR = "output/report_sections"
OUTPUT_PDF   = "output/progress_report.pdf"

TITLE_TEXT   = "Word2Vec Semantic Analysis &\nNetwork Textual Analysis"
SUBTITLE     = "Final Assignment – Computational Content Analysis"
COMPANY      = "Company: Cisco Systems, Inc. (CIK 858877)"
COURSE       = "Instructor: Dr Raja Shahzad Shaikh"


# ── Styles ────────────────────────────────────────────────────────────────────

def _build_styles():
    base = getSampleStyleSheet()
    styles = {
        "h1":    ParagraphStyle("H1",    parent=base["Heading1"],
                                fontSize=14, spaceAfter=8, spaceBefore=14,
                                textColor=colors.HexColor("#1a3a5c")),
        "h2":    ParagraphStyle("H2",    parent=base["Heading2"],
                                fontSize=11, spaceAfter=6, spaceBefore=10,
                                textColor=colors.HexColor("#2e6da4")),
        "h3":    ParagraphStyle("H3",    parent=base["Heading3"],
                                fontSize=10, spaceAfter=4, spaceBefore=8),
        "body":  ParagraphStyle("Body",  parent=base["Normal"],
                                fontSize=9,  spaceAfter=4, leading=13),
        "cover_title": ParagraphStyle("CoverTitle", parent=base["Title"],
                                fontSize=22, spaceAfter=12, leading=28,
                                textColor=colors.HexColor("#1a3a5c")),
        "cover_sub":   ParagraphStyle("CoverSub", parent=base["Normal"],
                                fontSize=12, spaceAfter=8,
                                textColor=colors.HexColor("#555555")),
    }
    return styles


# ── Cover page ────────────────────────────────────────────────────────────────

def _cover_page(styles):
    elements = []
    elements.append(Spacer(1, 4 * cm))
    elements.append(Paragraph(TITLE_TEXT.replace("\n", "<br/>"), styles["cover_title"]))
    elements.append(Spacer(1, 0.5 * cm))
    elements.append(HRFlowable(width="80%", thickness=1, color=colors.HexColor("#2e6da4")))
    elements.append(Spacer(1, 0.5 * cm))
    elements.append(Paragraph(SUBTITLE, styles["cover_sub"]))
    elements.append(Spacer(1, 0.3 * cm))
    elements.append(Paragraph(COMPANY,  styles["cover_sub"]))
    elements.append(Paragraph(COURSE,   styles["cover_sub"]))
    elements.append(PageBreak())
    return elements


# ── CSV → ReportLab Table ─────────────────────────────────────────────────────

def _csv_table(path):
    if not os.path.exists(path):
        return None
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            rows.append(row)
    if not rows:
        return None

    # Truncate very wide tables to first 8 columns, very long tables to 50 rows
    rows = [r[:8] for r in rows[:51]]

    col_count = max(len(r) for r in rows)
    page_w    = A4[0] - 4 * cm
    col_w     = page_w / col_count

    tbl = Table(rows, colWidths=[col_w] * col_count, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0),  colors.HexColor("#2e6da4")),
        ("TEXTCOLOR",   (0, 0), (-1, 0),  colors.white),
        ("FONTSIZE",    (0, 0), (-1, -1), 7),
        ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#eef3f9")]),
        ("GRID",        (0, 0), (-1, -1), 0.3, colors.HexColor("#aaaaaa")),
        ("TOPPADDING",  (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("WORDWRAP",    (0, 0), (-1, -1), "CJK"),
    ]))
    return tbl


# ── Image embed ───────────────────────────────────────────────────────────────

def _embed_image(path, max_w=15 * cm, max_h=10 * cm):
    if not os.path.exists(path):
        return None
    try:
        img = RLImage(path)
        ratio = img.imageWidth / img.imageHeight
        w = min(max_w, img.imageWidth)
        h = w / ratio
        if h > max_h:
            h = max_h
            w = h * ratio
        img.drawWidth  = w
        img.drawHeight = h
        return img
    except Exception:
        return None


# ── Section file parser ───────────────────────────────────────────────────────

_RE_H1    = re.compile(r"^#\s+(.+)$")
_RE_H2    = re.compile(r"^##\s+(.+)$")
_RE_H3    = re.compile(r"^###\s+(.+)$")
_RE_IMG   = re.compile(r"^!\[.*?\]\((.+?)\)$")
_RE_TABLE = re.compile(r"^\[TABLE\]\((.+?)\)$", re.IGNORECASE)


def _parse_section(path, styles):
    elements = []
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.rstrip("\n")

        m = _RE_H1.match(line)
        if m:
            elements.append(Paragraph(m.group(1), styles["h1"]))
            elements.append(HRFlowable(width="100%", thickness=0.5,
                                        color=colors.HexColor("#2e6da4")))
            continue

        m = _RE_H2.match(line)
        if m:
            elements.append(Paragraph(m.group(1), styles["h2"]))
            continue

        m = _RE_H3.match(line)
        if m:
            elements.append(Paragraph(m.group(1), styles["h3"]))
            continue

        m = _RE_IMG.match(line)
        if m:
            img = _embed_image(m.group(1))
            if img:
                elements.append(Spacer(1, 0.2 * cm))
                elements.append(img)
                elements.append(Spacer(1, 0.2 * cm))
            else:
                elements.append(Paragraph(f"[Image not found: {m.group(1)}]",
                                           styles["body"]))
            continue

        m = _RE_TABLE.match(line)
        if m:
            tbl = _csv_table(m.group(1))
            if tbl:
                elements.append(Spacer(1, 0.2 * cm))
                elements.append(tbl)
                elements.append(Spacer(1, 0.2 * cm))
            else:
                elements.append(Paragraph(f"[Table not found: {m.group(1)}]",
                                           styles["body"]))
            continue

        if line.strip() == "":
            elements.append(Spacer(1, 0.25 * cm))
            continue

        # Escape XML special chars for ReportLab
        safe = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        elements.append(Paragraph(safe, styles["body"]))

    return elements


# ── Main rebuild ──────────────────────────────────────────────────────────────

def rebuild():
    os.makedirs(SECTIONS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_PDF), exist_ok=True)

    styles   = _build_styles()
    elements = _cover_page(styles)

    section_files = sorted(
        f for f in os.listdir(SECTIONS_DIR)
        if f.endswith(".md") or f.endswith(".txt")
    )

    if not section_files:
        elements.append(Paragraph("No sections generated yet.", styles["body"]))
    else:
        for fname in section_files:
            path = os.path.join(SECTIONS_DIR, fname)
            elements.extend(_parse_section(path, styles))
            elements.append(Spacer(1, 0.5 * cm))

    doc = SimpleDocTemplate(
        OUTPUT_PDF,
        pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm,  bottomMargin=2 * cm,
        title=TITLE_TEXT,
    )
    doc.build(elements)
    print(f"PDF rebuilt -> {OUTPUT_PDF}  ({len(section_files)} section(s))")


def save_section(filename: str, content: str):
    """Write a markdown section file to output/report_sections/."""
    os.makedirs(SECTIONS_DIR, exist_ok=True)
    path = os.path.join(SECTIONS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Section saved -> {path}")


if __name__ == "__main__":
    rebuild()
