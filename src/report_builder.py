"""
src/report_builder.py

Builds output/progress_report.pdf with a fixed academic report structure:
  1. Cover Page
  2. Data Description
  3. Part 1 Results   (section files: 01_*, 02_*)
  4. Part 2 Results   (section files: 03_*, 04_*)
  5. Part 3 Results   (section files: 05_*, 06_*)
  6. Conclusion

Section files in output/report_sections/ feed question results into the
right part automatically. Missing parts show a placeholder sentence.

Section file markdown syntax:
  # Text        -> question heading (H2 inside the part)
  ## Text       -> subheading (H3)
  ### Text      -> sub-subheading (H4)
  ![alt](path)  -> embedded image with caption
  [TABLE](path) -> CSV rendered as table
  blank line    -> spacer
  other text    -> body paragraph
"""

import os
import re
import csv
import shutil
from datetime import date

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, PageBreak, HRFlowable,
)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

# ── Paths ─────────────────────────────────────────────────────────────────────
SECTIONS_DIR = "output/report_sections"
OUTPUT_PDF   = "output/progress_report.pdf"

# ── Fixed report content ──────────────────────────────────────────────────────
STUDENT_INFO = {
    "title":      "Word2Vec Semantic Analysis and\nNetwork Textual Analysis",
    "course":     "Computational Content Analysis",
    "company":    "Cisco Systems, Inc.",
    "cik":        "CIK: 858877",
    "student":    "Shayaan Arshad",
    "instructor": "Dr Raja Shahzad Shaikh",
}

DATA_DESCRIPTION = (
    "This analysis is based on five annual 10-K filings submitted by Cisco Systems, Inc. "
    "(CIK: 858877) to the U.S. Securities and Exchange Commission. The filings cover "
    "fiscal years 2021 through 2025, each corresponding to Cisco's fiscal year ending in "
    "late July of the respective year. The five documents used are: 2021_10K.txt, "
    "2022_10K.txt, 2023_10K.txt, 2024_10K.txt, and 2025_10K.txt. Each file was downloaded "
    "directly from the EDGAR database and contains the complete text of the primary 10-K "
    "document after HTML stripping. Each file is treated as a single document representing "
    "one fiscal year, producing a five-document corpus suitable for year-level comparison "
    "of semantic patterns in management disclosures. All subsequent analyses, including "
    "Word2Vec modelling, thematic clustering, and co-occurrence network construction, are "
    "performed on this corpus."
    "\n\n"
    "All analyses are implemented in Python 3.12. The following libraries are used as "
    "equivalents to the R packages described in course materials: gensim (Word2Vec model "
    "training and nearest-neighbour retrieval, equivalent to word2vec/text2vec), "
    "scikit-learn (k-means clustering and PCA dimensionality reduction, equivalent to "
    "stats::kmeans and FactoMineR), NetworkX (co-occurrence graph construction, "
    "centrality measures, and connected component analysis, equivalent to igraph), "
    "python-louvain (Louvain community detection with modularity maximisation, equivalent "
    "to igraph::cluster_louvain), nltk (lemmatisation using WordNetLemmatizer, equivalent "
    "to textstem), and reportlab (PDF report generation). All random seeds are fixed at "
    "42 for reproducibility."
)

PART_PLACEHOLDERS = {
    "Part 2": (
        "Part 2 results will be added after semantic proximity and thematic "
        "clustering analysis."
    ),
    "Part 3": (
        "Part 3 results will be added after co-occurrence network, centrality, "
        "and community detection analysis."
    ),
}

CONCLUSION = (
    "All six questions of this assignment have been completed for Cisco Systems, Inc.'s "
    "five-year 10-K corpus spanning fiscal years 2021 to 2025. The corpus construction "
    "and preprocessing pipeline (Q1) reduced the raw vocabulary from 9,457 to 3,682 "
    "lemmatised terms. A Word2Vec skip-gram model (Q2) was trained on this corpus, "
    "yielding a 2,799-word vocabulary with 100-dimensional embeddings. Semantic proximity "
    "analysis (Q3) retrieved nearest-neighbour terms for five financial dimensions using "
    "seed terms selected for meaningful financial neighbourhood structure. Thematic "
    "clustering (Q4) applied k-means to the top 300 corpus-frequency words, producing "
    "five clusters reflecting derivatives, governance, accounting, revenue, and strategic "
    "vocabulary. Co-occurrence network construction (Q5) built an undirected weighted "
    "graph of 2,858 nodes and 56,753 edges, confirming a sparse, small-world structure "
    "with a global clustering coefficient of 0.185."
    "\n\n"
    "Centrality analysis and Louvain community detection (Q6) identified the top "
    "betweenness-central bridge terms in the co-occurrence network and detected 12 "
    "thematically coherent communities with a modularity score of 0.31. The community "
    "structure reflects Cisco's distinct financial disclosure domains including revenue "
    "reporting, legal obligations, asset valuation, governance, and risk factors. Taken "
    "together, the six analyses confirm that Cisco's 10-K filing language has coherent "
    "semantic and structural organisation at multiple levels of analysis, consistent with "
    "the structured and legally mandated nature of SEC annual report disclosures."
    "\n\n"
    "Based solely on the textual patterns identified across Q1 to Q6, the following "
    "near-term financial performance observations can be drawn for Cisco Systems. The "
    "Revenue and Geographic Segments community (Q6) and the strong semantic coherence "
    "of the 'revenue' neighbourhood (Q3) — featuring terms such as deferral, recognition, "
    "predictability, and sustainable — indicate that Cisco's management language "
    "consistently emphasises a stable, subscription-based recurring revenue model, "
    "suggesting continued near-term revenue predictability from its software and "
    "services portfolio. The dominance of 'product' as the highest-betweenness node "
    "(Q6) alongside Supply Chain and Procurement as a distinct community signals "
    "ongoing product hardware exposure, with component sourcing risk remaining a "
    "material disclosure theme. The Legal and Intellectual Property community's large "
    "size reflects continued legal burden from patent litigation and compliance "
    "obligations, which could exert cost pressure on operating margins. The 'uncertain' "
    "neighbourhood (Q3) — clustering geopolitical, deteriorate, challenging, and "
    "instability — alongside the Risk Factors and Business Conditions community (Q6) "
    "confirms that macroeconomic and geopolitical uncertainty are persistently "
    "highlighted downside risks. The presence of a large Products, Services and Market "
    "community centred on 'security' co-occurrences within the governance cluster "
    "points to cybersecurity as a growth-driving theme in Cisco's strategic narrative. "
    "These observations are derived from textual frequency and co-occurrence patterns "
    "only and do not constitute investment advice or a forward-looking financial forecast."
)

# Section file prefixes that belong to each part
PART_PREFIXES = {
    "Part 1": ["01_", "02_"],
    "Part 2": ["03_", "04_"],
    "Part 3": ["05_", "06_"],
}

# Part titles used as section headings
PART_TITLES = {
    "Part 1": "Part 1: Corpus Construction and Word2Vec Model",
    "Part 2": "Part 2: Semantic Proximity and Thematic Clustering",
    "Part 3": "Part 3: Co-occurrence Network and Structural Analysis",
}


# ── Styles ────────────────────────────────────────────────────────────────────

def _styles():
    base_font  = "Times-Roman"
    bold_font  = "Times-Bold"
    body_size  = 11
    return {
        "cover_title": ParagraphStyle(
            "CoverTitle", fontName=bold_font, fontSize=16,
            alignment=TA_CENTER, spaceAfter=16, leading=22,
        ),
        "cover_body": ParagraphStyle(
            "CoverBody", fontName=base_font, fontSize=12,
            alignment=TA_CENTER, spaceAfter=8, leading=16,
        ),
        "part_heading": ParagraphStyle(
            "PartHeading", fontName=bold_font, fontSize=14,
            spaceBefore=0, spaceAfter=8, leading=18,
        ),
        "h2": ParagraphStyle(
            "H2", fontName=bold_font, fontSize=12,
            spaceBefore=14, spaceAfter=6, leading=16,
        ),
        "h3": ParagraphStyle(
            "H3", fontName=bold_font, fontSize=11,
            spaceBefore=10, spaceAfter=4, leading=14,
        ),
        "h4": ParagraphStyle(
            "H4", fontName=bold_font, fontSize=body_size,
            spaceBefore=8, spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "Body", fontName=base_font, fontSize=body_size,
            spaceAfter=6, leading=17, alignment=TA_JUSTIFY,
        ),
        "caption": ParagraphStyle(
            "Caption", fontName=base_font, fontSize=9,
            alignment=TA_CENTER, spaceBefore=2, spaceAfter=8,
        ),
    }


# ── Page number footer ────────────────────────────────────────────────────────

def _footer(canvas, doc):
    canvas.saveState()
    if canvas.getPageNumber() > 1:
        canvas.setFont("Times-Roman", 9)
        canvas.drawCentredString(A4[0] / 2.0, 1.0 * cm,
                                 f"Page {canvas.getPageNumber()}")
    canvas.restoreState()


# ── Cover page ────────────────────────────────────────────────────────────────

def _cover(s):
    el = []
    el.append(Spacer(1, 5 * cm))
    el.append(Paragraph(
        STUDENT_INFO["title"].replace("\n", "<br/>"), s["cover_title"]
    ))
    el.append(Spacer(1, 1 * cm))
    el.append(HRFlowable(width="60%", thickness=0.5, color=colors.black,
                          hAlign="CENTER"))
    el.append(Spacer(1, 1 * cm))
    for label, key in [
        ("Course",      "course"),
        ("Company",     "company"),
        ("",            "cik"),
        ("Student",     "student"),
        ("Instructor",  "instructor"),
    ]:
        line = f"{label}: {STUDENT_INFO[key]}" if label else STUDENT_INFO[key]
        el.append(Paragraph(line, s["cover_body"]))
    el.append(Spacer(1, 0.5 * cm))
    el.append(Paragraph(str(date.today().strftime("%B %d, %Y")), s["cover_body"]))
    el.append(PageBreak())
    return el


# ── CSV -> Table ──────────────────────────────────────────────────────────────

def _csv_table(path):
    if not os.path.exists(path):
        return None
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            rows.append(row)
    if not rows:
        return None

    rows = [r[:8] for r in rows[:125]]
    col_count = max(len(r) for r in rows)
    page_w    = A4[0] - 4 * cm
    col_w     = page_w / col_count

    tbl = Table(rows, colWidths=[col_w] * col_count, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("FONTNAME",      (0, 0), (-1, 0),  "Times-Bold"),
        ("FONTNAME",      (0, 1), (-1, -1), "Times-Roman"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("TEXTCOLOR",     (0, 0), (-1, -1), colors.black),
        ("BACKGROUND",    (0, 0), (-1, -1), colors.white),
        ("GRID",          (0, 0), (-1, -1), 0.5, colors.black),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, colors.white]),
    ]))
    return tbl


# ── Image embed ───────────────────────────────────────────────────────────────

def _image(path, max_w=15.5 * cm, max_h=13 * cm):
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
        img.hAlign     = "CENTER"
        return img
    except Exception:
        return None


# ── Section file parser ───────────────────────────────────────────────────────

_RE_H1    = re.compile(r"^#\s+(.+)$")
_RE_H2    = re.compile(r"^##\s+(.+)$")
_RE_H3    = re.compile(r"^###\s+(.+)$")
_RE_IMG   = re.compile(r"^!\[.*?\]\((.+?)\)$")
_RE_TABLE = re.compile(r"^\[TABLE\]\((.+?)\)$", re.IGNORECASE)


def _parse_section(path, s):
    el = []
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    fig_count = [0]

    for line in lines:
        line = line.rstrip("\n")

        m = _RE_H1.match(line)
        if m:
            el.append(Paragraph(m.group(1), s["h2"]))
            continue

        m = _RE_H2.match(line)
        if m:
            el.append(Paragraph(m.group(1), s["h3"]))
            continue

        m = _RE_H3.match(line)
        if m:
            el.append(Paragraph(m.group(1), s["h4"]))
            continue

        m = _RE_IMG.match(line)
        if m:
            img = _image(m.group(1))
            if img:
                fig_count[0] += 1
                el.append(Spacer(1, 0.3 * cm))
                el.append(img)
                # Caption from alt text if present, else generic
                alt = re.match(r"^!\[(.+?)\]", line)
                caption_text = alt.group(1) if alt and alt.group(1) else f"Figure {fig_count[0]}"
                el.append(Paragraph(caption_text, s["caption"]))
            else:
                el.append(Paragraph(
                    f"[Image not available: {m.group(1)}]", s["body"]
                ))
            continue

        m = _RE_TABLE.match(line)
        if m:
            tbl = _csv_table(m.group(1))
            if tbl:
                el.append(Spacer(1, 0.2 * cm))
                el.append(tbl)
                el.append(Spacer(1, 0.3 * cm))
            else:
                el.append(Paragraph(
                    f"[Table not available: {m.group(1)}]", s["body"]
                ))
            continue

        if line.strip() == "":
            el.append(Spacer(1, 0.2 * cm))
            continue

        safe = (line.replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;"))
        el.append(Paragraph(safe, s["body"]))

    return el


# ── Main rebuild ──────────────────────────────────────────────────────────────

def rebuild():
    os.makedirs(SECTIONS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_PDF), exist_ok=True)

    s  = _styles()
    el = []

    # 1. Cover page
    el.extend(_cover(s))

    # 2. Data Description
    el.append(Paragraph("Data Description", s["part_heading"]))
    el.append(HRFlowable(width="100%", thickness=0.5, color=colors.black))
    el.append(Spacer(1, 0.3 * cm))
    for para in DATA_DESCRIPTION.split("\n\n"):
        if para.strip():
            el.append(Paragraph(para.strip(), s["body"]))
    el.append(PageBreak())

    # 3-5. Parts 1, 2, 3
    all_files = sorted(os.listdir(SECTIONS_DIR))

    for part_key, part_title in PART_TITLES.items():
        el.append(Paragraph(part_title, s["part_heading"]))
        el.append(HRFlowable(width="100%", thickness=0.5, color=colors.black))
        el.append(Spacer(1, 0.3 * cm))

        prefixes      = PART_PREFIXES[part_key]
        part_files    = [
            f for f in all_files
            if any(f.startswith(p) for p in prefixes)
            and (f.endswith(".md") or f.endswith(".txt"))
        ]

        if part_files:
            for fname in part_files:
                el.extend(_parse_section(
                    os.path.join(SECTIONS_DIR, fname), s
                ))
                el.append(Spacer(1, 0.4 * cm))
        else:
            el.append(Paragraph(PART_PLACEHOLDERS.get(part_key, ""), s["body"]))

        el.append(PageBreak())

    # 6. Conclusion
    el.append(Paragraph("Conclusion", s["part_heading"]))
    el.append(HRFlowable(width="100%", thickness=0.5, color=colors.black))
    el.append(Spacer(1, 0.3 * cm))
    for para in CONCLUSION.split("\n\n"):
        if para.strip():
            el.append(Paragraph(para.strip(), s["body"]))

    # Build PDF
    doc = SimpleDocTemplate(
        OUTPUT_PDF,
        pagesize=A4,
        leftMargin=2.5 * cm, rightMargin=2.5 * cm,
        topMargin=2.5 * cm,  bottomMargin=2.0 * cm,
        title=STUDENT_INFO["title"],
        author=STUDENT_INFO["student"],
    )
    doc.build(el, onFirstPage=_footer, onLaterPages=_footer)

    section_count = sum(
        1 for f in all_files
        if f.endswith(".md") or f.endswith(".txt")
    )
    print(f"PDF rebuilt -> {OUTPUT_PDF}  ({section_count} section file(s))")

    final_pdf = "output/final_report.pdf"
    shutil.copy2(OUTPUT_PDF, final_pdf)
    print(f"Final report copied -> {final_pdf}")


def save_section(filename: str, content: str):
    """Write a markdown section file to output/report_sections/."""
    os.makedirs(SECTIONS_DIR, exist_ok=True)
    path = os.path.join(SECTIONS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Section saved -> {path}")


if __name__ == "__main__":
    rebuild()
