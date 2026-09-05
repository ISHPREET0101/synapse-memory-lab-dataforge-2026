#!/usr/bin/env python3
"""Generate the two publication-ready Synapse Memory Lab PDFs.

Inputs are read from docs/ and the two stable outputs are written to output/pdf/.
The script intentionally keeps all typography and layout decisions local so the
artifacts can be reproduced without a browser or network connection.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
OUT = ROOT / "output" / "pdf"

NAVY = colors.HexColor("#10243E")
INK = colors.HexColor("#203047")
MUTED = colors.HexColor("#5F6F82")
TEAL = colors.HexColor("#0D8B8F")
CYAN = colors.HexColor("#52C5C7")
PALE = colors.HexColor("#EAF6F6")
PALE_BLUE = colors.HexColor("#EDF3FA")
PAPER = colors.HexColor("#FAFCFE")
LINE = colors.HexColor("#D7E1EA")
WHITE = colors.white


def register_fonts() -> tuple[str, str, str, str]:
    """Register an embedded, deterministic TrueType font family."""
    candidates = [
        Path(r"C:\Windows\Fonts"),
        Path("/usr/share/fonts/truetype/dejavu"),
    ]
    for folder in candidates:
        files = {
            "regular": folder / ("arial.ttf" if "Windows" in str(folder) else "DejaVuSans.ttf"),
            "bold": folder / ("arialbd.ttf" if "Windows" in str(folder) else "DejaVuSans-Bold.ttf"),
            "italic": folder / ("ariali.ttf" if "Windows" in str(folder) else "DejaVuSans-Oblique.ttf"),
            "bolditalic": folder / ("arialbi.ttf" if "Windows" in str(folder) else "DejaVuSans-BoldOblique.ttf"),
        }
        if all(path.exists() for path in files.values()):
            names = ("SynapseSans", "SynapseSans-Bold", "SynapseSans-Italic", "SynapseSans-BoldItalic")
            for name, path in zip(names, files.values()):
                pdfmetrics.registerFont(TTFont(name, str(path)))
            pdfmetrics.registerFontFamily(
                "SynapseSans",
                normal=names[0],
                bold=names[1],
                italic=names[2],
                boldItalic=names[3],
            )
            return names
    raise FileNotFoundError("No supported embedded TrueType font family was found")


REGULAR, BOLD, ITALIC, BOLDITALIC = register_fonts()


def clean_math(text: str) -> str:
    replacements = {
        "\\gamma": "gamma",
        "\\tau": "tau",
        "\\top": "T",
        "\\times": "x",
        "\\in": " in ",
        "\\mathbb{R}": "R",
        "\\sum": "SUM",
        "\\left": "",
        "\\right": "",
        "\\qquad": "    ",
        "qquad": "    ",
        "\\begin{aligned}": "",
        "\\end{aligned}": "",
        "\\&": "",
        "&=": "=",
        "\\": " ",
    }
    text = text.replace("\t", " ")
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = text.replace("{", "").replace("}", "")
    text = text.replace("^", "").replace("_", "")
    return re.sub(r"\s+", " ", text).strip()


def normalize(text: str) -> str:
    return (
        text.replace("\u2011", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2212", "-")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\t", " ")
    )


LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")


def rich(text: str) -> str:
    """Convert the small inline Markdown subset used by the sources."""
    text = normalize(text)
    tokens: list[str] = []

    def hold_link(match: re.Match[str]) -> str:
        label = normalize(match.group(1)).replace("*", "")
        url = match.group(2)
        tokens.append(
            f'<link href="{html.escape(url, quote=True)}" color="#087B80">'
            f'<u>{html.escape(label)}</u></link>'
        )
        return f"@@LINK{len(tokens)-1}@@"

    text = LINK_RE.sub(hold_link, text)
    text = html.escape(text, quote=False)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"`([^`]+)`", r"<font name='SynapseSans'>\1</font>", text)
    for idx, token in enumerate(tokens):
        text = text.replace(f"@@LINK{idx}@@", token)
    return text


def equation_markup(text: str) -> str:
    """Turn the source equations into compact, typeset mathematical notation."""
    formulas = {
        "yt=SUMtau<tgammat-1-tau(qtT ktau)vtau.":
            "y<sub>t</sub> = Σ<sub>τ&lt;t</sub> γ<super>t-1-τ</super> (q<sub>t</sub><super>T</super> k<sub>τ</sub>) v<sub>τ</sub>",
        "St=gamma St-1+kt vtT, yt=qtT St-1.":
            "S<sub>t</sub> = γ S<sub>t-1</sub> + k<sub>t</sub> v<sub>t</sub><super>T</super>    y<sub>t</sub> = q<sub>t</sub><super>T</super> S<sub>t-1</sub>",
        "yt=SUMtau<t(qtT ktau)vtau.":
            "y<sub>t</sub> = Σ<sub>τ&lt;t</sub> (q<sub>t</sub><super>T</super> k<sub>τ</sub>) v<sub>τ</sub>",
        "yt =SUMtau<tqtT(ktau vtauT) =qtT(SUMtau<tktau vtauT).":
            "y<sub>t</sub> = Σ<sub>τ&lt;t</sub> q<sub>t</sub><super>T</super>(k<sub>τ</sub>v<sub>τ</sub><super>T</super>) = q<sub>t</sub><super>T</super>(Σ<sub>τ&lt;t</sub> k<sub>τ</sub>v<sub>τ</sub><super>T</super>)",
        "St=St-1+kt vtT.":
            "S<sub>t</sub> = S<sub>t-1</sub> + k<sub>t</sub>v<sub>t</sub><super>T</super>",
        "St=gamma St-1+kt vtT, 0 leqgamma leq1.":
            "S<sub>t</sub> = γ S<sub>t-1</sub> + k<sub>t</sub>v<sub>t</sub><super>T</super>    0 ≤ γ ≤ 1",
        "St-1=SUMtau<tgammat-1-tauktau vtauT.":
            "S<sub>t-1</sub> = Σ<sub>τ&lt;t</sub> γ<super>t-1-τ</super> k<sub>τ</sub>v<sub>τ</sub><super>T</super>",
        "Sijarrowgamma Sij+ki vj.":
            "S<sub>ij</sub> ← γ S<sub>ij</sub> + k<sub>i</sub>v<sub>j</sub>",
    }
    return formulas.get(text, rich(text))


def parse_blocks(path: Path) -> list[tuple[str, object]]:
    """Parse headings, prose, bullets, equations, and Markdown tables."""
    lines = normalize(path.read_text(encoding="utf-8")).splitlines()
    blocks: list[tuple[str, object]] = []
    i = 0
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            blocks.append(("p", " ".join(x.strip() for x in paragraph)))
            paragraph.clear()

    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()
        if not stripped:
            flush()
            i += 1
            continue
        if stripped.startswith("#"):
            flush()
            level = len(stripped) - len(stripped.lstrip("#"))
            blocks.append((f"h{level}", stripped[level:].strip()))
            i += 1
            continue
        if stripped.startswith("\\["):
            flush()
            formula: list[str] = []
            remainder = stripped[2:].strip()
            if remainder:
                formula.append(remainder)
            i += 1
            while i < len(lines) and not lines[i].strip().endswith("\\]"):
                formula.append(lines[i].strip())
                i += 1
            if i < len(lines):
                tail = lines[i].strip()[:-2].strip()
                if tail:
                    formula.append(tail)
                i += 1
            blocks.append(("eq", clean_math(" ".join(formula))))
            continue
        if stripped.startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|?\s*:?-+", lines[i + 1]):
            flush()
            rows: list[list[str]] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [cell.strip() for cell in lines[i].strip().strip("|").split("|")]
                rows.append(cells)
                i += 1
            if len(rows) >= 2:
                rows.pop(1)
            blocks.append(("table", rows))
            continue
        if re.match(r"^[-*]\s+", stripped):
            flush()
            blocks.append(("bullet", re.sub(r"^[-*]\s+", "", stripped)))
            i += 1
            continue
        if re.match(r"^\d+\.\s+", stripped):
            flush()
            number, body = stripped.split(".", 1)
            blocks.append(("number", (number, body.strip())))
            i += 1
            continue
        paragraph.append(stripped)
        i += 1
    flush()
    return blocks


class NumberedDocTemplate(BaseDocTemplate):
    pass


def footer(canvas, doc, label: str, compact: bool = False) -> None:
    width, _ = A4
    canvas.saveState()
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.5)
    canvas.line(doc.leftMargin, 14.8 * mm, width - doc.rightMargin, 14.8 * mm)
    canvas.setFont(REGULAR, 6.7 if compact else 7.2)
    canvas.setFillColor(MUTED)
    canvas.drawString(doc.leftMargin, 9.7 * mm, label)
    canvas.setFont(BOLD, 6.7 if compact else 7.2)
    canvas.setFillColor(TEAL)
    page = str(canvas.getPageNumber()).zfill(2) if compact else str(canvas.getPageNumber())
    canvas.drawRightString(width - doc.rightMargin, 9.7 * mm, page)
    canvas.restoreState()


def summary_page(canvas, doc) -> None:
    width, height = A4
    canvas.saveState()
    canvas.setFillColor(PAPER)
    canvas.rect(0, 0, width, height, fill=1, stroke=0)
    canvas.setFillColor(NAVY)
    canvas.rect(0, height - 8 * mm, width, 8 * mm, fill=1, stroke=0)
    canvas.setFillColor(CYAN)
    canvas.rect(0, height - 8 * mm, 49 * mm, 8 * mm, fill=1, stroke=0)
    canvas.restoreState()
    footer(canvas, doc, "SYNAPSE MEMORY LAB  /  CONCEPT SUMMARY", compact=True)


def blog_page(canvas, doc) -> None:
    width, height = A4
    canvas.saveState()
    canvas.setFillColor(WHITE)
    canvas.rect(0, 0, width, height, fill=1, stroke=0)
    canvas.setFillColor(NAVY)
    canvas.rect(0, height - 5 * mm, width, 5 * mm, fill=1, stroke=0)
    if canvas.getPageNumber() > 1:
        canvas.setFont(BOLD, 7.1)
        canvas.setFillColor(MUTED)
        canvas.drawString(doc.leftMargin, height - 13 * mm, "SYNAPSE MEMORY LAB")
        canvas.setFont(REGULAR, 7.1)
        canvas.drawRightString(width - doc.rightMargin, height - 13 * mm, "ATTENTION AS MEMORY")
    canvas.restoreState()
    footer(canvas, doc, "RESEARCH ARTICLE  /  SEPTEMBER 2026")


def summary_styles():
    base = getSampleStyleSheet()
    return {
        "kicker": ParagraphStyle("sum-kicker", parent=base["Normal"], fontName=BOLD, fontSize=6.5, leading=7.7, textColor=TEAL, spaceAfter=2.2 * mm, tracking=0.7),
        "title": ParagraphStyle("sum-title", parent=base["Title"], fontName=BOLD, fontSize=23, leading=24.5, textColor=NAVY, alignment=TA_LEFT, spaceAfter=1.4 * mm),
        "deck": ParagraphStyle("sum-deck", parent=base["Normal"], fontName=REGULAR, fontSize=8.7, leading=11, textColor=MUTED, spaceAfter=3.2 * mm),
        "h2": ParagraphStyle("sum-h2", parent=base["Heading2"], fontName=BOLD, fontSize=9.5, leading=11.3, textColor=NAVY, spaceBefore=2.3 * mm, spaceAfter=1.2 * mm, keepWithNext=True),
        "p": ParagraphStyle("sum-p", parent=base["BodyText"], fontName=REGULAR, fontSize=7.45, leading=9.2, textColor=INK, spaceAfter=1.45 * mm, splitLongWords=False),
        "bullet": ParagraphStyle("sum-bullet", parent=base["BodyText"], fontName=REGULAR, fontSize=7.15, leading=8.7, leftIndent=3.2 * mm, firstLineIndent=-2.5 * mm, bulletIndent=0, textColor=INK, spaceAfter=1.0 * mm),
        "eq": ParagraphStyle("sum-eq", parent=base["BodyText"], fontName=BOLD, fontSize=7.2, leading=8.8, alignment=TA_CENTER, textColor=NAVY, backColor=PALE, borderColor=colors.HexColor("#BFE2E3"), borderWidth=0.45, borderPadding=(2.2 * mm, 2.0 * mm, 2.2 * mm), spaceBefore=1.0 * mm, spaceAfter=1.5 * mm),
        "table_head": ParagraphStyle("sum-th", parent=base["Normal"], fontName=BOLD, fontSize=5.8, leading=7.0, textColor=WHITE),
        "table_cell": ParagraphStyle("sum-td", parent=base["Normal"], fontName=REGULAR, fontSize=5.7, leading=6.9, textColor=INK),
        "num": ParagraphStyle("sum-num", parent=base["BodyText"], fontName=REGULAR, fontSize=6.35, leading=7.7, leftIndent=3.5 * mm, firstLineIndent=-3.5 * mm, textColor=INK, spaceAfter=0.85 * mm),
    }


def blog_styles():
    base = getSampleStyleSheet()
    return {
        "kicker": ParagraphStyle("blog-kicker", parent=base["Normal"], fontName=BOLD, fontSize=7.4, leading=9, textColor=TEAL, spaceAfter=4 * mm, tracking=1.2),
        "title": ParagraphStyle("blog-title", parent=base["Title"], fontName=BOLD, fontSize=29, leading=32, textColor=NAVY, alignment=TA_LEFT, spaceAfter=4 * mm),
        "deck": ParagraphStyle("blog-deck", parent=base["Normal"], fontName=REGULAR, fontSize=11.6, leading=16, textColor=MUTED, spaceAfter=5 * mm),
        "meta": ParagraphStyle("blog-meta", parent=base["Normal"], fontName=BOLD, fontSize=7.4, leading=9, textColor=TEAL, spaceAfter=8 * mm),
        "h2": ParagraphStyle("blog-h2", parent=base["Heading2"], fontName=BOLD, fontSize=16.5, leading=19.5, textColor=NAVY, spaceBefore=7 * mm, spaceAfter=3 * mm, keepWithNext=True),
        "p": ParagraphStyle("blog-p", parent=base["BodyText"], fontName=REGULAR, fontSize=9.55, leading=13.65, textColor=INK, spaceAfter=2.9 * mm, splitLongWords=False),
        "bullet": ParagraphStyle("blog-bullet", parent=base["BodyText"], fontName=REGULAR, fontSize=8.9, leading=12.2, leftIndent=6 * mm, firstLineIndent=-3.7 * mm, bulletIndent=0, textColor=INK, spaceAfter=1.8 * mm),
        "eq": ParagraphStyle("blog-eq", parent=base["BodyText"], fontName=BOLD, fontSize=10.0, leading=13, alignment=TA_CENTER, textColor=NAVY, backColor=PALE, borderColor=colors.HexColor("#BFE2E3"), borderWidth=0.55, borderPadding=(3.2 * mm, 4 * mm, 3.2 * mm), spaceBefore=2.5 * mm, spaceAfter=3.8 * mm),
        "num": ParagraphStyle("blog-num", parent=base["BodyText"], fontName=REGULAR, fontSize=8.6, leading=12, leftIndent=6 * mm, firstLineIndent=-6 * mm, textColor=INK, spaceAfter=2.0 * mm),
    }


def block_flowables(blocks, styles, compact=False, skip_title=True):
    result = []
    for kind, payload in blocks:
        if kind == "h1" and skip_title:
            continue
        if kind == "h2":
            result.append(Paragraph(rich(str(payload)), styles["h2"]))
        elif kind == "p":
            result.append(Paragraph(rich(str(payload)), styles["p"]))
        elif kind == "bullet":
            result.append(Paragraph(rich(str(payload)), styles["bullet"], bulletText="-"))
        elif kind == "number":
            number, body = payload
            result.append(Paragraph(rich(body), styles["num"], bulletText=f"{number}."))
        elif kind == "eq":
            result.append(Paragraph(equation_markup(str(payload)), styles["eq"]))
        elif kind == "table":
            rows = payload
            table_data = []
            for row_idx, row in enumerate(rows):
                key = "table_head" if row_idx == 0 else "table_cell"
                table_data.append([Paragraph(rich(cell), styles[key]) for cell in row])
            table = Table(table_data, colWidths=[34 * mm, 31 * mm, 35 * mm, 35 * mm, 35 * mm] if not compact else [31 * mm, 25 * mm, 27 * mm, 30 * mm, 30 * mm], repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                ("GRID", (0, 0), (-1, -1), 0.35, LINE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2.0 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2.0 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 1.3 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1.3 * mm),
            ]))
            result.append(KeepTogether([table, Spacer(1, 1.5 * mm)]))
    return result


def make_summary() -> Path:
    path = OUT / "Synapse_Memory_Lab_Concept_Summary.pdf"
    styles = summary_styles()
    margin_x = 15 * mm
    top = 13 * mm
    bottom = 19 * mm
    gap = 8 * mm
    content_w = A4[0] - 2 * margin_x
    col_w = (content_w - gap) / 2
    frame_h = A4[1] - top - bottom
    frames = [
        Frame(margin_x, bottom, col_w, frame_h, id="left", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, showBoundary=0),
        Frame(margin_x + col_w + gap, bottom, col_w, frame_h, id="right", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, showBoundary=0),
    ]
    doc = NumberedDocTemplate(
        str(path), pagesize=A4, leftMargin=margin_x, rightMargin=margin_x,
        topMargin=top, bottomMargin=bottom,
        title="Synapse Memory Lab - Concept Summary",
        author="Synapse Memory Lab",
        subject="Causal linear attention, recurrent memory, BDH, and BDH-CQ",
    )
    doc.addPageTemplates(PageTemplate(id="summary", frames=frames, onPage=summary_page))
    blocks = parse_blocks(DOCS / "CONCEPT_SUMMARY.md")
    story = [
        Paragraph("RESEARCH BRIEF  /  SEPTEMBER 2026", styles["kicker"]),
        Paragraph("Synapse Memory Lab", styles["title"]),
        Paragraph("How causal linear attention compresses token history into a fixed-shape associative state - and where BDH and BDH-CQ extend the story.", styles["deck"]),
        HRFlowable(width="100%", thickness=1.2, color=TEAL, spaceAfter=2.1 * mm),
    ]
    story.extend(block_flowables(blocks, styles, compact=True))
    doc.build(story)
    return path


def make_blog() -> Path:
    path = OUT / "Synapse_Memory_Lab_Blog.pdf"
    styles = blog_styles()
    margin_x = 25 * mm
    top = 18 * mm
    bottom = 21 * mm
    frame = Frame(margin_x, bottom, A4[0] - 2 * margin_x, A4[1] - top - bottom, id="article", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc = NumberedDocTemplate(
        str(path), pagesize=A4, leftMargin=margin_x, rightMargin=margin_x,
        topMargin=top, bottomMargin=bottom,
        title="When attention becomes a memory: from token history to synaptic state",
        author="Synapse Memory Lab",
        subject="A technical article on recurrent linear attention, fast weights, BDH, and BDH-CQ",
    )
    doc.addPageTemplates(PageTemplate(id="article", frames=[frame], onPage=blog_page))
    blocks = parse_blocks(DOCS / "BLOG.md")
    intro_paragraphs = [payload for kind, payload in blocks if kind == "p"][:3]
    story = [
        Spacer(1, 7 * mm),
        Paragraph("TECHNICAL EXPLAINER  /  12 MIN READ", styles["kicker"]),
        Paragraph("When attention becomes a memory", styles["title"]),
        Paragraph("From token history to synaptic state", ParagraphStyle("blog-subtitle", parent=styles["title"], fontSize=17, leading=20, textColor=TEAL, spaceAfter=5 * mm)),
        HRFlowable(width="24%", thickness=3.2, color=CYAN, hAlign="LEFT", spaceAfter=5 * mm),
        Paragraph(rich(str(intro_paragraphs[0])), styles["deck"]),
        Paragraph("SYNAPSE MEMORY LAB  /  RESEARCH NOTE 01  /  SEPTEMBER 2026", styles["meta"]),
    ]
    skipped_first = False
    for kind, payload in blocks:
        if kind == "h1":
            continue
        if kind == "p" and not skipped_first and payload == intro_paragraphs[0]:
            skipped_first = True
            continue
        story.extend(block_flowables([(kind, payload)], styles, compact=False, skip_title=True))
    doc.build(story)
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    outputs = [make_summary(), make_blog()]
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
