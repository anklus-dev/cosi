"""
poster/build_poster.py

Builds the GSU STEM Conference poster (48 x 36 in) from the official template and
the evaluation results, so the numbers on the poster always match eval/results/.

    python poster/build_poster.py
    python poster/build_poster.py --variant prefixed/sentence   # report a different eval variant

Inputs (defaults in parentheses)
    --template      GSU poster template            (poster/template.pptx)
    --travel        travel-notes results folder    (eval/results/travel-notes)
    --meditations   Meditations results folder     (eval/results/meditations)
    --variant       which eval variant to report   (plain/sentence = what cosi ships)
    --repo          URL for the QR code            (https://github.com/anklus-dev/cosi)

Anything that needs results you haven't produced yet (vector / hybrid numbers need an
eval run with Ollama) is written as an orange "[pending ...]" marker, so a draft can
never be mistaken for a finished poster. Re-run this script after the eval and the
markers are replaced with real numbers.

Fonts: headings use Oswald Medium (the template's heading font), body text Fira Sans.
Both are free Google fonts, available in Google Slides; install them locally if you
open the poster in PowerPoint.
"""

import argparse
import csv
import io
import json
import math
import os
import re
import sys

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ---------------------------------------------------------------------------
# Style (taken from the template)
# ---------------------------------------------------------------------------
BLUE = RGBColor(0x00, 0x39, 0xA6)       # template headings
BRIGHT = RGBColor(0x35, 0x80, 0xFF)     # template secondary blue
ORANGE = RGBColor(0xFF, 0x5F, 0x26)     # template accent
INK = RGBColor(0x1C, 0x22, 0x33)
MUTED = RGBColor(0x55, 0x5E, 0x72)
TINT = RGBColor(0xEC, 0xF1, 0xFB)       # light blue card fill
LIGHT_BLUE = RGBColor(0xA9, 0xC1, 0xEE)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
HEAD_FONT = "Oswald Medium"
BODY_FONT = "Fira Sans"

BODY_PT = 30
SMALL_PT = 25
SECTION_PT = 56
CHAR_EM = 0.455  # average character width of Fira Sans in em, slightly conservative

COLS = [2.1, 13.3, 24.5, 35.6]   # column x positions (inches)
COL_W = 10.4
TOP = 11.05                      # first content line, below the column headers
BOTTOM = 34.9


def pending(what):
    return {"text": f"[pending: {what}]", "color": ORANGE, "bold": True}


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

class Results:
    """One dataset's eval output: summary.json + per_query.csv for a chosen variant."""

    def __init__(self, folder, variant):
        self.ok = False
        self.has_vectors = False
        self.rows = []
        path = os.path.join(folder, "summary.json")
        if not os.path.exists(path):
            return
        with open(path) as f:
            data = json.load(f)
        by_name = {r["variant"]: r for r in data["results"]}
        chosen = by_name.get(variant) or by_name.get("plain/sentence") or data["results"][0]
        self.variant = chosen["variant"]
        self.chunks = chosen["chunks"]
        self.summary = chosen["summary"]
        self.significance = chosen.get("significance", {})
        self.has_vectors = "cosi" in self.summary["all"]
        self.ok = True
        csv_path = os.path.join(folder, "per_query.csv")
        if os.path.exists(csv_path):
            with open(csv_path) as f:
                self.rows = [r for r in csv.DictReader(f) if r["variant"] == self.variant]

    def metric(self, group, method, name):
        try:
            return self.summary[group][method][name]
        except (KeyError, TypeError):
            return None

    def n(self, group="all"):
        return self.summary.get(group, {}).get("bm25", {}).get("n", 0) if self.ok else 0


def pct(x):
    return f"{round(100 * x)}%"


def chunk_stats(notes_dir):
    """Mid-sentence chunk ends under the old and new chunker (real numbers, computed here)."""
    import offline_ingest
    out = {}
    files = sorted(f for f in os.listdir(notes_dir) if f.endswith(".md")) if os.path.isdir(notes_dir) else []
    for mode in ("hard", "sentence"):
        chunks = []
        for name in files:
            with open(os.path.join(notes_dir, name), encoding="utf-8") as f:
                chunks += offline_ingest.chunk_text(f.read(), mode=mode)
        cut = sum(1 for c in chunks if not re.search(r"[.?!:;\"')_]$", c.strip()))
        out[mode] = (cut, len(chunks))
    return out


def load_passages(notes_dir):
    passages = {}
    if not os.path.isdir(notes_dir):
        return passages
    for name in sorted(os.listdir(notes_dir)):
        if name.endswith(".md"):
            with open(os.path.join(notes_dir, name), encoding="utf-8") as f:
                for p in re.split(r"\n\s*\n", f.read()):
                    if p.strip() and not p.startswith("#"):
                        label, _, text = p.strip().partition(" ")
                        passages[label] = text
    return passages


def pick_example(results, queries_path, passages):
    """A paraphrase query whose answer is short enough to quote, ideally a cosi win."""
    if not results.ok or not os.path.exists(queries_path):
        return None
    with open(queries_path) as f:
        queries = {q["id"]: q for q in json.load(f)["queries"]}
    rank = lambda v: int(v) if v else 10**6  # noqa: E731
    candidates = []
    for row in results.rows:
        q = queries.get(row["id"])
        if not q or q["category"] != "paraphrase" or "passage" not in q["relevant"][0]:
            continue
        label = q["relevant"][0]["passage"]
        text = passages.get(label, "")
        if not text or len(text) > 240:
            continue
        cosi = row.get("cosi_rank")
        if results.has_vectors and rank(cosi) != 1:
            continue
        candidates.append((-rank(row["bm25_rank"]), len(text), q["query"], label, text, row["bm25_rank"], cosi))
    if not candidates:
        return None
    _, _, query, label, text, bm25, cosi = sorted(candidates)[0]
    return {"query": query, "label": label, "text": text, "bm25_rank": bm25, "cosi_rank": cosi}


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def est_lines(text, width_in, pt):
    """Estimate wrapped line count for `text` in a box `width_in` wide."""
    per_line = max(1, int(width_in * 72 / (pt * CHAR_EM)))
    lines = 0
    for para in text.split("\n"):
        n, current = 1, 0
        for word in para.split():
            add = len(word) + (1 if current else 0)
            if current + add > per_line:
                n, current = n + 1, len(word)
            else:
                current += add
        lines += n
    return lines


def runs_text(para):
    if isinstance(para, str):
        return para
    if "runs" in para:
        return "".join(r if isinstance(r, str) else r["text"] for r in para["runs"])
    return para["text"]


def set_bullet(paragraph, indent_in=0.5):
    pPr = paragraph._p.get_or_add_pPr()
    pPr.set("marL", str(Emu(Inches(indent_in))))
    pPr.set("indent", str(-Emu(Inches(indent_in))))
    for tag in ("a:buClr", "a:buFont", "a:buChar", "a:buNone"):
        for el in pPr.findall(qn(tag)):
            pPr.remove(el)
    clr = pPr.makeelement(qn("a:buClr"), {})
    clr.append(clr.makeelement(qn("a:srgbClr"), {"val": "FF5F26"}))
    pPr.append(clr)
    pPr.append(pPr.makeelement(qn("a:buFont"), {"typeface": "Arial"}))
    pPr.append(pPr.makeelement(qn("a:buChar"), {"char": "•"}))


def style_run(run, size, color=INK, bold=False, italic=False, font=BODY_FONT):
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color


def add_text(slide, x, y, w, h, paras, size=BODY_PT, color=INK, align=PP_ALIGN.LEFT,
             anchor=MSO_ANCHOR.TOP, space_after=10, font=BODY_FONT, name=None, line_spacing=1.08):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        box.name = name
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space_after)
        p.line_spacing = line_spacing
        spec = {"text": para} if isinstance(para, str) else para
        if spec.get("bullet"):
            set_bullet(p)
        runs = spec.get("runs") or [spec]
        for r in runs:
            r = {"text": r} if isinstance(r, str) else r
            run = p.add_run()
            run.text = r["text"]
            style_run(run, r.get("size", spec.get("size", size)), r.get("color", spec.get("color", color)),
                      r.get("bold", spec.get("bold", False)), r.get("italic", spec.get("italic", False)),
                      r.get("font", spec.get("font", font)))
    return box


def text_height(paras, w, size=BODY_PT, space_after=10, line_spacing=1.08):
    total = 0.0
    for para in paras:
        spec = {"text": para} if isinstance(para, str) else para
        s = spec.get("size", size)
        width = w - (0.5 if spec.get("bullet") else 0)
        total += est_lines(runs_text(para), width, s) * s * 1.2 * line_spacing / 72 + space_after / 72
    return total


def add_rect(slide, x, y, w, h, fill=TINT, line=None, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08, name=None):
    shp = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        shp.name = name
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(2.5)
    shp.shadow.inherit = False
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        shp.adjustments[0] = radius
    return shp


def box_label(shape, lines, size=24, color=INK, bold_first=True):
    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.15)
    tf.margin_top = tf.margin_bottom = Inches(0.05)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = line
        style_run(run, size if i == 0 else size - 4, color, bold=(i == 0 and bold_first))


def arrow(slide, x1, y1, x2, y2, color=MUTED):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = color
    c.line.width = Pt(4)
    ln = c.line._get_or_add_ln()
    ln.append(ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"}))
    return c


class Column:
    """Stacks sections top-down in one poster column."""

    def __init__(self, slide, x, w=COL_W, y=TOP):
        self.slide, self.x, self.w, self.y = slide, x, w, y

    def heading(self, text):
        add_text(self.slide, self.x, self.y, self.w, 1.0, [text], size=SECTION_PT, color=BLUE,
                 font=HEAD_FONT, space_after=0, line_spacing=1.0, name=f"Heading: {text}")
        self.y += 1.15

    def text(self, paras, size=BODY_PT, gap=0.35, **kw):
        h = text_height(paras, self.w, size=size)
        add_text(self.slide, self.x, self.y, self.w, h, paras, size=size, **kw)
        self.y += h + gap

    def space(self, inches):
        self.y += inches

    def bottom_slot(self, h):
        """Y position that puts a block of height h on the shared bottom line."""
        y = BOTTOM - h
        if self.y > y + 0.6:  # estimates run ~0.5 in long; beyond that it may really collide
            print(f"warning: bottom block in column x={self.x} may overlap the text above it")
        self.y = BOTTOM
        return y


# ---------------------------------------------------------------------------
# Poster sections
# ---------------------------------------------------------------------------

def clear_template(slide):
    """Keep the frame (title, name lines, column headers, rule, GSU logos); drop sample content."""
    logos = sorted((s for s in slide.shapes if s.shape_type == 13), key=lambda s: s.top)[:2]
    for shp in list(slide.shapes):
        text = shp.text_frame.text.strip() if shp.has_text_frame else ""
        keep = (text in ("INTRODUCTION", "RESEARCH", "CONCLUSIONS")
                or text.startswith(("OCEAN DEEPS", "Your name"))
                or shp in logos or shp.shape_type == 9)  # 9 = line (the rule under the headers)
        if not keep:
            shp._element.getparent().remove(shp._element)
    # Drop relationships to the sample chart image and Slidesgo links that nothing uses now.
    xml = slide._element.xml
    for rId, rel in list(slide.part.rels.items()):
        if f'"{rId}"' not in xml and ("hyperlink" in rel.reltype or "image" in rel.reltype):
            slide.part.drop_rel(rId)
    if slide.has_notes_slide:
        slide.notes_slide.notes_text_frame.text = ""
    # The template's white inner panel (on the layout) carries a stray "v" at its left edge.
    for shp in slide.slide_layout.shapes:
        if shp.has_text_frame and shp.text_frame.text.strip() == "v":
            for para in shp.text_frame.paragraphs:
                for run in para.runs:
                    run.text = ""


def set_header_text(slide, title):
    for shp in slide.shapes:
        if not shp.has_text_frame:
            continue
        t = shp.text_frame.text
        if t.startswith("OCEAN DEEPS"):
            p = shp.text_frame.paragraphs[0]
            p.runs[0].text = title
            for r in p.runs[1:]:
                r.text = ""
            p.runs[0].font.size = Pt(128)
        elif t.startswith("Your name"):
            paras = shp.text_frame.paragraphs
            paras[0].runs[0].text = "Matthew Lindsey  ·  Faculty Mentor: Dr. Suxing Liu"
            for r in paras[0].runs[1:]:
                r.text = ""


def col_introduction(slide):
    c = Column(slide, COLS[0])
    c.heading("ABSTRACT")
    c.text(["cosi is a search engine for a folder of personal notes that runs entirely on the user's "
            "own computer. It ranks passages with two methods: BM25 keyword scoring, built on sets of "
            "stemmed words, and the cosine similarity of 768-dimensional text embeddings from a local "
            "AI model. Both scores are normalized and blended. We test keyword-only, meaning-only and "
            "blended search on two hand-labeled query sets."])
    c.heading("BACKGROUND")
    c.text([
        {"bullet": True, "runs": ["Keyword search only finds passages that share your exact words. "
                                  "Search for ", {"text": "aurora borealis", "italic": True},
                                  " and a note about the ", {"text": "northern lights", "italic": True},
                                  " stays hidden."]},
        {"bullet": True, "text": "Search by meaning usually runs in the cloud, so private notes leave your computer."},
        {"bullet": True, "text": "Meaning alone can blur exact names, numbers and terms."},
    ])
    c.heading("HYPOTHESIS")
    card_paras = [{"text": "Blending keyword and meaning scores finds the right passage more often than either "
                           "method alone, most of all when a question uses different words than the note.",
                   "italic": True}]
    h = text_height(card_paras, COL_W - 0.8) + 0.5
    add_rect(slide, c.x, c.y, COL_W, h, name="Hypothesis card")
    add_text(slide, c.x + 0.4, c.y + 0.3, COL_W - 0.8, h - 0.5, card_paras)
    c.space(h + 0.45)
    c.heading("OBJECTIVES")
    c.text([
        {"bullet": True, "text": "Run fully offline: no cloud services, no data leaves the machine."},
        {"bullet": True, "text": "Match exact terms and paraphrased meaning in one ranking."},
        {"bullet": True, "text": "Measure retrieval quality instead of assuming it."},
        {"bullet": True, "text": "Install in one command with Docker."},
    ])
    h = 2.45
    if True:
        y = c.bottom_slot(h)
        add_rect(slide, c.x, y, COL_W, h, name="Privacy callout")
        add_text(slide, c.x + 0.35, y + 0.2, 2.6, h - 0.4,
                 [{"runs": [{"text": "0", "size": 120, "color": ORANGE, "font": HEAD_FONT}]}],
                 align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, space_after=0, line_spacing=0.9)
        add_text(slide, c.x + 3.2, y + 0.3, COL_W - 3.55, h - 0.6,
                 [{"runs": [{"text": "cloud calls. ", "bold": True},
                            "Notes, index and AI model all stay on your computer; the web page loads no "
                            "outside fonts or scripts."]}],
                 size=SMALL_PT, anchor=MSO_ANCHOR.MIDDLE)
    return c


def col_methods(slide, chunks):
    c = Column(slide, COLS[1])
    c.heading("METHODOLOGY")
    # Two-lane pipeline diagram: indexing (left) and searching (right).
    x0, w, gap = c.x, (COL_W - 0.5) / 2, 0.5
    xl, xr = x0, x0 + w + gap
    lane_y = c.y
    add_text(slide, xl, lane_y, w, 0.5, ["Indexing (once)"], size=SMALL_PT, color=MUTED, align=PP_ALIGN.CENTER,
             space_after=0)
    add_text(slide, xr, lane_y, w, 0.5, ["Searching (each query)"], size=SMALL_PT, color=MUTED,
             align=PP_ALIGN.CENTER, space_after=0)
    y = lane_y + 0.65
    bh, step = 1.1, 1.55
    left = [
        (["Markdown notes"], TINT, INK),
        (["Sentence-aware chunker", "150–500 characters"], TINT, INK),
        (["Stem  +  embed", "word sets · 768-d vectors"], TINT, INK),
        (["SQLite index", "text, stems, vectors"], BLUE, WHITE),
    ]
    right = [
        (["Your question"], TINT, INK),
        (["BM25  +  cosine", "keywords · meaning"], TINT, INK),
        (["Blend scores", "0.4 · BM25 + 0.6 · cosine"], ORANGE, WHITE),
        (["Ranked passages", "with score breakdown"], BLUE, WHITE),
    ]
    for col_x, items in ((xl, left), (xr, right)):
        for i, (lines, fill, ink) in enumerate(items):
            shp = add_rect(slide, col_x, y + i * step, w, bh, fill=fill)
            box_label(shp, lines, size=25, color=ink)
            if i:
                arrow(slide, col_x + w / 2, y + (i - 1) * step + bh + 0.02, col_x + w / 2, y + i * step - 0.04)
    # index feeds the search branch
    arrow(slide, xl + w + 0.02, y + 3 * step + bh / 2, xr - 0.04, y + 1 * step + bh / 2, color=BLUE)
    c.y = y + 3 * step + bh + 0.35
    c.text(["Both branches score every question; the blend sets the order. Model offline: keywords only."],
           size=SMALL_PT, color=MUTED, gap=0.25)

    c.heading("MODELS")
    eq = lambda s: {"text": s, "italic": True, "size": 28, "color": BLUE}  # noqa: E731
    c.text([
        {"runs": [{"text": "Sets & equivalence classes. ", "bold": True},
                  "Stemming maps word forms to one representative; a keyword match is the intersection of "
                  "query and passage word sets."]},
        eq("stem(connecting) = stem(connection) = connect"),
        eq("keyword matches = Q ∩ D"),
        {"runs": [{"text": "BM25. ", "bold": True},
                  "Rare words count more (IDF), repeats have diminishing returns, long passages are normalized."]},
        eq("BM25(D,Q) = Σ IDF(t)·tf(k₁+1) / (tf + k₁(1 − b + b·|D|/avgdl))"),
        {"runs": [{"text": "Embedding function. ", "bold": True},
                  "A local model (nomic-embed-text) maps text to a point in 768-dimensional space; "
                  "relevance is the angle between points."]},
        eq("f : String → ℝ⁷⁶⁸,     cos θ = (q · d) / (‖q‖ ‖d‖)"),
        {"runs": [{"text": "Blend and relation R. ", "bold": True},
                  "Scores are normalized to [0, 1] and blended; results under half the top score are dropped. "
                  "x R y if similarity > τ: reflexive, symmetric, not transitive."]},
        eq("score = 0.4 · norm(BM25) + 0.6 · norm(cos)"),
    ], size=BODY_PT - 2, gap=0.3)

    chunk_callout(slide, c, chunks)
    return c


def chunk_callout(slide, c, chunks):
    """Before/after stat for the sentence-aware chunker (numbers computed from the dataset)."""
    (hard_cut, hard_n), (sent_cut, sent_n) = chunks["hard"], chunks["sentence"]
    h = 2.55
    y0 = c.bottom_slot(h)
    add_rect(slide, c.x, y0, COL_W, h, name="Chunking callout")
    add_text(slide, c.x + 0.35, y0 + 0.25, 3.6, h - 0.5,
             [{"runs": [{"text": pct(hard_cut / hard_n), "size": 66, "color": MUTED, "font": HEAD_FONT}]},
              {"runs": [{"text": "→ " + (f"{100 * sent_cut / sent_n:.1f}%" if sent_cut else "0%"), "size": 66,
                         "color": ORANGE, "font": HEAD_FONT}]}],
             space_after=0, line_spacing=0.9)
    add_text(slide, c.x + 4.1, y0 + 0.3, COL_W - 4.45, h - 0.5,
             [{"runs": [{"text": "Chunks cut mid-sentence. ", "bold": True},
                        f"Cutting every 500 characters split {hard_cut} of {hard_n} Meditations chunks "
                        f"mid-sentence. Splitting on sentence boundaries leaves {sent_cut} of {sent_n}."]}],
             size=SMALL_PT, anchor=MSO_ANCHOR.MIDDLE)


def results_chart(slide, x, y, w, h, med):
    has_vec = med.has_vectors
    data = CategoryChartData()
    data.categories = ["Keyword queries", "Paraphrase queries"]
    series = [("Keyword only (BM25)", "bm25", LIGHT_BLUE)]
    if has_vec:
        series += [("Meaning only (cosine)", "vector", BRIGHT), ("cosi blend", "cosi", ORANGE)]
    for label, method, _ in series:
        data.add_series(label, [med.metric("keyword", method, "mrr") or 0, med.metric("paraphrase", method, "mrr") or 0])
    gf = slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(x), Inches(y), Inches(w), Inches(h), data)
    gf.name = "Results chart"
    chart = gf.chart
    chart.font.name = BODY_FONT
    chart.font.size = Pt(22)
    chart.font.color.rgb = INK
    chart.has_title = True
    chart.chart_title.text_frame.text = "Meditations: mean reciprocal rank (higher is better)"
    tr = chart.chart_title.text_frame.paragraphs[0].runs[0]
    style_run(tr, 24, INK, bold=True)
    chart.has_legend = True
    chart.legend.position = XL_LEGEND_POSITION.BOTTOM
    chart.legend.include_in_layout = False
    chart.legend.font.size = Pt(22)
    plot = chart.plots[0]
    plot.gap_width = 55
    plot.overlap = -8
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.number_format = "0.00"
    dl.number_format_is_linked = False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    dl.font.size = Pt(22)
    dl.font.bold = True
    for s, (_, _, color) in zip(plot.series, series):
        s.format.fill.solid()
        s.format.fill.fore_color.rgb = color
    va = chart.value_axis
    va.minimum_scale, va.maximum_scale, va.major_unit = 0, 1.0, 0.25
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = RGBColor(0xDD, 0xE2, 0xEA)
    va.format.line.fill.background()
    va.tick_labels.font.size = Pt(20)
    va.tick_labels.font.color.rgb = MUTED
    ca = chart.category_axis
    ca.tick_labels.font.size = Pt(22)
    ca.format.line.color.rgb = RGBColor(0xBB, 0xC2, 0xCF)


def results_table(slide, x, y, w, travel, med):
    methods = [("Keyword only", "bm25"), ("Meaning only", "vector"), ("cosi blend", "cosi")]
    rows = [("Dataset / method", "Hit@1", "MRR", "Paraphrase MRR")]
    for name, res in (("Travel notes", travel), ("Meditations", med)):
        for label, m in methods:
            vals = []
            for group, metric in (("all", "hit@1"), ("all", "mrr"), ("paraphrase", "mrr")):
                v = res.metric(group, m, metric) if res.ok else None
                vals.append("pending" if v is None else (pct(v) if metric == "hit@1" else f"{v:.2f}"))
            rows.append((f"{name}: {label}", *vals))
    row_h = 0.62
    shape = slide.shapes.add_table(len(rows), 4, Inches(x), Inches(y), Inches(w), Inches(row_h * len(rows)))
    shape.name = "Results table"
    table = shape.table
    widths = [4.6, 1.6, 1.6, w - 7.8]
    for i, cw in enumerate(widths):
        table.columns[i].width = Inches(cw)
    tblPr = table._tbl.tblPr
    tblPr.set("bandRow", "0")
    tblPr.set("firstRow", "0")
    for r, row in enumerate(rows):
        table.rows[r].height = Inches(row_h)
        for col, val in enumerate(row):
            cell = table.cell(r, col)
            cell.margin_left = cell.margin_right = Inches(0.12)
            cell.margin_top = cell.margin_bottom = Inches(0.04)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cosi_row = r > 0 and row[0].endswith("cosi blend")
            cell.fill.fore_color.rgb = BLUE if r == 0 else (RGBColor(0xFF, 0xEC, 0xE4) if cosi_row else
                                                            (WHITE if r % 2 else TINT))
            tf = cell.text_frame
            tf.text = ""
            run = tf.paragraphs[0].add_run()
            run.text = val
            tf.paragraphs[0].alignment = PP_ALIGN.LEFT if col == 0 else PP_ALIGN.CENTER
            color = WHITE if r == 0 else (ORANGE if val == "pending" else INK)
            style_run(run, 22, color, bold=(r == 0 or cosi_row))
    return row_h * len(rows)


def col_results(slide, travel, med):
    c = Column(slide, COLS[2])
    c.heading("DATA & EVALUATION")
    c.text([
        {"bullet": True, "runs": [{"text": "Travel notes: ", "bold": True},
                                  f"6 notes, {travel.chunks if travel.ok else 24} passages, {travel.n() or 44} questions."]},
        {"bullet": True, "runs": [{"text": "Meditations ", "bold": True},
                                  "by Marcus Aurelius (Casaubon translation, public domain): 12 books, "
                                  f"{med.chunks if med.ok else 856} passages, {med.n() or 40} questions."]},
        {"bullet": True, "text": "Every question is labeled with its correct passage. Keyword questions reuse the "
                                 "text's words; paraphrase questions ask in everyday modern English."},
        {"bullet": True, "text": "Hit@1: right passage ranked first. MRR: average of 1 ÷ its rank (1.0 = always first)."},
    ], size=BODY_PT - 2)
    c.heading("RESULTS")
    chart_h = 6.6
    results_chart(slide, c.x - 0.1, c.y, COL_W + 0.2, chart_h, med)
    c.y += chart_h + 0.3
    if not med.has_vectors:
        c.text([{"runs": [pending("meaning-only and cosi bars; run eval/run_eval.py with Ollama")]}],
               size=SMALL_PT, gap=0.2)
    c.y += results_table(slide, c.x, c.y, COL_W, travel, med) + 0.4

    return c


def example_card(slide, c, example):
    if not example:
        return
    bm = example["bm25_rank"] or "not found"
    cosi = example["cosi_rank"]
    paras = [
        {"runs": [{"text": "Question: ", "bold": True, "color": BLUE},
                  {"text": f"“{example['query']}”", "italic": True}]},
        {"runs": [{"text": f"Answer ({example['label']}): ", "bold": True, "color": BLUE},
                  f"“{example['text']}”"]},
        {"runs": [{"text": "Rank: ", "bold": True},
                  f"keyword-only #{bm}  ·  cosi ",
                  {"text": f"#{cosi}", "bold": True, "color": ORANGE} if cosi else pending("cosi rank")]},
    ]
    h = text_height(paras, COL_W - 0.8, size=SMALL_PT) + 0.8
    y = c.bottom_slot(h)
    add_rect(slide, c.x, y, COL_W, h, fill=WHITE, line=LIGHT_BLUE, name="Example card")
    add_text(slide, c.x + 0.4, y + 0.25, COL_W - 0.8, h - 0.4, paras, size=SMALL_PT)


def col_conclusions(slide, travel, med, chunks, repo):
    c = Column(slide, COLS[3])
    c.heading("CONCLUSION")
    kw_hit = med.metric("keyword", "bm25", "hit@1")
    para_hit = med.metric("paraphrase", "bm25", "hit@1")
    findings = []
    if kw_hit is not None and para_hit is not None:
        findings.append({"bullet": True, "text": f"Keyword search is exact but brittle: on the Meditations it ranked "
                                                 f"the right passage first for {pct(kw_hit)} of keyword questions, "
                                                 f"but only {pct(para_hit)} of paraphrased ones."})
    if med.has_vectors:
        ch, cm = med.metric("paraphrase", "cosi", "hit@1"), med.metric("keyword", "cosi", "hit@1")
        findings.append({"bullet": True, "text": f"Blending in meaning raised paraphrase Hit@1 to {pct(ch)} while "
                                                 f"keyword questions stayed at {pct(cm)}."})
        p = med.significance.get("cosi vs bm25")
        if p is not None:
            findings.append({"bullet": True, "text": f"The gain over keyword-only search is "
                                                     f"{'statistically significant' if p < 0.05 else 'not statistically significant'} "
                                                     f"(paired randomization test, p = {p:.3f})."})
    else:
        findings.append({"bullet": True, "runs": [pending("how much the blend improves paraphrase questions")]})
    hard_cut, hard_n = chunks["hard"]
    sent_cut, sent_n = chunks["sentence"]
    findings.append({"bullet": True, "text": f"Sentence-aware chunking removed almost all mid-sentence cuts "
                                             f"({hard_cut} → {sent_cut})."})
    c.text(findings)
    c.heading("DISCUSSION")
    c.text([
        {"bullet": True, "text": "Small, hand-labeled query sets; the weight sweep is tuned on the questions it is scored on."},
        {"bullet": True, "text": "Labels name one or two passages, so scores are conservative."},
        {"bullet": True, "text": "No stopword removal yet: words like “the” add small keyword scores."},
        {"bullet": True, "text": "17th-century English is hard for both keyword and meaning search."},
    ], size=BODY_PT - 2)
    c.heading("FUTURE WORK")
    c.text([
        {"bullet": True, "text": "A browser-only version on GitHub Pages that uses the visitor's own local model."},
        {"bullet": True, "text": "Choose the blend weight per question, such as more keyword weight for names."},
        {"bullet": True, "text": "Larger test sets labeled by people other than the author."},
    ], size=BODY_PT - 2)
    c.heading("ACKNOWLEDGEMENTS")
    c.text(["Dr. Suxing Liu, faculty mentor (CSC 2510). Meditations text: Project Gutenberg eBook #2680."],
           size=SMALL_PT)
    qr_size = 2.6
    c.bottom_slot(qr_size + 1.15)
    c.y = BOTTOM - qr_size - 1.15
    c.heading("CONTACT INFO")
    add_text(slide, c.x, c.y, COL_W - qr_size - 0.3, qr_size,
             [{"text": "Matthew Lindsey", "bold": True}, repo.replace("https://", ""),
              {"runs": [pending("GSU email")]}],
             size=SMALL_PT, space_after=6)
    try:
        import qrcode
        img = qrcode.make(repo, border=1)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        slide.shapes.add_picture(buf, Inches(c.x + COL_W - qr_size), Inches(c.y), Inches(qr_size), Inches(qr_size))
    except ImportError:
        print("qrcode not installed; skipping QR code")
    c.y += qr_size
    return c


def main():
    ap = argparse.ArgumentParser(description="Build the cosi conference poster.")
    ap.add_argument("--template", default=os.path.join(ROOT, "poster", "template.pptx"))
    ap.add_argument("--travel", default=os.path.join(ROOT, "eval", "results", "travel-notes"))
    ap.add_argument("--meditations", default=os.path.join(ROOT, "eval", "results", "meditations"))
    ap.add_argument("--meditations-notes", default=os.path.join(ROOT, "datasets", "meditations"))
    ap.add_argument("--meditations-queries", default=os.path.join(ROOT, "eval", "queries_meditations.json"))
    ap.add_argument("--variant", default="plain/sentence")
    ap.add_argument("--repo", default="https://github.com/anklus-dev/cosi")
    ap.add_argument("--out", default=os.path.join(ROOT, "poster", "cosi_poster.pptx"))
    args = ap.parse_args()

    travel = Results(args.travel, args.variant)
    med = Results(args.meditations, args.variant)
    chunks = chunk_stats(args.meditations_notes)
    example = pick_example(med, args.meditations_queries, load_passages(args.meditations_notes))

    prs = Presentation(args.template)
    slide = prs.slides[0]
    clear_template(slide)
    set_header_text(slide, "COSI: OFFLINE HYBRID SEARCH FOR NOTES")
    c1 = col_introduction(slide)
    c2 = col_methods(slide, chunks)
    c3 = col_results(slide, travel, med)
    example_card(slide, c3, example)
    c4 = col_conclusions(slide, travel, med, chunks, args.repo)
    for col in (c1, c2, c3, c4):
        if col.y > BOTTOM:
            print(f"warning: column at x={col.x} overflows by {col.y - BOTTOM:.1f} in")
    prs.save(args.out)

    status = "complete" if med.has_vectors and travel.has_vectors else "draft (vector/hybrid results pending)"
    print(f"Saved {args.out}  [{status}]")


if __name__ == "__main__":
    main()
