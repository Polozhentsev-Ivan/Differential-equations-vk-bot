"""PDF рендерер: markdown с LaTeX → PDF с отрендеренными формулами."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import logging
logging.getLogger("fontTools.subset").setLevel(logging.WARNING)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from fpdf import FPDF

FONTS_DIR = Path(r"C:\Windows\Fonts")

# ── Unicode-замены для inline-формул ───────────────────────────────

_UNICODE_MAP = {
    r"\cdot": "·", r"\times": "×", r"\pm": "±",
    r"\neq": "≠", r"\leq": "≤", r"\geq": "≥", r"\approx": "≈",
    r"\Rightarrow": "⇒", r"\rightarrow": "→", r"\leftarrow": "←",
    r"\infty": "∞",
    r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ",
    r"\mu": "μ", r"\pi": "π", r"\theta": "θ", r"\lambda": "λ",
    r"\sigma": "σ", r"\omega": "ω", r"\phi": "φ", r"\varepsilon": "ε",
    r"\arctan": "arctan", r"\ln": "ln", r"\log": "log",
    r"\sin": "sin", r"\cos": "cos", r"\tan": "tan", r"\exp": "exp",
    r"\int": "∫", r"\sum": "∑", r"\prod": "∏", r"\partial": "∂",
    r"\,": " ", r"\ ": " ", r"\;": " ", r"\quad": "  ",
}

_SUP = {"2": "\u00B2", "3": "\u00B3"}


def _latex_to_unicode(latex: str) -> str:
    """Грубое приближение LaTeX -> Unicode для inline-текста (Arial-safe)."""
    s = latex
    for cmd, char in _UNICODE_MAP.items():
        s = s.replace(cmd, char)
    s = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"\1/\2", s)
    s = s.replace(r"\left(", "(").replace(r"\right)", ")")
    s = s.replace(r"\left[", "[").replace(r"\right]", "]")
    s = re.sub(r"\^{(\d)}", lambda m: _SUP.get(m.group(1), "^" + m.group(1)), s)
    s = re.sub(r"\^(\d)", lambda m: _SUP.get(m.group(1), "^" + m.group(1)), s)
    s = re.sub(r"_\{([^}]*)\}", r"_\1", s)
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    s = re.sub(r"[{}]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _process_inline_math(text: str) -> str:
    """Заменяет $...$ на Unicode-приближение."""
    return re.sub(r"\$(.*?)\$", lambda m: _latex_to_unicode(m.group(1)), text)


# ── Рендер display-формул через matplotlib ─────────────────────────

def _preprocess_for_mathtext(latex: str) -> str:
    latex = latex.replace(r"\text{", r"\mathrm{")
    latex = latex.replace("°", r"^{\circ}")
    return latex.strip()


def _render_latex_png(latex: str, fontsize: int = 15, dpi: int = 200) -> Path:
    """Рендерит LaTeX-формулу в PNG через matplotlib mathtext."""
    latex = _preprocess_for_mathtext(latex)
    fig = plt.figure(figsize=(0.01, 0.01))
    try:
        fig.text(0.5, 0.5, f"${latex}$",
                 fontsize=fontsize, ha="center", va="center")
    except Exception:
        plt.close(fig)
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.text(0.5, 0.5, _latex_to_unicode(latex),
                 fontsize=fontsize - 2, ha="center", va="center",
                 family="monospace")

    tmp = Path(tempfile.mktemp(suffix=".png"))
    fig.savefig(str(tmp), dpi=dpi, bbox_inches="tight",
                pad_inches=0.08, facecolor="white")
    plt.close(fig)
    return tmp


# ── PDF-документ ───────────────────────────────────────────────────

class SolutionPDF(FPDF):
    def __init__(self):
        super().__init__()
        self._temps: list[Path] = []

        if (FONTS_DIR / "arial.ttf").exists():
            self.add_font("Body", "", str(FONTS_DIR / "arial.ttf"))
            self.add_font("Body", "B", str(FONTS_DIR / "arialbd.ttf"))
            self.add_font("Body", "I", str(FONTS_DIR / "ariali.ttf"))
            self._ff = "Body"
        else:
            self._ff = "Helvetica"

        self.set_auto_page_break(auto=True, margin=20)
        self.add_page()
        self._body()

    def _body(self, style="", size=11):
        self.set_font(self._ff, style, size)

    def _page_w(self) -> float:
        return self.w - self.l_margin - self.r_margin

    # ── блоки ──

    def add_header(self, text: str, level: int):
        text = _process_inline_math(text)
        sizes = {1: 18, 2: 14, 3: 12}
        sz = sizes.get(level, 11)
        self.ln(4 if level == 1 else 2)
        self._body("B", sz)
        self.multi_cell(0, sz * 0.55, text)
        self.ln(2)
        if level <= 2:
            y = self.get_y()
            self.line(self.l_margin, y, self.l_margin + self._page_w(), y)
            self.ln(2)
        self._body()

    def add_text(self, text: str, bold: bool = False):
        text = _process_inline_math(text)
        self._body("B" if bold else "", 11)
        self.multi_cell(0, 6, text)
        self.ln(1)
        self._body()

    def add_formula(self, latex: str):
        png = _render_latex_png(latex)
        self._temps.append(png)
        try:
            img = Image.open(str(png))
            w_px, h_px = img.size
            img.close()

            w_mm = w_px / 200 * 25.4
            h_mm = h_px / 200 * 25.4

            max_w = self._page_w()
            if w_mm > max_w:
                ratio = max_w / w_mm
                w_mm *= ratio
                h_mm *= ratio

            if self.get_y() + h_mm + 5 > self.h - self.b_margin:
                self.add_page()

            x = self.l_margin + (max_w - w_mm) / 2
            self.image(str(png), x=x, y=self.get_y(), w=w_mm)
            self.set_y(self.get_y() + h_mm + 3)
        except Exception:
            self.add_text(f"  {_latex_to_unicode(latex)}")

    def add_table_row(self, cells: list[str], header: bool = False):
        cells = [_process_inline_math(c) for c in cells]
        col_w = self._page_w() / max(len(cells), 1)
        self._body("B" if header else "", 10)
        for cell in cells:
            self.cell(col_w, 7, cell, border=1, align="C")
        self.ln()
        self._body()

    def add_plot(self, plot_path: str):
        self.add_page()
        self.add_header("График", 2)
        self.image(plot_path, x=self.l_margin, w=self._page_w())

    def cleanup(self):
        for f in self._temps:
            f.unlink(missing_ok=True)


# ── Парсер markdown → PDF ─────────────────────────────────────────

def render_solution_pdf(
    md_content: str,
    output_path: str | Path,
    plot_path: str | Path | None = None,
) -> str:
    """Конвертирует markdown-решение с LaTeX в PDF."""
    content = _strip_frontmatter(md_content)
    pdf = SolutionPDF()

    lines = content.split("\n")
    i = 0
    in_table = False

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("![["):
            i += 1
            continue

        # ── display math $$...$$ ──
        if stripped.startswith("$$"):
            parts: list[str] = []
            if stripped == "$$":
                i += 1
                while i < len(lines) and lines[i].strip() != "$$":
                    parts.append(lines[i].strip())
                    i += 1
            else:
                inner = stripped[2:]
                if inner.endswith("$$"):
                    inner = inner[:-2]
                parts.append(inner)
            formula = " ".join(parts).strip()
            if formula:
                pdf.add_formula(formula)
            i += 1
            continue

        # ── headers ──
        hdr = re.match(r"^(#{1,3})\s+(.*)", line)
        if hdr:
            pdf.add_header(hdr.group(2).strip(), len(hdr.group(1)))
            in_table = False
            i += 1
            continue

        # ── table ──
        if stripped.startswith("|") and stripped.endswith("|"):
            sep_chars = set(stripped.replace("|", "").replace("-", "").replace(":", "").strip())
            if not sep_chars:
                i += 1
                continue
            cells = [c.strip() for c in stripped.split("|")[1:-1]]
            is_hdr = not in_table
            pdf.add_table_row(cells, header=is_hdr)
            in_table = True
            i += 1
            continue

        in_table = False

        # ── empty line ──
        if not stripped:
            pdf.ln(2)
            i += 1
            continue

        # ── bold-only line ──
        if re.match(r"^\*\*.*\*\*$", stripped):
            inner_text = stripped[2:-2]
            pdf.add_text(inner_text, bold=True)
            i += 1
            continue

        # ── list item ──
        list_m = re.match(r"^[-•]\s+(.*)", stripped)
        if list_m:
            item = re.sub(r"\*\*(.*?)\*\*", r"\1", list_m.group(1))
            pdf.add_text(f"  \u2022  {item}")
            i += 1
            continue

        # ── обычный текст ──
        clean = re.sub(r"\*\*(.*?)\*\*", r"\1", stripped)
        pdf.add_text(clean)
        i += 1

    if plot_path and Path(plot_path).exists():
        pdf.add_plot(str(plot_path))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output_path))
    pdf.cleanup()
    return str(output_path)


def _strip_frontmatter(text: str) -> str:
    lines = text.split("\n")
    if lines and lines[0].strip() == "---":
        for idx in range(1, len(lines)):
            if lines[idx].strip() == "---":
                return "\n".join(lines[idx + 1:])
    return text
