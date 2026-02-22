from __future__ import annotations

import hashlib
import re

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from docx import Document


@dataclass(frozen=True)
class LoadedDoc:
    """note: Represents a loaded document with stable identity for traceability and idempotent upserts."""
    path: Path
    sha256: str
    text: str


def sha256_file(path: Path) -> str:
    """note: Computes sha256 of a file for stable document identity even if file names change."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_text_file(path: Path) -> str:
    """note: Loads a UTF-8 text file (with replacement) into a string."""
    return path.read_text(encoding="utf-8", errors="replace")


def load_docx(path: Path) -> str:
    """note: Loads a .docx file using python-docx and returns concatenated paragraph text."""
    doc = Document(str(path))
    parts = []
    for p in doc.paragraphs:
        if p.text:
            parts.append(p.text)
    return "\n".join(parts).strip()


def load_pdf_optional(path: Path) -> Optional[str]:
    """note: Attempts to load a PDF; prefers PyMuPDF (fitz) for better layout, falls back to pypdf."""
    # Try PyMuPDF first
    try:
        import fitz  # type: ignore

        doc = fitz.open(str(path))
        parts: list[str] = []
        for i in range(doc.page_count):
            page = doc.load_page(i)
            t = page.get_text("text") or ""
            t = t.strip()
            if t:
                parts.append(f"--- Page {i + 1} ---\n{t}")
        return "\n\n".join(parts).strip()
    except Exception:
        pass

    # Fallback: pypdf
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        return None

    reader = PdfReader(str(path))
    parts: list[str] = []
    for i, page in enumerate(reader.pages, start=1):
        t = (page.extract_text() or "").strip()
        if t:
            parts.append(f"--- Page {i} ---\n{t}")
    return "\n\n".join(parts).strip()


def load_pptx_optional(path: Path) -> Optional[str]:
    """note: Attempts to load a PPTX using python-pptx; extracts text, tables, and speaker notes."""
    try:
        from pptx import Presentation  # type: ignore
    except Exception:
        return None

    prs = Presentation(str(path))
    parts: list[str] = []

    for si, slide in enumerate(prs.slides, start=1):
        slide_lines: list[str] = [f"--- Slide {si} ---"]

        for shape in slide.shapes:
            # Plain text
            if hasattr(shape, "text"):
                txt = (getattr(shape, "text") or "").strip()
                if txt:
                    slide_lines.append(txt)

            # Table cells (charts often have accompanying data tables)
            if getattr(shape, "has_table", False):
                for row in shape.table.rows:
                    for cell in row.cells:
                        txt = (cell.text or "").strip()
                        if txt:
                            slide_lines.append(txt)

        # Speaker notes (often contain the substantive content the chart illustrates)
        if getattr(slide, "has_notes_slide", False):
            notes_text = (slide.notes_slide.notes_text_frame.text or "").strip()
            if notes_text:
                slide_lines.append(f"[Notes] {notes_text}")

        if len(slide_lines) > 1:
            parts.append("\n".join(slide_lines))

    return "\n\n".join(parts).strip()


def _unwrap_pdf_lines(lines: list[str]) -> list[str]:
    """note: Heuristically joins hard-wrapped PDF lines while preserving blank lines as paragraph boundaries."""
    out: list[str] = []
    i = 0
    while i < len(lines):
        cur = lines[i]
        if cur == "":
            out.append("")
            i += 1
            continue

        # Merge subsequent wrapped lines until we hit a blank line or a "new paragraph" cue.
        merged = cur
        j = i + 1
        while j < len(lines):
            nxt = lines[j]
            if nxt == "":
                break

            # If current line ends in sentence-ish punctuation, treat next as new sentence/paragraph.
            if re.search(r"[.!?]\s*$", merged):
                break

            # If next line starts with lowercase (or punctuation) it's likely a hard wrap continuation.
            if re.match(r"^[a-z(]", nxt):
                merged = merged.rstrip() + " " + nxt.lstrip()
                j += 1
                continue

            # Otherwise, be conservative and stop merging.
            break

        out.append(merged)
        i = j
    return out


def _is_spoken_math_transcript(text: str) -> bool:
    """note: Detects auto-generated transcripts of spoken math (e.g. YouTube captions) which are
    unusable for generation — math notation read aloud produces near-jibberish chunks."""
    markers = [
        "superscript", "subscript", "open parenthesis", "close parenthesis",
        "divided by", "square root of", "equals sign", "times sign",
    ]
    text_lower = text.lower()
    hits = sum(1 for m in markers if m in text_lower)
    return hits >= 3


def preprocess_text(text: str, source_ext: str = "") -> str:
    """note: Normalizes extracted text while preserving paragraph boundaries for downstream block-based chunking."""
    t = (text or "")

    # Normalize line endings
    t = t.replace("\r\n", "\n").replace("\r", "\n")

    # De-hyphenate common PDF line wrap: "inter-\nnal" -> "internal"
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)

    # Normalize horizontal whitespace, but do not destroy newlines.
    t = re.sub(r"[ \t]+", " ", t)

    # Split into lines, stripping edges but preserving blank lines.
    raw_lines = t.split("\n")
    lines = [ln.strip() for ln in raw_lines]  # blank lines remain ""

    # Drop timestamp-only transcript lines (keep paragraph structure by turning them into blanks).
    ts_pat = re.compile(r"^\[?\d{1,2}:\d{2}(?::\d{2})?(?:\.\d{1,3})?\]?$")
    lines = [("" if ts_pat.match(ln) else ln) for ln in lines]

    # Drop standalone integer lines — PPTX/PDF slide and page number artifacts (e.g. "12", "13").
    int_pat = re.compile(r"^\d{1,3}$")
    lines = [("" if (ln and int_pat.match(ln)) else ln) for ln in lines]

    # Drop Wingdings/Symbol bullet artifacts rendered as bare 'z' by PyMuPDF/pptx.
    lines = [("" if ln == "z" else ln) for ln in lines]
    lines = [re.sub(r"^\s*z\s+", "", ln) for ln in lines]

    # Remove repeated "page furniture" lines that show up many times (headers/footers).
    # Raised length gate to 120 to catch longer repeated footer strings.
    # Only count non-empty lines; preserve blanks.
    freq: dict[str, int] = {}
    for ln in lines:
        if not ln:
            continue
        key = ln.lower()
        freq[key] = freq.get(key, 0) + 1

    cleaned: list[str] = []
    for ln in lines:
        if not ln:
            cleaned.append("")
            continue
        key = ln.lower()
        if freq.get(key, 0) >= 4 and len(ln) <= 120:
            cleaned.append("")
            continue
        cleaned.append(ln)

    # PDF-specific: unwrap hard line wraps, but preserve blank lines as paragraph separators.
    if source_ext.lower() == ".pdf":
        cleaned = _unwrap_pdf_lines(cleaned)

    # Collapse excessive blank lines (keep at most one blank line between paragraphs).
    out = "\n".join(cleaned)
    out = re.sub(r"\n{3,}", "\n\n", out)

    return out.strip()


def load_document(path: Path) -> Optional[LoadedDoc]:
    """note: Loads supported document types; returns None for unsupported or empty documents."""
    path = Path(path)
    ext = path.suffix.lower()
    sha = sha256_file(path)

    if ext in {".txt", ".md"}:
        text = load_text_file(path)
    elif ext == ".docx":
        text = load_docx(path)
    elif ext == ".pdf":
        text = load_pdf_optional(path)
        if text is None:
            raise RuntimeError("PDF support requires PyMuPDF or pypdf. Install one or remove PDFs.")
    elif ext == ".pptx":
        text = load_pptx_optional(path)
        if text is None:
            raise RuntimeError("PPTX support requires python-pptx. Install it or remove PPTX files.")
    else:
        return None

    if not text:
        return None

    text = preprocess_text(text, source_ext=ext).strip()

    if not text:
        return None

    # Flag auto-generated spoken-math transcripts — they produce unusable chunks and usually must be manually reviewed and edited at source doc.
    if _is_spoken_math_transcript(text):
        print(f"  [TRANSCRIPT FLAG] Spoken-math notation detected, manual review recommended @path:")
        print(f"  {path.name}")
        print(f"  [TRANSCRIPT FLAG] Spoken-math notation detected, manual review recommended: {path.name}", file=__import__("sys").stderr, flush=True)
        return None

    return LoadedDoc(path=path, sha256=sha, text=text)