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
        parts = []
        for i in range(doc.page_count):
            page = doc.load_page(i)
            t = page.get_text("text") or ""
            if t.strip():
                parts.append(t)
        return "\n\n".join(parts).strip()
    except Exception:
        pass

    # Fallback: pypdf
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        return None

    reader = PdfReader(str(path))
    parts = []
    for page in reader.pages:
        t = page.extract_text() or ""
        if t.strip():
            parts.append(t)
    return "\n\n".join(parts).strip()
    
    
def load_pptx_optional(path: Path) -> Optional[str]:
    """note: Attempts to load a PPTX using python-pptx; returns None if unavailable."""
    try:
        from pptx import Presentation  # type: ignore
    except Exception:
        return None

    prs = Presentation(str(path))
    parts: list[str] = []
    for si, slide in enumerate(prs.slides, start=1):
        slide_lines: list[str] = [f"--- Slide {si} ---"]
        for shape in slide.shapes:
            if not hasattr(shape, "text"):
                continue
            txt = (getattr(shape, "text") or "").strip()
            if txt:
                slide_lines.append(txt)
        if len(slide_lines) > 1:
            parts.append("\n".join(slide_lines))
    return "\n\n".join(parts).strip()


def preprocess_text(text: str, source_ext: str = "") -> str:
    """note: Normalizes extracted text and removes common extraction noise (headers/footers, timestamps, hyphenation)."""
    t = (text or "")

    # Normalize line endings
    t = t.replace("\r\n", "\n").replace("\r", "\n")

    # De-hyphenate common PDF line wrap: "inter-\nnal" -> "internal"
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)

    # Collapse excessive spaces and blank lines
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)

    lines = [ln.strip() for ln in t.split("\n")]
    lines = [ln for ln in lines if ln]

    # Drop timestamp-only transcript lines like:
    # 00:12, 1:02:33, [00:01:22], 00:01:22.123
    ts_pat = re.compile(r"^\[?\d{1,2}:\d{2}(?::\d{2})?(?:\.\d{1,3})?\]?$")
    lines = [ln for ln in lines if not ts_pat.match(ln)]

    # Remove very short “page furniture” lines that repeat a lot (headers/footers)
    # Heuristic: count repeats; drop lines that repeat on many pages/sections.
    freq: dict[str, int] = {}
    for ln in lines:
        key = ln.lower()
        freq[key] = freq.get(key, 0) + 1

    # Threshold tuned for small corpora: drop lines repeated 4+ times AND short-ish.
    cleaned: list[str] = []
    for ln in lines:
        key = ln.lower()
        if freq.get(key, 0) >= 4 and len(ln) <= 80:
            continue
        cleaned.append(ln)

    return "\n".join(cleaned).strip()


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

    return LoadedDoc(path=path, sha256=sha, text=text)


