from __future__ import annotations

import hashlib
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
    """note: Attempts to load a PDF using pypdf if installed; returns None if unavailable."""
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
    return "\n".join(parts).strip()


def load_document(path: Path) -> Optional[LoadedDoc]:
    """note: Loads supported document types; returns None for unsupported or empty documents."""
    path = Path(path)
    ext = path.suffix.lower()
    sha = sha256_file(path)

    if ext in {".txt", ".md"}:
        text = load_text_file(path).strip()
    elif ext == ".docx":
        text = load_docx(path).strip()
    elif ext == ".pdf":
        text = load_pdf_optional(path)
        if text is None:
            raise RuntimeError("PDF support requires pypdf. Install it or remove PDFs from the domain folder.")
        text = text.strip()
    else:
        return None

    if not text:
        return None

    return LoadedDoc(path=path, sha256=sha, text=text)
