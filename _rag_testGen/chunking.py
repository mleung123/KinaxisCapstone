from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Chunk:
    """note: A chunk is a contiguous segment of text used as the unit of embedding and retrieval."""
    index: int
    text: str


_SENTENCE_END_RE = re.compile(r"([.!?]+)\s+")
_WS_RE = re.compile(r"\s+")


def _normalize_ws(s: str) -> str:
    """note: Normalizes whitespace for more stable overlap behavior and downstream hashing/traceability."""
    return _WS_RE.sub(" ", (s or "").strip())


def _tail_overlap(prev_text: str, max_chars: int, max_sentences: int = 3) -> str:
    """note: Creates boundary-aware overlap by taking up to max_sentences from the end, capped by max_chars."""
    if max_chars <= 0:
        return ""

    t = (prev_text or "").strip()
    if not t:
        return ""

    # Prefer sentence-aware overlap. If sentence boundaries are not found, fall back to raw tail chars.
    sentences: list[str] = []
    start = 0
    for m in _SENTENCE_END_RE.finditer(t):
        end = m.end(1)  # include punctuation, exclude whitespace
        seg = t[start:end].strip()
        if seg:
            sentences.append(seg)
        start = m.end()
    if start < len(t):
        tail_seg = t[start:].strip()
        if tail_seg:
            sentences.append(tail_seg)

    if not sentences:
        return t[-max_chars:].strip()

    tail = " ".join(sentences[-max_sentences:]).strip()
    if len(tail) > max_chars:
        tail = tail[-max_chars:].strip()

    return tail


def chunk_text(text: str, chunk_chars: int = 1600, overlap_chars: int = 200) -> List[Chunk]:
    """note: Chunks text by paragraph-like blocks, then packs into roughly chunk_chars with boundary-aware overlap."""
    t = (text or "").strip()
    if not t:
        return []

    # Depends on preprocessing preserving blank-line paragraph breaks.
    blocks = [b.strip() for b in re.split(r"\n\s*\n+", t) if b and b.strip()]

    chunks: List[Chunk] = []
    idx = 0
    buf = ""

    for b in blocks:
        # If a single block is huge, hard-split so we still make progress.
        if len(b) > int(chunk_chars):
            start = 0
            step = max(1, int(chunk_chars) - int(overlap_chars))
            while start < len(b):
                piece = b[start : start + int(chunk_chars)].strip()
                if piece:
                    if buf:
                        chunks.append(Chunk(index=idx, text=buf.strip()))
                        idx += 1
                        buf = ""
                    chunks.append(Chunk(index=idx, text=piece))
                    idx += 1
                start += step
            continue

        if not buf:
            buf = b
            continue

        candidate = buf + "\n\n" + b
        if len(candidate) <= int(chunk_chars):
            buf = candidate
            continue

        chunks.append(Chunk(index=idx, text=buf.strip()))
        idx += 1

        tail = _tail_overlap(buf, max_chars=int(overlap_chars), max_sentences=3).strip()
        buf = (tail + "\n\n" + b).strip() if tail else b

    if buf.strip():
        chunks.append(Chunk(index=idx, text=buf.strip()))

    return chunks
