from __future__ import annotations

import re

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Chunk:
    """note: A chunk is a contiguous segment of text used as the unit of embedding and retrieval."""
    index: int
    text: str


def chunk_text(text: str, chunk_chars: int = 1600, overlap_chars: int = 200) -> List[Chunk]:
    """note: Chunks text by structural blocks (paragraph-like units) then packs into roughly chunk_chars with overlap."""
    t = (text or "").strip()
    if not t:
        return []

    # Split into blocks on blank lines. This preserves local structure better than raw char slicing.
    blocks = [b.strip() for b in re.split(r"\n\s*\n+", t) if b and b.strip()]

    chunks: List[Chunk] = []
    idx = 0

    buf = ""
    for b in blocks:
        # If a single block is huge, hard-split it so we still make progress.
        if len(b) > int(chunk_chars):
            start = 0
            while start < len(b):
                piece = b[start : start + int(chunk_chars)].strip()
                if piece:
                    if buf:
                        # flush buffer before inserting a huge block piece
                        chunks.append(Chunk(index=idx, text=buf.strip()))
                        idx += 1
                        buf = ""
                    chunks.append(Chunk(index=idx, text=piece))
                    idx += 1
                start += max(1, int(chunk_chars) - int(overlap_chars))
            continue

        # Normal packing
        if not buf:
            buf = b
            continue

        candidate = buf + "\n\n" + b
        if len(candidate) <= int(chunk_chars):
            buf = candidate
        else:
            chunks.append(Chunk(index=idx, text=buf.strip()))
            idx += 1

            # overlap: carry tail of prior buffer into next
            tail = buf[-int(overlap_chars) :].strip() if int(overlap_chars) > 0 else ""
            buf = (tail + "\n\n" + b).strip() if tail else b

    if buf.strip():
        chunks.append(Chunk(index=idx, text=buf.strip()))

    return chunks
