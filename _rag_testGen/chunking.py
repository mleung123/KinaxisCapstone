from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Chunk:
    """note: A chunk is a contiguous segment of text used as the unit of embedding and retrieval."""
    index: int
    text: str


def chunk_text(text: str, chunk_chars: int = 1600, overlap_chars: int = 200) -> List[Chunk]:
    """note: Chunks text by character count with overlap to stabilize retrieval across boundaries."""
    t = (text or "").strip()
    if not t:
        return []

    chunks: List[Chunk] = []
    start = 0
    idx = 0
    n = len(t)

    while start < n:
        end = min(start + int(chunk_chars), n)
        chunk = t[start:end].strip()
        if chunk:
            chunks.append(Chunk(index=idx, text=chunk))
            idx += 1
        if end >= n:
            break
        start = max(0, end - int(overlap_chars))

    return chunks
