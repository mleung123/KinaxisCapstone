from __future__ import annotations

import hashlib
import re
import sys
import psycopg

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List

from chunking import chunk_text
from db_pgvector import clear_chunks, ensure_schema, set_meta_if_absent, set_meta, upsert_chunks
from embed_lmstudio import EmbedConfig, embed_texts
from loaders import load_document


@dataclass(frozen=True)
class IngestConfig:
    """note: Configuration for end-to-end ingestion from domain folder into pgvector."""
    domain_dir: Path
    db_dsn: str
    embed_lm_url: str
    embed_model: str
    embedding_dim: int | None = None
    batch_size: int = 32
    chunk_chars: int = 1600
    overlap_chars: int = 200
    clear_first: bool = False


_WS_RE = re.compile(r"\s+")


def _chunk_id(doc_sha256: str, chunk_text: str) -> str:
    """note: Generates a stable chunk id derived from document identity and normalized chunk text."""
    norm = _WS_RE.sub(" ", (chunk_text or "").strip())
    h = hashlib.sha256()
    h.update((doc_sha256 + "\n" + norm).encode("utf-8"))
    return h.hexdigest()


def _chunk_content_ok(text: str, min_alpha_chars: int = 120, min_alpha_ratio: float = 0.35) -> bool:
    """note: Returns True if chunk has enough alphabetic content to be useful for generation.
    Filters out slide-number artifacts, footer-only chunks, chart-label noise, and OCR-corrupted text."""
    t = (text or "").strip()
    if not t:
        return False
    alpha_chars = sum(1 for c in t if c.isalpha())
    if alpha_chars < min_alpha_chars:
        return False
    if (alpha_chars / len(t)) < min_alpha_ratio:
        return False
    return True


def iter_domain_files(domain_dir: Path) -> Iterable[Path]:
    """note: Iterates all files recursively under domain_dir in a stable order."""
    for p in sorted(Path(domain_dir).rglob("*")):
        if p.is_file():
            yield p


def _infer_embedding_dim(embed_lm_url: str, embed_model: str) -> int:
    """note: Probes the embeddings endpoint once and returns the embedding vector dimension."""
    embs = embed_texts(EmbedConfig(lm_url=embed_lm_url, model=embed_model), ["dimension probe"])
    if not embs or not embs[0]:
        raise RuntimeError("Embedding dimension probe failed (no embedding returned).")
    return int(len(embs[0]))


def ingest_domain(cfg: IngestConfig) -> dict[str, Any]:
    """note: Loads docs, chunks, embeds, and upserts into Postgres; returns an ingestion summary."""
    domain_dir = Path(cfg.domain_dir).resolve()
    if not domain_dir.exists():
        raise RuntimeError(f"domain_dir not found: {domain_dir}")

    loaded = []
    skipped = 0

    for p in iter_domain_files(domain_dir):
        doc = load_document(p)
        if doc is None:
            skipped += 1
            continue
        loaded.append(doc)

    docs_total = len(loaded)

    embedding_dim = cfg.embedding_dim
    if embedding_dim is None:
        embedding_dim = _infer_embedding_dim(cfg.embed_lm_url, cfg.embed_model)

    rows_total = 0
    chunks_filtered_count = 0

    with psycopg.connect(cfg.db_dsn) as conn:
        ensure_schema(conn, int(embedding_dim))
        _set_meta = set_meta if cfg.clear_first else set_meta_if_absent
        _set_meta(conn, "embedding_dim", str(int(embedding_dim)))
        _set_meta(conn, "embed_model", str(cfg.embed_model))
        _set_meta(conn, "source_root", str(domain_dir))
        _set_meta(conn, "batch_size", str(int(cfg.batch_size)))
        _set_meta(conn, "chunk_chars", str(int(cfg.chunk_chars)))
        _set_meta(conn, "overlap_chars", str(int(cfg.overlap_chars)))

        if cfg.clear_first:
            cleared = clear_chunks(conn)
        else:
            cleared = 0

        print(f"Ingesting {docs_total} docs from {domain_dir}", file=sys.stderr, flush=True)

        pending_texts: List[str] = []
        pending_rows: List[dict[str, Any]] = []

        for doc in loaded:
            print(f"  Processing {doc.path.name} ...", file=sys.stderr, flush=True)

            raw_chunks = chunk_text(doc.text, chunk_chars=cfg.chunk_chars, overlap_chars=cfg.overlap_chars)
            raw_count = len(raw_chunks)
            chunks = [ch for ch in raw_chunks if _chunk_content_ok(ch.text)]
            filtered = raw_count - len(chunks)
            chunks_filtered_count += filtered

            if filtered:
                print(f"    Filtered {filtered}/{raw_count} chunks (low content density)", file=sys.stderr, flush=True)

            for ch in chunks:
                pending_texts.append(ch.text)
                pending_rows.append(
                    {
                        "doc_path": str(doc.path),
                        "doc_sha256": doc.sha256,
                        "chunk_index": int(ch.index),
                        "chunk_text": ch.text,
                        "embedding": None,
                        "meta": {
                            "source_root": str(domain_dir),
                            "rel_path": str(doc.path.resolve().relative_to(domain_dir)),
                            "chunk_id": _chunk_id(doc.sha256, ch.text),
                        },
                    }
                )

                if len(pending_texts) >= int(cfg.batch_size):
                    embs = embed_texts(
                        EmbedConfig(lm_url=cfg.embed_lm_url, model=cfg.embed_model),
                        pending_texts,
                    )
                    if not embs:
                        raise RuntimeError("Embeddings call returned no embeddings.")
                    if len(embs) != len(pending_rows):
                        raise RuntimeError("Embeddings count mismatch vs pending rows.")
                    for r, e in zip(pending_rows, embs):
                        r["embedding"] = e
                    rows_total += upsert_chunks(conn, pending_rows)
                    print(f"  Embedded {rows_total} chunks so far...", file=sys.stderr, flush=True)
                    pending_texts = []
                    pending_rows = []

        if pending_texts:
            embs = embed_texts(
                EmbedConfig(lm_url=cfg.embed_lm_url, model=cfg.embed_model),
                pending_texts,
            )
            if len(embs) != len(pending_rows):
                raise RuntimeError("Embeddings count mismatch vs pending rows.")
            for r, e in zip(pending_rows, embs):
                r["embedding"] = e
            rows_total += upsert_chunks(conn, pending_rows)

    print(
        f"Ingest complete: {rows_total} chunks upserted, {chunks_filtered_count} filtered",
        file=sys.stderr, flush=True,
    )
    print("", file=sys.stderr, flush=True)

    return {
        "domain_dir": str(domain_dir),
        "docs_loaded": docs_total,
        "files_skipped_or_unsupported": skipped,
        "chunks_cleared_first": int(cleared),
        "chunks_upserted": rows_total,
        "chunks_filtered": chunks_filtered_count,
        "embedding_dim": int(embedding_dim),
    }