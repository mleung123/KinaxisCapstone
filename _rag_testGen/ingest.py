# -*- coding: utf-8 -*-
from __future__ import annotations

# ingest.py - Knowledge extraction and embedding pipeline.
#
# Flow per document:
#   1. load_document() -> full preprocessed text
#   2. Extract knowledge chunks:
#      - INGEST_PROVIDER=api + PDF: send raw PDF natively to API (one call, no rendering)
#      - INGEST_PROVIDER=api + PPTX/DOCX/TXT: send text to API
#      - INGEST_PROVIDER=local + PDF: render pages, send to lmstudio vision in batches
#      - INGEST_PROVIDER=local + PPTX/DOCX/TXT: send text to lmstudio
#   3. Split LLM output into knowledge chunks
#   4. embed_texts() -> vector per chunk (always local)
#   5. upsert_chunks() -> pgvector

import hashlib
import os
import re
import sys
import time
import psycopg

from dataclasses import dataclass
from pathlib import Path

from db_pgvector import clear_chunks, ensure_schema, set_meta_if_absent, set_meta, upsert_chunks
from embed_lmstudio import EmbedConfig, embed_texts
from llm_client import call_llm, call_llm_vision, render_pdf_pages_b64, validate_provider_and_key
from loaders import load_document


@dataclass(frozen=True)
class IngestConfig:
    domain_dir: Path
    db_dsn: str
    embed_lm_url: str
    embed_model: str
    lm_url: str                    # LM Studio URL for local inference
    context_model: str             # local model for text/vision extraction
    api_provider: str              # anthropic | openai | gemini
    api_model: str                 # API model name
    ingest_provider: str           # "local" or "api"
    embedding_dim: object = None   # int or None
    batch_size: int = 32
    clear_first: bool = False
    context_temperature: float = 0.0
    context_max_tokens: int = 2000
    context_timeout_seconds: int = 600
    vision_timeout_seconds: int = 600
    render_dpi: int = 96
    vision_pages_per_batch: int = 4
    min_chunk_chars: int = 200
    max_chunk_chars: int = 1600
    ingest_delay_seconds: float = 0.0


_WS_RE = re.compile(r"\s+")


def _chunk_id(doc_sha256, chunk_text):
    norm = _WS_RE.sub(" ", (chunk_text or "").strip())
    h = hashlib.sha256()
    h.update((doc_sha256 + "\n" + norm).encode("utf-8"))
    return h.hexdigest()


def _split_knowledge_output(llm_output, min_chars, max_chars):
    if not llm_output or not llm_output.strip():
        return []

    raw_paragraphs = [p.strip() for p in re.split(r"\n\s*\n", llm_output) if p.strip()]

    merged = []
    buf = ""
    for para in raw_paragraphs:
        if not buf:
            buf = para
            continue
        candidate = buf + "\n\n" + para
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            if len(buf) >= min_chars:
                merged.append(buf)
            else:
                buf = candidate
                continue
            buf = para

    if buf.strip():
        if len(buf) < min_chars and merged:
            merged[-1] = merged[-1] + "\n\n" + buf
        else:
            merged.append(buf)

    final = []
    for chunk in merged:
        if len(chunk) <= max_chars:
            final.append(chunk)
        else:
            sentences = re.split(r"(?<=[.!?])\s+", chunk)
            current = ""
            for sent in sentences:
                if not current:
                    current = sent
                elif len(current) + 1 + len(sent) <= max_chars:
                    current = current + " " + sent
                else:
                    if current:
                        final.append(current)
                    current = sent
            if current:
                final.append(current)

    return [c for c in final if c.strip()]


def _load_context_prompts(prompts_dir):
    system_path = Path(prompts_dir) / "context_system.txt"
    user_path = Path(prompts_dir) / "context_user.txt"
    if not system_path.exists():
        raise RuntimeError("Missing prompt file: %s" % system_path)
    if not user_path.exists():
        raise RuntimeError("Missing prompt file: %s" % user_path)
    return (
        system_path.read_text(encoding="utf-8"),
        user_path.read_text(encoding="utf-8"),
    )


def _is_pdf(doc):
    return Path(str(doc.path)).suffix.lower() == ".pdf"


_VISION_USER_PROMPT = (
    "Extract the core knowledge from these slides as structured prose paragraphs. "
    "Each paragraph should capture one coherent concept, principle, equation, or finding. "
    "Separate paragraphs with a blank line. "
    "For mathematical content, describe equations and their meaning in plain language "
    "as well as preserving the symbolic notation. "
    "Write only the knowledge paragraphs - no headers, no lists, no commentary."
)


def _extract_api_pdf(doc, cfg, context_system):
    """Sends raw PDF to API provider natively (Anthropic) or as rendered images (OpenAI/Gemini)."""
    provider = cfg.api_provider
    model = cfg.api_model
    print(
        "    [%s/%s] Sending PDF natively..." % (provider, model),
        file=sys.stderr, flush=True,
    )
    t0 = time.perf_counter()
    result = call_llm_vision(
        lm_url=cfg.lm_url,
        model=model,
        system_prompt=context_system,
        user_prompt=_VISION_USER_PROMPT,
        pdf_path=doc.path,
        temperature=cfg.context_temperature,
        max_tokens=cfg.context_max_tokens,
        request_timeout_seconds=cfg.vision_timeout_seconds,
        provider=provider,
    )
    elapsed = time.perf_counter() - t0
    print(
        "    [%s/%s] Done in %.1fs" % (provider, model, elapsed),
        file=sys.stderr, flush=True,
    )
    return result


def _extract_api_text(doc, cfg, context_system, context_user_template):
    """Sends text to API provider for knowledge extraction."""
    provider = cfg.api_provider
    model = cfg.api_model
    print(
        "    [%s/%s] Extracting via text..." % (provider, model),
        file=sys.stderr, flush=True,
    )
    t0 = time.perf_counter()
    user_prompt = context_user_template.replace("{{DOCUMENT}}", doc.text)
    result = call_llm(
        lm_url=cfg.lm_url,
        model=model,
        system_prompt=context_system,
        user_prompt=user_prompt,
        temperature=cfg.context_temperature,
        max_tokens=cfg.context_max_tokens,
        request_timeout_seconds=cfg.context_timeout_seconds,
        provider=provider,
    )
    elapsed = time.perf_counter() - t0
    print(
        "    [%s/%s] Done in %.1fs" % (provider, model, elapsed),
        file=sys.stderr, flush=True,
    )
    return result


def _extract_local_pdf(doc, cfg, context_system):
    """Renders PDF pages, sends in batches to LM Studio vision endpoint."""
    model = cfg.context_model
    pages_per_batch = max(1, int(cfg.vision_pages_per_batch))
    print(
        "    [lmstudio/%s] Rendering %d pages at %d DPI in batches of %d..."
        % (model, doc.page_count, cfg.render_dpi, pages_per_batch),
        file=sys.stderr, flush=True,
    )
    all_pages = render_pdf_pages_b64(doc.path, dpi=cfg.render_dpi)
    n_pages = len(all_pages)
    outputs = []
    import datetime as _dt
    for batch_start in range(0, n_pages, pages_per_batch):
        batch = all_pages[batch_start: batch_start + pages_per_batch]
        batch_end = min(batch_start + pages_per_batch, n_pages)
        print(
            "    [%s] Vision batch pages %d-%d of %d..."
            % (_dt.datetime.now().strftime("%H:%M:%S"), batch_start + 1, batch_end, n_pages),
            file=sys.stderr, flush=True,
        )
        t0 = time.perf_counter()
        out = call_llm_vision(
            lm_url=cfg.lm_url,
            model=model,
            system_prompt=context_system,
            user_prompt=_VISION_USER_PROMPT,
            image_b64_list=batch,
            temperature=cfg.context_temperature,
            max_tokens=cfg.context_max_tokens,
            request_timeout_seconds=cfg.vision_timeout_seconds,
            provider="lmstudio",
        )
        elapsed = time.perf_counter() - t0
        print(
            "    [%s] Batch done in %.1fs" % (_dt.datetime.now().strftime("%H:%M:%S"), elapsed),
            file=sys.stderr, flush=True,
        )
        if out and out.strip():
            outputs.append(out.strip())
    return "\n\n".join(outputs)


def _extract_local_text(doc, cfg, context_system, context_user_template):
    """Sends text to LM Studio context model for knowledge extraction."""
    model = cfg.context_model
    print(
        "    [lmstudio/%s] Extracting via text..." % model,
        file=sys.stderr, flush=True,
    )
    t0 = time.perf_counter()
    user_prompt = context_user_template.replace("{{DOCUMENT}}", doc.text)
    result = call_llm(
        lm_url=cfg.lm_url,
        model=model,
        system_prompt=context_system,
        user_prompt=user_prompt,
        temperature=cfg.context_temperature,
        max_tokens=cfg.context_max_tokens,
        request_timeout_seconds=cfg.context_timeout_seconds,
        provider="lmstudio",
    )
    elapsed = time.perf_counter() - t0
    print(
        "    [lmstudio/%s] Done in %.1fs" % (model, elapsed),
        file=sys.stderr, flush=True,
    )
    return result


def extract_knowledge_chunks(doc, cfg, context_system, context_user_template):
    """Routes extraction based on ingest_provider and file type."""
    is_pdf = _is_pdf(doc)
    provider = cfg.ingest_provider  # "local" or "api"

    print(
        "    Extracting knowledge from %s (%d chars, %d pages/slides, provider=%s, mode=%s)..."
        % (doc.path.name, len(doc.text), doc.page_count,
           provider, "vision" if is_pdf else "text"),
        file=sys.stderr, flush=True,
    )

    if provider == "api":
        if is_pdf:
            llm_output = _extract_api_pdf(doc, cfg, context_system)
        else:
            llm_output = _extract_api_text(doc, cfg, context_system, context_user_template)
    else:
        # local
        if is_pdf:
            llm_output = _extract_local_pdf(doc, cfg, context_system)
        else:
            llm_output = _extract_local_text(doc, cfg, context_system, context_user_template)

    if not llm_output or not llm_output.strip():
        print(
            "    [WARNING] Context model returned empty output for %s" % doc.path.name,
            file=sys.stderr, flush=True,
        )
        return []

    chunks = _split_knowledge_output(
        llm_output,
        min_chars=cfg.min_chunk_chars,
        max_chars=cfg.max_chunk_chars,
    )

    print(
        "    -> %d knowledge chunks extracted" % len(chunks),
        file=sys.stderr, flush=True,
    )
    return chunks


def iter_domain_files(domain_dir):
    for p in sorted(Path(domain_dir).rglob("*")):
        if p.is_file():
            yield p


def _infer_embedding_dim(embed_lm_url, embed_model):
    embs = embed_texts(EmbedConfig(lm_url=embed_lm_url, model=embed_model), ["dimension probe"])
    if not embs or not embs[0]:
        raise RuntimeError("Embedding dimension probe failed.")
    return int(len(embs[0]))


def ingest_domain(cfg, prompts_dir):
    domain_dir = Path(cfg.domain_dir).resolve()
    if not domain_dir.exists():
        raise RuntimeError("domain_dir not found: %s" % domain_dir)

    context_system, context_user_template = _load_context_prompts(prompts_dir)

    # Validate API key up front if using API for ingest
    if cfg.ingest_provider == "api":
        api_key = (os.environ.get("LLM_API_KEY") or "").strip()
        validate_provider_and_key(cfg.api_provider, api_key, context="ingest")

    loaded = []
    skipped = 0
    for p in iter_domain_files(domain_dir):
        doc = load_document(p)
        if doc is None:
            skipped += 1
            continue
        loaded.append(doc)

    docs_total = len(loaded)
    print("Ingesting %d docs from %s" % (docs_total, domain_dir), file=sys.stderr, flush=True)

    embedding_dim = cfg.embedding_dim
    if embedding_dim is None:
        embedding_dim = _infer_embedding_dim(cfg.embed_lm_url, cfg.embed_model)

    rows_total = 0
    chunks_total = 0

    with psycopg.connect(cfg.db_dsn) as conn:
        ensure_schema(conn, int(embedding_dim))
        _set_meta = set_meta if cfg.clear_first else set_meta_if_absent
        _set_meta(conn, "embedding_dim", str(int(embedding_dim)))
        _set_meta(conn, "embed_model", str(cfg.embed_model))
        _set_meta(conn, "context_model", str(cfg.context_model))
        _set_meta(conn, "ingest_provider", str(cfg.ingest_provider))
        _set_meta(conn, "api_model", str(cfg.api_model))
        _set_meta(conn, "source_root", str(domain_dir))
        _set_meta(conn, "batch_size", str(int(cfg.batch_size)))

        cleared = clear_chunks(conn) if cfg.clear_first else 0

        pending_texts = []
        pending_rows = []

        def _flush_batch():
            nonlocal rows_total
            if not pending_texts:
                return
            embs = embed_texts(
                EmbedConfig(lm_url=cfg.embed_lm_url, model=cfg.embed_model),
                pending_texts,
            )
            if not embs:
                raise RuntimeError("Embeddings call returned no embeddings.")
            if len(embs) != len(pending_rows):
                raise RuntimeError(
                    "Embeddings count mismatch: got %d, expected %d"
                    % (len(embs), len(pending_rows))
                )
            for r, e in zip(pending_rows, embs):
                r["embedding"] = e
            rows_total += upsert_chunks(conn, pending_rows)
            print("  Embedded %d chunks so far..." % rows_total, file=sys.stderr, flush=True)
            pending_texts.clear()
            pending_rows.clear()

        for i, doc in enumerate(loaded):
            print("  Processing %s ..." % doc.path.name, file=sys.stderr, flush=True)

            knowledge_chunks = extract_knowledge_chunks(
                doc, cfg, context_system, context_user_template
            )
            chunks_total += len(knowledge_chunks)

            for chunk_index, chunk_text in enumerate(knowledge_chunks):
                pending_texts.append(chunk_text)
                pending_rows.append({
                    "doc_path": str(doc.path),
                    "doc_sha256": doc.sha256,
                    "chunk_index": chunk_index,
                    "chunk_text": chunk_text,
                    "embedding": None,
                    "meta": {
                        "source_root": str(domain_dir),
                        "rel_path": str(doc.path.resolve().relative_to(domain_dir)),
                        "chunk_id": _chunk_id(doc.sha256, chunk_text),
                        "page_count": doc.page_count,
                        "extraction_method": "vision" if _is_pdf(doc) else "text",
                        "ingest_provider": cfg.ingest_provider,
                    },
                })
                if len(pending_texts) >= int(cfg.batch_size):
                    _flush_batch()

            # Delay between documents to avoid API rate limits
            if cfg.ingest_delay_seconds > 0 and i < len(loaded) - 1:
                print(
                    "    Waiting %.0fs before next document..." % cfg.ingest_delay_seconds,
                    file=sys.stderr, flush=True,
                )
                time.sleep(cfg.ingest_delay_seconds)

        _flush_batch()

    print(
        "Ingest complete: %d chunks upserted from %d docs"
        % (rows_total, docs_total),
        file=sys.stderr, flush=True,
    )

    return {
        "domain_dir": str(domain_dir),
        "docs_loaded": docs_total,
        "files_skipped_or_unsupported": skipped,
        "chunks_cleared_first": int(cleared),
        "knowledge_chunks_extracted": chunks_total,
        "chunks_upserted": rows_total,
        "embedding_dim": int(embedding_dim),
    }
