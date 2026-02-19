from __future__ import annotations

import json
import os
import time
import psycopg
import sys

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from openpyxl import Workbook

from embed_lmstudio import EmbedConfig, embed_texts
from ingest import IngestConfig, ingest_domain
from lmstudio_client import call_llm

from db_pgvector import (
    chunks_rowcount,
    clear_chunks,
    ensure_schema,
    get_db_snapshot_per_doc,
    get_db_snapshot_summary,
    get_random_chunks,
    similarity_search,
)

from text_utils import (
    clean_generator_text,
    enforce_hygiene_on_review,
    extract_first_json_obj,
    hard_trim_after_difficulty,
    validate_generator_schema,
)

# ---------------------------------------------------------------------
# Config dataclasses (CLI/env parsing stays in cli.py/config.py)
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class GenerateConfig:
    """note: Parameters for generate-only runs that read existing embedded chunks from Postgres+pgvector."""

    db_dsn: str
    lm_url: str
    embed_model: str
    sme_model: str
    review_model: str
    n_items: int
    run_id: str
    prompts_dir: Path
    out_dir: Path
    top_k: int = 6
    temperature_gen: float = 0.2
    temperature_review: float = 0.0
    max_tokens_gen: int = 700
    max_tokens_review: int = 700
    request_timeout_seconds: int = 120
    sleep_seconds: float = 0.0


@dataclass(frozen=True)
class PipelineConfig:
    """note: Flat pipeline config matching cli.py construction for the 'pipeline' command."""

    db_dsn: str
    domain_dir: Path
    lm_url: str
    embed_model: str
    embedding_dim: int | None = None
    batch_size: int = 32
    chunk_chars: int = 1600
    overlap_chars: int = 200
    clear_first: bool = False
    force_ingest: bool = False
    n_items: int = 5
    sme_model: str = ""
    review_model: str = ""
    run_id: str = ""
    prompts_dir: Path = Path("_prompts")
    out_dir: Path = Path("runs")
    top_k: int = 6
    sleep_seconds: float = 0.0


# ---------------------------------------------------------------------
# Excel writer
# ---------------------------------------------------------------------


def _parse_item_fields(gen_text: str) -> dict[str, str]:
    """note: Extracts question/choices/correct_key/difficulty from the clean generator text using simple label heuristics."""
    t = gen_text or ""
    lines = [ln.rstrip() for ln in t.splitlines()]

    def _grab_after(prefix: str) -> str:
        pref = prefix.lower()
        for ln in lines:
            if ln.lower().startswith(pref):
                return ln.split(":", 1)[1].strip() if ":" in ln else ""
        return ""

    out: dict[str, str] = {}
    out["question"] = _grab_after("question")
    for opt in ["a)", "b)", "c)", "d)"]:
        val = ""
        for ln in lines:
            if ln.lower().startswith(opt):
                val = ln.split(")", 1)[1].strip() if ")" in ln else ""
                break
        out[opt[0]] = val
    out["correct_key"] = _grab_after("correct_key") or _grab_after("correct key")
    out["difficulty"] = _grab_after("difficulty").lower()
    return out


def _xlsx_write_sheet(ws, headers: list[str], rows: Iterable[list[Any]]) -> None:
    """note: Writes headers + rows to an openpyxl worksheet in a simple, predictable way."""
    ws.append(headers)
    for r in rows:
        ws.append([("" if v is None else v) for v in r])


def write_run_xlsx(
    out_dir: Path,
    run_id: str,
    items_rows: list[dict[str, Any]],
    decisions_rows: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    db_snapshot_summary: dict[str, Any] | None = None,
    db_snapshot_per_doc: list[dict[str, Any]] | None = None,
) -> Path:
    """note: Creates a single XLSX per run with normalized sheets and returns the written path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    xlsx_path = out_dir / f"run_{run_id}.xlsx"

    wb = Workbook()

    ws_items = wb.active
    ws_items.title = "Items"
    items_headers = [
        "run_id", "item_id", "question", "a", "b", "c", "d",
        "correct_key", "difficulty", "decision", "schema_ok",
        "schema_violations", "gen_text_clean",
    ]
    items_data: list[list[Any]] = []
    for r in items_rows:
        items_data.append([
            r.get("run_id"), r.get("item_id"), r.get("question"),
            r.get("a"), r.get("b"), r.get("c"), r.get("d"),
            r.get("correct_key"), r.get("difficulty"), r.get("decision"),
            r.get("schema_ok"), r.get("schema_violations"), r.get("gen_text_clean"),
        ])
    _xlsx_write_sheet(ws_items, items_headers, items_data)

    ws_rev = wb.create_sheet("Reviewer Decisions")
    rev_headers = [
        "run_id", "item_id", "decision", "failure_layer",
        "reason_codes", "revision_instructions", "reviewer_parse_ok",
    ]
    rev_data: list[list[Any]] = []
    for r in decisions_rows:
        rev_data.append([
            r.get("run_id"), r.get("item_id"), r.get("decision"),
            r.get("failure_layer"),
            json.dumps(r.get("reason_codes", []), ensure_ascii=False),
            r.get("revision_instructions"), r.get("reviewer_parse_ok"),
        ])
    _xlsx_write_sheet(ws_rev, rev_headers, rev_data)

    ws_trace = wb.create_sheet("Traceability")
    trace_headers = ["run_id", "item_id", "doc_path", "chunk_index", "distance", "chunk_text"]
    trace_data: list[list[Any]] = []
    for r in trace_rows:
        trace_data.append([
            r.get("run_id"), r.get("item_id"), r.get("doc_path"),
            r.get("chunk_index"), r.get("distance"), r.get("chunk_text"),
        ])
    _xlsx_write_sheet(ws_trace, trace_headers, trace_data)

    ws_meta = wb.create_sheet("Run Metadata")
    meta_headers = ["key", "value"]
    meta_rows: list[list[Any]] = []
    for k in sorted(metadata.keys()):
        v = metadata[k]
        if isinstance(v, (dict, list)):
            v = json.dumps(v, ensure_ascii=False)
        meta_rows.append([k, v])
    _xlsx_write_sheet(ws_meta, meta_headers, meta_rows)

    # ---- DB Snapshot sheet ----
    ws_snap = wb.create_sheet("DB Snapshot")
    ws_snap.append(["--- Summary ---", ""])
    for k, v in sorted((db_snapshot_summary or {}).items()):
        ws_snap.append([k, ("" if v is None else v)])
    ws_snap.append(["", ""])
    ws_snap.append(["--- Per-Document Inventory ---", ""])
    per_doc_headers = [
        "doc_path", "chunk_count", "doc_sha256",
        "first_created_at", "last_updated_at",
    ]
    ws_snap.append(per_doc_headers)
    for row in (db_snapshot_per_doc or []):
        ws_snap.append([
            row.get("doc_path", ""),
            row.get("chunk_count", ""),
            row.get("doc_sha256", ""),
            row.get("first_created_at", "") or "",
            row.get("last_updated_at", "") or "",
        ])

    wb.save(xlsx_path)
    return xlsx_path


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _redact_dsn(dsn: str) -> str:
    """note: Redacts password in a Postgres DSN for logging/metadata."""
    if not dsn:
        return ""
    # crude but safe: hide anything between : and @ in a URL-like DSN
    try:
        if "://" in dsn and "@" in dsn:
            left, right = dsn.split("://", 1)
            creds_host = right
            if "@" in creds_host and ":" in creds_host.split("@", 1)[0]:
                creds, host = creds_host.split("@", 1)
                user = creds.split(":", 1)[0]
                return f"{left}://{user}:***@{host}"
    except Exception:
        pass
    return dsn


def _infer_embedding_dim_from_db(conn: psycopg.Connection) -> int:
    """note: Infers embedding dimension from an existing rag_chunks row (uses embedding::text to avoid type adapter issues)."""
    with conn.cursor() as cur:
        cur.execute("SELECT embedding::text FROM rag_chunks LIMIT 1;")
        row = cur.fetchone()
    if not row or not row[0]:
        raise RuntimeError("Cannot infer embedding_dim: rag_chunks is empty.")
    s = str(row[0]).strip()
    # expected like: [0.1,0.2,...]
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if not inner:
            raise RuntimeError("Cannot infer embedding_dim: empty embedding literal.")
        return len([x for x in inner.split(",") if x.strip() != ""])
    raise RuntimeError(f"Cannot infer embedding_dim from embedding::text: {s[:80]}")


# ---------------------------------------------------------------------
# Generation (retrieval + agentic review)
# ---------------------------------------------------------------------


def generate_from_db(cfg: GenerateConfig) -> dict[str, Any]:
    """note: Generates items by retrieving top-k chunks from pgvector, running generator+reviewer prompts, and writing one XLSX artifact."""
    prompts_dir = Path(cfg.prompts_dir)
    generator_system = (prompts_dir / "generator_system.txt").read_text(encoding="utf-8")
    generator_user_template = (prompts_dir / "generator_user.txt").read_text(encoding="utf-8")
    reviewer_system = (prompts_dir / "reviewer_system.txt").read_text(encoding="utf-8")
    reviewer_user_template = (prompts_dir / "reviewer_user.txt").read_text(encoding="utf-8")

    keep_raw_csv = (os.environ.get("KEEP_RAW_CSV") or "").strip() in {"1", "true", "TRUE", "yes", "YES"}

    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": created_at,
        "config": {
            "db_dsn_redacted": _redact_dsn(cfg.db_dsn),
            "lm_url": cfg.lm_url,
            "embed_model": cfg.embed_model,
            "sme_model": cfg.sme_model,
            "review_model": cfg.review_model,
            "n_items": int(cfg.n_items),
            "run_id": cfg.run_id,
            "prompts_dir": str(cfg.prompts_dir),
            "out_dir": str(cfg.out_dir),
            "top_k": int(cfg.top_k),
        },
        "prompt_files": {
            "generator_system": "generator_system.txt",
            "generator_user": "generator_user.txt",
            "reviewer_system": "reviewer_system.txt",
            "reviewer_user": "reviewer_user.txt",
        },
    }
    manifest_path = out_dir / f"run_manifest_{cfg.run_id}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    items_rows: list[dict[str, Any]] = []
    decisions_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []

    schema_ok_count = 0
    reviewer_json_ok = 0
    decisions_count: dict[str, int] = {}
    
    db_snap_summary: dict[str, Any] | None = None
    db_snap_per_doc: list[dict[str, Any]] | None = None

    with psycopg.connect(cfg.db_dsn) as conn:
        if chunks_rowcount(conn) <= 0:
            raise RuntimeError("DB has 0 chunks. Run 'ingest' (or 'pipeline') first.")

        embedding_dim = _infer_embedding_dim_from_db(conn)
        ensure_schema(conn, embedding_dim)

        for i in range(int(cfg.n_items)):
            item_id = f"item_{i+1}"
            print(f"Currently processing... {item_id}", file=sys.stderr, flush=True)

            seed_rows = get_random_chunks(conn, n=1)
            seed_text = seed_rows[0]["chunk_text"]
            seed_doc = seed_rows[0].get("doc_path", "")

            seed_emb = embed_texts(
                EmbedConfig(lm_url=cfg.lm_url, model=cfg.embed_model),
                [seed_text],
            )[0]

            retrieved = similarity_search(conn, seed_emb, int(cfg.top_k))

            for r in retrieved:
                trace_rows.append(
                    {
                        "run_id": cfg.run_id,
                        "item_id": item_id,
                        "doc_path": r.get("doc_path"),
                        "chunk_index": r.get("chunk_index"),
                        "distance": r.get("distance"),
                        "chunk_text": r.get("chunk_text"),
                    }
                )

            context_block = "\n\n".join(f"[{j+1}] {r['chunk_text']}" for j, r in enumerate(retrieved))
            generator_user = generator_user_template.replace("{{CONTEXT}}", context_block)

            gen_raw = call_llm(
                lm_url=cfg.lm_url,
                model=cfg.sme_model,
                system_prompt=generator_system,
                user_prompt=generator_user,
                temperature=cfg.temperature_gen,
                max_tokens=cfg.max_tokens_gen,
                request_timeout_seconds=cfg.request_timeout_seconds,
            )

            gen_text = hard_trim_after_difficulty(clean_generator_text(gen_raw))
            schema_ok, violations = validate_generator_schema(gen_text)
            if schema_ok:
                schema_ok_count += 1

            parsed_fields = _parse_item_fields(gen_text)

            reviewer_user = reviewer_user_template.replace("{{GEN_ITEM}}", gen_text)
            rev_raw = call_llm(
                lm_url=cfg.lm_url,
                model=cfg.review_model,
                system_prompt=reviewer_system,
                user_prompt=reviewer_user,
                temperature=cfg.temperature_review,
                max_tokens=cfg.max_tokens_review,
                request_timeout_seconds=cfg.request_timeout_seconds,
            )

            rev_json = extract_first_json_obj(rev_raw) or {}
            rev_clean = enforce_hygiene_on_review(rev_json)
            if rev_clean.get("reviewer_parse_ok"):
                reviewer_json_ok += 1

            decisions_count[rev_clean.get("decision", "")] = decisions_count.get(rev_clean.get("decision", ""), 0) + 1

            items_rows.append(
                {
                    "run_id": cfg.run_id,
                    "item_id": item_id,
                    "question": parsed_fields.get("question", ""),
                    "a": parsed_fields.get("a", ""),
                    "b": parsed_fields.get("b", ""),
                    "c": parsed_fields.get("c", ""),
                    "d": parsed_fields.get("d", ""),
                    "correct_key": parsed_fields.get("correct_key", ""),
                    "difficulty": parsed_fields.get("difficulty", ""),
                    "decision": rev_clean.get("decision", ""),
                    "schema_ok": bool(schema_ok),
                    "schema_violations": "|".join(violations),
                    "gen_text_clean": gen_text,
                    "seed_doc_path": seed_doc,
                }
            )

            decisions_rows.append(
                {
                    "run_id": cfg.run_id,
                    "item_id": item_id,
                    "decision": rev_clean.get("decision", ""),
                    "failure_layer": rev_clean.get("failure_layer", ""),
                    "reason_codes": rev_clean.get("reason_codes", []),
                    "revision_instructions": rev_clean.get("revision_instructions", ""),
                    "reviewer_parse_ok": bool(rev_clean.get("reviewer_parse_ok", False)),
                }
            )          

            if cfg.sleep_seconds:
                time.sleep(float(cfg.sleep_seconds))
                
        db_snap_summary = get_db_snapshot_summary(conn)
        db_snap_per_doc = get_db_snapshot_per_doc(conn)

    meta = {
        "created_at": created_at,
        "run_id": cfg.run_id,
        "lm_url": cfg.lm_url,
        "embed_model": cfg.embed_model,
        "sme_model": cfg.sme_model,
        "review_model": cfg.review_model,
        "n_items": int(cfg.n_items),
        "top_k": int(cfg.top_k),
        "db_dsn_redacted": _redact_dsn(cfg.db_dsn),
        "prompts_dir": str(cfg.prompts_dir),
        "out_dir": str(cfg.out_dir),
        "keep_raw_csv": bool(keep_raw_csv),
    }

    xlsx_path = write_run_xlsx(
        out_dir=Path(cfg.out_dir),
        run_id=cfg.run_id,
        items_rows=items_rows,
        decisions_rows=decisions_rows,
        trace_rows=trace_rows,
        metadata=meta,
        db_snapshot_summary=db_snap_summary,
        db_snapshot_per_doc=db_snap_per_doc,
    )

    return {
        "run_id": cfg.run_id,
        "out_dir": str(cfg.out_dir),
        "items_total": int(cfg.n_items),
        "items_schema_ok": int(schema_ok_count),
        "reviewer_json_ok": int(reviewer_json_ok),
        "decisions": decisions_count,
        "files": {
            "xlsx": str(xlsx_path),
            "manifest_json": str(manifest_path),
        },
    }


def run_pipeline(cfg: PipelineConfig) -> dict[str, Any]:
    """note: Orchestrates ingest-if-needed and then generate; avoids re-ingesting unless forced."""
    ingest_ran = False
    ingest_summary: dict[str, Any] | None = None

    with psycopg.connect(cfg.db_dsn) as conn:
        has_chunks = chunks_rowcount(conn) > 0

    if cfg.force_ingest or not has_chunks:
        ingest_ran = True
        ingest_cfg = IngestConfig(
            domain_dir=Path(cfg.domain_dir),
            db_dsn=cfg.db_dsn,
            embed_lm_url=cfg.lm_url,
            embed_model=cfg.embed_model,
            embedding_dim=int(cfg.embedding_dim) if cfg.embedding_dim is not None else None,
            batch_size=int(cfg.batch_size),
            chunk_chars=int(cfg.chunk_chars),
            overlap_chars=int(cfg.overlap_chars),
            clear_first=bool(cfg.clear_first),
        )
        ingest_summary = ingest_domain(ingest_cfg)

    gen_cfg = GenerateConfig(
        db_dsn=cfg.db_dsn,
        lm_url=cfg.lm_url,
        embed_model=cfg.embed_model,
        sme_model=cfg.sme_model,
        review_model=cfg.review_model,
        n_items=int(cfg.n_items),
        run_id=cfg.run_id,
        prompts_dir=Path(cfg.prompts_dir),
        out_dir=Path(cfg.out_dir),
        top_k=int(cfg.top_k),
        sleep_seconds=float(cfg.sleep_seconds),
    )

    generate_summary = generate_from_db(gen_cfg)

    return {
        "ingest_ran": ingest_ran,
        "ingest_summary": ingest_summary,
        "generate_summary": generate_summary,
    }
