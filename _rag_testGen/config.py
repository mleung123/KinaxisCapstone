# -*- coding: utf-8 -*-
from __future__ import annotations

# config.py - Loads all settings from environment into ResolvedConfig.
#
# Two-lane provider model:
#   Local lane:  LLM_PROVIDER=lmstudio, CONTEXT_MODEL, GENERATOR_MODEL, REVIEW_MODEL
#   API lane:    API_PROVIDER=anthropic|openai|gemini, API_MODEL
#
# Per-step routing (each can be "local" or "api"):
#   INGEST_PROVIDER    - who does knowledge extraction (default: local)
#   GENERATE_PROVIDER  - who generates MCQ items (default: local)
#   REVIEW_PROVIDER    - who reviews items (default: local)
#
# Embedding is always local (LM Studio).
# LLM_API_KEY is never written to config.env — entered at runtime only.

import os
import re
from dataclasses import dataclass, replace
from pathlib import Path


def _env(name):
    v = os.environ.get(name)
    if v is None:
        return None
    v = str(v).strip()
    return v if v else None


def _env_bool(name):
    v = _env(name)
    if v is None:
        return False
    return v.lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name):
    v = _env(name)
    if v is None:
        return None
    try:
        return int(v)
    except ValueError as e:
        raise ValueError("Invalid int for %s: %s" % (name, v)) from e


def _redact_dsn(dsn):
    return re.sub(r":([^:@/]+)@", ":***@", dsn)


@dataclass(frozen=True)
class ResolvedConfig:
    """Fully-resolved configuration loaded from environment variables."""

    rag_root: Path
    domain_dir: Path
    db_dsn: str

    # Local lane
    lm_url: str
    embed_model: str
    context_model: str      # lmstudio model for PPTX/DOCX/TXT text extraction
    generator_model: str    # lmstudio model for MCQ generation
    review_model: str       # lmstudio model for MCQ review

    # API lane
    api_provider: str       # anthropic | openai | gemini
    api_model: str          # model name on the API provider

    # Per-step routing
    ingest_provider: str    # "local" or "api"
    generate_provider: str  # "local" or "api"
    review_provider: str    # "local" or "api"

    # LLM_API_KEY is read from env only, never stored here

    n_items: int
    run_id: object          # str or None

    prompts_dir: Path
    out_dir: Path

    force_ingest: bool
    ingest_only: bool
    baseline_mode: bool

    embedding_dim: object   # int or None
    batch_size: int
    top_k: int
    sleep_seconds: float

    checkpoint_chunks: bool
    checkpoint_items: bool
    checkpoint_review: bool

    def with_overrides(self, **kwargs):
        norm = dict(kwargs)
        for path_field in ("rag_root", "domain_dir", "prompts_dir", "out_dir"):
            if path_field in norm and norm[path_field] is not None:
                norm[path_field] = Path(norm[path_field])
        return replace(self, **norm)

    def effective_ingest_provider(self):
        return self.api_provider if self.ingest_provider == "api" else "lmstudio"

    def effective_ingest_model(self):
        return self.api_model if self.ingest_provider == "api" else self.context_model

    def effective_generate_provider(self):
        return self.api_provider if self.generate_provider == "api" else "lmstudio"

    def effective_generate_model(self):
        return self.api_model if self.generate_provider == "api" else self.generator_model

    def effective_review_provider(self):
        return self.api_provider if self.review_provider == "api" else "lmstudio"

    def effective_review_model(self):
        return self.api_model if self.review_provider == "api" else self.review_model

    def startup_diagnostics(self):
        lines = [
            "rag_root=%s" % self.rag_root,
            "domain_dir=%s" % self.domain_dir,
            "db_dsn=%s" % _redact_dsn(self.db_dsn),
            "lm_url=%s" % self.lm_url,
            "\n--------------- local lane ---------------",
            "context_model=%s" % self.context_model,
            "generator_model=%s" % self.generator_model,
            "review_model=%s" % self.review_model,
            "embed_model=%s" % self.embed_model,
            "\n--------------- api lane ---------------",
            "api_provider=%s" % self.api_provider,
            "api_model=%s" % self.api_model,
            "\n--------------- routing ---------------",
            "ingest=%s -> %s/%s" % (self.ingest_provider, self.effective_ingest_provider(), self.effective_ingest_model()),
            "generate=%s -> %s/%s" % (self.generate_provider, self.effective_generate_provider(), self.effective_generate_model()),
            "review=%s -> %s/%s" % (self.review_provider, self.effective_review_provider(), self.effective_review_model()),
            "\n--------------- run ---------------",
            "n_items=%s" % self.n_items,
            "run_id=%s" % (self.run_id or ""),
            "prompts_dir=%s" % self.prompts_dir,
            "out_dir=%s" % self.out_dir,
            "force_ingest=%s" % self.force_ingest,
            "ingest_only=%s" % self.ingest_only,
            "baseline_mode=%s" % self.baseline_mode,
            "checkpoint_chunks=%s" % self.checkpoint_chunks,
            "checkpoint_items=%s" % self.checkpoint_items,
            "checkpoint_review=%s" % self.checkpoint_review,
        ]
        return "\n".join(lines)


def load_config_from_env():
    """Loads all configuration from environment variables."""
    rag_root = Path(_env("RAG_ROOT") or Path(__file__).resolve().parent).resolve()
    domain_dir = Path(_env("DOMAIN_DIR") or "").resolve() if _env("DOMAIN_DIR") else rag_root
    db_dsn = _env("DB_DSN") or ""
    lm_url = _env("LM_URL") or "http://localhost:1234"

    # Local lane
    embed_model = _env("EMBED_MODEL") or ""
    context_model = _env("CONTEXT_MODEL") or ""
    generator_model = _env("GENERATOR_MODEL") or _env("SME_MODEL") or ""
    review_model = _env("REVIEW_MODEL") or generator_model

    # API lane
    api_provider = (_env("API_PROVIDER") or "").lower()
    api_model = _env("API_MODEL") or ""

    # Per-step routing
    ingest_provider = (_env("INGEST_PROVIDER") or "local").lower()
    generate_provider = (_env("GENERATE_PROVIDER") or "local").lower()
    review_prov = (_env("REVIEW_PROVIDER") or "local").lower()

    for label, val in (("INGEST_PROVIDER", ingest_provider),
                       ("GENERATE_PROVIDER", generate_provider),
                       ("REVIEW_PROVIDER", review_prov)):
        if val not in ("local", "api"):
            raise SystemExit("Invalid %s=%r - must be 'local' or 'api'" % (label, val))

    n_items = _env_int("N_ITEMS") or 5
    run_id = _env("RUN_ID")
    prompts_dir = Path(_env("PROMPTS_DIR") or str(rag_root / "_prompts")).resolve()
    out_dir = Path(_env("OUT_DIR") or str(rag_root / "runs")).resolve()

    force_ingest = _env_bool("FORCE_INGEST")
    ingest_only = _env_bool("INGEST_ONLY")
    baseline_mode = _env_bool("BASELINE_MODE")

    embedding_dim = _env_int("EMBED_DIM")
    batch_size = _env_int("BATCH_SIZE") or 32
    top_k = _env_int("TOP_K") or 6
    sleep_seconds = float(_env("SLEEP_SECONDS") or "0.0")

    cp_chunks = _env("CHECKPOINT_CHUNKS")
    checkpoint_chunks = _env_bool("CHECKPOINT_CHUNKS") if cp_chunks is not None else True
    cp_items = _env("CHECKPOINT_ITEMS")
    checkpoint_items = _env_bool("CHECKPOINT_ITEMS") if cp_items is not None else True
    cp_review = _env("CHECKPOINT_REVIEW")
    checkpoint_review = _env_bool("CHECKPOINT_REVIEW") if cp_review is not None else True

    # Validation
    if not baseline_mode:
        if not db_dsn.strip():
            raise SystemExit("Missing required setting: DB_DSN")
        if not embed_model.strip():
            raise SystemExit("Missing required setting: EMBED_MODEL")

    needs_api = (ingest_provider == "api" or generate_provider == "api" or review_prov == "api")
    if needs_api:
        if not api_provider:
            raise SystemExit(
                "A step is set to 'api' but API_PROVIDER is not set.\n"
                "Set API_PROVIDER=anthropic (or openai/gemini) in settings."
            )
        if not api_model:
            raise SystemExit(
                "A step is set to 'api' but API_MODEL is not set.\n"
                "Set API_MODEL to the model name for your API provider."
            )
        api_key = (os.environ.get("LLM_API_KEY") or "").strip()
        if not api_key:
            print(
                "[WARNING] API step configured but LLM_API_KEY is not set. "
                "API calls will fail. Enter your key when prompted."
            )

    needs_local = (ingest_provider == "local" or generate_provider == "local" or review_prov == "local")
    if needs_local and not generator_model.strip():
        raise SystemExit("Missing required setting: GENERATOR_MODEL (needed for local steps)")

    if not baseline_mode and not context_model.strip():
        context_model = generator_model

    return ResolvedConfig(
        rag_root=rag_root,
        domain_dir=domain_dir,
        db_dsn=db_dsn,
        lm_url=lm_url,
        embed_model=embed_model,
        context_model=context_model,
        generator_model=generator_model,
        review_model=review_model,
        api_provider=api_provider,
        api_model=api_model,
        ingest_provider=ingest_provider,
        generate_provider=generate_provider,
        review_provider=review_prov,
        n_items=int(n_items),
        run_id=run_id,
        prompts_dir=prompts_dir,
        out_dir=out_dir,
        force_ingest=bool(force_ingest),
        ingest_only=bool(ingest_only),
        baseline_mode=bool(baseline_mode),
        embedding_dim=embedding_dim,
        batch_size=int(batch_size),
        top_k=int(top_k),
        sleep_seconds=float(sleep_seconds),
        checkpoint_chunks=bool(checkpoint_chunks),
        checkpoint_items=bool(checkpoint_items),
        checkpoint_review=bool(checkpoint_review),
    )
