from __future__ import annotations

import os
import re
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path


def _env(name: str) -> str | None:
    """note: Reads an environment variable and normalizes empty strings to None."""
    v = os.environ.get(name)
    if v is None:
        return None
    v = str(v).strip()
    return v if v else None


def _env_bool(name: str) -> bool:
    """note: Parses a boolean-ish environment variable (1/true/yes/on) into a bool."""
    v = _env(name)
    if v is None:
        return False
    return v.lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str) -> int | None:
    """note: Parses an integer environment variable into int, returning None if missing/blank."""
    v = _env(name)
    if v is None:
        return None
    try:
        return int(v)
    except ValueError as e:
        raise ValueError(f"Invalid int for {name}: {v}") from e


def _redact_dsn(dsn: str) -> str:
    """note: Redacts password-like segments in a DSN for safe logging."""
    return re.sub(r":([^:@/]+)@", ":***@", dsn)


@dataclass(frozen=True)
class ResolvedConfig:
    """note: Holds fully-resolved env-first configuration (with optional CLI overrides applied)."""

    rag_root: Path
    domain_dir: Path
    db_dsn: str

    lm_url: str
    embed_model: str
    sme_model: str
    review_model: str

    n_items: int
    run_id: str | None

    prompts_dir: Path
    out_dir: Path

    force_ingest: bool

    embedding_dim: int | None
    batch_size: int
    chunk_chars: int
    overlap_chars: int

    top_k: int
    sleep_seconds: float

    def with_overrides(self, **kwargs) -> "ResolvedConfig":
        """note: Returns a copy of the config with the provided keyword overrides applied."""
        norm = dict(kwargs)
        if "rag_root" in norm and norm["rag_root"] is not None:
            norm["rag_root"] = Path(norm["rag_root"])
        if "domain_dir" in norm and norm["domain_dir"] is not None:
            norm["domain_dir"] = Path(norm["domain_dir"])
        if "prompts_dir" in norm and norm["prompts_dir"] is not None:
            norm["prompts_dir"] = Path(norm["prompts_dir"])
        if "out_dir" in norm and norm["out_dir"] is not None:
            norm["out_dir"] = Path(norm["out_dir"])
        return replace(self, **norm)

    def startup_diagnostics(self) -> str:
        """note: Returns concise startup diagnostics with sensitive values redacted."""
        lines = []
        lines.append(f"rag_root={self.rag_root}")
        lines.append(f"domain_dir={self.domain_dir}")
        lines.append(f"db_dsn={_redact_dsn(self.db_dsn)}")
        lines.append(f"lm_url={self.lm_url}")
        lines.append(f"embed_model={self.embed_model}")
        lines.append(f"sme_model={self.sme_model}")
        lines.append(f"review_model={self.review_model}")
        lines.append(f"n_items={self.n_items}")
        lines.append(f"run_id={(self.run_id or '')}")
        lines.append(f"prompts_dir={self.prompts_dir}")
        lines.append(f"out_dir={self.out_dir}")
        lines.append(f"force_ingest={self.force_ingest}")
        return "\n".join(lines)


def load_config_from_env() -> ResolvedConfig:
    """note: Loads env-first configuration with backward-compatible aliases for older BAT vars."""
    rag_root = Path(_env("RAG_ROOT") or Path(__file__).resolve().parent).resolve()

    domain_dir = Path(_env("DOMAIN_DIR") or "").resolve() if _env("DOMAIN_DIR") else rag_root
    db_dsn = _env("DB_DSN") or ""

    lm_url = _env("LM_URL") or "http://localhost:1234"

    embed_model = _env("EMBED_MODEL") or ""
    sme_model = _env("SME_MODEL") or ""
    review_model = _env("REVIEW_MODEL") or sme_model

    n_items = _env_int("N_ITEMS") or 5
    run_id = _env("RUN_ID")

    prompts_dir = Path(_env("PROMPTS_DIR") or (rag_root / "_prompts")).resolve()
    out_dir = Path(_env("OUT_DIR") or (rag_root / "runs")).resolve()

    force_ingest = _env_bool("FORCE_INGEST")

    embedding_dim = _env_int("EMBED_DIM")
    batch_size = _env_int("BATCH_SIZE") or 32
    chunk_chars = _env_int("CHUNK_CHARS") or 1600
    overlap_chars = _env_int("OVERLAP_CHARS") or 200

    top_k = _env_int("TOP_K") or 6
    sleep_seconds = float(_env("SLEEP_SECONDS") or "0.0")

    if not db_dsn.strip():
        raise SystemExit("Missing required setting: DB_DSN (env var DB_DSN).")
    if not embed_model.strip():
        raise SystemExit("Missing required setting: EMBED_MODEL (env var EMBED_MODEL).")
    if not sme_model.strip():
        raise SystemExit("Missing required setting: SME_MODEL (env var SME_MODEL).")

    return ResolvedConfig(
        rag_root=rag_root,
        domain_dir=domain_dir,
        db_dsn=db_dsn,
        lm_url=lm_url,
        embed_model=embed_model,
        sme_model=sme_model,
        review_model=review_model,
        n_items=int(n_items),
        run_id=run_id,
        prompts_dir=prompts_dir,
        out_dir=out_dir,
        force_ingest=bool(force_ingest),
        embedding_dim=embedding_dim,
        batch_size=int(batch_size),
        chunk_chars=int(chunk_chars),
        overlap_chars=int(overlap_chars),
        top_k=int(top_k),
        sleep_seconds=float(sleep_seconds),
    )
