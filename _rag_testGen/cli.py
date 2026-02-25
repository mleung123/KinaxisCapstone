from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from config import ResolvedConfig, load_config_from_env
from ingest import IngestConfig, ingest_domain
from pipeline import (
    BaselineConfig,
    GenerateConfig,
    PipelineConfig,
    generate_baseline,
    generate_from_db,
    run_pipeline,
)


def _default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rag_testgen", description="RAG TestGen dispatcher.")
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- ingest ----
    ingest_p = sub.add_parser("ingest", help="Ingest domain folder into pgvector.")
    ingest_p.add_argument("--domain-dir", type=str, default=None)
    ingest_p.add_argument("--db-dsn", type=str, default=None)
    ingest_p.add_argument("--lm-url", type=str, default=None)
    ingest_p.add_argument("--embed-model", type=str, default=None)
    ingest_p.add_argument("--context-model", type=str, default=None)
    ingest_p.add_argument("--embedding-dim", type=int, default=None)
    ingest_p.add_argument("--batch-size", type=int, default=None)
    ingest_p.add_argument("--clear-first", action="store_true")
    ingest_p.add_argument("--run-id", type=str, default=None)

    # ---- generate (RAG) ----
    gen_p = sub.add_parser("generate", help="Generate items from existing pgvector chunks.")
    gen_p.add_argument("--db-dsn", type=str, default=None)
    gen_p.add_argument("--lm-url", type=str, default=None)
    gen_p.add_argument("--embed-model", type=str, default=None)
    gen_p.add_argument("--generator-model", type=str, default=None)
    gen_p.add_argument("--review-model", type=str, default=None)
    gen_p.add_argument("--n-items", type=int, default=None)
    gen_p.add_argument("--run-id", type=str, default=None)
    gen_p.add_argument("--prompts-dir", type=str, default=None)
    gen_p.add_argument("--out-dir", type=str, default=None)
    gen_p.add_argument("--top-k", type=int, default=None)
    gen_p.add_argument("--sleep-seconds", type=float, default=None)
    gen_p.add_argument("--no-checkpoint-items", action="store_true")
    gen_p.add_argument("--no-checkpoint-review", action="store_true")

    # ---- baseline (no-RAG) ----
    base_p = sub.add_parser("baseline", help="Baseline: generate directly from docs (no pgvector).")
    base_p.add_argument("--domain-dir", type=str, default=None)
    base_p.add_argument("--lm-url", type=str, default=None)
    base_p.add_argument("--generator-model", type=str, default=None)
    base_p.add_argument("--review-model", type=str, default=None)
    base_p.add_argument("--n-items", type=int, default=None)
    base_p.add_argument("--run-id", type=str, default=None)
    base_p.add_argument("--prompts-dir", type=str, default=None)
    base_p.add_argument("--out-dir", type=str, default=None)
    base_p.add_argument("--sleep-seconds", type=float, default=None)
    base_p.add_argument("--no-checkpoint-items", action="store_true")
    base_p.add_argument("--no-checkpoint-review", action="store_true")

    # ---- pipeline (ingest + generate) ----
    pipe_p = sub.add_parser("pipeline", help="Orchestrate ingest + generate.")
    pipe_p.add_argument("--force-ingest", action="store_true")
    pipe_p.add_argument("--clear-first", action="store_true")
    pipe_p.add_argument("--ingest-only", action="store_true")
    pipe_p.add_argument("--run-id", type=str, default=None)
    pipe_p.add_argument("--no-checkpoint-chunks", action="store_true")
    pipe_p.add_argument("--no-checkpoint-items", action="store_true")
    pipe_p.add_argument("--no-checkpoint-review", action="store_true")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg: ResolvedConfig = load_config_from_env()

    # Apply CLI overrides
    overrides: dict = {}
    for attr, cfg_attr in [
        ("db_dsn", "db_dsn"), ("lm_url", "lm_url"), ("domain_dir", "domain_dir"),
        ("embed_model", "embed_model"), ("context_model", "context_model"),
        ("generator_model", "generator_model"), ("review_model", "review_model"),
        ("prompts_dir", "prompts_dir"), ("out_dir", "out_dir"),
    ]:
        val = getattr(args, attr, None)
        if val is not None:
            overrides[cfg_attr] = val
    for attr, cfg_attr in [
        ("n_items", "n_items"), ("top_k", "top_k"), ("embedding_dim", "embedding_dim"),
        ("batch_size", "batch_size"),
    ]:
        val = getattr(args, attr, None)
        if val is not None:
            overrides[cfg_attr] = int(val)
    val = getattr(args, "sleep_seconds", None)
    if val is not None:
        overrides["sleep_seconds"] = float(val)
    if getattr(args, "ingest_only", False):
        overrides["ingest_only"] = True
    if overrides:
        cfg = cfg.with_overrides(**overrides)

    program_root = Path(__file__).resolve().parent
    print("STARTUP")
    print(f"python_exe={sys.executable}")
    print(f"program_root={program_root}")
    print(f"cmd={args.cmd}")
    print(cfg.startup_diagnostics())
    print("")

    run_id = getattr(args, "run_id", None) or cfg.run_id or _default_run_id()

    # ---- ingest ----
    if args.cmd == "ingest":
        from pathlib import Path as P
        ingest_cfg = IngestConfig(
            domain_dir=cfg.domain_dir,
            db_dsn=cfg.db_dsn,
            embed_lm_url=cfg.lm_url,
            embed_model=cfg.embed_model,
            context_lm_url=cfg.lm_url,
            context_model=cfg.context_model,
            embedding_dim=cfg.embedding_dim,
            batch_size=cfg.batch_size,
            clear_first=bool(args.clear_first),
        )
        summary = ingest_domain(ingest_cfg, prompts_dir=cfg.prompts_dir)
        print("INGEST_SUMMARY")
        for k, v in summary.items():
            print(f"{k}={v}")
        return 0

    # ---- generate (RAG) ----
    if args.cmd == "generate":
        gen_cfg = GenerateConfig(
            db_dsn=cfg.db_dsn,
            lm_url=cfg.lm_url,
            embed_model=cfg.embed_model,
            generator_model=cfg.generator_model,
            review_model=cfg.review_model,
            n_items=cfg.n_items,
            run_id=run_id,
            prompts_dir=cfg.prompts_dir,
            out_dir=cfg.out_dir,
            top_k=cfg.top_k,
            sleep_seconds=cfg.sleep_seconds,
            checkpoint_items=not bool(getattr(args, "no_checkpoint_items", False)),
            checkpoint_review=not bool(getattr(args, "no_checkpoint_review", False)),
        )
        summary = generate_from_db(gen_cfg)
        print("GENERATE_SUMMARY")
        for k, v in summary.items():
            print(f"{k}={v}")
        return 0

    # ---- baseline ----
    if args.cmd == "baseline":
        base_cfg = BaselineConfig(
            domain_dir=cfg.domain_dir,
            lm_url=cfg.lm_url,
            generator_model=cfg.generator_model,
            review_model=cfg.review_model,
            n_items=cfg.n_items,
            run_id=run_id,
            prompts_dir=cfg.prompts_dir,
            out_dir=cfg.out_dir,
            sleep_seconds=cfg.sleep_seconds,
            checkpoint_items=not bool(getattr(args, "no_checkpoint_items", False)),
            checkpoint_review=not bool(getattr(args, "no_checkpoint_review", False)),
        )
        summary = generate_baseline(base_cfg)
        print("BASELINE_SUMMARY")
        for k, v in summary.items():
            print(f"{k}={v}")
        return 0

    # ---- pipeline ----
    if args.cmd == "pipeline":
        pipe_cfg = PipelineConfig(
            db_dsn=cfg.db_dsn,
            domain_dir=cfg.domain_dir,
            lm_url=cfg.lm_url,
            embed_model=cfg.embed_model,
            context_model=cfg.context_model,
            api_provider=cfg.api_provider,
            api_model=cfg.api_model,
            ingest_provider=cfg.ingest_provider,
            generate_provider=cfg.generate_provider,
            review_provider=cfg.review_provider,
            embedding_dim=cfg.embedding_dim,
            batch_size=cfg.batch_size,
            clear_first=bool(args.clear_first),
            force_ingest=bool(args.force_ingest) or bool(cfg.force_ingest),
            ingest_only=bool(getattr(args, "ingest_only", False)) or bool(cfg.ingest_only),
            n_items=cfg.n_items,
            generator_model=cfg.generator_model,
            review_model=cfg.review_model,
            run_id=run_id,
            prompts_dir=cfg.prompts_dir,
            out_dir=cfg.out_dir,
            top_k=cfg.top_k,
            sleep_seconds=cfg.sleep_seconds,
            checkpoint_chunks=not bool(getattr(args, "no_checkpoint_chunks", False)) and cfg.checkpoint_chunks,
            checkpoint_items=not bool(getattr(args, "no_checkpoint_items", False)) and cfg.checkpoint_items,
            checkpoint_review=not bool(getattr(args, "no_checkpoint_review", False)) and cfg.checkpoint_review,
            ingest_delay_seconds=float(os.environ.get("INGEST_DELAY_SECONDS") or "0"),
        )
        summary = run_pipeline(pipe_cfg)
        print("PIPELINE_SUMMARY")
        for k, v in summary.items():
            print(f"{k}={v}")
        return 0

    raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
