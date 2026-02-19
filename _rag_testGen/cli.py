from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from config import ResolvedConfig, load_config_from_env
from ingest import IngestConfig, ingest_domain
from pipeline import GenerateConfig, PipelineConfig, generate_from_db, run_pipeline


def _default_run_id() -> str:
    """note: Returns a UTC timestamp run_id suitable for folder/file names and traceability."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def build_parser() -> argparse.ArgumentParser:
    """note: Builds a dumb CLI dispatcher; real logic lives in ingest.py and pipeline.py."""
    p = argparse.ArgumentParser(prog="rag_testgen", description="Kinaxis Capstone RAG TestGen (dispatcher).")
    sub = p.add_subparsers(dest="cmd", required=True)

    ingest_p = sub.add_parser("ingest", help="Ingest domain folder into Postgres+pgvector.")
    ingest_p.add_argument("--domain-dir", type=str, default=None)
    ingest_p.add_argument("--db-dsn", type=str, default=None)
    ingest_p.add_argument("--lm-base-url", type=str, default=None)
    ingest_p.add_argument("--embed-model", type=str, default=None)
    ingest_p.add_argument("--embedding-dim", type=int, default=None)
    ingest_p.add_argument("--batch-size", type=int, default=None)
    ingest_p.add_argument("--chunk-chars", type=int, default=None)
    ingest_p.add_argument("--overlap-chars", type=int, default=None)
    ingest_p.add_argument("--clear-first", action="store_true", help="If set, deletes existing chunks before ingesting.")

    gen_p = sub.add_parser("generate", help="Generate test items using existing pgvector chunks (no ingest).")
    gen_p.add_argument("--db-dsn", type=str, default=None)
    gen_p.add_argument("--lm-base-url", type=str, default=None)
    gen_p.add_argument("--embed-model", type=str, default=None)
    gen_p.add_argument("--sme-model", type=str, default=None)
    gen_p.add_argument("--review-model", type=str, default=None)
    gen_p.add_argument("--n-items", type=int, default=None)
    gen_p.add_argument("--run-id", type=str, default=None)
    gen_p.add_argument("--prompts-dir", type=str, default=None)
    gen_p.add_argument("--out-dir", type=str, default=None)
    gen_p.add_argument("--top-k", type=int, default=None)
    gen_p.add_argument("--sleep-seconds", type=float, default=None)

    pipe_p = sub.add_parser("pipeline", help="Orchestrate ingest (if needed) and then generate.")
    pipe_p.add_argument("--force-ingest", action="store_true", help="Force ingestion even if DB already has chunks.")
    pipe_p.add_argument("--clear-first", action="store_true", help="If set, deletes existing chunks before ingesting.")
    pipe_p.add_argument("--run-id", type=str, default=None)

    return p


def main(argv: list[str] | None = None) -> int:
    """note: Entrypoint that parses args and dispatches to the requested subcommand."""
    args = build_parser().parse_args(argv)

    cfg: ResolvedConfig = load_config_from_env()

    if getattr(args, "db_dsn", None):
        cfg = cfg.with_overrides(db_dsn=args.db_dsn)
    if getattr(args, "lm_url", None):
        cfg = cfg.with_overrides(lm_url=args.lm_url)
    if getattr(args, "domain_dir", None):
        cfg = cfg.with_overrides(domain_dir=args.domain_dir)
    if getattr(args, "embed_model", None):
        cfg = cfg.with_overrides(embed_model=args.embed_model)
    if getattr(args, "sme_model", None):
        cfg = cfg.with_overrides(sme_model=args.sme_model)
    if getattr(args, "review_model", None):
        cfg = cfg.with_overrides(review_model=args.review_model)
    if getattr(args, "n_items", None) is not None:
        cfg = cfg.with_overrides(n_items=int(args.n_items))
    if getattr(args, "prompts_dir", None):
        cfg = cfg.with_overrides(prompts_dir=args.prompts_dir)
    if getattr(args, "out_dir", None):
        cfg = cfg.with_overrides(out_dir=args.out_dir)
    if getattr(args, "top_k", None) is not None:
        cfg = cfg.with_overrides(top_k=int(args.top_k))
    if getattr(args, "sleep_seconds", None) is not None:
        cfg = cfg.with_overrides(sleep_seconds=float(args.sleep_seconds))

    program_root = Path(__file__).resolve().parent
    print("STARTUP")
    print(f"python_exe={sys.executable}")
    print(f"program_root={program_root}")
    print(f"cmd={args.cmd}")
    print(cfg.startup_diagnostics())
    print("")

    if args.cmd == "ingest":
        ingest_cfg = IngestConfig(
            domain_dir=cfg.domain_dir,
            db_dsn=cfg.db_dsn,
            embed_lm_url=cfg.lm_url,
            embed_model=cfg.embed_model,
            embedding_dim=cfg.embedding_dim,
            batch_size=cfg.batch_size,
            chunk_chars=cfg.chunk_chars,
            overlap_chars=cfg.overlap_chars,
            clear_first=bool(args.clear_first),
        )
        summary = ingest_domain(ingest_cfg)
        print("INGEST_SUMMARY")
        for k, v in summary.items():
            print(f"{k}={v}")
        return 0

    if args.cmd == "generate":
        run_id = args.run_id or cfg.run_id or _default_run_id()
        gen_cfg = GenerateConfig(
            db_dsn=cfg.db_dsn,
            lm_url=cfg.lm_url,
            embed_model=cfg.embed_model,
            sme_model=cfg.sme_model,
            review_model=cfg.review_model,
            n_items=cfg.n_items,
            run_id=run_id,
            prompts_dir=cfg.prompts_dir,
            out_dir=cfg.out_dir,
            top_k=cfg.top_k,
            sleep_seconds=cfg.sleep_seconds,
        )
        summary = generate_from_db(gen_cfg)
        print("GENERATE_SUMMARY")
        for k, v in summary.items():
            print(f"{k}={v}")
        return 0

    if args.cmd == "pipeline":
        run_id = args.run_id or cfg.run_id or _default_run_id()
        pipe_cfg = PipelineConfig(
            db_dsn=cfg.db_dsn,
            domain_dir=cfg.domain_dir,
            lm_url=cfg.lm_url,
            embed_model=cfg.embed_model,
            embedding_dim=cfg.embedding_dim,
            batch_size=cfg.batch_size,
            chunk_chars=cfg.chunk_chars,
            overlap_chars=cfg.overlap_chars,
            clear_first=bool(args.clear_first),
            force_ingest=bool(args.force_ingest) or bool(cfg.force_ingest),
            n_items=cfg.n_items,
            sme_model=cfg.sme_model,
            review_model=cfg.review_model,
            run_id=run_id,
            prompts_dir=cfg.prompts_dir,
            out_dir=cfg.out_dir,
            top_k=cfg.top_k,
            sleep_seconds=cfg.sleep_seconds,
        )
        summary = run_pipeline(pipe_cfg)
        print("PIPELINE_SUMMARY")
        for k, v in summary.items():
            print(f"{k}={v}")
        return 0

    raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
