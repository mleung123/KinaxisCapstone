#!/usr/bin/env python3
"""runner.py — Cross-platform launcher for RAG TestGen.

Replaces the orchestration logic formerly in _run_testGen.bat.
Called by thin platform shims (_run_testGen.bat / _run_testGen.sh).

Responsibilities:
  - Load and save config.env
  - Prompt user: update settings? show current values
  - Prompt user: run mode (F / I / G / B)
  - Generate RUN_ID (UTC timestamp)
  - Create runs/logs_RUNID/ directory
  - Set environment variables for subprocess
  - Call cli.py via subprocess, streaming stdout to console + log file
  - Capture docker logs and LM Studio logs into run folder
  - Write run_info.txt with config + timing
  - Ask "Run again?" loop
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Config persistence
# ---------------------------------------------------------------------------

CONFIG_KEYS = [
    "RAG_ROOT",
    "DOMAIN_DIR",
    "N_ITEMS",
    "DB_DSN",
    "DOCKER_CONTAINER",
    "LM_URL",
    "EMBED_MODEL",
    "CONTEXT_MODEL",
    "GENERATOR_MODEL",
    "REVIEW_MODEL",
    "API_PROVIDER",
    "API_MODEL",
    "INGEST_PROVIDER",
    "GENERATE_PROVIDER",
    "REVIEW_PROVIDER",
    "INGEST_DELAY_SECONDS",
    "LMSTUDIO_LOGPATH",
    "CHECKPOINT_CHUNKS",
    "CHECKPOINT_ITEMS",
    "CHECKPOINT_REVIEW",
]
CONFIG_KEYS_SENSITIVE = ["LLM_API_KEY"]  # loaded from env only, never written to config.env

CONFIG_LABELS = {
    "RAG_ROOT":             "Program folder (contains .py files)",
    "DOMAIN_DIR":           "Domain folder (subject matter documents)",
    "N_ITEMS":              "Number of items to generate",
    "DB_DSN":               "Postgres DSN",
    "DOCKER_CONTAINER":     "Docker container name (for log capture)",
    "LM_URL":               "LM Studio URL",
    "EMBED_MODEL":          "LM Studio embedding model name",
    "CONTEXT_MODEL":        "LM Studio context model (for local text extraction of PPTX/DOCX)",
    "GENERATOR_MODEL":      "LM Studio generator model (generates exam items)",
    "REVIEW_MODEL":         "LM Studio review model (evaluates items; can match generator)",
    "API_PROVIDER":         "API provider (anthropic / openai / gemini)",
    "API_MODEL": (
        "API model name\n"
        "  anthropic: claude-haiku-4-5-20251001 | claude-sonnet-4-6-20250514\n"
        "  openai:    gpt-4o-mini | gpt-4o\n"
        "  gemini:    gemini-1.5-flash | gemini-1.5-pro"
    ),
    "INGEST_PROVIDER":      "Ingest provider: local or api",
    "GENERATE_PROVIDER":    "Generate provider: local or api",
    "REVIEW_PROVIDER":      "Review provider: local or api",
    "INGEST_DELAY_SECONDS": "Seconds to wait between documents during ingest (helps avoid API rate limits)",
    "LMSTUDIO_LOGPATH":     "LM Studio log file path",
    "CHECKPOINT_CHUNKS":    "Pause to review knowledge chunks? (true/false)",
    "CHECKPOINT_ITEMS":     "Pause to review generated items? (true/false)",
    "CHECKPOINT_REVIEW":    "Pause to review flagged items? (true/false)",
}


DEFAULTS = {
    "N_ITEMS":              "5",
    "DB_DSN":               "postgresql://postgres:postgres@localhost:5435/kinaxis_ragtestdb",
    "LM_URL":               "http://localhost:1234",
    "DOCKER_CONTAINER":     "pgvector17",
    "API_PROVIDER":         "",
    "API_MODEL":            "",
    "INGEST_PROVIDER":      "local",
    "GENERATE_PROVIDER":    "local",
    "REVIEW_PROVIDER":      "local",
    "INGEST_DELAY_SECONDS": "10",
    "CHECKPOINT_CHUNKS":    "true",
    "CHECKPOINT_ITEMS":     "true",
    "CHECKPOINT_REVIEW":    "true",
}

if platform.system() == "Windows":
    DEFAULTS["LMSTUDIO_LOGPATH"] = str(
        Path(os.environ.get("APPDATA", "C:\\Users\\user\\AppData\\Roaming"))
        / "LM Studio" / "logs" / "main.log"
    )
else:
    DEFAULTS["LMSTUDIO_LOGPATH"] = str(
        Path.home() / "Library" / "Logs" / "LM Studio" / "main.log"
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _find_config_env(rag_root: Path) -> Path:
    return rag_root / "config.env"


def load_config_env(path: Path) -> dict[str, str]:
    """Loads KEY=VALUE pairs from config.env into a dict."""
    cfg: dict[str, str] = {}
    if not path.exists():
        return cfg
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, _, v = line.partition("=")
            cfg[k.strip()] = v.strip()
    return cfg


def save_config_env(path: Path, values: dict[str, str]) -> None:
    """Writes KEY=VALUE pairs to config.env. Never writes sensitive keys."""
    lines = []
    for k in CONFIG_KEYS:
        v = values.get(k, "")
        lines.append(f"{k}={v}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _prompt(label: str, current: str, secret: bool = False) -> str:
    """Prompts user for a new value; returns current if user presses Enter."""
    print(f"\n  {label}")
    print(f"  Current:  {'[hidden]' if secret else current}")
    try:
        if secret:
            val = getpass.getpass("  New value (Enter to keep): ")
        else:
            val = input("  New value (Enter to keep): ")
    except (EOFError, KeyboardInterrupt):
        return current
    return val.strip() if val.strip() else current


def _masked_input(prompt):
    """Shows * for each character on Windows; falls back to getpass on Mac/Linux."""
    if sys.platform == "win32":
        import msvcrt
        sys.stdout.write(prompt)
        sys.stdout.flush()
        chars = []
        while True:
            ch = msvcrt.getwch()
            if ch in ("\r", "\n"):
                sys.stdout.write("\n")
                sys.stdout.flush()
                break
            elif ch == "\x08":  # backspace
                if chars:
                    chars.pop()
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
            elif ch == "\x03":  # ctrl+c
                raise KeyboardInterrupt
            else:
                chars.append(ch)
                sys.stdout.write("*")
                sys.stdout.flush()
        return "".join(chars)
    else:
        import getpass
        return getpass.getpass(prompt)


def configure(values: dict[str, str], rag_root: Path) -> dict[str, str]:
    """Interactive settings update. Returns updated values dict."""
    print("\nEnter new values or press Enter to keep current.\n")

    updated = dict(values)

    # RAG_ROOT first — affects config.env path
    new_root = _prompt(CONFIG_LABELS["RAG_ROOT"], updated.get("RAG_ROOT", str(rag_root)))
    updated["RAG_ROOT"] = new_root

    for key in CONFIG_KEYS:
        if key == "RAG_ROOT":
            continue
        current = updated.get(key, DEFAULTS.get(key, ""))
        updated[key] = _prompt(CONFIG_LABELS.get(key, key), current)

    # Sensitive: LLM_API_KEY
    # Prompt if any step routes to an API provider
    api_prov = updated.get("API_PROVIDER", "").strip().lower()
    ingest_p = updated.get("INGEST_PROVIDER", "local").strip().lower()
    generate_p = updated.get("GENERATE_PROVIDER", "local").strip().lower()
    review_p = updated.get("REVIEW_PROVIDER", "local").strip().lower()
    needs_key = bool(api_prov) and (ingest_p == "api" or generate_p == "api" or review_p == "api")
    current_key = os.environ.get("LLM_API_KEY", "")
    if needs_key or not current_key:
        print("")
        new_key = _masked_input("  LLM API key (not saved to config.env)\n  Current: %s\n  New value (Enter to keep): " % ("[set]" if current_key else "[not set]"))
        if new_key and new_key.strip():
            os.environ["LLM_API_KEY"] = new_key.strip()
    else:
        print("\n  LLM API key: [not required for lmstudio]")

    return updated


# ---------------------------------------------------------------------------
# Subprocess runner with tee
# ---------------------------------------------------------------------------

def _run_tee(cmd: list[str], log_path: Path, env: dict[str, str]) -> int:
    """Runs cmd, streaming stdout to both console and log_path."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log_f:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_f.write(line)
        proc.wait()
    return proc.returncode


# ---------------------------------------------------------------------------
# Log capture helpers
# ---------------------------------------------------------------------------

def _capture_docker_logs(container: str, dest: Path) -> None:
    if not container:
        return
    try:
        result = subprocess.run(
            ["docker", "logs", "--tail", "5000", container],
            capture_output=True, text=True, timeout=30,
        )
        dest.write_text(result.stdout + result.stderr, encoding="utf-8")
    except Exception as e:
        dest.write_text(f"docker logs capture failed: {e}\n", encoding="utf-8")


def _capture_lmstudio_logs(log_path_str: str, dest: Path) -> None:
    if not log_path_str:
        return
    src = Path(log_path_str)
    if not src.exists():
        return
    try:
        # Read last 5000 lines
        lines = src.read_text(encoding="utf-8", errors="replace").splitlines()
        dest.write_text("\n".join(lines[-5000:]), encoding="utf-8")
    except Exception as e:
        dest.write_text(f"LM Studio log capture failed: {e}\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def _build_env(values: dict[str, str], log_dir: Path, run_id: str) -> dict[str, str]:
    """Builds the subprocess environment from current os.environ + config values."""
    env = dict(os.environ)
    for k, v in values.items():
        if v:
            env[k] = v
    env["RUN_ID"] = run_id
    env["LOG_DIR"] = str(log_dir)
    # Never inherit LLM_API_KEY from config — only from os.environ (set in configure())
    # It's already in env if the user entered it above
    return env


def _write_run_info(path: Path, values: dict[str, str], run_id: str, extra: dict | None = None) -> None:
    lines = [f"RUN_ID={run_id}"]
    for k in CONFIG_KEYS:
        v = values.get(k, "")
        lines.append(f"{k}={v}")
    if extra:
        for k, v in extra.items():
            lines.append(f"{k}={v}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    # Determine rag_root: the directory containing runner.py
    rag_root = Path(__file__).resolve().parent
    config_path = _find_config_env(rag_root)

    print(f"\nRAG TestGen Runner")
    print(f"  rag_root: {rag_root}")
    print(f"  config:   {config_path}")

    # Load persisted config, fill defaults
    values: dict[str, str] = {**DEFAULTS}
    values["RAG_ROOT"] = str(rag_root)
    persisted = load_config_env(config_path)
    values.update(persisted)

    while True:
        print("\n" + "-"*50)
        try:
            update = input("Update settings? (Y/N, default N): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if update in {"y", "yes"}:
            values = configure(values, rag_root)
            save_config_env(config_path, values)
            print(f"\nSettings saved to {config_path}")
        else:
            # Even when skipping settings, ensure API key is set if needed
            api_prov = values.get("API_PROVIDER", "").strip().lower()
            ingest_p = values.get("INGEST_PROVIDER", "local").strip().lower()
            generate_p = values.get("GENERATE_PROVIDER", "local").strip().lower()
            review_p = values.get("REVIEW_PROVIDER", "local").strip().lower()
            needs_key = bool(api_prov) and (ingest_p == "api" or generate_p == "api" or review_p == "api")
            current_key = os.environ.get("LLM_API_KEY", "")
            if needs_key and not current_key:
                print("\n  API key required for API_PROVIDER=%s" % api_prov)
                new_key = _masked_input("  LLM API key (not saved to config.env)\n  New value: ")
                if new_key and new_key.strip():
                    os.environ["LLM_API_KEY"] = new_key.strip()

        print("\nRun mode:")
        print("  F = Full pipeline (ingest + generate, RAG mode)")
        print("  I = Ingest only   (extract knowledge chunks, write XLSX)")
        print("  G = Generate only (use existing DB chunks, RAG mode)")
        print("  B = Baseline      (no-RAG, load docs directly)")

        try:
            run_choice = input("Choice (F / I / G / B): ").strip().upper()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if run_choice not in {"F", "I", "G", "B"}:
            print("Invalid choice. Try again.")
            continue

        # Validate required paths
        rag_root_val = Path(values.get("RAG_ROOT", str(rag_root)))
        cli_path = rag_root_val / "cli.py"
        if not cli_path.exists():
            print(f"\nERROR: cli.py not found at {cli_path}")
            print("Update RAG_ROOT in settings.")
            continue

        if run_choice != "B":
            domain_dir = values.get("DOMAIN_DIR", "")
            if not domain_dir or not Path(domain_dir).exists():
                print(f"\nERROR: DOMAIN_DIR not found: {domain_dir}")
                continue

        run_id = _utc_now()
        log_dir = rag_root_val / "runs" / f"logs_{run_id}"
        log_dir.mkdir(parents=True, exist_ok=True)

        env = _build_env(values, log_dir, run_id)

        python = sys.executable
        cli = str(cli_path)

        if run_choice == "F":
            cmd = [python, cli, "pipeline", "--force-ingest", "--clear-first"]
            log_file = log_dir / "console_pipeline.txt"
        elif run_choice == "I":
            cmd = [python, cli, "pipeline", "--force-ingest", "--clear-first", "--ingest-only"]
            log_file = log_dir / "console_pipeline.txt"
        elif run_choice == "G":
            cmd = [python, cli, "generate"]
            log_file = log_dir / "console_generate.txt"
        else:  # B
            cmd = [python, cli, "baseline"]
            log_file = log_dir / "console_baseline.txt"

        # Handle checkpoint flags from config
        for flag, env_key in [
            ("--no-checkpoint-chunks", "CHECKPOINT_CHUNKS"),
            ("--no-checkpoint-items", "CHECKPOINT_ITEMS"),
            ("--no-checkpoint-review", "CHECKPOINT_REVIEW"),
        ]:
            if values.get(env_key, "true").lower() in {"0", "false", "no", "n"}:
                if flag in ("--no-checkpoint-items", "--no-checkpoint-review"):
                    if run_choice in {"F", "G", "B"}:
                        cmd.append(flag)
                elif flag == "--no-checkpoint-chunks":
                    if run_choice in {"F", "I"}:
                        cmd.append(flag)

        _write_run_info(log_dir / "run_info.txt", values, run_id, {"RUN_START": _utc_iso()})

        print(f"\nRUN_ID: {run_id}")
        print(f"Log dir: {log_dir}")
        print(f"Started: {_utc_iso()}\n")

        t0 = time.perf_counter()

        # Note: pipeline/generate handle checkpoints interactively themselves.
        # We can't tee + interact at the same time, so for checkpoint-enabled
        # runs we use direct subprocess (no tee) so stdin passes through.
        checkpoint_enabled = any(
            values.get(k, "true").lower() not in {"0", "false", "no", "n"}
            for k in ["CHECKPOINT_CHUNKS", "CHECKPOINT_ITEMS", "CHECKPOINT_REVIEW"]
        ) and run_choice in {"F", "G", "B", "I"}

        if checkpoint_enabled:
            # Direct run — stdin passes through for interactive checkpoints
            result = subprocess.run(cmd, env=env)
            returncode = result.returncode
            # Capture stdout separately after the fact is not possible here;
            # pipeline writes to stderr for progress, stdout for summaries.
            # For now, note this in run_info.
            with open(log_file, "w", encoding="utf-8") as f:
                f.write(f"[interactive run — stdout not captured when checkpoints enabled]\n")
                f.write(f"returncode={returncode}\n")
        else:
            returncode = _run_tee(cmd, log_file, env)

        elapsed = time.perf_counter() - t0
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = int(elapsed % 60)
        duration_str = f"{h}h {m}m {s}s"

        print(f"\nFinished: {_utc_iso()}")
        print(f"Duration: {duration_str}")
        print(f"Exit code: {returncode}")

        _write_run_info(
            log_dir / "run_info.txt", values, run_id,
            {"RUN_END": _utc_iso(), "RUN_DURATION": duration_str, "EXIT_CODE": str(returncode)},
        )

        # Capture external logs
        docker_container = values.get("DOCKER_CONTAINER", "")
        if docker_container:
            _capture_docker_logs(docker_container, log_dir / f"docker_{docker_container}.log")

        lmstudio_log = values.get("LMSTUDIO_LOGPATH", "")
        if lmstudio_log:
            _capture_lmstudio_logs(lmstudio_log, log_dir / "lmstudio.log")

        try:
            again = input("\nRun again? (Y/N, default N): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if again not in {"y", "yes"}:
            break

    print("\nDone.")


if __name__ == "__main__":
    main()
