from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


def _utc_now_iso() -> str:
    """note: Returns a UTC timestamp string for log records."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _truncate(s: str, limit: int) -> str:
    """note: Truncates a string to a safe length for logs/exceptions without breaking the pipeline."""
    if s is None:
        return ""
    s = str(s)
    return s if len(s) <= limit else (s[:limit] + "...<truncated>")


def _http_logpath() -> Path | None:
    """note: Resolves where to write LM Studio request-level logs; returns None if disabled."""
    explicit = (os.environ.get("LMSTUDIO_HTTP_LOGPATH") or "").strip()
    if explicit:
        return Path(explicit)

    log_dir = (os.environ.get("LOG_DIR") or "").strip()
    if log_dir:
        return Path(log_dir) / "lmstudio_http.jsonl"

    return None


def _append_http_log(record: dict[str, Any]) -> None:
    """note: Appends one JSONL record to the request-level log if logging is enabled."""
    path = _http_logpath()
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        return


def call_llm(
    lm_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    request_timeout_seconds: int,
) -> str:
    """note: Calls LM Studio's OpenAI-compatible chat completions endpoint and returns the first message content."""
    run_id = (os.environ.get("RUN_ID") or "").strip() or None
    url = lm_url.rstrip("/") + "/v1/chat/completions"

    payload = {
        "model": model,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "messages": [
            {"role": "system", "content": system_prompt or ""},
            {"role": "user", "content": user_prompt or ""},
        ],
    }

    t0 = time.perf_counter()
    status_code: int | None = None
    response_text: str = ""

    try:
        r = requests.post(url, json=payload, timeout=int(request_timeout_seconds))
        status_code = int(r.status_code)
        response_text = r.text or ""

        if not r.ok:
            err_body = _truncate(response_text, 4000)
            _append_http_log(
                {
                    "ts_utc": _utc_now_iso(),
                    "run_id": run_id,
                    "endpoint": "/v1/chat/completions",
                    "url": url,
                    "model": model,
                    "temperature": float(temperature),
                    "max_tokens": int(max_tokens),
                    "timeout_s": int(request_timeout_seconds),
                    "status": status_code,
                    "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                    "system_chars": len(system_prompt or ""),
                    "user_chars": len(user_prompt or ""),
                    "ok": False,
                    "error_type": "HTTPError",
                    "error": f"HTTP {status_code}",
                    "response_body_trunc": err_body,
                }
            )
            raise RuntimeError(
                "LM Studio chat completion failed.\n"
                f"HTTP {status_code}\n"
                f"url={url}\n"
                f"model={model!r}\n"
                f"response_body={err_body}"
            )

        data = r.json()
        try:
            out = data["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Unexpected LM Studio response shape: {data!r}") from e

        _append_http_log(
            {
                "ts_utc": _utc_now_iso(),
                "run_id": run_id,
                "endpoint": "/v1/chat/completions",
                "url": url,
                "model": model,
                "temperature": float(temperature),
                "max_tokens": int(max_tokens),
                "timeout_s": int(request_timeout_seconds),
                "status": status_code,
                "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                "system_chars": len(system_prompt or ""),
                "user_chars": len(user_prompt or ""),
                "response_chars": len(out or ""),
                "ok": True,
            }
        )

        return out

    except Exception as e:  # noqa: BLE001
        _append_http_log(
            {
                "ts_utc": _utc_now_iso(),
                "run_id": run_id,
                "endpoint": "/v1/chat/completions",
                "url": url,
                "model": model,
                "temperature": float(temperature),
                "max_tokens": int(max_tokens),
                "timeout_s": int(request_timeout_seconds),
                "status": status_code,
                "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                "system_chars": len(system_prompt or ""),
                "user_chars": len(user_prompt or ""),
                "ok": False,
                "error_type": type(e).__name__,
                "error": _truncate(str(e), 4000),
            }
        )
        raise
