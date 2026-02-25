# -*- coding: utf-8 -*-
from __future__ import annotations

# llm_client.py - Unified LLM provider abstraction with vision support.
#
# Supported providers: lmstudio, openai, anthropic, gemini
# Vision: lmstudio/openai/gemini use image_url blocks (base64 PNG)
#         anthropic uses document block (native PDF) or image blocks
#
# API keys read from env only, never logged or saved.

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Known valid model prefixes per provider (for guardrail validation)
# ---------------------------------------------------------------------------

_KNOWN_PREFIXES = {
    "anthropic": ("claude-",),
    "openai":    ("gpt-", "o1-", "o3-", "o4-"),
    "gemini":    ("gemini-",),
    "lmstudio":  (),   # any string accepted for local models
}

_ANTHROPIC_MODELS_HINT = (
    "claude-haiku-4-5-20251001  (fast, cheap - good for extraction)\n"
    "  claude-sonnet-4-6-20250514 (better quality, moderate cost)\n"
    "  claude-opus-4-6-20250401   (highest quality, expensive)"
)

_OPENAI_MODELS_HINT = (
    "gpt-4o-mini  (fast, cheap)\n"
    "  gpt-4o       (better quality)"
)

_GEMINI_MODELS_HINT = (
    "gemini-1.5-flash  (fast, cheap)\n"
    "  gemini-1.5-pro    (better quality)"
)

_API_KEY_PATTERNS = {
    "anthropic": re.compile(r"^sk-ant-[A-Za-z0-9\-_]{20,}$"),
    "openai":    re.compile(r"^sk-[A-Za-z0-9\-_]{20,}$"),
    "gemini":    re.compile(r"^[A-Za-z0-9\-_]{20,}$"),
}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_provider_and_key(provider, api_key, context=""):
    """Checks provider name, key format, and key presence. Raises RuntimeError with helpful message."""
    valid = ("lmstudio", "openai", "anthropic", "gemini")
    if provider not in valid:
        raise RuntimeError(
            "Unknown LLM provider: %r\nValid options: %s" % (provider, ", ".join(valid))
        )

    if provider == "lmstudio":
        return  # no key needed

    if not api_key or not api_key.strip():
        hints = {
            "anthropic": _ANTHROPIC_MODELS_HINT,
            "openai": _OPENAI_MODELS_HINT,
            "gemini": _GEMINI_MODELS_HINT,
        }
        raise RuntimeError(
            "LLM_API_KEY is required for provider=%r but is not set.%s\n"
            "Set it by running the launcher and entering your key when prompted,\n"
            "or: set LLM_API_KEY=your-key-here  (in cmd before launching)"
            % (provider, (" (%s)" % context) if context else "")
        )

    pattern = _API_KEY_PATTERNS.get(provider)
    if pattern and not pattern.match(api_key.strip()):
        raise RuntimeError(
            "LLM_API_KEY does not look like a valid %s key.\n"
            "Expected format: %s keys start with %r.\n"
            "Check that you copied the full key correctly."
            % (provider, provider, {"anthropic": "sk-ant-", "openai": "sk-", "gemini": "AIza..."}.get(provider, ""))
        )


def validate_model_name(provider, model):
    """Warns if model name doesn't match known prefixes for the provider."""
    prefixes = _KNOWN_PREFIXES.get(provider, ())
    if not prefixes:
        return  # lmstudio - any name ok
    if not any(model.strip().lower().startswith(p) for p in prefixes):
        hints = {
            "anthropic": _ANTHROPIC_MODELS_HINT,
            "openai": _OPENAI_MODELS_HINT,
            "gemini": _GEMINI_MODELS_HINT,
        }.get(provider, "")
        print(
            "\n[WARNING] Model name %r may not be valid for provider=%r.\n"
            "  Common %s models:\n  %s\n"
            % (model, provider, provider, hints),
            file=sys.stderr, flush=True
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _utc_now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _truncate(s, limit=4000):
    if s is None:
        return ""
    s = str(s)
    return s if len(s) <= limit else (s[:limit] + "...<truncated>")


def _http_logpath():
    for key in ("LLM_HTTP_LOGPATH", "LMSTUDIO_HTTP_LOGPATH"):
        explicit = (os.environ.get(key) or "").strip()
        if explicit:
            return Path(explicit)
    log_dir = (os.environ.get("LOG_DIR") or "").strip()
    if log_dir:
        return Path(log_dir) / "llm_http.jsonl"
    return None


def _append_http_log(record):
    path = _http_logpath()
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _get_api_key():
    return (os.environ.get("LLM_API_KEY") or "").strip()


# ---------------------------------------------------------------------------
# Rate limit retry helper
# ---------------------------------------------------------------------------

def _post_with_retry(post_fn, max_retries=4, base_delay=15):
    """Calls post_fn(), retrying on HTTP 429 with exponential backoff.
    post_fn should be a zero-argument callable that returns the response text.
    Raises RuntimeError on final failure.
    """
    delay = base_delay
    for attempt in range(max_retries + 1):
        try:
            return post_fn()
        except RuntimeError as e:
            msg = str(e)
            if "429" in msg and attempt < max_retries:
                print(
                    "    [rate limit] HTTP 429 - waiting %ds before retry %d/%d..."
                    % (delay, attempt + 1, max_retries),
                    file=sys.stderr, flush=True,
                )
                time.sleep(delay)
                delay = min(delay * 2, 120)
                continue
            raise


# ---------------------------------------------------------------------------
# Content block builders
# ---------------------------------------------------------------------------

def _image_block_openai(b64_png):
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64_png}}

def _text_block_openai(text):
    return {"type": "text", "text": text or ""}

def _image_block_anthropic(b64_png):
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64_png}}

def _text_block_anthropic(text):
    return {"type": "text", "text": text or ""}

def _pdf_block_anthropic(b64_pdf):
    return {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": b64_pdf}}

def _image_block_gemini(b64_png):
    return {"inline_data": {"mime_type": "image/png", "data": b64_png}}

def _text_block_gemini(text):
    return {"text": text or ""}


# ---------------------------------------------------------------------------
# PDF rendering helpers
# ---------------------------------------------------------------------------

def render_pdf_pages_b64(pdf_path, dpi=96):
    """Renders all PDF pages to base64 PNG. MuPDF warnings suppressed."""
    import fitz
    import base64
    fitz.TOOLS.mupdf_display_errors(False)
    doc = fitz.open(str(pdf_path))
    pages = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        pix = page.get_pixmap(dpi=dpi)
        b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")
        pages.append(b64)
    fitz.TOOLS.mupdf_display_errors(True)
    return pages


def pdf_to_b64(pdf_path):
    """Reads PDF file as base64 string."""
    import base64
    with open(str(pdf_path), "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ---------------------------------------------------------------------------
# Provider implementations - text
# ---------------------------------------------------------------------------

def _post_openai_compat(url, headers, payload, timeout):
    import requests
    r = requests.post(url, json=payload, headers=headers, timeout=int(timeout))
    if not r.ok:
        raise RuntimeError(
            "HTTP %d from %s\n%s" % (r.status_code, url, _truncate(r.text))
        )
    data = r.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError("Unexpected response shape: %r" % data) from e


def _call_lmstudio_text(lm_url, model, system_prompt, user_prompt, temperature, max_tokens, timeout):
    url = lm_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "messages": [
            {"role": "system", "content": system_prompt or ""},
            {"role": "user",   "content": user_prompt or ""},
        ],
    }
    return _post_openai_compat(url, {}, payload, timeout)


def _call_openai_text(model, system_prompt, user_prompt, temperature, max_tokens, timeout, api_key):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}
    payload = {
        "model": model,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "messages": [
            {"role": "system", "content": system_prompt or ""},
            {"role": "user",   "content": user_prompt or ""},
        ],
    }
    return _post_openai_compat(url, headers, payload, timeout)


def _call_anthropic_text(model, system_prompt, user_prompt, temperature, max_tokens, timeout, api_key):
    import requests
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "system": system_prompt or "",
        "messages": [{"role": "user", "content": user_prompt or ""}],
    }
    def _do_post():
        r = requests.post(url, json=payload, headers=headers, timeout=int(timeout))
        if not r.ok:
            raise RuntimeError("Anthropic API HTTP %d\n%s" % (r.status_code, _truncate(r.text)))
        data = r.json()
        try:
            return "\n".join(b["text"] for b in data["content"] if b.get("type") == "text")
        except Exception as e:
            raise RuntimeError("Unexpected Anthropic response: %r" % data) from e
    return _post_with_retry(_do_post)


def _call_gemini_text(model, system_prompt, user_prompt, temperature, max_tokens, timeout, api_key):
    import requests
    url = "https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent" % model
    payload = {
        "contents": [{"role": "user", "parts": [{"text": user_prompt or ""}]}],
        "generationConfig": {"temperature": float(temperature), "maxOutputTokens": int(max_tokens)},
    }
    if system_prompt:
        payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
    r = requests.post(url, json=payload, headers={"Content-Type": "application/json"},
                      params={"key": api_key}, timeout=int(timeout))
    if not r.ok:
        raise RuntimeError("Gemini API HTTP %d\n%s" % (r.status_code, _truncate(r.text)))
    data = r.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        raise RuntimeError("Unexpected Gemini response: %r" % data) from e


# ---------------------------------------------------------------------------
# Provider implementations - vision
# ---------------------------------------------------------------------------

def _call_lmstudio_vision(lm_url, model, system_prompt, user_prompt, image_b64_list,
                           temperature, max_tokens, timeout):
    url = lm_url.rstrip("/") + "/v1/chat/completions"
    content = [_image_block_openai(b) for b in (image_b64_list or [])]
    content.append(_text_block_openai(user_prompt or ""))
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})
    payload = {"model": model, "temperature": float(temperature),
               "max_tokens": int(max_tokens), "messages": messages}
    return _post_openai_compat(url, {}, payload, timeout)


def _call_openai_vision(model, system_prompt, user_prompt, image_b64_list,
                         temperature, max_tokens, timeout, api_key):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}
    content = [_image_block_openai(b) for b in (image_b64_list or [])]
    content.append(_text_block_openai(user_prompt or ""))
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})
    payload = {"model": model, "temperature": float(temperature),
               "max_tokens": int(max_tokens), "messages": messages}
    return _post_openai_compat(url, headers, payload, timeout)


def _call_anthropic_vision(model, system_prompt, user_prompt, image_b64_list,
                            pdf_b64, temperature, max_tokens, timeout, api_key):
    import requests
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    content = []
    if pdf_b64:
        content.append(_pdf_block_anthropic(pdf_b64))
    else:
        for b in (image_b64_list or []):
            content.append(_image_block_anthropic(b))
    content.append(_text_block_anthropic(user_prompt or ""))
    payload = {
        "model": model,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "system": system_prompt or "",
        "messages": [{"role": "user", "content": content}],
    }
    def _do_post():
        r = requests.post(url, json=payload, headers=headers, timeout=int(timeout))
        if not r.ok:
            raise RuntimeError("Anthropic vision API HTTP %d\n%s" % (r.status_code, _truncate(r.text)))
        data = r.json()
        try:
            return "\n".join(b["text"] for b in data["content"] if b.get("type") == "text")
        except Exception as e:
            raise RuntimeError("Unexpected Anthropic response: %r" % data) from e
    return _post_with_retry(_do_post)


def _call_gemini_vision(model, system_prompt, user_prompt, image_b64_list,
                         temperature, max_tokens, timeout, api_key):
    import requests
    url = "https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent" % model
    parts = [_image_block_gemini(b) for b in (image_b64_list or [])]
    parts.append(_text_block_gemini(user_prompt or ""))
    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"temperature": float(temperature), "maxOutputTokens": int(max_tokens)},
    }
    if system_prompt:
        payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
    r = requests.post(url, json=payload, headers={"Content-Type": "application/json"},
                      params={"key": api_key}, timeout=int(timeout))
    if not r.ok:
        raise RuntimeError("Gemini vision API HTTP %d\n%s" % (r.status_code, _truncate(r.text)))
    data = r.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        raise RuntimeError("Unexpected Gemini response: %r" % data) from e


# ---------------------------------------------------------------------------
# Public interface - text
# ---------------------------------------------------------------------------

def call_llm(lm_url, model, system_prompt, user_prompt, temperature,
             max_tokens, request_timeout_seconds, provider=None):
    """Unified text LLM call.
    provider: explicit provider string, or None to read LLM_PROVIDER env var.
    """
    if provider is None:
        provider = (os.environ.get("LLM_PROVIDER") or "lmstudio").strip().lower()
    api_key = _get_api_key()
    run_id = (os.environ.get("RUN_ID") or "").strip() or None

    validate_provider_and_key(provider, api_key, context="call_llm")
    validate_model_name(provider, model)

    t0 = time.perf_counter()
    ok = False
    error_info = None
    response_chars = 0

    try:
        if provider == "lmstudio":
            out = _call_lmstudio_text(lm_url, model, system_prompt, user_prompt,
                                       temperature, max_tokens, request_timeout_seconds)
        elif provider == "openai":
            out = _call_openai_text(model, system_prompt, user_prompt,
                                     temperature, max_tokens, request_timeout_seconds, api_key)
        elif provider == "anthropic":
            out = _call_anthropic_text(model, system_prompt, user_prompt,
                                        temperature, max_tokens, request_timeout_seconds, api_key)
        elif provider == "gemini":
            out = _call_gemini_text(model, system_prompt, user_prompt,
                                     temperature, max_tokens, request_timeout_seconds, api_key)
        else:
            raise RuntimeError("Unknown provider: %r" % provider)

        ok = True
        response_chars = len(out or "")
        return out

    except Exception as e:
        error_info = _truncate(str(e))
        raise

    finally:
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        _append_http_log({
            "ts_utc": _utc_now_iso(), "run_id": run_id,
            "provider": provider, "model": model, "mode": "text",
            "temperature": float(temperature), "max_tokens": int(max_tokens),
            "timeout_s": int(request_timeout_seconds), "elapsed_ms": elapsed_ms,
            "system_chars": len(system_prompt or ""), "user_chars": len(user_prompt or ""),
            "response_chars": response_chars, "ok": ok,
            **({"error": error_info} if error_info else {}),
        })


# ---------------------------------------------------------------------------
# Public interface - vision
# ---------------------------------------------------------------------------

def call_llm_vision(lm_url, model, system_prompt, user_prompt,
                    image_b64_list=None, pdf_path=None,
                    temperature=0.0, max_tokens=2000,
                    request_timeout_seconds=600, render_dpi=96,
                    provider=None):
    """Vision-capable LLM call.

    provider: explicit provider string, or None to read LLM_PROVIDER env var.
              NOTE: for PDF ingest, pass provider explicitly from _effective_provider()
              so that PDF_PROVIDER override is respected.

    anthropic: sends raw PDF as document block (preferred, no rendering needed)
    lmstudio/openai/gemini: renders PDF pages to images
    """
    if provider is None:
        provider = (os.environ.get("LLM_PROVIDER") or "lmstudio").strip().lower()
    api_key = _get_api_key()
    run_id = (os.environ.get("RUN_ID") or "").strip() or None

    validate_provider_and_key(provider, api_key, context="call_llm_vision")
    validate_model_name(provider, model)

    # Build image list / pdf bytes
    images = list(image_b64_list or [])
    pdf_b64 = None

    if pdf_path:
        if provider == "anthropic":
            pdf_b64 = pdf_to_b64(pdf_path)
        else:
            if not images:
                images = render_pdf_pages_b64(pdf_path, dpi=render_dpi)

    t0 = time.perf_counter()
    ok = False
    error_info = None
    response_chars = 0

    try:
        if provider == "lmstudio":
            out = _call_lmstudio_vision(lm_url, model, system_prompt, user_prompt,
                                         images, temperature, max_tokens,
                                         request_timeout_seconds)
        elif provider == "openai":
            out = _call_openai_vision(model, system_prompt, user_prompt,
                                       images, temperature, max_tokens,
                                       request_timeout_seconds, api_key)
        elif provider == "anthropic":
            out = _call_anthropic_vision(model, system_prompt, user_prompt,
                                          images, pdf_b64,
                                          temperature, max_tokens,
                                          request_timeout_seconds, api_key)
        elif provider == "gemini":
            out = _call_gemini_vision(model, system_prompt, user_prompt,
                                       images, temperature, max_tokens,
                                       request_timeout_seconds, api_key)
        else:
            raise RuntimeError("Unknown provider: %r" % provider)

        ok = True
        response_chars = len(out or "")
        return out

    except Exception as e:
        error_info = _truncate(str(e))
        raise

    finally:
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        _append_http_log({
            "ts_utc": _utc_now_iso(), "run_id": run_id,
            "provider": provider, "model": model, "mode": "vision",
            "n_images": len(images), "pdf_native": bool(pdf_b64),
            "temperature": float(temperature), "max_tokens": int(max_tokens),
            "timeout_s": int(request_timeout_seconds), "elapsed_ms": elapsed_ms,
            "system_chars": len(system_prompt or ""), "user_chars": len(user_prompt or ""),
            "response_chars": response_chars, "ok": ok,
            **({"error": error_info} if error_info else {}),
        })
