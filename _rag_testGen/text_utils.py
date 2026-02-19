from __future__ import annotations

import json
import re
from typing import Any


def extract_first_json_obj(text: str) -> dict[str, Any] | None:
    """note: Extracts the first JSON object found in text using a conservative brace-scan; returns None if not found/parsable."""
    if not text:
        return None

    s = text.strip()
    start = s.find("{")
    if start < 0:
        return None

    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = s[start : i + 1]
                try:
                    return json.loads(candidate)
                except Exception:  # noqa: BLE001
                    return None
    return None


def normalize_decision(raw: str) -> str:
    """note: Normalizes reviewer decision labels into a small controlled vocabulary."""
    t = (raw or "").strip().lower()
    if t in {"accept", "accepted"}:
        return "ACCEPT"
    if t in {"revise", "revision", "revise_and_resubmit"}:
        return "REVISE"
    if t in {"reject", "rejected"}:
        return "REJECT"
    return "UNKNOWN"


def clean_generator_text(gen_raw: str) -> str:
    """note: Cleans generator output by removing code fences and trimming excess whitespace."""
    t = gen_raw or ""
    t = re.sub(r"^```[a-zA-Z0-9]*\s*", "", t.strip())
    t = re.sub(r"```\s*$", "", t.strip())
    return t.strip()


def hard_trim_after_difficulty(gen_text: str) -> str:
    """note: Trims any trailing text after a 'difficulty:' field to keep outputs contract-like."""
    if not gen_text:
        return gen_text
    m = re.search(r"(difficulty\s*:\s*[^\n\r]+)", gen_text, flags=re.IGNORECASE)
    if not m:
        return gen_text.strip()
    end = m.end()
    return gen_text[:end].strip()


def validate_generator_schema(gen_text: str) -> tuple[bool, list[str]]:
    """note: Performs a lightweight schema gate on generator output to catch contract violations before review."""
    t = gen_text or ""
    violations: list[str] = []

    required_labels = ["question:", "a)", "b)", "c)", "d)", "correct_key:", "correct key:", "difficulty:"]
    lower = t.lower()
    for lab in required_labels:
        if lab not in lower:
            violations.append(f"missing_{lab.replace(':','').replace(')','')}")
    # validate correct key
    m = re.search(r"correct[_ ]key\s*:\s*([A-Da-d])", t, flags=re.IGNORECASE)
    if not m:
        violations.append("bad_correct_key")
    # validate difficulty tag
    m2 = re.search(r"difficulty\s*:\s*(easy|medium|hard)", t, flags=re.IGNORECASE)
    if not m2:
        violations.append("bad_difficulty")

    return (len(violations) == 0, violations)


def enforce_hygiene_on_review(review_json: dict[str, Any] | None) -> dict[str, Any]:
    """note: Ensures reviewer JSON has expected keys with safe defaults and normalized decision."""
    review_json = review_json or {}
    out: dict[str, Any] = {}

    out["decision"] = normalize_decision(str(review_json.get("decision", "")))
    out["failure_layer"] = str(review_json.get("failure_layer", "") or "")
    out["reason_codes"] = review_json.get("reason_codes", [])
    out["revision_instructions"] = str(review_json.get("revision_instructions", "") or "")
    out["reviewer_parse_ok"] = bool(review_json.get("decision", ""))
    return out
