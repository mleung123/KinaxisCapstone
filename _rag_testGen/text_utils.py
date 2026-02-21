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

    # Required structural elements (line-anchored, case-insensitive).
    if not re.search(r"^\s*question\s*:", t, flags=re.IGNORECASE | re.MULTILINE):
        violations.append("missing_question")

    for opt in ["a", "b", "c", "d"]:
        if not re.search(rf"^\s*{opt}\)\s*", t, flags=re.IGNORECASE | re.MULTILINE):
            violations.append(f"missing_{opt}")

    # Correct key label: allow "correct_key:" or "correct key:"
    if not re.search(r"^\s*correct[_ ]key\s*:", t, flags=re.IGNORECASE | re.MULTILINE):
        violations.append("missing_correct_key")

    # Difficulty label
    if not re.search(r"^\s*difficulty\s*:", t, flags=re.IGNORECASE | re.MULTILINE):
        violations.append("missing_difficulty")

    # Validate correct key value
    m = re.search(r"correct[_ ]key\s*:\s*([A-Da-d])", t, flags=re.IGNORECASE)
    if not m:
        violations.append("bad_correct_key")

    # Validate difficulty value
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

    def _to_int(v: Any) -> int | None:
        try:
            if v is None:
                return None
            return int(v)
        except Exception:
            return None

    def _score_1_5(v: Any) -> int | None:
        n = _to_int(v)
        if n is None:
            return None
        if 1 <= n <= 5:
            return n
        return None

    def _to_bool(v: Any) -> bool | None:
        if isinstance(v, bool):
            return v
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return bool(int(v))
        s = str(v).strip().lower()
        if s in {"true", "t", "1", "yes", "y"}:
            return True
        if s in {"false", "f", "0", "no", "n"}:
            return False
        return None

    out["source_alignment"] = _score_1_5(review_json.get("source_alignment"))
    out["distractor_quality"] = _score_1_5(review_json.get("distractor_quality"))
    out["stem_clarity"] = _score_1_5(review_json.get("stem_clarity"))
    out["difficulty_match"] = _to_bool(review_json.get("difficulty_match"))

    violations: list[str] = []
    if out["decision"] == "UNKNOWN":
        violations.append("bad_decision")
    if out["source_alignment"] is None:
        violations.append("bad_source_alignment")
    if out["distractor_quality"] is None:
        violations.append("bad_distractor_quality")
    if out["stem_clarity"] is None:
        violations.append("bad_stem_clarity")
    if out["difficulty_match"] is None:
        violations.append("bad_difficulty_match")

    out["reviewer_schema_ok"] = (len(violations) == 0)
    out["reviewer_schema_violations"] = violations

    # Keep backward-compatible meaning: parse ok means “we got a usable structured review record”.
    out["reviewer_parse_ok"] = bool(out["reviewer_schema_ok"])

    return out

