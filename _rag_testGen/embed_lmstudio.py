from __future__ import annotations

from dataclasses import dataclass
from typing import List

import requests


@dataclass(frozen=True)
class EmbedConfig:
    """note: Configuration for LM Studio OpenAI-compatible embeddings endpoint."""
    lm_url: str
    model: str
    timeout_seconds: int = 180


def embed_texts(cfg: EmbedConfig, texts: List[str]) -> list[list[float]]:
    """note: Calls LM Studio /v1/embeddings and returns embeddings in the same order as input texts."""
    url = cfg.lm_url.rstrip("/") + "/v1/embeddings"
    payload = {"model": cfg.model, "input": texts}

    r = requests.post(url, json=payload, timeout=int(cfg.timeout_seconds))
    if not r.ok:
        # Include response body because LM Studio usually returns a useful error JSON.
        raise RuntimeError(
            f"Embeddings request failed: HTTP {r.status_code}\n"
            f"url={url}\n"
            f"model={cfg.model!r}\n"
            f"response={r.text}"
        )

    data = r.json()

    try:
        items = data["data"]
        items_sorted = sorted(items, key=lambda x: x["index"])
        return [it["embedding"] for it in items_sorted]
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Unexpected embeddings response shape: {data!r}") from e
