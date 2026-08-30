"""Zhipu embedding-3 adapter used by the RAG Core.

The API key is read only from ``ZAI_API_KEY`` at runtime and is never stored
in source, cache keys, or Qdrant payloads.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from math import sqrt
from typing import Any


class ZhipuEmbeddingModel:
    """Minimal synchronous adapter for Zhipu ``embedding-3``."""

    name = "embedding-3"
    dimension = 1024
    endpoint = "https://open.bigmodel.cn/api/paas/v4/embeddings"

    def __init__(self, *, api_key: str | None = None, timeout_seconds: int = 30) -> None:
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds

    def embed(self, text: str) -> tuple[float, ...]:
        api_key = self._api_key or os.getenv("ZAI_API_KEY")
        if not api_key:
            raise RuntimeError("ZAI_API_KEY is required for Zhipu embedding-3.")
        if not text.strip():
            raise ValueError("embedding text must not be empty")

        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(
                {"model": self.name, "input": [text], "dimensions": self.dimension},
                ensure_ascii=False,
            ).encode("utf-8"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:
                payload: dict[str, Any] = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Zhipu embedding request failed: HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Zhipu embedding request failed: {exc.reason}") from exc

        data = payload.get("data", [])
        if len(data) != 1 or "embedding" not in data[0]:
            raise RuntimeError("Zhipu embedding response did not contain exactly one vector")
        vector = tuple(float(value) for value in data[0]["embedding"])
        if len(vector) != self.dimension:
            raise RuntimeError(
                f"Zhipu embedding dimension mismatch: expected {self.dimension}, got {len(vector)}"
            )
        norm = sqrt(sum(value * value for value in vector))
        if norm == 0:
            raise RuntimeError("Zhipu embedding response was a zero vector")
        return tuple(value / norm for value in vector)
