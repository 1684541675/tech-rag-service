"""Small, secret-safe request tracing primitives for the Agent API."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import logging
from threading import Lock
from time import time
from typing import Mapping, Protocol


_SENSITIVE_MARKERS = ("api_key", "authorization", "token", "secret", "password")


@dataclass(frozen=True)
class TraceEvent:
    """A structured event containing identifiers and safe operational fields only."""

    name: str
    request_id: str
    run_id: str | None
    timestamp_ms: int
    fields: dict[str, object]

    def public(self) -> dict[str, object]:
        return asdict(self)


class TraceSink(Protocol):
    def emit(self, event: TraceEvent) -> None: ...


class InMemoryTraceSink:
    """Process-local sink for development and tests; replace in a real deployment."""

    def __init__(self) -> None:
        self.events: list[TraceEvent] = []
        self._lock = Lock()

    def emit(self, event: TraceEvent) -> None:
        with self._lock:
            self.events.append(event)


class LoggingTraceSink:
    """Emit JSON trace events through the application's normal logger."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger("rag_core.agent.trace")

    def emit(self, event: TraceEvent) -> None:
        self._logger.info("agent_trace=%s", json.dumps(event.public(), ensure_ascii=False, separators=(",", ":")))


def task_fingerprint(task: str) -> dict[str, object]:
    """Describe a request without retaining the original user text in logs."""
    normalized = " ".join(task.split())
    return {"task_length": len(normalized), "task_sha256_12": sha256(normalized.encode("utf-8")).hexdigest()[:12]}


def safe_fields(values: Mapping[str, object]) -> dict[str, object]:
    """Drop fields whose names could contain credentials and bound string values."""
    result: dict[str, object] = {}
    for key, value in values.items():
        lowered = key.lower()
        if any(marker in lowered for marker in _SENSITIVE_MARKERS):
            result[key] = "[redacted]"
        elif isinstance(value, str):
            result[key] = value[:128]
        elif isinstance(value, (bool, int, float)) or value is None:
            result[key] = value
        else:
            result[key] = str(value)[:128]
    return result


def emit(sink: TraceSink, *, name: str, request_id: str, run_id: str | None, fields: Mapping[str, object]) -> None:
    sink.emit(TraceEvent(name=name, request_id=request_id, run_id=run_id, timestamp_ms=int(time() * 1000), fields=safe_fields(fields)))
