"""Conservative, explainable evidence gate before answer generation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .hybrid import HybridRetrievalResult, ParentWindow


@dataclass(frozen=True)
class EvidenceDecision:
    """Whether retrieved context is sufficient to allow an LLM answer."""

    sufficient: bool
    reason: str
    fused_hit_count: int
    window_count: int
    max_dense_score: float | None
    diagnostics: tuple[str, ...]


@dataclass(frozen=True)
class RetrievedContext:
    """Retrieval output that an API handler may safely pass to generation."""

    windows: tuple[ParentWindow, ...]
    evidence: EvidenceDecision


class EvidenceGate:
    """Reject weak retrieval evidence rather than letting the LLM fill gaps.

    The threshold is revision/model dependent. ``0.71`` is only an initial
    guardrail for the current AI-25 baseline and must later be calibrated on
    an expanded, independent gap set.
    """

    def __init__(self, *, min_dense_score: float = 0.71) -> None:
        if not 0.0 <= min_dense_score <= 1.0:
            raise ValueError("min_dense_score must be within [0, 1]")
        self.min_dense_score = min_dense_score

    def assess(self, *, retrieval: HybridRetrievalResult, windows: Sequence[ParentWindow]) -> EvidenceDecision:
        dense_scores = [
            score
            for hit in retrieval.hits
            for source, score in hit.source_scores
            if source == "dense"
        ]
        max_dense_score = max(dense_scores, default=None)
        if not retrieval.hits:
            return EvidenceDecision(False, "no_retrieval_hits", 0, len(windows), max_dense_score, retrieval.diagnostics)
        if not windows:
            return EvidenceDecision(False, "no_parent_context", len(retrieval.hits), 0, max_dense_score, retrieval.diagnostics)
        if max_dense_score is None:
            return EvidenceDecision(False, "dense_evidence_unavailable", len(retrieval.hits), len(windows), None, retrieval.diagnostics)
        if max_dense_score < self.min_dense_score:
            return EvidenceDecision(False, "dense_score_below_threshold", len(retrieval.hits), len(windows), max_dense_score, retrieval.diagnostics)
        return EvidenceDecision(True, "sufficient_evidence", len(retrieval.hits), len(windows), max_dense_score, retrieval.diagnostics)
