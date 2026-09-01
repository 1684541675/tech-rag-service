"""Conservative, explainable evidence gate before answer generation."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from .hybrid import HybridRetrievalResult, ParentWindow

TECH_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+.#()/-]{1,}")
COMMON_ENGLISH_TOKENS = frozenset({
    "a", "an", "and", "are", "can", "do", "does", "explain", "for", "how",
    "in", "is", "of", "or", "the", "to", "what", "why", "with",
})


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

    def assess(self, *, retrieval: HybridRetrievalResult, windows: Sequence[ParentWindow], query: str | None = None) -> EvidenceDecision:
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
        has_query = bool(query and query.strip())
        anchors = _technical_anchors(query or "")
        shared_retrieval = any(len(hit.source_ranks) >= 2 for hit in retrieval.hits)
        diagnostics = (*retrieval.diagnostics, f"retrieval_consensus:{'shared' if shared_retrieval else 'missing'}")
        if anchors and not _has_query_anchor(anchors, windows):
            return EvidenceDecision(
                False,
                "query_anchor_missing",
                len(retrieval.hits),
                len(windows),
                max_dense_score,
                (*diagnostics, "query_anchor_missing:" + ",".join(sorted(anchors))),
            )
        if has_query and not anchors and not shared_retrieval:
            return EvidenceDecision(
                False,
                "retrieval_consensus_missing",
                len(retrieval.hits),
                len(windows),
                max_dense_score,
                diagnostics,
            )
        return EvidenceDecision(True, "sufficient_evidence", len(retrieval.hits), len(windows), max_dense_score, diagnostics)


def _technical_anchors(query: str) -> set[str]:
    """Extract explicit Latin/technical identifiers without requiring Chinese NLP."""
    return {
        _normalize_technical_text(token)
        for token in TECH_TOKEN_RE.findall(query)
        if len(_normalize_technical_text(token)) >= 3
        and _normalize_technical_text(token) not in COMMON_ENGLISH_TOKENS
    }


def _has_query_anchor(anchors: set[str], windows: Sequence[ParentWindow]) -> bool:
    context = _normalize_technical_text("\n".join(
        " ".join((*window.parent.heading_path, window.parent.content))
        for window in windows
    ))
    return any(anchor in context for anchor in anchors)


def _normalize_technical_text(text: str) -> str:
    """Align harmless spelling variants such as ``TopK`` and ``Top(K)``."""
    return re.sub(r"[^a-z0-9]+", "", text.casefold())
