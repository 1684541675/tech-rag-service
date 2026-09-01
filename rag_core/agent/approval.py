"""Validated, append-only mastery-update proposals for the controlled Agent."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


class ApprovalError(RuntimeError):
    """Stable failure codes for approval/resume callers."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class MasteryUpdateProposal:
    proposal_id: str
    run_id: str
    thread_id: str
    topic: str
    mastered: bool
    note: str
    citations: tuple[dict[str, object], ...]
    status: str = "pending"

    def public(self) -> dict[str, object]:
        value = asdict(self)
        value["citations"] = list(self.citations)
        return value


class MasteryUpdateStore:
    """Own proposals and approved records; never accepts model-selected paths."""

    def __init__(self, *, record_path: Path | None = None) -> None:
        self._record_path = record_path
        self._proposals: dict[str, MasteryUpdateProposal] = {}

    def propose(
        self, *, run_id: str, thread_id: str, topic: str, mastered: bool, note: str, citations: list[dict[str, object]]
    ) -> MasteryUpdateProposal:
        topic = topic.strip()
        note = note.strip()
        if not 1 <= len(topic) <= 120 or not 1 <= len(note) <= 500:
            raise ApprovalError("proposal_validation_error", "掌握度提案字段不符合长度约束。")
        proposal = MasteryUpdateProposal(uuid4().hex, run_id, thread_id, topic, mastered, note, tuple(citations))
        self._proposals[proposal.proposal_id] = proposal
        return proposal

    def resolve(self, *, proposal_id: str, run_id: str, thread_id: str, action: str) -> dict[str, object]:
        proposal = self._proposals.get(proposal_id)
        if proposal is None or proposal.run_id != run_id or proposal.thread_id != thread_id:
            raise ApprovalError("approval_mismatch", "提案不存在、已过期或不属于当前运行。")
        if action not in {"approve", "reject"}:
            raise ApprovalError("approval_validation_error", "审批动作必须为 approve 或 reject。")
        result = {**proposal.public(), "status": "approved" if action == "approve" else "rejected"}
        if action == "approve":
            self._append(result)
        return result

    def _append(self, record: dict[str, object]) -> None:
        if self._record_path is None:
            return
        self._record_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {**record, "approved_at": datetime.now(timezone.utc).isoformat()}
        with self._record_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
