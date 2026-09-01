"""Fail-closed GLM Function Calling loop for the controlled Agent."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from rag_core.generation.service import ZHIPU_CHAT_URL

from .tools import AgentToolError, ControlledAgentTools

JsonObject = dict[str, Any]
Transport = Callable[[JsonObject], JsonObject]


@dataclass(frozen=True)
class ToolCallingResult:
    status: str
    answer: str | None
    tool_calls: tuple[str, ...]
    model: str
    usage: Mapping[str, object] | None


class GlmToolCallingAgent:
    """Execute only the two read-only Agent tools via GLM Function Calling.

    The model never receives an arbitrary execution capability.  A missing
    retrieval call or an insufficient evidence decision terminates locally,
    before a final model answer can be requested.
    """

    def __init__(self, *, tools: ControlledAgentTools, model: str = "glm-4.7-flash", api_key: str | None = None, timeout_seconds: int = 300, transport: Transport | None = None) -> None:
        self.tools, self.model = tools, model
        self.api_key, self.timeout_seconds, self._transport = api_key, timeout_seconds, transport

    def run(self, *, user_task: str) -> ToolCallingResult:
        if not user_task.strip():
            raise ValueError("user_task must not be empty")
        messages: list[JsonObject] = [
            {"role": "system", "content": "你是受证据约束的 C++ 后端面试知识库助手。回答资料性问题前必须调用 retrieve_evidence。仅当工具返回 sufficient=true 时，才可依据工具结果回答并标注 citation；否则不得回答。"},
            {"role": "user", "content": user_task.strip()},
        ]
        calls: list[str] = []
        retrieved = False
        usage: Mapping[str, object] | None = None
        for _ in range(4):
            payload: JsonObject = {"model": self.model, "messages": messages, "tools": _tool_schema(), "tool_choice": "auto", "temperature": 0.1, "max_tokens": 800}
            try:
                response = self._request(payload)
            except RuntimeError:
                # An unavailable model must never be replaced with a guessed answer.
                return ToolCallingResult("llm_unavailable", None, tuple(calls), self.model, usage)
            usage = response.get("usage") if isinstance(response.get("usage"), Mapping) else usage
            message = _message(response)
            raw_calls = message.get("tool_calls") or []
            if not raw_calls:
                if not retrieved:
                    return ToolCallingResult("agent_protocol_error", None, tuple(calls), self.model, usage)
                return ToolCallingResult("ok", _text(message.get("content")), tuple(calls), self.model, usage)
            messages.append({"role": "assistant", "content": message.get("content") or "", "tool_calls": raw_calls})
            for call in raw_calls:
                name = str(call.get("function", {}).get("name", ""))
                call_id = str(call.get("id", ""))
                calls.append(name)
                result = self._execute(name, call.get("function", {}).get("arguments", "{}"))
                if name == "retrieve_evidence":
                    retrieved = True
                    if not result.get("sufficient"):
                        return ToolCallingResult("no_knowledge", None, tuple(calls), self.model, usage)
                messages.append({"role": "tool", "tool_call_id": call_id, "content": json.dumps(result, ensure_ascii=False)})
        return ToolCallingResult("agent_step_limit", None, tuple(calls), self.model, usage)

    def _execute(self, name: str, raw_arguments: object) -> JsonObject:
        if name not in {"retrieve_evidence", "inspect_source"}:
            return {"status": "error", "code": "tool_not_allowed"}
        try:
            arguments = json.loads(raw_arguments if isinstance(raw_arguments, str) else "{}")
            return self.tools.retrieve_evidence(arguments) if name == "retrieve_evidence" else self.tools.inspect_source(arguments)
        except (json.JSONDecodeError, AgentToolError) as exc:
            return {"status": "error", "code": getattr(exc, "code", "tool_validation_error")}

    def _request(self, payload: JsonObject) -> JsonObject:
        if self._transport is not None:
            return self._transport(payload)
        api_key = self.api_key or os.getenv("ZAI_API_KEY")
        if not api_key:
            raise RuntimeError("ZAI_API_KEY is not configured")
        request = urllib.request.Request(ZHIPU_CHAT_URL, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                decoded = json.loads(response.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError) as exc:
            raise RuntimeError(f"GLM tool request failed: {exc}") from exc
        if not isinstance(decoded, dict):
            raise RuntimeError("GLM tool response is not an object")
        return decoded


def _tool_schema() -> list[JsonObject]:
    return [
        {"type": "function", "function": {"name": "retrieve_evidence", "description": "检索当前知识库的可引用证据。必须先调用此工具。", "parameters": {"type": "object", "properties": {"query": {"type": "string", "minLength": 1, "maxLength": 500}}, "required": ["query"], "additionalProperties": False}}},
        {"type": "function", "function": {"name": "inspect_source", "description": "读取 retrieve_evidence 在当前运行返回的来源上下文。", "parameters": {"type": "object", "properties": {"ref_id": {"type": "string"}}, "required": ["ref_id"], "additionalProperties": False}}},
    ]


def _message(response: JsonObject) -> JsonObject:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        raise RuntimeError("GLM tool response has no choices")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise RuntimeError("GLM tool response has no message")
    return message


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""
