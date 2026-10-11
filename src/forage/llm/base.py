"""LLM adapter contract.

The engine asks for one thing only: a validated instance of a pydantic model.
Adapters have no tools and return data, never actions.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """Base class for adapter failures."""


class LLMUnavailable(LLMError):
    """The model server could not be reached or the model is missing."""


class LLMOutputError(LLMError):
    """The model answered, but not with data that fits the requested schema."""


@dataclass
class LLMUsage:
    prompt_tokens: int = 0
    output_tokens: int = 0
    duration_ms: int = 0
    attempts: int = 1
    load_ms: int = 0  # model (re)load time reported by the server
    prompt_eval_ms: int = 0
    eval_ms: int = 0

    def add(self, other: "LLMUsage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.output_tokens += other.output_tokens
        self.duration_ms += other.duration_ms
        self.load_ms += other.load_ms
        self.prompt_eval_ms += other.prompt_eval_ms
        self.eval_ms += other.eval_ms


class LLMClient(Protocol):
    model: str

    def generate(
        self, *, purpose: str, system: str, user: str, schema: type[T], max_tokens: int,
        payload: dict[str, Any] | None = None,
    ) -> tuple[T, LLMUsage]:
        """Return a validated `schema` instance.

        `payload` carries the same inputs as the prompt in structured form. Real
        adapters ignore it; the simulated adapter uses it instead of parsing prose.
        """
        ...


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_json_object(text: str) -> Any:
    """Parse a JSON object from model output, tolerating fences, chatter and truncation."""
    text = _FENCE_RE.sub("", (text or "").strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in output")
    body = text[start:]
    end = body.rfind("}")
    if end > 0:
        try:
            return json.loads(body[: end + 1])
        except json.JSONDecodeError:
            pass
    repaired = _close_truncated(body)
    if repaired is None:
        raise ValueError("output is not valid JSON")
    return json.loads(repaired)


def _close_truncated(body: str) -> str | None:
    """Cut a truncated JSON document back to its last complete value and close open brackets."""
    stack: list[str] = []
    in_string = False
    escaped = False
    last_safe = -1  # index after which the document can be cut and closed
    safe_stack: list[str] = []
    for i, ch in enumerate(body):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if not stack:
                return None
            stack.pop()
            last_safe, safe_stack = i + 1, list(stack)
            if not stack:
                break
        elif ch == "," and stack and stack[-1] == "]":
            last_safe, safe_stack = i, list(stack)
    if last_safe <= 0:
        return None
    candidate = body[:last_safe].rstrip().rstrip(",") + "".join(reversed(safe_stack))
    try:
        json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return candidate


def validate_output(text: str, schema: type[T]) -> T:
    try:
        data = parse_json_object(text)
    except ValueError as exc:
        raise LLMOutputError(str(exc)) from exc
    try:
        return schema.model_validate(data)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first.get("loc", ()))
        raise LLMOutputError(f"schema mismatch at '{where}': {first.get('msg')}") from exc
