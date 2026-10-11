"""Ollama adapter: model discovery and schema-constrained generation over the HTTP API.

Read-only with respect to the server: it never pulls, deletes or reconfigures models.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx

from forage.llm.base import LLMOutputError, LLMUnavailable, LLMUsage, T, validate_output


@dataclass
class InstalledModel:
    name: str
    size_bytes: int
    parameter_size: str
    family: str

    @property
    def size_gb(self) -> float:
        return self.size_bytes / 1e9


def _is_embedding_model(model: InstalledModel) -> bool:
    name = model.name.lower()
    return "embed" in name or model.family.lower() in {"bert", "nomic-bert"}


def select_model(installed: list[InstalledModel], preferred: str = "", max_gb: float = 12.0) -> str:
    """Use the configured model if installed, otherwise the largest installed chat model under `max_gb`."""
    names = {m.name for m in installed}
    if preferred:
        if preferred in names:
            return preferred
        if f"{preferred}:latest" in names:
            return f"{preferred}:latest"
        raise LLMUnavailable(f"configured model '{preferred}' is not installed; installed: {sorted(names) or 'none'}")
    candidates = [m for m in installed if not _is_embedding_model(m) and m.size_gb <= max_gb]
    if not candidates:
        raise LLMUnavailable(f"no installed chat model under {max_gb} GB; installed: {sorted(names) or 'none'}")
    return max(candidates, key=lambda m: m.size_bytes).name


class OllamaClient:
    def __init__(
        self, base_url: str = "http://127.0.0.1:11434", model: str = "", *, num_ctx: int = 8192,
        temperature: float = 0.1, timeout: float = 180.0, think: bool = False,
        http: httpx.Client | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.think = think
        self._http = http or httpx.Client(timeout=httpx.Timeout(timeout, connect=5.0))

    # -- discovery --------------------------------------------------------

    def version(self) -> str:
        try:
            resp = self._http.get(f"{self.base_url}/api/version", timeout=5.0)
            resp.raise_for_status()
            return str(resp.json().get("version", "unknown"))
        except (httpx.HTTPError, ValueError) as exc:
            raise LLMUnavailable(f"Ollama not reachable at {self.base_url}: {exc}") from exc

    def list_models(self) -> list[InstalledModel]:
        try:
            resp = self._http.get(f"{self.base_url}/api/tags", timeout=10.0)
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise LLMUnavailable(f"Ollama not reachable at {self.base_url}: {exc}") from exc
        out = []
        for item in data.get("models", []):
            details = item.get("details") or {}
            out.append(
                InstalledModel(
                    name=item.get("name", ""), size_bytes=int(item.get("size", 0)),
                    parameter_size=str(details.get("parameter_size", "")), family=str(details.get("family", "")),
                )
            )
        return out

    def ensure_model(self, preferred: str = "", max_gb: float = 12.0) -> str:
        self.model = select_model(self.list_models(), preferred or self.model, max_gb)
        return self.model

    # -- generation -------------------------------------------------------

    def _chat(self, messages: list[dict[str, str]], schema_json: dict[str, Any], max_tokens: int) -> tuple[str, LLMUsage, str]:
        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": self.think,
            "format": schema_json,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "num_predict": max_tokens,
                # Model defaults include a strong presence penalty, which corrupts verbatim quotes.
                "presence_penalty": 0.0,
                "repeat_penalty": 1.0,
            },
        }
        started = time.monotonic()
        try:
            resp = self._http.post(f"{self.base_url}/api/chat", json=body)
        except httpx.HTTPError as exc:
            raise LLMUnavailable(f"Ollama request failed: {type(exc).__name__}: {exc}") from exc
        if resp.status_code == 404:
            raise LLMUnavailable(f"model '{self.model}' not found on the Ollama server")
        if resp.status_code >= 400:
            raise LLMUnavailable(f"Ollama returned HTTP {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        usage = LLMUsage(
            prompt_tokens=int(data.get("prompt_eval_count") or 0),
            output_tokens=int(data.get("eval_count") or 0),
            duration_ms=int((time.monotonic() - started) * 1000),
            load_ms=int((data.get("load_duration") or 0) / 1e6),
            prompt_eval_ms=int((data.get("prompt_eval_duration") or 0) / 1e6),
            eval_ms=int((data.get("eval_duration") or 0) / 1e6),
        )
        content = (data.get("message") or {}).get("content", "")
        return content, usage, str(data.get("done_reason", ""))

    def generate(self, *, purpose: str, system: str, user: str, schema: type[T], max_tokens: int,
                 payload: dict[str, Any] | None = None) -> tuple[T, LLMUsage]:
        if not self.model:
            self.ensure_model()
        schema_json = schema.model_json_schema()
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        total = LLMUsage(attempts=0)
        last_error = "no attempt made"
        for attempt in range(2):
            content, usage, done_reason = self._chat(messages, schema_json, max_tokens)
            total.add(usage)
            total.attempts += 1
            try:
                return validate_output(content, schema), total
            except LLMOutputError as exc:
                last_error = f"{exc} (done_reason={done_reason})"
                if attempt == 0:
                    messages = messages + [
                        {"role": "assistant", "content": content[:2000]},
                        {"role": "user", "content": f"That reply was rejected: {exc}. "
                                                    "Reply again with one complete JSON object that fits the schema. Be concise."},
                    ]
        error = LLMOutputError(f"{purpose}: {last_error}")
        error.usage = total  # type: ignore[attr-defined]
        raise error
