"""Client for a local Ollama server.

Uses the native /api/chat endpoint because it is the only one that accepts
num_ctx per request and returns exact prompt and completion token counts.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

import httpx

from membench.config import ModelConfig

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_RETRIES = 3
_BACKOFF_SECONDS = (1.0, 4.0)


@dataclass(frozen=True)
class ChatResponse:
    message: dict
    tool_calls: list[dict]
    prompt_tokens: int
    completion_tokens: int
    seconds: float
    truncated: bool


@dataclass(frozen=True)
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float
    truncated: bool


class LLMClient:
    def __init__(self, cfg: ModelConfig, transport: httpx.BaseTransport | None = None) -> None:
        self.cfg = cfg
        self._client = httpx.Client(base_url=cfg.base_url, timeout=300.0, transport=transport)

    def complete(
        self,
        system: str,
        user: str,
        seed: int,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        payload = {
            "model": model or self.cfg.answer_model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {
                "num_ctx": self.cfg.num_ctx,
                "temperature": self.cfg.temperature,
                "seed": seed,
                "num_predict": max_tokens or self.cfg.max_answer_tokens,
            },
        }
        if self.cfg.think is not None:
            # Reasoning models emit a long <think> block by default. It costs time
            # and tokens without changing the answer, so the benchmark turns it off
            # identically for every system. See METHODS.md.
            payload["think"] = self.cfg.think
        started = time.perf_counter()
        data = self._post("/api/chat", payload)
        elapsed = time.perf_counter() - started
        text = _THINK.sub("", data.get("message", {}).get("content", "")).strip()
        return LLMResponse(
            text=text,
            prompt_tokens=int(data.get("prompt_eval_count", 0)),
            completion_tokens=int(data.get("eval_count", 0)),
            seconds=elapsed,
            truncated=data.get("done_reason") == "length",
        )

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        seed: int,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> ChatResponse:
        """Multi-turn chat with optional tool calling, for tool-loop systems."""
        payload = {
            "model": model or self.cfg.answer_model,
            "stream": False,
            "messages": messages,
            "options": {
                "num_ctx": self.cfg.num_ctx,
                "temperature": self.cfg.temperature,
                "seed": seed,
                "num_predict": max_tokens or self.cfg.max_answer_tokens,
            },
        }
        if tools:
            payload["tools"] = tools
        if self.cfg.think is not None:
            payload["think"] = self.cfg.think
        started = time.perf_counter()
        data = self._post("/api/chat", payload)
        message = dict(data.get("message", {}))
        message["content"] = _THINK.sub("", message.get("content", "") or "").strip()
        return ChatResponse(
            message=message,
            tool_calls=list(message.get("tool_calls") or []),
            prompt_tokens=int(data.get("prompt_eval_count", 0)),
            completion_tokens=int(data.get("eval_count", 0)),
            seconds=time.perf_counter() - started,
            truncated=data.get("done_reason") == "length",
        )

    def count_tokens(self, text: str) -> int:
        data = self._post("/api/embed", {"model": self.cfg.embed_model, "input": text})
        return int(data.get("prompt_eval_count", 0))

    def version(self) -> str:
        return str(self._client.get("/api/version").json().get("version", "unknown"))

    def _post(self, path: str, payload: dict) -> dict:
        last = ""
        for attempt in range(_RETRIES):
            try:
                response = self._client.post(path, json=payload)
                if response.status_code == 200:
                    return response.json()
                last = f"HTTP {response.status_code}: {response.text[:200]}"
            except httpx.HTTPError as exc:
                last = f"{type(exc).__name__}: {exc}"
            if attempt < _RETRIES - 1:
                time.sleep(_BACKOFF_SECONDS[attempt])
        raise RuntimeError(f"Ollama call to {path} failed after {_RETRIES} attempts. {last}")
