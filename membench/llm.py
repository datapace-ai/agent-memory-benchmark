"""Model client with two backends.

`ollama`: the native /api/chat endpoint, which accepts num_ctx and a think flag
per request and returns exact prompt and completion token counts.

`openai_compat`: any OpenAI-compatible /chat/completions endpoint such as
OpenRouter. Token counts come from the response's usage block. Reasoning is
controlled through the `reasoning` parameter where the provider supports it.
Rate limits are respected with a minimum request interval and Retry-After.
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from membench.config import ModelConfig

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_OLLAMA_RETRIES = 3
_OLLAMA_BACKOFF = (1.0, 4.0)
_COMPAT_RETRIES = 6
_COMPAT_BACKOFF = (2.0, 5.0, 15.0, 30.0, 60.0)


@dataclass(frozen=True)
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float
    truncated: bool


@dataclass(frozen=True)
class ChatResponse:
    message: dict
    tool_calls: list[dict]
    prompt_tokens: int
    completion_tokens: int
    seconds: float
    truncated: bool


class LLMClient:
    def __init__(self, cfg: ModelConfig, transport: httpx.BaseTransport | None = None) -> None:
        self.cfg = cfg
        headers = {}
        if cfg.provider == "openai_compat":
            key = os.environ.get(cfg.api_key_env or "", "") if cfg.api_key_env else ""
            if cfg.api_key_env and not key:
                raise RuntimeError(f"environment variable {cfg.api_key_env} is not set")
            headers["Authorization"] = f"Bearer {key or 'none'}"
            headers["HTTP-Referer"] = "https://github.com/datapace-ai/agent-memory-benchmark"
            headers["X-Title"] = "agent-memory-benchmark"
        self._client = httpx.Client(
            base_url=cfg.base_url, timeout=300.0, transport=transport, headers=headers
        )
        self._last_request = 0.0
        self._pace_lock = threading.Lock()

    # ---- public API -------------------------------------------------------

    def complete(
        self,
        system: str,
        user: str,
        seed: int,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        out = self.chat(messages, tools=None, seed=seed, max_tokens=max_tokens, model=model)
        return LLMResponse(
            text=out.message.get("content", ""),
            prompt_tokens=out.prompt_tokens,
            completion_tokens=out.completion_tokens,
            seconds=out.seconds,
            truncated=out.truncated,
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
        if self.cfg.provider == "openai_compat":
            return self._chat_compat(messages, tools, seed, max_tokens, model)
        return self._chat_ollama(messages, tools, seed, max_tokens, model)

    def tool_message(self, name: str, tool_call_id: str | None, content: str) -> dict:
        """A tool-result message in the shape the active backend expects."""
        if self.cfg.provider == "openai_compat":
            return {"role": "tool", "tool_call_id": tool_call_id or name, "content": content}
        return {"role": "tool", "content": content, "tool_name": name}

    def count_tokens(self, text: str) -> int:
        if self.cfg.provider == "openai_compat":
            return len(text) // 4
        data = self._post("/api/embed", {"model": self.cfg.embed_model, "input": text})
        return int(data.get("prompt_eval_count", 0))

    def version(self) -> str:
        if self.cfg.provider == "openai_compat":
            return f"openai-compat:{urlparse(self.cfg.base_url).netloc}"
        return str(self._client.get("/api/version").json().get("version", "unknown"))

    # ---- backends ---------------------------------------------------------

    def _chat_ollama(self, messages, tools, seed, max_tokens, model) -> ChatResponse:
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
        calls = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function", {})
            calls.append({"id": tc.get("id"), "function": {"name": fn.get("name", ""), "arguments": fn.get("arguments", {}) or {}}})
        return ChatResponse(
            message=message,
            tool_calls=calls,
            prompt_tokens=int(data.get("prompt_eval_count", 0)),
            completion_tokens=int(data.get("eval_count", 0)),
            seconds=time.perf_counter() - started,
            truncated=data.get("done_reason") == "length",
        )

    def _chat_compat(self, messages, tools, seed, max_tokens, model) -> ChatResponse:
        payload = {
            "model": model or self.cfg.answer_model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "seed": seed,
            "max_tokens": max_tokens or self.cfg.max_answer_tokens,
        }
        if tools:
            payload["tools"] = tools
        if self.cfg.think is not None:
            payload["reasoning"] = {"enabled": bool(self.cfg.think)}
        started = time.perf_counter()
        data = self._post("/chat/completions", payload)
        choice = (data.get("choices") or [{}])[0]
        message = dict(choice.get("message") or {})
        message["content"] = _THINK.sub("", message.get("content") or "").strip()
        calls = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function", {})
            raw = fn.get("arguments", "{}")
            if isinstance(raw, str):
                try:
                    args = json.loads(raw) if raw.strip() else {}
                except json.JSONDecodeError:
                    args = {}
            else:
                args = dict(raw or {})
            calls.append({"id": tc.get("id"), "function": {"name": fn.get("name", ""), "arguments": args}})
        usage = data.get("usage") or {}
        return ChatResponse(
            message=message,
            tool_calls=calls,
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            seconds=time.perf_counter() - started,
            truncated=choice.get("finish_reason") == "length",
        )

    # ---- transport --------------------------------------------------------

    def _sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def _pace(self) -> None:
        """Global minimum gap between requests, shared across worker threads."""
        gap = self.cfg.min_request_interval_seconds
        with self._pace_lock:
            if gap > 0:
                wait = self._last_request + gap - time.monotonic()
                if wait > 0:
                    self._sleep(wait)
            self._last_request = time.monotonic()

    def _post(self, path: str, payload: dict) -> dict:
        compat = self.cfg.provider == "openai_compat"
        retries = _COMPAT_RETRIES if compat else _OLLAMA_RETRIES
        backoff = _COMPAT_BACKOFF if compat else _OLLAMA_BACKOFF
        last = ""
        for attempt in range(retries):
            self._pace()
            try:
                response = self._client.post(path, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    if compat and isinstance(data, dict) and data.get("error"):
                        last = f"provider error: {json.dumps(data['error'])[:300]}"
                    else:
                        return data
                else:
                    last = f"HTTP {response.status_code}: {response.text[:300]}"
                    retry_after = response.headers.get("Retry-After")
                    if response.status_code == 429:
                        print(f"[llm] rate limited (attempt {attempt + 1}/{retries}), retry-after={retry_after}",
                              file=sys.stderr, flush=True)
                    if retry_after and attempt < retries - 1:
                        try:
                            self._sleep(min(float(retry_after), 120.0))
                            continue
                        except ValueError:
                            pass
            except httpx.HTTPError as exc:
                last = f"{type(exc).__name__}: {exc}"
            if attempt < retries - 1:
                self._sleep(backoff[min(attempt, len(backoff) - 1)])
        raise RuntimeError(f"model call to {path} failed after {retries} attempts. {last}")
