"""Helpers every product adapter uses.

Products receive sessions as chat messages with the session date on the first
turn, because none of the open-source engines accept a timestamp on ingestion.
Every adapter answers through the benchmark's own client with the shared prompt,
so the answer step and its token counts are identical across systems.
"""

from __future__ import annotations

import asyncio
import sys
import threading

import time

from membench.config import ModelConfig
from membench.data.types import Session
from membench.llm import LLMClient
from membench.systems.base import ANSWER_SYSTEM_PROMPT, Answer, build_user_prompt

EMBEDDING_DIMS = 768


def session_messages(session: Session) -> list[dict]:
    messages = []
    dated = False
    for turn in session.turns:
        content = turn.content
        if not dated and turn.role == "user":
            content = f"[{session.date}] {content}"
            dated = True
        messages.append({"role": turn.role, "content": content})
    return messages


def session_text(session: Session) -> str:
    return session.as_text()


def answer_from_context(
    llm: LLMClient,
    context: str,
    question: str,
    question_date: str,
    seed: int,
    retrieval_seconds: float,
    started: float,
) -> Answer:
    response = llm.complete(
        system=ANSWER_SYSTEM_PROMPT,
        user=build_user_prompt(context, question, question_date),
        seed=seed,
    )
    return Answer(
        text=response.text,
        context=context,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        retrieval_seconds=retrieval_seconds,
        total_seconds=time.perf_counter() - started,
        truncated=response.truncated,
    )


def api_key(cfg: ModelConfig) -> str:
    import os

    return os.environ.get(cfg.api_key_env or "", "") if cfg.api_key_env else "none"


def chat_model(cfg: ModelConfig, seed: int):
    """The products' internal model: a LangChain chat model with reasoning off.

    Ollama gets ChatOllama with its native flag. An OpenAI-compatible provider
    gets ChatOpenAI pointed at the provider with the `reasoning` extra that
    OpenRouter understands; providers that do not know it ignore it.
    """
    if cfg.provider == "openai_compat":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=cfg.openai_compat_model,
            base_url=cfg.base_url,
            api_key=api_key(cfg) or "none",
            temperature=cfg.temperature,
            seed=seed,
            max_retries=6,
            extra_body={"reasoning": {"enabled": False}},
        )
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=cfg.answer_model,
        base_url=cfg.base_url,
        temperature=cfg.temperature,
        seed=seed,
        num_ctx=cfg.num_ctx,
        reasoning=False,
    )


def embeddings(cfg: ModelConfig):
    """Embeddings for the products. Ollama serves its own; otherwise a small
    open model runs on the CPU, because no free API serves embeddings."""
    if cfg.provider == "openai_compat":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=cfg.embed_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(model=cfg.embed_model, base_url=cfg.base_url)


_LOOPS = threading.local()


def run_async(coro):
    """Run a coroutine on one persistent event loop per thread.

    Cognee and Graphiti keep asyncio locks and clients bound to the loop that
    created them. asyncio.run() makes a new loop per call, so the second call
    fails with "is bound to a different event loop". One loop per worker thread
    keeps every call of a unit on the loop that built its objects.
    """
    loop = getattr(_LOOPS, "loop", None)
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        _LOOPS.loop = loop
    return loop.run_until_complete(coro)


def retry_transient(call, *, what: str, attempts: int = 3, base_delay: float = 5.0, sleep=time.sleep):
    """Call again after a failure, with growing delays; raise the last error.

    Free API endpoints answer a share of requests with a fast 502 or a bare
    404, and the products' own clients do not retry all of them. One retry
    around each ingest or search call costs seconds; a lost unit costs minutes.
    """
    last: BaseException | None = None
    for attempt in range(attempts):
        try:
            return call()
        except Exception as exc:  # noqa: BLE001, any upstream failure is worth one more try
            last = exc
            if attempt == attempts - 1:
                break
            delay = base_delay * (3**attempt)
            print(f"[retry] {what} attempt {attempt + 1}/{attempts} failed: {str(exc)[:160]!r}; "
                  f"retrying in {delay:.0f}s", file=sys.stderr, flush=True)
            sleep(delay)
    assert last is not None
    raise last
