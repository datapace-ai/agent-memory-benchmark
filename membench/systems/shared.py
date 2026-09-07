"""Helpers every product adapter uses.

Products receive sessions as chat messages with the session date on the first
turn, because none of the open-source engines accept a timestamp on ingestion.
Every adapter answers through the benchmark's own client with the shared prompt,
so the answer step and its token counts are identical across systems.
"""

from __future__ import annotations

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
