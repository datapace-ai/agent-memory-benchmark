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


def chat_model(cfg: ModelConfig, seed: int):
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
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(model=cfg.embed_model, base_url=cfg.base_url)
