"""The no-product baseline: keep recent session text, drop what does not fit."""

from __future__ import annotations

import time

from membench.data.types import Session
from membench.llm import LLMClient
from membench.systems.base import (
    ANSWER_SYSTEM_PROMPT,
    Answer,
    IngestStats,
    MemorySystem,
    build_user_prompt,
)

CHARS_PER_TOKEN = 4


class ContextWindowSystem(MemorySystem):
    """Holds session text and keeps the most recent `token_budget` tokens of it."""

    def __init__(self, name: str, llm: LLMClient, token_budget: int, seed: int = 0) -> None:
        self.name = name
        self._llm = llm
        self._budget = token_budget
        self._seed = seed
        self._chunks: list[str] = []

    def reset(self, namespace: str) -> None:
        self._chunks = []

    def ingest(self, session: Session) -> IngestStats:
        started = time.perf_counter()
        self._chunks.append(session.as_text())
        elapsed = time.perf_counter() - started
        return IngestStats(
            seconds=elapsed,
            store_items=len(self._chunks),
            store_tokens=sum(len(c) for c in self._chunks) // CHARS_PER_TOKEN,
        )

    def _context(self) -> str:
        budget_chars = self._budget * CHARS_PER_TOKEN
        kept: list[str] = []
        used = 0
        for chunk in reversed(self._chunks):
            if used + len(chunk) > budget_chars and kept:
                break
            kept.append(chunk)
            used += len(chunk)
        return "\n\n".join(reversed(kept))

    def answer(self, question: str, question_date: str) -> Answer:
        started = time.perf_counter()
        context = self._context()
        retrieval_seconds = time.perf_counter() - started
        response = self._llm.complete(
            system=ANSWER_SYSTEM_PROMPT,
            user=build_user_prompt(context, question, question_date),
            seed=self._seed,
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
