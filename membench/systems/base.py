"""The one interface every memory system implements.

A system receives sessions and a question string. It never receives the gold
answer, the question type, the ability, or which sessions hold the evidence.
tests/test_integrity.py enforces that by inspecting this package's source.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from membench.data.types import Session

ANSWER_SYSTEM_PROMPT = (
    "You are answering a question about a user, using only the context provided. "
    "The context comes from earlier conversations with that user. "
    "Answer in one or two short sentences. "
    "If the context does not contain the information needed, reply exactly: "
    "The information is not available."
)


@dataclass(frozen=True)
class IngestStats:
    seconds: float
    store_items: int
    store_tokens: int


@dataclass(frozen=True)
class Answer:
    text: str
    context: str
    prompt_tokens: int
    completion_tokens: int
    retrieval_seconds: float
    total_seconds: float
    truncated: bool = False


def build_user_prompt(context: str, question: str, question_date: str) -> str:
    return (
        f"Today is {question_date}.\n\n"
        f"Context from earlier conversations:\n{context}\n\n"
        f"Question: {question}"
    )


class MemorySystem(ABC):
    name: str

    @abstractmethod
    def reset(self, namespace: str) -> None:
        """Discard all stored state and start fresh under a new namespace."""

    @abstractmethod
    def ingest(self, session: Session) -> IngestStats:
        """Store one session. Called once per session, in chronological order."""

    @abstractmethod
    def answer(self, question: str, question_date: str) -> Answer:
        """Retrieve from storage and answer. Only stored state may be used."""
