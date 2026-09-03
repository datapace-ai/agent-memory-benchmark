"""The session clock.

Reset, then ingest sessions one at a time in chronological order, then ask the
question with its own date as today. A system never sees the whole history at
once and never sees the question before ingestion is finished.
"""

from __future__ import annotations

import traceback
from dataclasses import asdict, dataclass

from membench.data.types import Question
from membench.systems.base import MemorySystem


@dataclass(frozen=True)
class QuestionRun:
    question_id: str
    system: str
    seed: int
    answer_text: str
    context: str
    prompt_tokens: int
    completion_tokens: int
    retrieval_seconds: float
    answer_seconds: float
    ingest_seconds: float
    sessions_ingested: int
    store_items: int
    store_tokens: int
    truncated: bool
    error: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def _empty(question: Question, system: MemorySystem, seed: int, error: str, **overrides) -> QuestionRun:
    base = dict(
        question_id=question.question_id,
        system=system.name,
        seed=seed,
        answer_text="",
        context="",
        prompt_tokens=0,
        completion_tokens=0,
        retrieval_seconds=0.0,
        answer_seconds=0.0,
        ingest_seconds=0.0,
        sessions_ingested=0,
        store_items=0,
        store_tokens=0,
        truncated=False,
        error=error,
    )
    base.update(overrides)
    return QuestionRun(**base)


def run_question(
    system: MemorySystem, question: Question, seed: int, evidence_only: bool = False
) -> QuestionRun:
    sessions = question.sessions
    if evidence_only:
        sessions = tuple(s for s in sessions if s.session_id in question.evidence_session_ids)

    system.reset(f"{question.question_id}:{seed}")

    ingest_seconds = 0.0
    store_items = 0
    store_tokens = 0
    ingested = 0
    for session in sessions:
        try:
            stats = system.ingest(session)
        except Exception as exc:
            return _empty(
                question,
                system,
                seed,
                f"ingest failed on {session.session_id}: {exc}\n{traceback.format_exc(limit=3)}",
                ingest_seconds=ingest_seconds,
                sessions_ingested=ingested,
            )
        ingest_seconds += stats.seconds
        store_items = stats.store_items
        store_tokens = stats.store_tokens
        ingested += 1

    try:
        answer = system.answer(question.question, question.question_date)
    except Exception as exc:
        return _empty(
            question,
            system,
            seed,
            f"answer failed: {exc}\n{traceback.format_exc(limit=3)}",
            ingest_seconds=ingest_seconds,
            sessions_ingested=ingested,
            store_items=store_items,
            store_tokens=store_tokens,
        )

    return QuestionRun(
        question_id=question.question_id,
        system=system.name,
        seed=seed,
        answer_text=answer.text,
        context=answer.context,
        prompt_tokens=answer.prompt_tokens,
        completion_tokens=answer.completion_tokens,
        retrieval_seconds=answer.retrieval_seconds,
        answer_seconds=answer.total_seconds,
        ingest_seconds=ingest_seconds,
        sessions_ingested=ingested,
        store_items=store_items,
        store_tokens=store_tokens,
        truncated=answer.truncated,
        error=None,
    )
