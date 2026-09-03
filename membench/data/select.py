"""Select 100 balanced questions and reduce each history to a fixed session count.

The reduction keeps every evidence session, samples distractors from the same
question's own haystack with a fixed seed, and preserves the original order,
which longmemeval_s guarantees is chronological.

Per-question seeds come from sha256, not Python's built-in hash, because
PYTHONHASHSEED randomizes string hashing per process and would make the
committed question set irreproducible.
"""

from __future__ import annotations

import hashlib
import random

from membench.data.types import Question, Session, Turn

ABILITIES = ("extraction", "multi_session", "temporal", "knowledge_update", "abstention")

_TYPE_TO_ABILITY = {
    "single-session-user": "extraction",
    "single-session-assistant": "extraction",
    "single-session-preference": "extraction",
    "multi-session": "multi_session",
    "temporal-reasoning": "temporal",
    "knowledge-update": "knowledge_update",
}


def ability_of(question_id: str, question_type: str) -> str:
    if question_id.endswith("_abs"):
        return "abstention"
    try:
        return _TYPE_TO_ABILITY[question_type]
    except KeyError:
        raise ValueError(f"Unknown question_type: {question_type}")


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256(":".join(str(p) for p in parts).encode()).hexdigest()
    return int(digest[:8], 16)


def _sessions_of(record: dict) -> list[Session]:
    sessions = []
    for order, (sid, date, turns) in enumerate(
        zip(record["haystack_session_ids"], record["haystack_dates"], record["haystack_sessions"])
    ):
        sessions.append(
            Session(
                session_id=sid,
                date=date,
                order=order,
                turns=tuple(
                    Turn(
                        role=t["role"],
                        content=t["content"],
                        has_answer=bool(t.get("has_answer", False)),
                    )
                    for t in turns
                ),
            )
        )
    return sessions


def reduce_history(record: dict, target_sessions: int, rng: random.Random) -> tuple[Session, ...]:
    sessions = _sessions_of(record)
    if len(sessions) <= target_sessions:
        return tuple(sessions)

    evidence_ids = set(record.get("answer_session_ids", []))
    kept = [s for s in sessions if s.session_id in evidence_ids]
    distractors = [s for s in sessions if s.session_id not in evidence_ids]
    room = max(0, target_sessions - len(kept))
    kept.extend(rng.sample(distractors, min(room, len(distractors))))
    return tuple(sorted(kept, key=lambda s: s.order))


def select_questions(
    records: list[dict], per_ability: int, target_sessions: int, seed: int
) -> list[Question]:
    by_ability: dict[str, list[dict]] = {a: [] for a in ABILITIES}
    for record in records:
        ability = ability_of(record["question_id"], record["question_type"])
        by_ability[ability].append(record)

    rng = random.Random(seed)
    picked: list[Question] = []
    for ability in ABILITIES:
        pool = sorted(by_ability[ability], key=lambda r: r["question_id"])
        if len(pool) < per_ability:
            raise ValueError(f"Only {len(pool)} questions available for ability {ability}")
        for record in rng.sample(pool, per_ability):
            picked.append(
                Question(
                    question_id=record["question_id"],
                    question_type=record["question_type"],
                    ability=ability,
                    question=record["question"],
                    answer=record["answer"],
                    question_date=record["question_date"],
                    is_abstention=ability == "abstention",
                    sessions=reduce_history(
                        record,
                        target_sessions,
                        random.Random(stable_seed(seed, record["question_id"])),
                    ),
                    evidence_session_ids=frozenset(record.get("answer_session_ids", [])),
                )
            )
    return picked
