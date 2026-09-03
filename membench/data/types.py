"""Dataset value objects. These are the only shapes the rest of the harness sees."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Turn:
    role: str
    content: str
    has_answer: bool = False


@dataclass(frozen=True)
class Session:
    session_id: str
    date: str
    order: int
    turns: tuple[Turn, ...] = ()

    def as_text(self) -> str:
        body = "\n".join(f"{t.role}: {t.content}" for t in self.turns)
        return f"[session {self.session_id} on {self.date}]\n{body}"


@dataclass(frozen=True)
class Question:
    question_id: str
    question_type: str
    ability: str
    question: str
    answer: str
    question_date: str
    is_abstention: bool
    sessions: tuple[Session, ...] = field(default_factory=tuple)
    evidence_session_ids: frozenset[str] = field(default_factory=frozenset)

    @property
    def evidence_gap(self) -> int:
        """Sessions between the last evidence session and the end of the history."""
        positions = [
            i for i, s in enumerate(self.sessions) if s.session_id in self.evidence_session_ids
        ]
        if not positions:
            return len(self.sessions)
        return len(self.sessions) - 1 - max(positions)

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "question_type": self.question_type,
            "ability": self.ability,
            "question": self.question,
            "answer": self.answer,
            "question_date": self.question_date,
            "is_abstention": self.is_abstention,
            "evidence_session_ids": sorted(self.evidence_session_ids),
            "sessions": [
                {
                    "session_id": s.session_id,
                    "date": s.date,
                    "order": s.order,
                    "turns": [
                        {"role": t.role, "content": t.content, "has_answer": t.has_answer}
                        for t in s.turns
                    ],
                }
                for s in self.sessions
            ],
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "Question":
        return cls(
            question_id=raw["question_id"],
            question_type=raw["question_type"],
            ability=raw["ability"],
            question=raw["question"],
            answer=raw["answer"],
            question_date=raw["question_date"],
            is_abstention=raw["is_abstention"],
            evidence_session_ids=frozenset(raw["evidence_session_ids"]),
            sessions=tuple(
                Session(
                    session_id=s["session_id"],
                    date=s["date"],
                    order=s["order"],
                    turns=tuple(
                        Turn(
                            role=t["role"],
                            content=t["content"],
                            has_answer=t.get("has_answer", False),
                        )
                        for t in s["turns"]
                    ),
                )
                for s in raw["sessions"]
            ),
        )


def write_jsonl(questions: list[Question], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for question in questions:
            handle.write(json.dumps(question.to_dict()) + "\n")


def read_jsonl(path: Path) -> list[Question]:
    return [Question.from_dict(json.loads(line)) for line in path.read_text().splitlines() if line]
