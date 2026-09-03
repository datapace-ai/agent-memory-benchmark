from membench.data.types import Question, Session, Turn
from membench.protocol.clock import QuestionRun, run_question
from membench.systems.base import Answer, IngestStats, MemorySystem


class RecordingSystem(MemorySystem):
    name = "recording"

    def __init__(self, fail_on: str | None = None):
        self.resets: list[str] = []
        self.ingested: list[str] = []
        self.asked: tuple[str, str] | None = None
        self.fail_on = fail_on

    def reset(self, namespace: str) -> None:
        self.resets.append(namespace)
        self.ingested = []

    def ingest(self, session: Session) -> IngestStats:
        if self.fail_on == "ingest":
            raise RuntimeError("ingest exploded")
        self.ingested.append(session.session_id)
        return IngestStats(
            seconds=0.01, store_items=len(self.ingested), store_tokens=10 * len(self.ingested)
        )

    def answer(self, question: str, question_date: str) -> Answer:
        if self.fail_on == "answer":
            raise RuntimeError("answer exploded")
        self.asked = (question, question_date)
        return Answer(
            text="an answer",
            context="ctx",
            prompt_tokens=42,
            completion_tokens=7,
            retrieval_seconds=0.02,
            total_seconds=0.05,
        )


def make_question(n=4, evidence=("s1",)):
    sessions = tuple(
        Session(
            session_id=f"s{i}",
            date=f"2023/05/{i + 1:02d} (Mon) 10:00",
            order=i,
            turns=(Turn(role="user", content=f"turn {i}"),),
        )
        for i in range(n)
    )
    return Question(
        question_id="q1",
        question_type="multi-session",
        ability="multi_session",
        question="What?",
        answer="gold",
        question_date="2023/06/01 (Thu) 09:00",
        is_abstention=False,
        sessions=sessions,
        evidence_session_ids=frozenset(evidence),
    )


def test_run_question_resets_then_ingests_in_order_then_asks():
    system = RecordingSystem()
    out = run_question(system, make_question(), seed=11)

    assert system.resets == ["q1:11"]
    assert system.ingested == ["s0", "s1", "s2", "s3"]
    assert system.asked == ("What?", "2023/06/01 (Thu) 09:00")
    assert isinstance(out, QuestionRun)
    assert out.answer_text == "an answer"
    assert out.sessions_ingested == 4
    assert out.prompt_tokens == 42
    assert out.error is None


def test_evidence_only_feeds_just_the_evidence_sessions():
    system = RecordingSystem()
    run_question(system, make_question(evidence=("s1", "s3")), seed=11, evidence_only=True)
    assert system.ingested == ["s1", "s3"]


def test_ingest_failure_is_recorded_not_raised():
    out = run_question(RecordingSystem(fail_on="ingest"), make_question(), seed=11)
    assert out.error is not None
    assert "ingest exploded" in out.error
    assert out.answer_text == ""


def test_answer_failure_is_recorded_not_raised():
    out = run_question(RecordingSystem(fail_on="answer"), make_question(), seed=11)
    assert out.error is not None
    assert "answer exploded" in out.error
    assert out.sessions_ingested == 4
