import random

from membench.data.select import ABILITIES, ability_of, reduce_history, select_questions
from membench.data.types import Question, Session


def make_record(qid, qtype, n_sessions=20, evidence=(2, 5)):
    return {
        "question_id": qid,
        "question_type": qtype,
        "question": f"q for {qid}",
        "answer": "gold",
        "question_date": "2023/06/01 (Thu) 09:00",
        "haystack_session_ids": [f"{qid}-s{i}" for i in range(n_sessions)],
        "haystack_dates": [f"2023/05/{i + 1:02d} (Mon) 10:00" for i in range(n_sessions)],
        "haystack_sessions": [
            [{"role": "user", "content": f"turn {i}"}, {"role": "assistant", "content": "ok"}]
            for i in range(n_sessions)
        ],
        "answer_session_ids": [f"{qid}-s{i}" for i in evidence],
    }


def test_ability_of_maps_types_and_abstention():
    assert ability_of("a1", "single-session-user") == "extraction"
    assert ability_of("a2", "single-session-assistant") == "extraction"
    assert ability_of("a3", "single-session-preference") == "extraction"
    assert ability_of("a4", "multi-session") == "multi_session"
    assert ability_of("a5", "temporal-reasoning") == "temporal"
    assert ability_of("a6", "knowledge-update") == "knowledge_update"
    assert ability_of("a7_abs", "single-session-user") == "abstention"
    assert set(ABILITIES) == {
        "extraction", "multi_session", "temporal", "knowledge_update", "abstention"
    }


def test_reduce_history_keeps_evidence_and_hits_target():
    record = make_record("q1", "multi-session", n_sessions=30, evidence=(3, 17))
    sessions = reduce_history(record, target_sessions=12, rng=random.Random(42))
    ids = [s.session_id for s in sessions]
    assert len(sessions) == 12
    assert "q1-s3" in ids and "q1-s17" in ids


def test_reduce_history_preserves_original_order():
    record = make_record("q2", "multi-session", n_sessions=30, evidence=(3, 17))
    sessions = reduce_history(record, target_sessions=12, rng=random.Random(42))
    assert [s.order for s in sessions] == sorted(s.order for s in sessions)


def test_reduce_history_is_deterministic_for_a_seed():
    record = make_record("q3", "multi-session", n_sessions=30, evidence=(1,))
    first = reduce_history(record, 12, random.Random(42))
    second = reduce_history(record, 12, random.Random(42))
    assert [s.session_id for s in first] == [s.session_id for s in second]


def test_reduce_history_keeps_all_when_history_is_short():
    record = make_record("q4", "multi-session", n_sessions=5, evidence=(1,))
    assert len(reduce_history(record, 12, random.Random(42))) == 5


def test_select_questions_balances_abilities_and_is_deterministic():
    records = []
    for i in range(40):
        records.append(make_record(f"u{i}", "single-session-user"))
        records.append(make_record(f"m{i}", "multi-session"))
        records.append(make_record(f"t{i}", "temporal-reasoning"))
        records.append(make_record(f"k{i}", "knowledge-update"))
        records.append(make_record(f"x{i}_abs", "single-session-user"))

    picked = select_questions(records, per_ability=20, target_sessions=12, seed=42)
    counts = {a: sum(1 for q in picked if q.ability == a) for a in ABILITIES}

    assert len(picked) == 100
    assert counts == {a: 20 for a in ABILITIES}
    again = select_questions(records, per_ability=20, target_sessions=12, seed=42)
    assert [q.question_id for q in picked] == [q.question_id for q in again]


def test_select_questions_seeding_does_not_use_python_string_hash():
    """Cross-process reproducibility: the committed set must not depend on PYTHONHASHSEED."""
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import json, sys
        sys.path.insert(0, ".")
        from membench.data.select import select_questions
        records = []
        for i in range(40):
            for prefix, qtype in (("u", "single-session-user"), ("m", "multi-session"),
                                  ("t", "temporal-reasoning"), ("k", "knowledge-update")):
                records.append({
                    "question_id": f"{prefix}{i}", "question_type": qtype, "question": "q",
                    "answer": "a", "question_date": "d",
                    "haystack_session_ids": [f"{prefix}{i}-s{j}" for j in range(30)],
                    "haystack_dates": [f"2023/05/{j+1:02d}" for j in range(30)],
                    "haystack_sessions": [[{"role": "user", "content": str(j)}] for j in range(30)],
                    "answer_session_ids": [f"{prefix}{i}-s3"],
                })
            records.append({
                "question_id": f"x{i}_abs", "question_type": "single-session-user", "question": "q",
                "answer": "a", "question_date": "d",
                "haystack_session_ids": [f"x{i}-s{j}" for j in range(30)],
                "haystack_dates": [f"2023/05/{j+1:02d}" for j in range(30)],
                "haystack_sessions": [[{"role": "user", "content": str(j)}] for j in range(30)],
                "answer_session_ids": [f"x{i}-s3"],
            })
        picked = select_questions(records, 20, 12, 42)
        print(json.dumps([[q.question_id, [s.session_id for s in q.sessions]] for q in picked]))
        """
    )
    outs = []
    for hashseed in ("0", "12345"):
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, env={"PYTHONHASHSEED": hashseed, "PATH": "/usr/bin:/bin"},
        )
        assert proc.returncode == 0, proc.stderr
        outs.append(proc.stdout)
    assert outs[0] == outs[1]


def test_question_evidence_gap_counts_sessions_after_last_evidence():
    sessions = tuple(
        Session(session_id=f"s{i}", date="2023/05/01 (Mon) 10:00", order=i, turns=())
        for i in range(6)
    )
    question = Question(
        question_id="q",
        question_type="multi-session",
        ability="multi_session",
        question="?",
        answer="a",
        question_date="2023/06/01 (Thu) 09:00",
        is_abstention=False,
        sessions=sessions,
        evidence_session_ids=frozenset({"s2"}),
    )
    assert question.evidence_gap == 3
