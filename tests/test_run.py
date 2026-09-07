import json

from membench.config import SystemConfig
from membench.data.types import Question
import os

import pytest

from membench.run import acquire_lock, append_record, load_done, plan_work, record_key, release_lock


def question(qid):
    return Question(
        question_id=qid, question_type="multi-session", ability="multi_session",
        question="?", answer="a", question_date="2023/06/01 (Thu) 09:00", is_abstention=False,
    )


def systems():
    return (
        SystemConfig(name="oracle", kind="context_window",
                     params={"evidence_only": True, "token_budget": 40000}),
        SystemConfig(name="window", kind="context_window",
                     params={"evidence_only": False, "token_budget": 32000}),
    )


def test_record_key_is_system_question_seed():
    assert record_key({"system": "window", "question_id": "q1", "seed": 11}) == ("window", "q1", 11)


def test_append_and_load_done_round_trip(tmp_path):
    path = tmp_path / "runs.jsonl"
    append_record(path, {"system": "window", "question_id": "q1", "seed": 11, "answer_text": "x"})
    append_record(path, {"system": "oracle", "question_id": "q2", "seed": 22, "answer_text": "y"})
    assert load_done(path) == {("window", "q1", 11), ("oracle", "q2", 22)}


def test_load_done_on_missing_file_is_empty(tmp_path):
    assert load_done(tmp_path / "nope.jsonl") == set()


def test_load_done_skips_corrupt_lines(tmp_path):
    path = tmp_path / "runs.jsonl"
    path.write_text('{"system":"window","question_id":"q1","seed":11}\nnot json\n')
    assert load_done(path) == {("window", "q1", 11)}


def test_plan_work_is_the_full_grid_when_nothing_is_done():
    assert len(plan_work(systems(), [question("q1"), question("q2")], (11, 22), done=set())) == 8


def test_plan_work_skips_completed_units():
    done = {("window", "q1", 11), ("oracle", "q2", 22)}
    work = plan_work(systems(), [question("q1"), question("q2")], (11, 22), done)
    assert len(work) == 6
    assert all((s.name, q.question_id, seed) not in done for s, q, seed in work)


def test_plan_work_groups_by_system_then_question():
    work = plan_work(systems(), [question("q1"), question("q2")], (11,), done=set())
    assert [s.name for s, _, _ in work] == ["oracle", "oracle", "window", "window"]


def test_appended_records_are_valid_json_lines(tmp_path):
    path = tmp_path / "runs.jsonl"
    append_record(path, {"system": "window", "question_id": "q1", "seed": 11, "nested": {"a": 1}})
    assert json.loads(path.read_text().strip())["nested"] == {"a": 1}


def test_acquire_lock_refuses_while_holder_is_alive(tmp_path):
    out = tmp_path / "runs.jsonl"
    lock = acquire_lock(out)
    assert lock.read_text() == str(os.getpid())
    with pytest.raises(RuntimeError):
        acquire_lock(out)
    release_lock(lock)
    assert not lock.exists()


def test_acquire_lock_replaces_a_stale_lock(tmp_path):
    out = tmp_path / "runs.jsonl"
    stale = out.with_name("runs.jsonl.lock")
    dead_pid = 2**22 - 7
    try:
        os.kill(dead_pid, 0)
        pytest.skip("pid unexpectedly alive")
    except ProcessLookupError:
        pass
    stale.write_text(str(dead_pid))
    lock = acquire_lock(out)
    assert lock.read_text() == str(os.getpid())
    release_lock(lock)
