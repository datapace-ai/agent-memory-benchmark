import json

from membench.config import SystemConfig
from membench.data.types import Question
import os

import pytest

from membench.config import ModelConfig
from membench.run import (
    _run_units,
    acquire_lock,
    append_record,
    apply_model_overrides,
    load_done,
    plan_work,
    record_key,
    release_lock,
)


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


def test_apply_model_overrides_changes_answerer_and_product_model_but_not_judge_unless_asked():
    base = ModelConfig(
        base_url="https://router.test/api/v1", answer_model="a", judge_model="j", embed_model="e",
        num_ctx=1, temperature=0.0, max_answer_tokens=1, seeds=(1,), openai_compat_model="a",
        provider="openai_compat", api_key_env=None,
    )
    track = apply_model_overrides(base, "vendor/other:free", "")
    assert track.answer_model == "vendor/other:free"
    assert track.openai_compat_model == "vendor/other:free"
    assert track.judge_model == "j"
    assert apply_model_overrides(base, "", "").answer_model == "a"
    assert apply_model_overrides(base, "", "judge2").judge_model == "judge2"


def test_run_units_with_workers_writes_every_unit_once(tmp_path, monkeypatch):
    """Four workers, twelve units, no duplicates, no losses, every record graded."""
    import membench.run as run_mod
    from membench.judge.judge import Verdicts
    from membench.systems.base import Answer, IngestStats, MemorySystem

    class Slow(MemorySystem):
        def __init__(self, name):
            self.name = name

        def reset(self, namespace):
            pass

        def ingest(self, session):
            return IngestStats(0.0, 1, 1)

        def answer(self, q, d):
            import time

            time.sleep(0.01)
            return Answer("a", "ctx", 5, 1, 0.0, 0.01)

    class FakeJudge:
        def grade(self, question, response, seed):
            return Verdicts(True, True, False, None, {})

    monkeypatch.setattr(run_mod, "build", lambda cfg, llm, seed=0, models=None: Slow(cfg.name))
    out = tmp_path / "runs.jsonl"
    work = [(s, question(f"q{i}"), 11) for s in systems() for i in range(6)]
    _run_units(work, llm=None, models=None, judge=FakeJudge(), provenance={"p": 1}, out=out, workers=4)
    lines = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(lines) == 12
    assert len({(l["system"], l["question_id"], l["seed"]) for l in lines}) == 12
    assert all(l["correct_longmemeval"] is True and l["provenance"] == {"p": 1} for l in lines)


def test_load_done_can_exclude_error_records_for_retry(tmp_path):
    path = tmp_path / "runs.jsonl"
    append_record(path, {"system": "window", "question_id": "q1", "seed": 11, "error": "boom"})
    append_record(path, {"system": "window", "question_id": "q2", "seed": 11, "error": None})
    assert load_done(path) == {("window", "q1", 11), ("window", "q2", 11)}
    assert load_done(path, retry_errors=True) == {("window", "q2", 11)}


def test_run_unit_records_a_build_failure_instead_of_raising(monkeypatch):
    import membench.run as run_mod

    def boom(cfg, llm, seed=0, models=None):
        raise RuntimeError("store is locked")

    monkeypatch.setattr(run_mod, "build", boom)
    record = run_mod.run_unit(systems()[1], question("q1"), 11, llm=None, models=None, judge=None, provenance={})
    assert record["error"] and "store is locked" in record["error"]
    assert record["correct_longmemeval"] is False and record["system"] == "window"


def test_serial_systems_run_after_the_pool_one_at_a_time(tmp_path, monkeypatch):
    import threading

    import membench.run as run_mod
    from membench.judge.judge import Verdicts
    from membench.systems.base import Answer, IngestStats, MemorySystem

    active = {"n": 0, "max_serial": 0}
    lock = threading.Lock()

    class Sys(MemorySystem):
        def __init__(self, name):
            self.name = name

        def reset(self, ns):
            pass

        def ingest(self, s):
            return IngestStats(0.0, 1, 1)

        def answer(self, q, d):
            import time

            if self.name == "cognee":
                with lock:
                    active["n"] += 1
                    active["max_serial"] = max(active["max_serial"], active["n"])
                time.sleep(0.02)
                with lock:
                    active["n"] -= 1
            return Answer("a", "", 1, 1, 0.0, 0.0)

    class J:
        def grade(self, *a):
            return Verdicts(True, True, True, None, {})

    monkeypatch.setattr(run_mod, "build", lambda cfg, llm, seed=0, models=None: Sys(cfg.name))
    cognee = SystemConfig(name="cognee", kind="cognee", params={"serial": True})
    window = SystemConfig(name="window", kind="context_window", params={"token_budget": 1})
    work = [(cognee, question(f"c{i}"), 11) for i in range(6)] + [(window, question(f"w{i}"), 11) for i in range(6)]
    out = tmp_path / "runs.jsonl"
    run_mod._run_units(work, None, None, J(), {}, out, workers=4)
    assert active["max_serial"] == 1
    assert len(out.read_text().splitlines()) == 12
