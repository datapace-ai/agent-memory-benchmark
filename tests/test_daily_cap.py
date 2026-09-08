import json

import httpx
import pytest

from membench import run as run_mod
from membench.config import ModelConfig
from membench.llm import DailyCapExceeded, LLMClient, is_daily_cap
from membench.systems.shared import retry_transient

CAP_BODY = {"error": {"message": "Rate limit exceeded: free-models-per-day-high-balance. ", "code": 429,
                      "metadata": {"limit_source": "openrouter_free_tier_daily"}}}

CFG = ModelConfig(
    base_url="http://router.test/api/v1", answer_model="m", judge_model="m", embed_model="e",
    num_ctx=4096, temperature=0.0, max_answer_tokens=64, seeds=(11,), think=False,
    provider="openai_compat", openai_compat_model="m", api_key_env="",
)


def test_client_raises_the_daily_cap_after_one_attempt(monkeypatch):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(429, json=CAP_BODY)

    client = LLMClient(CFG, transport=httpx.MockTransport(handler))
    monkeypatch.setattr(client, "_sleep", lambda s: None)
    with pytest.raises(DailyCapExceeded):
        client.complete("s", "u", seed=11)
    assert calls["n"] == 1


def test_is_daily_cap_recognises_vendor_exceptions_by_text():
    assert is_daily_cap(RuntimeError("Error code: 429 - " + json.dumps(CAP_BODY)))
    assert not is_daily_cap(RuntimeError("Upstream error from Nvidia: overloaded"))


def test_retry_transient_does_not_retry_the_daily_cap():
    slept = []

    def capped():
        raise RuntimeError("free-models-per-day-high-balance")

    with pytest.raises(RuntimeError):
        retry_transient(capped, what="x", sleep=slept.append)
    assert slept == []


def test_run_units_stops_at_the_daily_cap_and_keeps_finished_units(tmp_path, monkeypatch):
    from membench.data.types import Question

    q = lambda i: Question(question_id=f"q{i}", question_type="t", ability="extraction", question="?",
                           answer="a", question_date="2023/01/01 (Sun) 00:00", is_abstention=False)
    from membench.config import SystemConfig

    sys_cfg = SystemConfig(name="window", kind="context_window", params={})
    work = [(sys_cfg, q(1), 11), (sys_cfg, q(2), 11), (sys_cfg, q(3), 11)]
    calls = {"n": 0}

    def fake_run_unit(system_cfg, question, seed, llm, models, judge, provenance):
        calls["n"] += 1
        if question.question_id == "q2":
            raise DailyCapExceeded("HTTP 429: free-models-per-day")
        return {"system": system_cfg.name, "question_id": question.question_id, "seed": seed,
                "correct_longmemeval": True, "prompt_tokens": 1, "error": None}

    monkeypatch.setattr(run_mod, "run_unit", fake_run_unit)
    out = tmp_path / "runs.jsonl"
    run_mod._run_units(work, None, None, None, {}, out, workers=1)
    records = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert [r["question_id"] for r in records] == ["q1"]
    assert calls["n"] == 2
