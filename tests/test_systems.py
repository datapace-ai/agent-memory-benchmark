import json as _json

import httpx

from membench.config import ModelConfig, SystemConfig
from membench.data.types import Session, Turn
from membench.llm import LLMClient
from membench.systems.base import Answer, IngestStats, MemorySystem
from membench.systems.registry import build, evidence_only

CFG = ModelConfig(
    base_url="http://ollama.test",
    answer_model="qwen3:14b",
    judge_model="qwen3:14b",
    embed_model="nomic-embed-text",
    num_ctx=40960,
    temperature=0.0,
    max_answer_tokens=512,
    seeds=(11,),
)


def fake_llm(captured=None):
    def handler(request: httpx.Request) -> httpx.Response:
        body = _json.loads(request.content)
        if captured is not None:
            captured.append(body)
        if request.url.path == "/api/embed":
            return httpx.Response(200, json={"embeddings": [[0.0]], "prompt_eval_count": 7})
        return httpx.Response(
            200,
            json={
                "message": {"content": "answer text"},
                "prompt_eval_count": 100,
                "eval_count": 5,
                "done_reason": "stop",
            },
        )

    return LLMClient(CFG, transport=httpx.MockTransport(handler))


def session(i: int, text: str) -> Session:
    return Session(
        session_id=f"s{i}",
        date=f"2023/05/{i + 1:02d} (Mon) 10:00",
        order=i,
        turns=(Turn(role="user", content=text), Turn(role="assistant", content="noted")),
    )


def window_system(token_budget=32000):
    cfg = SystemConfig(
        name="window",
        kind="context_window",
        params={"evidence_only": False, "token_budget": token_budget},
    )
    return build(cfg, fake_llm())


def test_build_returns_a_memory_system():
    system = window_system()
    assert isinstance(system, MemorySystem)
    assert system.name == "window"


def test_ingest_reports_stats_and_grows_the_store():
    system = window_system()
    system.reset("ns")
    first = system.ingest(session(0, "hello"))
    second = system.ingest(session(1, "again"))
    assert isinstance(first, IngestStats)
    assert first.store_items == 1
    assert second.store_items == 2
    assert second.store_tokens > 0
    assert second.seconds >= 0


def test_reset_clears_the_store():
    system = window_system()
    system.reset("ns")
    system.ingest(session(0, "hello"))
    system.reset("ns2")
    assert system.ingest(session(0, "hello")).store_items == 1


def test_answer_returns_context_and_counts():
    system = window_system()
    system.reset("ns")
    system.ingest(session(0, "the capital is Paris"))
    out = system.answer("What is the capital?", "2023/06/01 (Thu) 09:00")
    assert isinstance(out, Answer)
    assert out.text == "answer text"
    assert "Paris" in out.context
    assert out.prompt_tokens == 100
    assert out.completion_tokens == 5
    assert out.total_seconds >= out.retrieval_seconds


def test_window_drops_oldest_sessions_over_budget():
    system = window_system(token_budget=1)
    system.reset("ns")
    system.ingest(session(0, "oldest fact"))
    system.ingest(session(1, "newest fact"))
    out = system.answer("q", "2023/06/01 (Thu) 09:00")
    assert "newest fact" in out.context
    assert "oldest fact" not in out.context


def test_answer_prompt_carries_the_question_date():
    captured = []
    cfg = SystemConfig(
        name="window", kind="context_window", params={"evidence_only": False, "token_budget": 32000}
    )
    system = build(cfg, fake_llm(captured))
    system.reset("ns")
    system.ingest(session(0, "fact"))
    system.answer("What?", "2023/06/01 (Thu) 09:00")
    chat = [c for c in captured if "messages" in c][-1]
    assert "2023/06/01" in chat["messages"][1]["content"]


def test_evidence_only_flag_is_read_from_config():
    oracle = SystemConfig(
        name="oracle", kind="context_window", params={"evidence_only": True, "token_budget": 40000}
    )
    window = SystemConfig(
        name="window", kind="context_window", params={"evidence_only": False, "token_budget": 32000}
    )
    assert evidence_only(oracle) is True
    assert evidence_only(window) is False
