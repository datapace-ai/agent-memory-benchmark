import json
from datetime import timezone

import httpx

from membench.config import ModelConfig
from membench.data.types import Session, Turn
from membench.llm import LLMClient
from membench.systems.base import Answer, IngestStats, MemorySystem
from membench.systems.graphiti_system import GraphitiSystem, format_edges, parse_session_date, reasoning_off_client

CFG = ModelConfig(
    base_url="http://ollama.test", answer_model="qwen3:14b", judge_model="qwen3:14b",
    embed_model="nomic-embed-text", num_ctx=40960, temperature=0.0, max_answer_tokens=512,
    seeds=(11,), think=False,
)


class FakeEdge:
    def __init__(self, fact):
        self.fact = fact


class FakeGraphiti:
    def __init__(self):
        self.built = 0
        self.episodes: list[dict] = []
        self.searches: list[dict] = []
        self.closed = False

    async def build_indices_and_constraints(self):
        self.built += 1

    async def add_episode(self, **kwargs):
        self.episodes.append(kwargs)

    async def search(self, query, group_ids=None, num_results=10):
        self.searches.append({"query": query, "group_ids": group_ids, "num_results": num_results})
        return [FakeEdge("Rex is a beagle"), FakeEdge("User adopted Rex in May 2023")]

    async def close(self):
        self.closed = True


def fake_llm():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return httpx.Response(200, json={"message": {"content": "Rex"},
                                         "prompt_eval_count": len(body["messages"][1]["content"]) // 4,
                                         "eval_count": 1, "done_reason": "stop"})

    return LLMClient(CFG, transport=httpx.MockTransport(handler))


def session(i=0):
    return Session(session_id=f"s{i}", date="2023/05/20 (Sat) 02:21", order=i,
                   turns=(Turn(role="user", content="I adopted a dog named Rex."),
                          Turn(role="assistant", content="Nice.")))


def make(tmp_path):
    made: list[FakeGraphiti] = []

    def factory(namespace):
        g = FakeGraphiti()
        made.append(g)
        return g

    system = GraphitiSystem(
        "graphiti", fake_llm(), CFG, 11, tmp_path / "graphiti", top_k=5, graphiti_factory=factory
    )
    return system, made


def test_parse_session_date_is_timezone_aware_utc():
    dt = parse_session_date("2023/05/20 (Sat) 02:21")
    assert (dt.year, dt.month, dt.day, dt.hour, dt.minute) == (2023, 5, 20, 2, 21)
    assert dt.tzinfo == timezone.utc
    assert parse_session_date("garbage").tzinfo == timezone.utc


def test_is_a_memory_system(tmp_path):
    system, _ = make(tmp_path)
    assert isinstance(system, MemorySystem) and system.name == "graphiti"


def test_reset_builds_a_fresh_graph_with_indices_and_closes_the_old_one(tmp_path):
    system, made = make(tmp_path)
    system.reset("q1:11")
    system.reset("q2:11")
    assert len(made) == 2 and made[0].closed and not made[1].closed
    assert made[1].built == 1


def test_ingest_adds_one_episode_per_session_with_group_and_reference_time(tmp_path):
    system, made = make(tmp_path)
    system.reset("q1:11")
    stats = system.ingest(session())
    stats = system.ingest(session(1))
    eps = made[-1].episodes
    assert len(eps) == 2
    assert eps[0]["group_id"] == "q1_11"
    assert eps[0]["reference_time"].tzinfo == timezone.utc
    body = eps[0]["episode_body"]
    assert body.startswith("user: I adopted a dog named Rex.") and "\nassistant: Nice." in body
    assert eps[0]["name"] == "s0" and eps[1]["name"] == "s1"
    assert isinstance(stats, IngestStats) and stats.store_items == 2


def test_reasoning_off_client_sends_the_flag_on_every_completion():
    import asyncio

    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "x", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "{}"}}],
        })

    client = reasoning_off_client(
        "k", "http://router.test/api/v1", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    asyncio.run(client.chat.completions.create(
        model="m", messages=[{"role": "user", "content": "hi"}], extra_body={"seed": 1}
    ))
    assert seen[0]["reasoning"] == {"enabled": False} and seen[0]["seed"] == 1


def test_answer_searches_the_group_and_answers_from_facts(tmp_path):
    system, made = make(tmp_path)
    system.reset("q1:11")
    system.ingest(session())
    out = system.answer("What is my dog's name?", "2023/06/01 (Thu) 09:00")
    call = made[-1].searches[0]
    assert call["query"] == "What is my dog's name?"
    assert call["group_ids"] == ["q1_11"] and call["num_results"] == 5
    assert isinstance(out, Answer) and "beagle" in out.context and out.text == "Rex"


def test_format_edges_lists_facts():
    assert format_edges([FakeEdge("a"), FakeEdge("b"), {"fact": "c"}]) == "- a\n- b\n- c"
    assert format_edges([]) == "(no facts retrieved)"
