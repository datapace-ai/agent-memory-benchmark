import json

import httpx

from membench.config import ModelConfig
from membench.data.types import Session, Turn
from membench.llm import LLMClient
from membench.systems.base import Answer, IngestStats, MemorySystem
from membench.systems.langmem_system import LangMemSystem, format_memories

CFG = ModelConfig(
    base_url="http://ollama.test", answer_model="qwen3:14b", judge_model="qwen3:14b",
    embed_model="nomic-embed-text", num_ctx=40960, temperature=0.0, max_answer_tokens=512,
    seeds=(11,), think=False,
)


class FakeItem:
    def __init__(self, content):
        self.value = {"content": content}


class FakeStore:
    instances = 0

    def __init__(self):
        FakeStore.instances += 1
        self.items: list[FakeItem] = []


class FakeManager:
    def __init__(self, store):
        self.store = store
        self.invoked: list[tuple[dict, dict]] = []
        self.searched: list[tuple[str, dict]] = []

    def invoke(self, payload, config):
        self.invoked.append((payload, config))
        for m in payload["messages"]:
            if m["role"] == "user":
                self.store.items.append(FakeItem(f"fact from: {m['content'][:30]}"))
        return []

    def search(self, query, config):
        self.searched.append((query, config))
        return list(self.store.items)


def fake_llm():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json={"message": {"content": "Rex"},
                  "prompt_eval_count": len(body["messages"][1]["content"]) // 4,
                  "eval_count": 1, "done_reason": "stop"},
        )

    return LLMClient(CFG, transport=httpx.MockTransport(handler))


def session(i=0):
    return Session(
        session_id=f"s{i}", date="2023/05/20 (Sat) 02:21", order=i,
        turns=(Turn(role="user", content="I adopted a dog named Rex."),
               Turn(role="assistant", content="Nice.")),
    )


def make():
    managers: list[FakeManager] = []

    def manager_factory(store):
        manager = FakeManager(store)
        managers.append(manager)
        return manager

    system = LangMemSystem(
        name="langmem", llm=fake_llm(), cfg=CFG, seed=11, top_k=5,
        store_factory=FakeStore, manager_factory=manager_factory,
    )
    return system, managers


def test_is_a_memory_system():
    system, _ = make()
    assert isinstance(system, MemorySystem)
    assert system.name == "langmem"


def test_reset_builds_a_fresh_store_and_manager_each_time():
    system, managers = make()
    before = FakeStore.instances
    system.reset("q1:11")
    system.reset("q1:22")
    assert FakeStore.instances == before + 2
    assert len(managers) == 2
    assert managers[0].store is not managers[1].store


def test_ingest_invokes_the_manager_with_dated_messages_and_the_namespace_user():
    system, managers = make()
    system.reset("q1:11")
    stats = system.ingest(session())
    payload, config = managers[-1].invoked[0]
    assert payload["messages"][0]["content"].startswith("[2023/05/20")
    assert config["configurable"]["langgraph_user_id"] == "q1:11"
    assert isinstance(stats, IngestStats)
    assert stats.store_items == 1


def test_answer_searches_then_answers_from_the_memories():
    system, managers = make()
    system.reset("q1:11")
    system.ingest(session())
    out = system.answer("What is my dog's name?", "2023/06/01 (Thu) 09:00")
    query, config = managers[-1].searched[-1]
    assert query == "What is my dog's name?"
    assert config["configurable"]["langgraph_user_id"] == "q1:11"
    assert isinstance(out, Answer)
    assert "fact from:" in out.context
    assert out.text == "Rex"


def test_format_memories_handles_items_dicts_and_strings():
    assert format_memories([FakeItem("a"), {"value": {"content": "b"}}, "c"]) == "- a\n- b\n- c"
    assert format_memories([]) == "(no memories retrieved)"
