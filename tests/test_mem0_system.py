import json

import httpx

from membench.config import ModelConfig
from membench.data.types import Session, Turn
from membench.llm import LLMClient
from membench.systems.base import Answer, IngestStats, MemorySystem
from membench.systems.mem0_system import Mem0System, format_results

CFG = ModelConfig(
    base_url="http://ollama.test", answer_model="qwen3:14b", judge_model="qwen3:14b",
    embed_model="nomic-embed-text", num_ctx=40960, temperature=0.0, max_answer_tokens=512,
    seeds=(11,), think=False,
)


class FakeMemory:
    def __init__(self):
        self.added: list[tuple[list, str]] = []
        self.deleted: list[str] = []
        self.searched: list[tuple[str, dict, int]] = []
        self.results = [
            {"memory": "User adopted a dog named Rex", "score": 0.9},
            {"memory": "Rex is a beagle", "score": 0.8},
        ]

    def add(self, messages, *, user_id, **kwargs):
        self.added.append((messages, user_id))
        return {"results": [{"event": "ADD"}]}

    def search(self, query, *, filters, top_k, **kwargs):
        self.searched.append((query, filters, top_k))
        return {"results": self.results}

    def delete_all(self, *, user_id):
        self.deleted.append(user_id)

    def get_all(self, *, filters, **kwargs):
        return {"results": self.results[: len(self.added)]}


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


def make(tmp_path) -> tuple[Mem0System, FakeMemory]:
    memory = FakeMemory()
    system = Mem0System(
        name="mem0", llm=fake_llm(), cfg=CFG, seed=11, store_dir=tmp_path / "mem0",
        top_k=5, memory_factory=lambda namespace: memory,
    )
    return system, memory


def test_is_a_memory_system(tmp_path):
    system, _ = make(tmp_path)
    assert isinstance(system, MemorySystem)
    assert system.name == "mem0"


def test_reset_deletes_the_namespace_and_sets_the_user(tmp_path):
    system, memory = make(tmp_path)
    system.reset("q1:11")
    assert memory.deleted == ["q1:11"]


def test_ingest_adds_dated_messages_under_the_namespace(tmp_path):
    system, memory = make(tmp_path)
    system.reset("q1:11")
    stats = system.ingest(session())
    assert isinstance(stats, IngestStats)
    messages, user_id = memory.added[0]
    assert user_id == "q1:11"
    assert messages[0]["content"].startswith("[2023/05/20")
    assert messages[1] == {"role": "assistant", "content": "Nice."}
    assert stats.store_items == 1
    assert stats.seconds >= 0


def test_answer_searches_with_user_filter_and_answers_from_results(tmp_path):
    system, memory = make(tmp_path)
    system.reset("q1:11")
    system.ingest(session())
    out = system.answer("What is my dog's name?", "2023/06/01 (Thu) 09:00")
    assert isinstance(out, Answer)
    query, filters, top_k = memory.searched[0]
    assert query == "What is my dog's name?"
    assert filters == {"user_id": "q1:11"}
    assert top_k == 5
    assert "Rex is a beagle" in out.context
    assert out.text == "Rex"
    assert out.prompt_tokens > 0
    assert out.retrieval_seconds >= 0 and out.total_seconds >= out.retrieval_seconds


def test_format_results_lists_memories_one_per_line():
    assert format_results([{"memory": "a"}, {"memory": "b"}]) == "- a\n- b"
    assert format_results([]) == "(no memories retrieved)"


def test_answer_before_reset_raises(tmp_path):
    system, _ = make(tmp_path)
    try:
        system.answer("q", "d")
    except RuntimeError as exc:
        assert "reset" in str(exc)
    else:
        raise AssertionError("answer without reset must fail loudly")
