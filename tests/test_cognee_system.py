import asyncio
import json

import httpx

from membench.config import ModelConfig
from membench.data.types import Session, Turn
from membench.llm import LLMClient
from membench.systems.base import Answer, IngestStats, MemorySystem
from membench.systems.cognee_system import CogneeSystem, cognee_environment, format_context

CFG = ModelConfig(
    base_url="http://ollama.test", answer_model="qwen3:14b", judge_model="qwen3:14b",
    embed_model="nomic-embed-text", num_ctx=40960, temperature=0.0, max_answer_tokens=512,
    seeds=(11,), think=False,
)


class FakePrune:
    def __init__(self):
        self.data_calls = 0
        self.system_calls = []

    async def prune_data(self):
        self.data_calls += 1

    async def prune_system(self, **kwargs):
        self.system_calls.append(kwargs)


class FakeSearchType:
    GRAPH_COMPLETION = "GRAPH_COMPLETION"
    CHUNKS = "CHUNKS"


class FakeCognee:
    def __init__(self):
        self.prune = FakePrune()
        self.SearchType = FakeSearchType
        self.added: list[tuple[str, str]] = []
        self.cognified: list[list[str]] = []
        self.searched: list[dict] = []

    async def add(self, data, dataset_name):
        self.added.append((data, dataset_name))

    async def cognify(self, datasets):
        self.cognified.append(list(datasets))

    async def search(self, **kwargs):
        self.searched.append(kwargs)
        return ["Rex is the user's beagle", "Adopted in May"]


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


def make(tmp_path):
    cognee = FakeCognee()
    system = CogneeSystem(
        name="cognee", llm=fake_llm(), cfg=CFG, seed=11, store_dir=tmp_path / "cognee",
        top_k=5, cognee_module=cognee,
    )
    return system, cognee


def test_environment_points_cognee_at_ollama_with_reasoning_off(tmp_path):
    env = cognee_environment(CFG, seed=11, store_dir=tmp_path)
    assert env["LLM_PROVIDER"] == "ollama"
    assert env["LLM_MODEL"] == "qwen3:14b"
    assert env["LLM_ENDPOINT"] == "http://ollama.test/v1"
    assert env["EMBEDDING_PROVIDER"] == "ollama"
    assert env["EMBEDDING_MODEL"] == "nomic-embed-text"
    assert env["EMBEDDING_ENDPOINT"] == "http://ollama.test/api/embed"
    assert env["EMBEDDING_DIMENSIONS"] == "768"
    assert json.loads(env["LLM_ARGS"]) == {"think": False}
    assert env["LLM_TEMPERATURE"] == "0.0" and env["LLM_SEED"] == "11"
    assert env["DB_PROVIDER"] == "sqlite"
    assert env["GRAPH_DATABASE_PROVIDER"] == "kuzu"
    assert env["VECTOR_DB_PROVIDER"] == "lancedb"
    assert env["DATA_ROOT_DIRECTORY"].startswith(str(tmp_path))
    assert env["SYSTEM_ROOT_DIRECTORY"].startswith(str(tmp_path))


def test_is_a_memory_system(tmp_path):
    system, _ = make(tmp_path)
    assert isinstance(system, MemorySystem)
    assert system.name == "cognee"


def test_first_reset_prunes_once_and_later_resets_do_not(tmp_path):
    system, cognee = make(tmp_path)
    system.reset("q1:11")
    system.reset("q2:11")
    assert cognee.prune.data_calls == 1
    assert cognee.prune.system_calls == [{"metadata": True}]


def test_ingest_adds_session_text_to_the_namespace_dataset_and_cognifies(tmp_path):
    system, cognee = make(tmp_path)
    system.reset("q1:11")
    stats = system.ingest(session())
    data, dataset = cognee.added[0]
    assert "Rex" in data and "2023/05/20" in data
    assert dataset == "q1_11"
    assert cognee.cognified == [["q1_11"]]
    assert isinstance(stats, IngestStats)
    assert stats.store_items == 1 and stats.store_tokens > 0


def test_answer_searches_graph_context_in_the_dataset_and_answers(tmp_path):
    system, cognee = make(tmp_path)
    system.reset("q1:11")
    system.ingest(session())
    out = system.answer("What is my dog's name?", "2023/06/01 (Thu) 09:00")
    call = cognee.searched[0]
    assert call["query_text"] == "What is my dog's name?"
    assert call["query_type"] == "GRAPH_COMPLETION"
    assert call["datasets"] == ["q1_11"]
    assert call["top_k"] == 5
    assert call["only_context"] is True
    assert isinstance(out, Answer)
    assert "beagle" in out.context
    assert out.text == "Rex"


def test_format_context_accepts_lists_dicts_and_strings():
    assert format_context(["a", {"text": "b"}, {"content": "c"}]) == "- a\n- b\n- c"
    assert format_context("plain") == "plain"
    assert format_context([]) == "(no context retrieved)"


def test_calls_run_even_when_an_event_loop_is_not_running(tmp_path):
    system, cognee = make(tmp_path)
    system.reset("q1:11")
    assert asyncio.iscoroutinefunction(cognee.add)
    system.ingest(session())
    assert len(cognee.added) == 1
