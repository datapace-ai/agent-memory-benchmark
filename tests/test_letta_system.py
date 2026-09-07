from types import SimpleNamespace

import httpx

from membench.config import ModelConfig
from membench.data.types import Session, Turn
from membench.llm import LLMClient
from membench.systems.base import Answer, IngestStats, MemorySystem
from membench.systems.letta_system import LettaSystem, extract_answer, extract_usage

CFG = ModelConfig(
    base_url="http://ollama.test", answer_model="qwen3:14b", judge_model="qwen3:14b",
    embed_model="nomic-embed-text", num_ctx=40960, temperature=0.0, max_answer_tokens=512,
    seeds=(11,), think=False,
)


class FakeAgents:
    def __init__(self):
        self.created: list[dict] = []
        self.deleted: list[str] = []
        self.sent: list[tuple[str, list, int]] = []
        self.messages = self
        self.passages = self
        self.n = 0

    def create(self, **params):
        if "agent_id" in params:
            self.sent.append((params["agent_id"], params.get("messages", []), params.get("max_steps", 0)))
            return SimpleNamespace(
                messages=[SimpleNamespace(message_type="assistant_message", content="Rex")],
                usage=SimpleNamespace(prompt_tokens=120, completion_tokens=6),
            )
        self.created.append(params)
        self.n += 1
        return SimpleNamespace(id=f"agent-{self.n}")

    def delete(self, agent_id):
        self.deleted.append(agent_id)


class FakeClient:
    def __init__(self):
        self.agents = FakeAgents()


def session(i=0):
    return Session(session_id=f"s{i}", date="2023/05/20 (Sat) 02:21", order=i,
                   turns=(Turn(role="user", content="I adopted a dog named Rex."),
                          Turn(role="assistant", content="Nice.")))


def make():
    client = FakeClient()
    llm = LLMClient(CFG, transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    system = LettaSystem("letta", llm, CFG, 11, client_factory=lambda: client)
    return system, client


def test_is_a_memory_system():
    system, _ = make()
    assert isinstance(system, MemorySystem) and system.name == "letta"


def test_reset_creates_an_agent_with_ollama_handles_and_reasoning_off_and_deletes_the_previous():
    system, client = make()
    system.reset("q1:11")
    system.reset("q2:11")
    assert client.agents.deleted == ["agent-1"]
    params = client.agents.created[-1]
    assert params["model"] == "ollama/qwen3:14b"
    assert params["embedding"] == "ollama/nomic-embed-text"
    assert params["reasoning"] is False and params["enable_reasoner"] is False
    assert params["include_base_tools"] is True
    assert any(b["label"] == "human" for b in params["memory_blocks"])
    assert ":" not in params["name"]


def test_ingest_sends_the_session_as_dated_user_messages_with_bounded_steps():
    system, client = make()
    system.reset("q1:11")
    stats = system.ingest(session())
    agent_id, messages, max_steps = client.agents.sent[0]
    assert agent_id == "agent-1"
    assert messages[0]["role"] == "user" and messages[0]["content"].startswith("[2023/05/20")
    assert "Full conversation on 2023/05/20" in messages[1]["content"]
    assert max_steps >= 1
    assert isinstance(stats, IngestStats) and stats.store_items == 1


def test_answer_asks_the_agent_and_reads_letta_usage():
    system, client = make()
    system.reset("q1:11")
    system.ingest(session())
    out = system.answer("What is my dog's name?", "2023/06/01 (Thu) 09:00")
    _, messages, _ = client.agents.sent[-1]
    assert "What is my dog's name?" in messages[-1]["content"]
    assert "2023/06/01" in messages[-1]["content"]
    assert isinstance(out, Answer) and out.text == "Rex"
    assert out.prompt_tokens == 120 and out.completion_tokens == 6
    assert out.context == "(answered inside Letta; see agent memory)"


def test_extract_answer_picks_the_last_assistant_message_and_tolerates_dicts():
    resp = SimpleNamespace(messages=[
        SimpleNamespace(message_type="reasoning_message", reasoning="thinking"),
        {"message_type": "assistant_message", "content": "first"},
        SimpleNamespace(message_type="assistant_message", content="final"),
    ])
    assert extract_answer(resp) == "final"
    assert extract_answer(SimpleNamespace(messages=[])) == ""


def test_extract_usage_defaults_to_zero():
    assert extract_usage(SimpleNamespace()) == (0, 0)
    assert extract_usage(SimpleNamespace(usage={"prompt_tokens": 3, "completion_tokens": 4})) == (3, 4)
