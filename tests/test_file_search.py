import json

import httpx

from membench.config import ModelConfig
from membench.data.types import Session, Turn
from membench.llm import LLMClient
from membench.systems.base import Answer, IngestStats, MemorySystem
from membench.systems.file_search import FileSearchSystem

CFG = ModelConfig(
    base_url="http://ollama.test", answer_model="qwen3:14b", judge_model="qwen3:14b",
    embed_model="nomic-embed-text", num_ctx=40960, temperature=0.0, max_answer_tokens=512,
    seeds=(11,), think=False,
)


def scripted_llm(script):
    state = {"i": 0, "bodies": []}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        state["bodies"].append(body)
        step = script[min(state["i"], len(script) - 1)]
        state["i"] += 1
        if isinstance(step, str):
            message = {"role": "assistant", "content": step}
        else:
            message = {"role": "assistant", "content": "", "tool_calls": step}
        return httpx.Response(200, json={"message": message, "prompt_eval_count": 40,
                                         "eval_count": 5, "done_reason": "stop"})

    return LLMClient(CFG, transport=httpx.MockTransport(handler)), state


def session(i, text):
    return Session(session_id=f"s{i}", date=f"2023/05/{i + 1:02d} (Mon) 10:00", order=i,
                   turns=(Turn(role="user", content=text), Turn(role="assistant", content="ok")))


def call(tool, **args):
    return {"function": {"name": tool, "arguments": args}}


def test_is_a_memory_system(tmp_path):
    llm, _ = scripted_llm(["x"])
    system = FileSearchSystem("file", llm, tmp_path, seed=11)
    assert isinstance(system, MemorySystem) and system.name == "file"


def test_reset_makes_an_empty_namespace_dir_and_ingest_writes_dated_files(tmp_path):
    llm, _ = scripted_llm(["x"])
    system = FileSearchSystem("file", llm, tmp_path, seed=11)
    system.reset("q1:11")
    stats = system.ingest(session(0, "I adopted a dog named Rex."))
    assert isinstance(stats, IngestStats) and stats.store_items == 1
    files = sorted(p.name for p in (tmp_path / "q1_11").iterdir())
    assert files == ["000_s0.txt"]
    assert "Rex" in (tmp_path / "q1_11" / "000_s0.txt").read_text()
    system.reset("q1:11")
    assert list((tmp_path / "q1_11").iterdir()) == []


def test_answer_runs_the_tool_loop_and_sums_tokens(tmp_path):
    llm, state = scripted_llm([
        [call("list_files")],
        [call("grep", pattern="dog")],
        [call("read_file", name="000_s0.txt")],
        "Rex",
    ])
    system = FileSearchSystem("file", llm, tmp_path, seed=11, max_tool_calls=8)
    system.reset("q1:11")
    system.ingest(session(0, "I adopted a dog named Rex."))
    system.ingest(session(1, "The weather is nice."))
    out = system.answer("What is my dog's name?", "2023/06/01 (Thu) 09:00")
    assert isinstance(out, Answer)
    assert out.text == "Rex"
    assert out.prompt_tokens == 160 and out.completion_tokens == 20
    tool_msgs = [m for m in state["bodies"][-1]["messages"] if m["role"] == "tool"]
    assert any("000_s0.txt" in m["content"] for m in tool_msgs)
    assert any("Rex" in m["content"] for m in tool_msgs)
    assert "000_s0.txt" in out.context and "Rex" in out.context


def test_loop_stops_at_max_tool_calls_and_forces_an_answer(tmp_path):
    """Three tool calls are allowed; the fourth request carries no tools and its
    content is the answer, however many more tool calls the model would like."""
    llm, state = scripted_llm([[call("list_files")]] * 3 + ["forced"])
    system = FileSearchSystem("file", llm, tmp_path, seed=11, max_tool_calls=3)
    system.reset("q1:11")
    system.ingest(session(0, "a"))
    out = system.answer("q", "d")
    assert out.text == "forced"
    assert len(state["bodies"]) == 4
    assert all("tools" in b for b in state["bodies"][:3])
    assert "tools" not in state["bodies"][-1]
    assert "no longer available" in state["bodies"][-1]["messages"][-1]["content"]


def test_unknown_tool_and_missing_file_return_errors_not_exceptions(tmp_path):
    llm, state = scripted_llm([[call("read_file", name="nope.txt")], [call("bogus")], "done"])
    system = FileSearchSystem("file", llm, tmp_path, seed=11)
    system.reset("q1:11")
    system.ingest(session(0, "a"))
    out = system.answer("q", "d")
    assert out.text == "done"
    tool_msgs = [m["content"] for m in state["bodies"][-1]["messages"] if m["role"] == "tool"]
    assert any("not found" in t for t in tool_msgs)
    assert any("unknown tool" in t for t in tool_msgs)
