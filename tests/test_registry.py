import httpx
import pytest

from membench.config import REPO_ROOT, ModelConfig, SystemConfig
from membench.llm import LLMClient
from membench.systems.cognee_system import CogneeSystem
from membench.systems.file_search import FileSearchSystem
from membench.systems.graphiti_system import GraphitiSystem
from membench.systems.letta_system import LettaSystem
from membench.systems.context_window import ContextWindowSystem
from membench.systems.langmem_system import LangMemSystem
from membench.systems.mem0_system import Mem0System
from membench.systems.registry import STORE_ROOT, VENDOR_KINDS, build

CFG = ModelConfig(
    base_url="http://ollama.test", answer_model="qwen3:14b", judge_model="qwen3:14b",
    embed_model="nomic-embed-text", num_ctx=40960, temperature=0.0, max_answer_tokens=512,
    seeds=(11,), think=False,
)


def llm():
    return LLMClient(CFG, transport=httpx.MockTransport(lambda r: httpx.Response(500)))


def test_build_each_kind():
    assert isinstance(
        build(SystemConfig("window", "context_window", {"token_budget": 32000}), llm(), 11, CFG),
        ContextWindowSystem,
    )
    assert isinstance(
        build(SystemConfig("file", "file_search", {"max_tool_calls": 8}), llm(), 11, CFG),
        FileSearchSystem,
    )
    assert isinstance(build(SystemConfig("mem0", "mem0", {"top_k": 10}), llm(), 11, CFG), Mem0System)
    assert isinstance(
        build(SystemConfig("langmem", "langmem", {"top_k": 10}), llm(), 11, CFG), LangMemSystem
    )
    assert isinstance(
        build(SystemConfig("cognee", "cognee", {"top_k": 10}), llm(), 11, CFG), CogneeSystem
    )
    assert isinstance(
        build(SystemConfig("graphiti", "graphiti", {"top_k": 10}), llm(), 11, CFG), GraphitiSystem
    )
    assert isinstance(build(SystemConfig("letta", "letta", {}), llm(), 11, CFG), LettaSystem)


def test_unknown_kind_raises():
    with pytest.raises(ValueError, match="nope"):
        build(SystemConfig("x", "nope", {}), llm(), 11, CFG)


def test_vendor_kinds_require_model_config():
    for kind in VENDOR_KINDS:
        with pytest.raises(ValueError, match="models"):
            build(SystemConfig(kind, kind, {}), llm(), 11, None)


def test_store_root_is_under_data():
    assert STORE_ROOT == REPO_ROOT / "data" / "vendor-stores"


def test_building_a_vendor_system_makes_no_network_call():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(500)

    client = LLMClient(CFG, transport=httpx.MockTransport(handler))
    for kind in VENDOR_KINDS:
        build(SystemConfig(kind, kind, {}), client, 11, CFG)
    assert calls["n"] == 0
