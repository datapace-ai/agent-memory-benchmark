import json

import httpx
import pytest

from membench.config import ModelConfig
from membench.llm import LLMClient

CFG = ModelConfig(
    base_url="http://ollama.test",
    answer_model="qwen3:14b",
    judge_model="qwen3:14b",
    embed_model="nomic-embed-text",
    num_ctx=40960,
    temperature=0.0,
    max_answer_tokens=512,
    seeds=(11,),
    think=False,
)


def make_client(handler):
    return LLMClient(CFG, transport=httpx.MockTransport(handler))


def test_complete_sends_options_and_parses_counts():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": "Paris"},
                "prompt_eval_count": 120,
                "eval_count": 3,
                "done_reason": "stop",
            },
        )

    out = make_client(handler).complete(system="sys", user="usr", seed=11)

    assert out.text == "Paris"
    assert out.prompt_tokens == 120
    assert out.completion_tokens == 3
    assert out.truncated is False
    assert out.seconds >= 0
    assert seen["url"] == "http://ollama.test/api/chat"
    assert seen["body"]["model"] == "qwen3:14b"
    assert seen["body"]["stream"] is False
    assert seen["body"]["options"]["num_ctx"] == 40960
    assert seen["body"]["options"]["temperature"] == 0.0
    assert seen["body"]["options"]["seed"] == 11
    assert seen["body"]["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]


def test_complete_marks_length_truncation():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "message": {"content": "cut off"},
                "prompt_eval_count": 5,
                "eval_count": 512,
                "done_reason": "length",
            },
        )

    assert make_client(handler).complete("s", "u", seed=11).truncated is True


def test_complete_strips_thinking_tags():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "message": {"content": "<think>weighing it up</think>\n\nThe answer is Paris."},
                "prompt_eval_count": 5,
                "eval_count": 5,
                "done_reason": "stop",
            },
        )

    assert make_client(handler).complete("s", "u", seed=11).text == "The answer is Paris."


def test_complete_retries_then_raises():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(500, json={"error": "boom"})

    with pytest.raises(RuntimeError):
        make_client(handler).complete("s", "u", seed=11)
    assert calls["n"] == 3


def test_count_tokens_uses_embed_endpoint():
    def handler(request):
        assert str(request.url) == "http://ollama.test/api/embed"
        return httpx.Response(200, json={"embeddings": [[0.0]], "prompt_eval_count": 42})

    assert make_client(handler).count_tokens("some text") == 42


def test_complete_sends_think_when_configured():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"message": {"content": "x"}, "prompt_eval_count": 1, "eval_count": 1,
                  "done_reason": "stop"},
        )

    make_client(handler).complete("s", "u", seed=11)
    assert seen["body"]["think"] is False


def test_complete_omits_think_when_unset():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"message": {"content": "x"}, "prompt_eval_count": 1, "eval_count": 1,
                  "done_reason": "stop"},
        )

    from dataclasses import replace

    LLMClient(replace(CFG, think=None), transport=httpx.MockTransport(handler)).complete(
        "s", "u", seed=11
    )
    assert "think" not in seen["body"]


def test_chat_passes_tools_and_returns_tool_calls():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "message": {"role": "assistant", "content": "",
                        "tool_calls": [{"function": {"name": "grep", "arguments": {"pattern": "dog"}}}]},
            "prompt_eval_count": 50, "eval_count": 9, "done_reason": "stop"})

    out = make_client(handler).chat(
        messages=[{"role": "user", "content": "find the dog"}],
        tools=[{"type": "function", "function": {"name": "grep", "parameters": {}}}], seed=11)
    assert out.tool_calls[0]["function"]["name"] == "grep"
    assert out.tool_calls[0]["function"]["arguments"] == {"pattern": "dog"}
    assert out.prompt_tokens == 50 and out.completion_tokens == 9
    assert seen["body"]["tools"][0]["function"]["name"] == "grep"
    assert seen["body"]["options"]["seed"] == 11


def test_chat_without_tools_returns_plain_message():
    def handler(request):
        assert "tools" not in json.loads(request.content)
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "hi"},
                                         "prompt_eval_count": 3, "eval_count": 1, "done_reason": "stop"})

    out = make_client(handler).chat(messages=[{"role": "user", "content": "x"}], tools=None, seed=11)
    assert out.message["content"] == "hi" and out.tool_calls == []


# ---- OpenAI-compatible backend ------------------------------------------

from dataclasses import replace as _replace

COMPAT = _replace(
    CFG, provider="openai_compat", base_url="https://router.test/api/v1", api_key_env="TEST_ROUTER_KEY",
    answer_model="vendor/model:free", judge_model="vendor/model:free", min_request_interval_seconds=0.0,
)


def compat_client(handler, monkeypatch):
    monkeypatch.setenv("TEST_ROUTER_KEY", "sk-test")
    client = LLMClient(COMPAT, transport=httpx.MockTransport(handler))
    client._sleep = lambda s: None
    return client


def test_compat_complete_sends_bearer_reasoning_and_parses_usage(monkeypatch):
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("Authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": "Paris"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 33, "completion_tokens": 2}})

    out = compat_client(handler, monkeypatch).complete("sys", "usr", seed=11)
    assert out.text == "Paris" and out.prompt_tokens == 33 and out.completion_tokens == 2
    assert out.truncated is False
    assert seen["url"] == "https://router.test/api/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-test"
    assert seen["body"]["model"] == "vendor/model:free"
    assert seen["body"]["seed"] == 11 and seen["body"]["temperature"] == 0.0
    assert seen["body"]["max_tokens"] == 512
    assert seen["body"]["reasoning"] == {"enabled": False}
    assert "options" not in seen["body"]


def test_compat_missing_key_fails_loudly(monkeypatch):
    monkeypatch.delenv("TEST_ROUTER_KEY", raising=False)
    with pytest.raises(RuntimeError, match="TEST_ROUTER_KEY"):
        LLMClient(COMPAT)


def test_compat_chat_parses_string_tool_arguments_and_keeps_raw_message(monkeypatch):
    def handler(request):
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": None,
                                     "tool_calls": [{"id": "call_1", "type": "function",
                                                     "function": {"name": "grep", "arguments": '{"pattern": "dog"}'}}]},
                         "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}})

    out = compat_client(handler, monkeypatch).chat([{"role": "user", "content": "x"}], tools=[{"type": "function", "function": {"name": "grep"}}], seed=11)
    assert out.tool_calls == [{"id": "call_1", "function": {"name": "grep", "arguments": {"pattern": "dog"}}}]
    assert out.message["tool_calls"][0]["function"]["arguments"] == '{"pattern": "dog"}'
    assert out.message["content"] == ""


def test_compat_tool_message_shape_differs_from_ollama(monkeypatch):
    client = compat_client(lambda r: httpx.Response(500), monkeypatch)
    assert client.tool_message("grep", "call_1", "hits") == {"role": "tool", "tool_call_id": "call_1", "content": "hits"}
    assert make_client(lambda r: httpx.Response(500)).tool_message("grep", None, "hits") == {"role": "tool", "content": "hits", "tool_name": "grep"}


def test_compat_retries_on_429_with_retry_after_then_succeeds(monkeypatch):
    calls = {"n": 0}
    slept = []

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "7"}, json={"error": {"message": "slow down"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    client = compat_client(handler, monkeypatch)
    client._sleep = lambda s: slept.append(s)
    assert client.complete("s", "u", seed=11).text == "ok"
    assert calls["n"] == 2 and 7.0 in slept


def test_compat_provider_error_in_200_body_is_retried_then_raised(monkeypatch):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(200, json={"error": {"code": 502, "message": "upstream"}})

    with pytest.raises(RuntimeError, match="upstream"):
        compat_client(handler, monkeypatch).complete("s", "u", seed=11)
    assert calls["n"] == 8


def test_compat_paces_requests(monkeypatch):
    cfg = _replace(COMPAT, min_request_interval_seconds=3.0)
    monkeypatch.setenv("TEST_ROUTER_KEY", "sk-test")
    slept = []
    client = LLMClient(cfg, transport=httpx.MockTransport(lambda r: httpx.Response(200, json={
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {}})))
    client._sleep = lambda s: slept.append(s)
    client.complete("s", "u", seed=11)
    client.complete("s", "u", seed=11)
    assert len(slept) == 1 and 0 < slept[0] <= 3.0
