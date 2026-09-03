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
