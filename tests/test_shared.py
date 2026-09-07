import json
import time

import httpx

from membench.config import ModelConfig
from membench.data.types import Session, Turn
from membench.llm import LLMClient
from membench.systems.base import ANSWER_SYSTEM_PROMPT, Answer
from membench.systems.shared import (
    EMBEDDING_DIMS,
    answer_from_context,
    chat_model,
    embeddings,
    session_messages,
    session_text,
)

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


def session():
    return Session(
        session_id="s1",
        date="2023/05/20 (Sat) 02:21",
        order=0,
        turns=(
            Turn(role="user", content="I adopted a dog named Rex."),
            Turn(role="assistant", content="Congratulations on Rex."),
            Turn(role="user", content="He is a beagle."),
        ),
    )


def test_session_messages_keep_roles_and_prefix_the_date_once():
    msgs = session_messages(session())
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[0]["content"].startswith("[2023/05/20 (Sat) 02:21] ")
    assert msgs[0]["content"].endswith("I adopted a dog named Rex.")
    assert msgs[1]["content"] == "Congratulations on Rex."
    assert msgs[2]["content"] == "He is a beagle."


def test_session_text_carries_id_date_and_every_turn():
    text = session_text(session())
    assert "s1" in text and "2023/05/20" in text
    assert "Rex" in text and "beagle" in text


def test_answer_from_context_uses_the_shared_prompt_and_counts():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"message": {"content": "Rex"}, "prompt_eval_count": 77, "eval_count": 2,
                  "done_reason": "stop"},
        )

    llm = LLMClient(CFG, transport=httpx.MockTransport(handler))
    started = time.perf_counter()
    out = answer_from_context(
        llm, context="memory: dog named Rex", question="What is my dog's name?",
        question_date="2023/06/01 (Thu) 09:00", seed=11, retrieval_seconds=0.25, started=started,
    )
    assert isinstance(out, Answer)
    assert out.text == "Rex"
    assert out.context == "memory: dog named Rex"
    assert out.prompt_tokens == 77 and out.completion_tokens == 2
    assert out.retrieval_seconds == 0.25
    assert out.total_seconds >= 0.0
    assert seen["body"]["messages"][0]["content"] == ANSWER_SYSTEM_PROMPT
    assert "2023/06/01" in seen["body"]["messages"][1]["content"]
    assert seen["body"]["options"]["seed"] == 11


def test_chat_model_has_reasoning_off_and_the_benchmark_settings():
    model = chat_model(CFG, seed=22)
    assert model.model == "qwen3:14b"
    assert model.reasoning is False
    assert model.temperature == 0.0
    assert model.seed == 22
    assert model.num_ctx == 40960
    assert model.base_url == "http://ollama.test"


def test_embeddings_use_the_configured_model():
    emb = embeddings(CFG)
    assert emb.model == "nomic-embed-text"
    assert emb.base_url == "http://ollama.test"
    assert EMBEDDING_DIMS == 768
