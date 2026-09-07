import json

import httpx

from membench.config import ModelConfig
from membench.data.types import Question
from membench.judge.judge import GRADER_SYSTEM, JUDGE_MAX_TOKENS, Judge, Verdicts, parse_yes_no
from membench.judge.prompts import RULES, longmemeval_prompt
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
)


def question(ability="multi_session", qtype="multi-session", is_abs=False):
    return Question(
        question_id="q1",
        question_type=qtype,
        ability=ability,
        question="What is the capital?",
        answer="Paris",
        question_date="2023/06/01 (Thu) 09:00",
        is_abstention=is_abs,
    )


def judge_with(replies):
    state = {"i": 0}
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        text = replies[min(state["i"], len(replies) - 1)]
        state["i"] += 1
        return httpx.Response(
            200,
            json={
                "message": {"content": text},
                "prompt_eval_count": 10,
                "eval_count": 2,
                "done_reason": "stop",
            },
        )

    return Judge(LLMClient(CFG, transport=httpx.MockTransport(handler))), seen


def test_longmemeval_prompt_uses_the_temporal_off_by_one_clause():
    prompt = longmemeval_prompt("temporal-reasoning", "q", "a", "r", abstention=False)
    assert "do not penalize off-by-one errors" in prompt
    assert "Answer yes or no only." in prompt


def test_longmemeval_prompt_uses_the_abstention_template():
    assert "unanswerable" in longmemeval_prompt("single-session-user", "q", "a", "r", abstention=True)


def test_longmemeval_prompt_uses_the_preference_rubric():
    assert "Rubric" in longmemeval_prompt("single-session-preference", "q", "a", "r", abstention=False)


def test_rules_contain_the_three_published_judges():
    assert set(RULES) == {"strict", "zep", "mem0"}
    assert "SAME TOPIC" in RULES["zep"]
    assert "PARTIAL CREDIT" in RULES["mem0"]


def test_grade_returns_three_verdicts():
    judge, _ = judge_with(["yes", '{"correct": true}', '{"correct": false}'])
    out = judge.grade(question(), "Paris", seed=11)
    assert isinstance(out, Verdicts)
    assert out.longmemeval is True
    assert out.zep is True
    assert out.mem0 is False
    assert out.stale is None
    assert set(out.raw) == {"longmemeval", "zep", "mem0"}


def test_grade_parses_no_and_false():
    judge, _ = judge_with(["no", '{"correct": false}', '{"correct": false}'])
    assert judge.grade(question(), "Berlin", seed=11).longmemeval is False


def test_grade_adds_a_stale_verdict_for_knowledge_update():
    judge, _ = judge_with(["no", '{"correct": false}', '{"correct": false}', "yes"])
    out = judge.grade(
        question(ability="knowledge_update", qtype="knowledge-update"), "the old value", seed=11
    )
    assert out.stale is True
    assert "stale" in out.raw


def test_judge_never_sees_the_system_name():
    """The judge payload must not name the system under test.

    "system" itself is excluded: it is the JSON role of a chat message.
    """
    judge, seen = judge_with(["yes", '{"correct": true}', '{"correct": true}'])
    judge.grade(question(), "Paris", seed=11)
    blob = json.dumps(seen).lower()
    for name in ("window", "oracle", "graphiti", "letta", "langmem", "cognee"):
        assert name not in blob, f"judge payload leaked the system name {name}"


def test_unparseable_judge_output_is_false_not_an_error():
    judge, _ = judge_with(["maybe?", "not json", "also not json"])
    out = judge.grade(question(), "Paris", seed=11)
    assert out.longmemeval is False and out.zep is False and out.mem0 is False


def test_json_verdict_survives_surrounding_prose():
    judge, _ = judge_with(["yes", 'Sure. {"correct": true} done', 'Nope {"correct": false}'])
    out = judge.grade(question(), "Paris", seed=11)
    assert out.zep is True and out.mem0 is False


def test_parse_yes_no_reads_a_trailing_verdict_after_analysis():
    assert parse_yes_no("1. The answer matches.\n2. Nothing missing.\n\nyes") is True
    assert parse_yes_no("Analysis: the response says 15 days, gold says not enough.\n**Answer: No**") is False
    assert parse_yes_no("1. Scenario Interpretation: the user asks") is None
    assert parse_yes_no("") is None
    assert parse_yes_no("Yes, the response is correct. No issues.") is True


def test_grade_flags_an_unparsed_verdict_and_uses_a_one_word_instruction():
    judge, seen = judge_with(["1. Scenario Interpretation: cut off", '{"correct": true}', '{"correct": true}'])
    out = judge.grade(question(), "Paris", seed=11)
    assert out.longmemeval is False
    assert out.raw.get("longmemeval_unparsed") == "1"
    first = seen[0]
    assert first["messages"][0]["content"] == GRADER_SYSTEM
    assert "exactly one word" in GRADER_SYSTEM
    assert first["options"]["num_predict"] == JUDGE_MAX_TOKENS >= 200
