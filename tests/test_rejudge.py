import json

from membench.data.types import Question
from membench.judge.judge import Verdicts
from membench.rejudge import regrade


class FlipJudge:
    def grade(self, question, response, seed):
        return Verdicts(True, True, True, None, {"longmemeval": "yes"})


def test_regrade_replaces_verdicts_keeps_answers_and_skips_errors():
    q = Question(question_id="q1", question_type="multi-session", ability="multi_session", question="?",
                 answer="a", question_date="d", is_abstention=False)
    records = [
        {"question_id": "q1", "system": "window", "seed": 11, "answer_text": "a", "prompt_tokens": 9,
         "error": None, "correct_longmemeval": False, "judge_raw": {"longmemeval": "1. cut"}, "provenance": {"x": 1}},
        {"question_id": "q1", "system": "mem0", "seed": 11, "answer_text": "", "error": "boom",
         "correct_longmemeval": False},
    ]
    out = regrade(records, {"q1": q}, FlipJudge(), "judge/new")
    assert out[0]["correct_longmemeval"] is True and out[0]["prompt_tokens"] == 9
    assert out[0]["provenance"] == {"x": 1, "judge_model": "judge/new", "rejudged": True}
    assert out[1] == records[1]
