"""Grades one answer under three published rules.

The judge is never told which system produced the answer.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from membench.data.types import Question
from membench.judge.prompts import RULE_USER_TEMPLATE, RULES, STALE_PROMPT, longmemeval_prompt
from membench.llm import LLMClient

_YES = re.compile(r"\byes\b", re.IGNORECASE)
_NO = re.compile(r"\bno\b", re.IGNORECASE)
_JSON_TRUE = re.compile(r'"correct"\s*:\s*true', re.IGNORECASE)


@dataclass(frozen=True)
class Verdicts:
    longmemeval: bool
    zep: bool
    mem0: bool
    stale: bool | None
    raw: dict[str, str]


def _parse_yes_no(text: str) -> bool:
    """Read the last line that commits to yes or no. Anything else is a no."""
    for line in reversed([l.strip() for l in text.strip().splitlines() if l.strip()]):
        if _YES.search(line):
            return True
        if _NO.search(line):
            return False
    return False


def _parse_json_correct(text: str) -> bool:
    stripped = text.strip()
    try:
        return bool(json.loads(stripped).get("correct", False))
    except Exception:
        pass
    match = re.search(r"\{.*?\}", stripped, re.DOTALL)
    if match:
        try:
            return bool(json.loads(match.group(0)).get("correct", False))
        except Exception:
            pass
    return bool(_JSON_TRUE.search(stripped))


class Judge:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def _ask(self, system: str, user: str, seed: int) -> str:
        return self._llm.complete(
            system=system, user=user, seed=seed, model=self._llm.cfg.judge_model, max_tokens=64
        ).text

    def grade(self, question: Question, response: str, seed: int) -> Verdicts:
        raw: dict[str, str] = {}

        lme_prompt = longmemeval_prompt(
            question.question_type,
            question.question,
            question.answer,
            response,
            question.is_abstention,
        )
        raw["longmemeval"] = self._ask("You are a strict grader.", lme_prompt, seed)
        longmemeval = _parse_yes_no(raw["longmemeval"])

        rule_user = RULE_USER_TEMPLATE.format(question.question, question.answer, response)
        raw["zep"] = self._ask(RULES["zep"], rule_user, seed)
        raw["mem0"] = self._ask(RULES["mem0"], rule_user, seed)

        stale: bool | None = None
        if question.ability == "knowledge_update" and not longmemeval:
            raw["stale"] = self._ask(
                "You are a strict grader.",
                STALE_PROMPT.format(question.question, question.answer, response),
                seed,
            )
            stale = _parse_yes_no(raw["stale"])

        return Verdicts(
            longmemeval=longmemeval,
            zep=_parse_json_correct(raw["zep"]),
            mem0=_parse_json_correct(raw["mem0"]),
            stale=stale,
            raw=raw,
        )
