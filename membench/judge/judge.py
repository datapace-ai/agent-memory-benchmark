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


GRADER_SYSTEM = (
    "You are a strict grader. Reply with exactly one word, yes or no. "
    "No analysis, no explanation."
)
JUDGE_MAX_TOKENS = 256


def parse_yes_no(text: str) -> bool | None:
    """The verdict, or None when the text never commits to yes or no.

    The last line wins; failing that, the last yes or no anywhere in the text,
    which covers graders that explain first and conclude at the end."""
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if lines:
        # Graders lead with the verdict, so within the last line the first
        # yes or no wins ("Yes, correct. No issues." is a yes).
        last = lines[-1]
        first = [(m.start(), True) for m in _YES.finditer(last)] + [(m.start(), False) for m in _NO.finditer(last)]
        if first:
            return min(first)[1]
    # A multi-line analysis concludes at the end, so the last yes or no wins.
    hits = [(m.start(), True) for m in _YES.finditer(text)] + [(m.start(), False) for m in _NO.finditer(text)]
    if hits:
        return max(hits)[1]
    return None


def _parse_yes_no(text: str) -> bool:
    return bool(parse_yes_no(text))


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
            system=system, user=user, seed=seed, model=self._llm.cfg.judge_model,
            max_tokens=JUDGE_MAX_TOKENS,
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
        raw["longmemeval"] = self._ask(GRADER_SYSTEM, lme_prompt, seed)
        verdict = parse_yes_no(raw["longmemeval"])
        if verdict is None:
            raw["longmemeval_unparsed"] = "1"
        longmemeval = bool(verdict)

        rule_user = RULE_USER_TEMPLATE.format(question.question, question.answer, response)
        raw["zep"] = self._ask(RULES["zep"], rule_user, seed)
        raw["mem0"] = self._ask(RULES["mem0"], rule_user, seed)

        stale: bool | None = None
        if question.ability == "knowledge_update" and not longmemeval:
            raw["stale"] = self._ask(
                GRADER_SYSTEM,
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
