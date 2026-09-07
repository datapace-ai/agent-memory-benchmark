"""Re-grade existing run records without re-running the systems.

Usage: python -m membench.rejudge --runs results/runs/x.jsonl [--out results/runs/x.jsonl]

Answers, tokens and timings are kept; the four verdict fields and judge_raw are
replaced; provenance gains rejudged_with. Writing back to the same path
replaces the file atomically.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from membench.config import REPO_ROOT, load_models
from membench.data.types import read_jsonl
from membench.judge.judge import Judge
from membench.llm import LLMClient


def regrade(records: list[dict], questions: dict, judge, judge_label: str) -> list[dict]:
    out = []
    for record in records:
        if record.get("error"):
            out.append(record)
            continue
        verdicts = judge.grade(questions[record["question_id"]], record["answer_text"], int(record["seed"]))
        updated = dict(record)
        updated.update(
            correct_longmemeval=verdicts.longmemeval, correct_zep=verdicts.zep, correct_mem0=verdicts.mem0,
            stale=verdicts.stale, judge_raw=verdicts.raw,
        )
        prov = dict(updated.get("provenance") or {})
        prov["judge_model"] = judge_label
        prov["rejudged"] = True
        updated["provenance"] = prov
        out.append(updated)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Re-grade run records with the current judge.")
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--questions", type=Path, default=REPO_ROOT / "data" / "questions_s12.jsonl")
    parser.add_argument("--models-config", type=Path, default=REPO_ROOT / "configs" / "models.yaml")
    parser.add_argument("--judge-model", default="")
    args = parser.parse_args(argv)

    models = load_models(args.models_config)
    if args.judge_model:
        from dataclasses import replace

        models = replace(models, judge_model=args.judge_model)
    questions = {q.question_id: q for q in read_jsonl(args.questions)}
    records = [json.loads(l) for l in args.runs.read_text().splitlines() if l.strip()]
    judge = Judge(LLMClient(models))
    graded = regrade(records, questions, judge, models.judge_model)

    out = args.out or args.runs
    fd, tmp = tempfile.mkstemp(dir=str(out.parent), prefix=out.name, suffix=".tmp")
    with os.fdopen(fd, "w") as handle:
        for record in graded:
            handle.write(json.dumps(record) + "\n")
    os.replace(tmp, out)
    changed = sum(1 for a, b in zip(records, graded) if a.get("correct_longmemeval") != b.get("correct_longmemeval"))
    print(f"re-judged {len(graded)} records into {out}; LongMemEval verdicts changed: {changed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
