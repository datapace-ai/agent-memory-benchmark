"""Resumable benchmark runner.

Writes one JSON line per (system, question, seed). Restarting skips units that
already have a line, so a run can be interrupted and continued.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

from membench.config import REPO_ROOT, SystemConfig, load_models, load_systems
from membench.data.types import Question, read_jsonl
from membench.judge.judge import Judge
from membench.llm import LLMClient
from membench.protocol.clock import run_question
from membench.systems.registry import build, evidence_only

DEFAULT_RUNS = REPO_ROOT / "results" / "runs" / "runs.jsonl"
DEFAULT_QUESTIONS = REPO_ROOT / "data" / "questions_s12.jsonl"


def record_key(record: dict) -> tuple[str, str, int]:
    return (record["system"], record["question_id"], int(record["seed"]))


def load_done(path: Path) -> set[tuple[str, str, int]]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            done.add(record_key(json.loads(line)))
        except Exception:
            continue
    return done


def append_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")


def plan_work(
    systems: tuple[SystemConfig, ...],
    questions: list[Question],
    seeds: tuple[int, ...],
    done: set[tuple[str, str, int]],
) -> list[tuple[SystemConfig, Question, int]]:
    work = []
    for system in systems:
        for question in questions:
            for seed in seeds:
                if (system.name, question.question_id, seed) not in done:
                    work.append((system, question, seed))
    return work


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the agent memory benchmark.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--out", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--limit", type=int, default=0, help="0 means every question")
    parser.add_argument("--systems", default="", help="comma separated subset of system names")
    parser.add_argument("--seeds", default="", help="comma separated subset of seeds")
    args = parser.parse_args(argv)

    models = load_models(REPO_ROOT / "configs" / "models.yaml")
    systems = load_systems(REPO_ROOT / "configs" / "systems.yaml")
    if args.systems:
        wanted = set(args.systems.split(","))
        systems = tuple(s for s in systems if s.name in wanted)
    seeds = tuple(int(s) for s in args.seeds.split(",")) if args.seeds else models.seeds

    questions = read_jsonl(args.questions)
    if args.limit:
        by_ability: dict[str, list[Question]] = {}
        for question in questions:
            by_ability.setdefault(question.ability, []).append(question)
        balanced: list[Question] = []
        index = 0
        while len(balanced) < args.limit and any(by_ability.values()):
            for ability in sorted(by_ability):
                if len(balanced) >= args.limit:
                    break
                if index < len(by_ability[ability]):
                    balanced.append(by_ability[ability][index])
            index += 1
        questions = balanced

    llm = LLMClient(models)
    judge = Judge(llm)
    provenance = {
        "answer_model": models.answer_model,
        "judge_model": models.judge_model,
        "num_ctx": models.num_ctx,
        "ollama_version": llm.version(),
        "python": platform.python_version(),
    }

    done = load_done(args.out)
    work = plan_work(systems, questions, seeds, done)
    print(f"{len(done)} units done, {len(work)} to run", flush=True)

    for index, (system_cfg, question, seed) in enumerate(work, start=1):
        started = time.perf_counter()
        system = build(system_cfg, llm, seed=seed)
        run = run_question(system, question, seed, evidence_only=evidence_only(system_cfg))
        record = run.to_dict()

        if run.error is None:
            verdicts = judge.grade(question, run.answer_text, seed)
            record.update(
                correct_longmemeval=verdicts.longmemeval,
                correct_zep=verdicts.zep,
                correct_mem0=verdicts.mem0,
                stale=verdicts.stale,
                judge_raw=verdicts.raw,
            )
        else:
            record.update(
                correct_longmemeval=False,
                correct_zep=False,
                correct_mem0=False,
                stale=None,
                judge_raw={},
            )

        record["provenance"] = provenance
        append_record(args.out, record)
        print(
            f"[{index}/{len(work)}] {system_cfg.name} {question.question_id} "
            f"({question.ability}) seed={seed} correct={record['correct_longmemeval']} "
            f"tok={record['prompt_tokens']} {time.perf_counter() - started:.1f}s",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
