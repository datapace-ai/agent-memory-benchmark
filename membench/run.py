"""Resumable benchmark runner.

Writes one JSON line per (system, question, seed). Restarting skips units that
already have a line, so a run can be interrupted and continued.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

from membench.config import REPO_ROOT, ModelConfig, SystemConfig, load_models, load_systems
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


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def acquire_lock(out: Path) -> Path:
    """One writer per run file. Two concurrent runners would both append and
    produce duplicate records, which is exactly what happened once."""
    lock = out.with_name(out.name + ".lock")
    if lock.exists():
        try:
            holder = int(lock.read_text().strip())
        except ValueError:
            holder = 0
        if holder and _pid_alive(holder):
            raise RuntimeError(f"another runner (pid {holder}) is writing {out}; refusing to start")
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(str(os.getpid()))
    return lock


def release_lock(lock: Path) -> None:
    try:
        lock.unlink()
    except FileNotFoundError:
        pass


def apply_model_overrides(models: ModelConfig, answer_model: str, judge_model: str) -> ModelConfig:
    """A model track: same questions, same judge, a different answerer.

    The override also applies to the products' internal model, so a track is
    one model for everything except the judge."""
    from dataclasses import replace

    if answer_model:
        models = replace(models, answer_model=answer_model, openai_compat_model=answer_model)
    if judge_model:
        models = replace(models, judge_model=judge_model)
    return models


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
    parser.add_argument("--models-config", type=Path, default=REPO_ROOT / "configs" / "models.yaml")
    parser.add_argument("--answer-model", default="", help="override the answerer for this run (a model track)")
    parser.add_argument("--judge-model", default="", help="override the judge; keep one judge across tracks")
    args = parser.parse_args(argv)

    models = apply_model_overrides(load_models(args.models_config), args.answer_model, args.judge_model)
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
        "provider": models.provider,
        "base_url": models.base_url,
        "answer_model": models.answer_model,
        "judge_model": models.judge_model,
        "embed_model": models.embed_model,
        "num_ctx": models.num_ctx,
        "ollama_version": llm.version(),
        "python": platform.python_version(),
    }

    lock = acquire_lock(args.out)
    done = load_done(args.out)
    work = plan_work(systems, questions, seeds, done)
    print(f"{len(done)} units done, {len(work)} to run", flush=True)

    try:
        _run_units(work, llm, models, judge, provenance, args.out)
    finally:
        release_lock(lock)
    return 0


def _run_units(
    work, llm: LLMClient, models: ModelConfig, judge: Judge, provenance: dict, out: Path
) -> None:
    for index, (system_cfg, question, seed) in enumerate(work, start=1):
        started = time.perf_counter()
        system = build(system_cfg, llm, seed=seed, models=models)
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
        append_record(out, record)
        print(
            f"[{index}/{len(work)}] {system_cfg.name} {question.question_id} "
            f"({question.ability}) seed={seed} correct={record['correct_longmemeval']} "
            f"tok={record['prompt_tokens']} {time.perf_counter() - started:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    sys.exit(main())
