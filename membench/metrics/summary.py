"""Turns run records into the published summary."""

from __future__ import annotations

from collections import defaultdict

from membench.data.select import ABILITIES
from membench.data.types import Question
from membench.metrics.stats import bootstrap_ci, percentile

JUDGES = ("longmemeval", "zep", "mem0")
BASELINES = ("window", "file")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _accuracy(records: list[dict], judge: str) -> float:
    return _mean([1.0 if r.get(f"correct_{judge}") else 0.0 for r in records])


def summarize(records: list[dict], questions: dict[str, Question]) -> dict:
    by_system: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_system[record["system"]].append(record)

    systems: dict[str, dict] = {}
    for name, rows in by_system.items():
        accuracy = {judge: _accuracy(rows, judge) for judge in JUDGES}

        by_ability: dict[str, dict] = {}
        for ability in ABILITIES:
            subset = [r for r in rows if questions[r["question_id"]].ability == ability]
            if subset:
                by_ability[ability] = {judge: _accuracy(subset, judge) for judge in JUDGES}

        curve: dict[str, float] = {}
        gaps: dict[int, list[dict]] = defaultdict(list)
        for record in rows:
            gaps[questions[record["question_id"]].evidence_gap].append(record)
        for gap in sorted(gaps):
            curve[str(gap)] = _accuracy(gaps[gap], "longmemeval")

        ku = [r for r in rows if questions[r["question_id"]].ability == "knowledge_update"]
        stale_rate = _mean([1.0 if r.get("stale") else 0.0 for r in ku]) if ku else 0.0

        per_answer = [r["prompt_tokens"] + r["completion_tokens"] for r in rows]
        answer_seconds = [r["answer_seconds"] for r in rows]
        retrieval_seconds = [r["retrieval_seconds"] for r in rows]
        ingest_per_session = [
            r["ingest_seconds"] / r["sessions_ingested"] for r in rows if r["sessions_ingested"]
        ]
        low, high = bootstrap_ci([1.0 if r.get("correct_longmemeval") else 0.0 for r in rows])

        systems[name] = {
            "questions": len(rows),
            "errors": sum(1 for r in rows if r.get("error")),
            "judge_unparsed": sum(1 for r in rows if (r.get("judge_raw") or {}).get("longmemeval_unparsed")),
            "truncated": sum(1 for r in rows if r.get("truncated")),
            "accuracy": accuracy,
            "accuracy_ci95": [low, high],
            "by_ability": by_ability,
            "forgetting_curve": curve,
            "stale_rate": stale_rate,
            "tokens_per_answer": _mean(per_answer),
            "answer_latency_p50": percentile(answer_seconds, 50),
            "answer_latency_p95": percentile(answer_seconds, 95),
            "retrieval_latency_p50": percentile(retrieval_seconds, 50),
            "retrieval_latency_p95": percentile(retrieval_seconds, 95),
            "ingest_seconds_per_session": _mean(ingest_per_session),
            "store_tokens_final": _mean([float(r["store_tokens"]) for r in rows]),
        }

    oracle = systems.get("oracle", {}).get("accuracy", {}).get("longmemeval", 0.0)
    best_baseline = max(
        (systems[n]["accuracy"]["longmemeval"] for n in BASELINES if n in systems), default=0.0
    )
    for entry in systems.values():
        entry["share_of_oracle"] = entry["accuracy"]["longmemeval"] / oracle if oracle else 0.0
        entry["gap_to_best_baseline"] = entry["accuracy"]["longmemeval"] - best_baseline

    competitors = [n for n in systems if n != "oracle"]
    rankings = {
        judge: sorted(competitors, key=lambda n: -systems[n]["accuracy"][judge]) for judge in JUDGES
    }
    ranking_stable = len({tuple(order) for order in rankings.values()}) == 1

    return {
        "systems": systems,
        "rankings": rankings,
        "ranking_stable": ranking_stable,
        "oracle_accuracy": oracle,
        "best_baseline_accuracy": best_baseline,
    }
