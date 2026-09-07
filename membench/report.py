"""Turns run records into results/summary.json and results/summary.md."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from membench.config import REPO_ROOT
from membench.data.types import read_jsonl
from membench.metrics.summary import summarize

CEILING_NOTE = (
    "The oracle row is a ceiling, not a competitor. It sees only the evidence "
    "sessions, so it measures the best this answerer and judge can do on this set."
)


def render_markdown(summary: dict) -> str:
    lines = ["# Agent memory benchmark results", ""]
    if summary.get("answer_model") or summary.get("judge_model"):
        lines.append(f"Answerer: {summary.get('answer_model', 'unknown')}. Judge: {summary.get('judge_model', 'unknown')}.")
        lines.append("")
    lines += [CEILING_NOTE, ""]
    lines += [
        "| System | LongMemEval rule | Zep rule | Mem0 rule | Share of ceiling | "
        "Gap to best baseline | Tokens per answer | Answer p50 | Answer p95 | Errors |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    ordered = sorted(
        summary["systems"], key=lambda n: -summary["systems"][n]["accuracy"]["longmemeval"]
    )
    for name in ordered:
        s = summary["systems"][name]
        lines.append(
            f"| {name} | {s['accuracy']['longmemeval'] * 100:.1f} | "
            f"{s['accuracy']['zep'] * 100:.1f} | {s['accuracy']['mem0'] * 100:.1f} | "
            f"{s['share_of_oracle'] * 100:.1f} | {s['gap_to_best_baseline'] * 100:+.1f} | "
            f"{s['tokens_per_answer']:.0f} | {s['answer_latency_p50']:.1f} | "
            f"{s['answer_latency_p95']:.1f} | {s['errors']} |"
        )

    unparsed = sum(s.get("judge_unparsed", 0) for s in summary["systems"].values())
    if unparsed:
        lines += ["", f"Judge verdicts that never said yes or no, scored as wrong: {unparsed}. Re-judge before publishing."]
    stability = (
        "Ranking is stable across all three judge rules."
        if summary["ranking_stable"]
        else "Ranking is NOT stable across judge rules. Treat the order as unresolved."
    )
    lines += ["", stability, "", "## Accuracy by ability, LongMemEval rule", ""]

    abilities = sorted({a for s in summary["systems"].values() for a in s["by_ability"]})
    lines.append("| System | " + " | ".join(abilities) + " |")
    lines.append("| --- | " + " | ".join("---:" for _ in abilities) + " |")
    for name in ordered:
        s = summary["systems"][name]
        cells = [
            f"{s['by_ability'][a]['longmemeval'] * 100:.1f}" if a in s["by_ability"] else "n/a"
            for a in abilities
        ]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    lines += ["", "## Forgetting curve, accuracy by sessions since the evidence", ""]
    for name in ordered:
        curve = summary["systems"][name]["forgetting_curve"]
        points = ", ".join(
            f"{gap}: {value * 100:.0f}"
            for gap, value in sorted(curve.items(), key=lambda kv: int(kv[0]))
        )
        lines.append(f"- {name}: {points}")

    lines += ["", "## Stale answers on knowledge updates", ""]
    for name in ordered:
        rate = summary["systems"][name]["stale_rate"]
        lines.append(f"- {name}: {rate * 100:.1f} percent of wrong answers gave a superseded value")

    return "\n".join(lines) + "\n"


def compare_tracks(named_summaries: dict[str, dict]) -> str:
    """One table across model tracks: rows are systems, columns are tracks.

    Accuracy under the LongMemEval rule, with tokens per answer in parentheses,
    so a reader sees both what each model recalls and what it costs to ask."""
    tracks = list(named_summaries)
    systems = sorted({s for summary in named_summaries.values() for s in summary["systems"]})
    lines = ["# Model tracks, LongMemEval rule (tokens per answer)", ""]
    lines.append("| System | " + " | ".join(tracks) + " |")
    lines.append("| --- | " + " | ".join("---:" for _ in tracks) + " |")
    for name in systems:
        cells = []
        for track in tracks:
            entry = named_summaries[track]["systems"].get(name)
            if entry is None:
                cells.append("n/a")
            else:
                cells.append(f"{entry['accuracy']['longmemeval'] * 100:.1f} ({entry['tokens_per_answer']:.0f})")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "Judge: " + ", ".join(sorted({str(s.get("judge_model", "unknown")) for s in named_summaries.values()})), ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize benchmark runs.")
    parser.add_argument("--runs", type=Path, default=REPO_ROOT / "results" / "runs" / "runs.jsonl")
    parser.add_argument("--questions", type=Path, default=REPO_ROOT / "data" / "questions_s12.jsonl")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results")
    args = parser.parse_args(argv)

    questions = {q.question_id: q for q in read_jsonl(args.questions)}
    latest: dict[tuple, dict] = {}
    for line in args.runs.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            latest[(record["system"], record["question_id"], record["seed"])] = record
    records = list(latest.values())
    summary = summarize(records, questions)
    judges = {str((r.get("provenance") or {}).get("judge_model", "")) for r in records}
    answerers = {str((r.get("provenance") or {}).get("answer_model", "")) for r in records}
    summary["judge_model"] = ", ".join(sorted(j for j in judges if j)) or "unknown"
    summary["answer_model"] = ", ".join(sorted(a for a in answerers if a)) or "unknown"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    text = render_markdown(summary)
    (args.out_dir / "summary.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
