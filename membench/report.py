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
    lines = ["# Agent memory benchmark results", "", CEILING_NOTE, ""]
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize benchmark runs.")
    parser.add_argument("--runs", type=Path, default=REPO_ROOT / "results" / "runs" / "runs.jsonl")
    parser.add_argument("--questions", type=Path, default=REPO_ROOT / "data" / "questions_s12.jsonl")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results")
    args = parser.parse_args(argv)

    questions = {q.question_id: q for q in read_jsonl(args.questions)}
    records = [json.loads(line) for line in args.runs.read_text().splitlines() if line.strip()]
    summary = summarize(records, questions)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    text = render_markdown(summary)
    (args.out_dir / "summary.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
