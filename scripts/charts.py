"""Draw the results charts as static SVG files that GitHub renders.

Usage: uv run python scripts/charts.py [--out results/charts]

Reads results/tracks/<track>/summary.json for accuracy, intervals and tokens,
and results/runs/track-<track>.jsonl for per-question ingest and answer
seconds. Writes four SVGs with explicit colours and system fonts, so they read
the same in a README on either GitHub theme.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from membench.config import REPO_ROOT

INK, MUTED, RULE, GRID, GREEN, OCHRE, WHITE = "#171d24", "#667080", "#d6dbe1", "#e6eaee", "#0f8a5f", "#a8791a", "#ffffff"
HEAT = {0: "#f4f6f8", 50: "#cfe9dc", 100: GREEN}
FONT = "Helvetica, Arial, sans-serif"
MONO = "Menlo, Consolas, monospace"
NAMES = {
    "oracle": "Oracle ceiling", "window": "Window 32k", "file": "File search", "mem0": "Mem0",
    "langmem": "LangMem", "cognee": "Cognee", "graphiti": "Graphiti",
}
ORDER = ["oracle", "window", "file", "mem0", "langmem", "cognee", "graphiti"]
ABILITIES = ["abstention", "extraction", "knowledge_update", "multi_session", "temporal"]
ABILITY_LABELS = {
    "abstention": "Abstention", "extraction": "Extraction", "knowledge_update": "Knowledge update",
    "multi_session": "Multi-session", "temporal": "Temporal",
}


def esc(text: str) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Svg:
    def __init__(self, width: int, height: int) -> None:
        self.w, self.h = width, height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
            f'font-family="{FONT}" font-size="12">',
            f'<rect x="0" y="0" width="{width}" height="{height}" fill="{WHITE}"/>',
        ]

    def line(self, x1, y1, x2, y2, stroke=RULE, width=1.0, dash: str | None = None, opacity=1.0):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" '
            f'stroke-width="{width}" stroke-opacity="{opacity}"{d}/>'
        )

    def rect(self, x, y, w, h, fill, stroke: str | None = None, rx=2):
        s = f' stroke="{stroke}" stroke-width="1"' if stroke else ""
        self.parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}"{s}/>')

    def circle(self, cx, cy, r, fill, stroke: str | None = None, width=2):
        s = f' stroke="{stroke}" stroke-width="{width}"' if stroke else ""
        self.parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{fill}"{s}/>')

    def text(self, x, y, s, anchor="start", fill=INK, size=12, weight=400, mono=False, rotate: float | None = None):
        fam = MONO if mono else FONT
        tr = f' transform="rotate({rotate} {x:.1f} {y:.1f})"' if rotate is not None else ""
        self.parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" fill="{fill}" font-size="{size}" '
            f'font-weight="{weight}" font-family="{fam}"{tr}>{esc(s)}</text>'
        )

    def write(self, path: Path) -> None:
        self.parts.append("</svg>")
        path.write_text("\n".join(self.parts) + "\n")


def load_summary(track: str, tracks_dir: Path) -> dict:
    return json.loads((tracks_dir / track / "summary.json").read_text())["systems"]


def timing_medians(track: str) -> dict[str, dict]:
    """Median ingest and answer seconds per finished question, per system."""
    last: dict[tuple[str, str], dict] = {}
    path = REPO_ROOT / "results" / "runs" / f"track-{track}.jsonl"
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            last[(r["system"], r["question_id"])] = r
    per: dict[str, list] = defaultdict(list)
    for (system, _), r in last.items():
        if r.get("error"):
            continue
        ingest = r.get("ingest_seconds") or 0
        ingest = sum(ingest) if isinstance(ingest, list) else ingest
        per[system].append((ingest, r.get("answer_seconds") or 0))
    return {
        s: {"n": len(v), "ingest": statistics.median(i for i, _ in v), "answer": statistics.median(a for _, a in v)}
        for s, v in per.items()
    }


def pct(x: float) -> int:
    return int(round(x * 100))


def chart_accuracy_vs_tokens(ling: dict, out: Path) -> None:
    W, H, L, R, T, B = 880, 400, 56, 24, 28, 56
    svg = Svg(W, H)
    x = lambda t: L + (math.log10(t) - 2) / (math.log10(40000) - 2) * (W - L - R)
    y = lambda a: T + (100 - a) / 100 * (H - T - B)
    for v in (0, 20, 40, 60, 80, 100):
        svg.line(L, y(v), W - R, y(v), GRID)
        svg.text(L - 10, y(v) + 4, str(v), "end", MUTED, mono=True)
    for v in (100, 300, 1000, 3000, 10000, 30000):
        svg.line(x(v), T, x(v), H - B, GRID)
        svg.text(x(v), H - B + 18, f"{v // 1000}k" if v >= 1000 else str(v), "middle", MUTED, mono=True)
    svg.line(L, H - B, W - R, H - B, RULE)
    svg.text((L + W - R) / 2, H - 14, "Prompt tokens per answer, log scale", "middle", MUTED)
    svg.text(14, T + 10, "Accuracy, LongMemEval rule", "end", MUTED, rotate=-90)
    ceiling = pct(ling["oracle"]["accuracy"]["longmemeval"])
    svg.line(L, y(ceiling), W - R, y(ceiling), INK, dash="4 4")
    svg.text(W - R, y(ceiling) - 8, f"Oracle ceiling {ceiling}", "end", MUTED)
    placement = {"oracle": (12, -12, "start"), "window": (12, -12, "start"), "file": (-12, -12, "end"),
                 "mem0": (12, -12, "start"), "langmem": (12, 22, "start")}
    for key in ("oracle", "window", "file", "mem0", "langmem"):
        s = ling[key]
        acc, (lo, hi) = pct(s["accuracy"]["longmemeval"]), s["accuracy_ci95"]
        cx, cy = x(s["tokens_per_answer"]), y(acc)
        svg.line(cx, y(pct(lo)), cx, y(pct(hi)), RULE, 2)
        fill = GREEN if key in ("mem0", "langmem") else (WHITE if key == "oracle" else INK)
        svg.circle(cx, cy, 7, fill, INK if key == "oracle" else None)
        dx, dy, anchor = placement[key]
        svg.text(cx + dx, cy + dy, f"{NAMES[key]} · {acc}", anchor, INK, weight=600)
        svg.text(cx + dx, cy + dy + 14, f"{round(s['tokens_per_answer']):,} tok", anchor, MUTED, mono=True)
    svg.text(L, 16, "Whiskers: 95% bootstrap interval. Green: memory products. Black: baselines. Hollow: oracle.", "start", MUTED, 11)
    svg.write(out / "accuracy-vs-tokens.svg")


def chart_tracks(ling: dict, nemo: dict, out: Path) -> None:
    rows = ORDER
    W, L, R, T, row_h, B = 880, 150, 40, 30, 44, 44
    H = T + len(rows) * row_h + B
    svg = Svg(W, H)
    x = lambda a: L + a / 100 * (W - L - R)
    for v in (0, 20, 40, 60, 80, 100):
        svg.line(x(v), T - 8, x(v), H - B, GRID)
        svg.text(x(v), H - B + 18, str(v), "middle", MUTED, mono=True)
    svg.text((L + W - R) / 2, H - 8, "Accuracy, LongMemEval rule, with 95% interval. Green: Ling answering. Ochre: Nemotron answering.", "middle", MUTED)
    for i, key in enumerate(rows):
        yc = T + i * row_h + row_h / 2
        svg.text(L - 14, yc + 4, NAMES[key], "end", INK, weight=600)
        a, b = ling.get(key), nemo.get(key)
        if a:
            lo, hi = a["accuracy_ci95"]
            svg.line(x(pct(lo)), yc - 8, x(pct(hi)), yc - 8, GREEN, 3, opacity=0.35)
            svg.circle(x(pct(a["accuracy"]["longmemeval"])), yc - 8, 6, GREEN)
        else:
            svg.text(x(0) + 4, yc - 4, "not runnable: Ling has no JSON mode", "start", MUTED)
        if b:
            done, errors = b["questions"] - b["errors"], b["errors"]
            lo, hi = b["accuracy_ci95"]
            svg.line(x(pct(lo)), yc + 8, x(pct(hi)), yc + 8, OCHRE, 3, opacity=0.35)
            svg.circle(x(pct(b["accuracy"]["longmemeval"])), yc + 8, 6, OCHRE)
            if done < 10 or errors:
                svg.text(x(pct(hi)) + 12, yc + 12, f"{done} of 10 done", "start", MUTED, mono=True)
        if i < len(rows) - 1:
            svg.line(10, yc + row_h / 2, W - R, yc + row_h / 2, GRID)
    svg.write(out / "tracks.svg")


def chart_abilities(ling: dict, out: Path) -> None:
    rows = ["oracle", "window", "file", "mem0", "langmem"]
    W, L, T, cell_h = 880, 150, 40, 42
    cell_w = (W - L - 20) / len(ABILITIES)
    H = T + len(rows) * cell_h + 24
    svg = Svg(W, H)
    for j, ab in enumerate(ABILITIES):
        svg.text(L + j * cell_w + cell_w / 2, T - 14, ABILITY_LABELS[ab], "middle", MUTED)
    for i, key in enumerate(rows):
        yc = T + i * cell_h
        svg.text(L - 14, yc + cell_h / 2 + 4, NAMES[key], "end", INK, weight=600)
        for j, ab in enumerate(ABILITIES):
            v = pct(ling[key]["by_ability"][ab]["longmemeval"])
            v = 100 if v >= 100 else 50 if v >= 50 else 0
            svg.rect(L + j * cell_w + 3, yc + 3, cell_w - 6, cell_h - 6, HEAT[v], RULE if v == 0 else None, rx=3)
            svg.text(L + j * cell_w + cell_w / 2, yc + cell_h / 2 + 4, str(v), "middle", WHITE if v == 100 else INK, weight=600, mono=True)
    svg.text(L, H - 6, "Two questions per ability: 100 = both right, 50 = one, 0 = none. Ling track.", "start", MUTED, 11, mono=True)
    svg.write(out / "abilities.svg")


def chart_time(timing: dict, out: Path) -> None:
    rows = [k for k in ("oracle", "window", "file", "langmem", "mem0", "graphiti", "cognee") if k in timing]
    W, L, R, T, row_h, B = 880, 150, 120, 20, 40, 44
    H = T + len(rows) * row_h + B
    max_s = max(540.0, max(timing[k]["ingest"] + timing[k]["answer"] for k in rows) * 1.05)
    svg = Svg(W, H)
    x = lambda s: L + s / max_s * (W - L - R)
    v = 0
    while v <= max_s:
        svg.line(x(v), T - 6, x(v), H - B, GRID)
        if v % 120 == 0:
            svg.text(x(v), H - B + 18, f"{v // 60} min", "middle", MUTED, mono=True)
        v += 60
    svg.text((L + W - R) / 2, H - 8, "Median seconds per question, Nemotron track. Grey: ingest of twelve sessions. Green: answer.", "middle", MUTED)
    for i, key in enumerate(rows):
        t = timing[key]
        yc, h = T + i * row_h + 8, row_h - 16
        svg.text(L - 14, yc + h / 2 + 4, NAMES[key], "end", INK, weight=600)
        if t["ingest"] > 0:
            svg.rect(x(0), yc, x(t["ingest"]) - x(0), h, MUTED)
        svg.rect(x(t["ingest"]), yc, max(2.0, x(t["ingest"] + t["answer"]) - x(t["ingest"])), h, GREEN)
        label = (f"{t['ingest']:.0f} s ingest + " if t["ingest"] else "") + f"{t['answer']:.1f} s answer"
        if t["n"] < 10:
            label += f" · {t['n']} question{'s' if t['n'] != 1 else ''}"
        svg.text(x(t["ingest"] + t["answer"]) + 8, yc + h / 2 + 4, label, "start", MUTED, mono=True)
    svg.write(out / "ingest-vs-answer.svg")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Draw the results charts as SVG.")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "charts")
    # The write-up's figures are the pilot's, frozen under results/pilot-2026-09-07;
    # results/tracks moves with every nightly run.
    parser.add_argument("--tracks-dir", type=Path, default=REPO_ROOT / "results" / "pilot-2026-09-07" / "tracks")
    parser.add_argument("--ling", default="ling-3.0-flash-fin")
    parser.add_argument("--nemotron", default="nemotron-3-super-120b")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    ling, nemo = load_summary(args.ling, args.tracks_dir), load_summary(args.nemotron, args.tracks_dir)
    chart_accuracy_vs_tokens(ling, args.out)
    chart_tracks(ling, nemo, args.out)
    chart_abilities(ling, args.out)
    chart_time(timing_medians(args.nemotron), args.out)
    for name in ("accuracy-vs-tokens", "tracks", "abilities", "ingest-vs-answer"):
        print(f"wrote {args.out / (name + '.svg')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
