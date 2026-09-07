"""Print judged answers for hand labelling, to measure judge agreement.

Usage: uv run python scripts/judge_sample.py results/runs/smoke.jsonl [N]

Shows question, gold answer, the model's answer, and the three verdicts,
without the system name, so the human label is as blind as the judge's.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from membench.config import REPO_ROOT
from membench.data.types import read_jsonl


def main() -> int:
    runs = Path(sys.argv[1])
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    questions = {q.question_id: q for q in read_jsonl(REPO_ROOT / "data" / "questions_s12.jsonl")}
    records = [json.loads(l) for l in runs.read_text().splitlines() if l.strip()]
    records = [r for r in records if not r.get("error")]
    random.Random(7).shuffle(records)
    for i, r in enumerate(records[:limit], start=1):
        q = questions[r["question_id"]]
        print(f"=== {i}. [{q.ability}] {r['question_id']} seed={r['seed']}")
        print(f"Q:    {q.question}")
        print(f"GOLD: {q.answer}")
        print(f"ANS:  {r['answer_text'][:400]}")
        print(
            f"JUDGE longmemeval={r['correct_longmemeval']} zep={r['correct_zep']} "
            f"mem0={r['correct_mem0']} stale={r.get('stale')}"
        )
        print(f"RAW:  {json.dumps(r.get('judge_raw', {}))[:300]}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
