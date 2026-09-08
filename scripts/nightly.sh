#!/usr/bin/env bash
# One night of the full run within OpenRouter's free daily cap.
#
# Advances every track by STEP questions on the current seed, seed by seed:
# seed 11 completes on every system before seed 22 starts, so partial reports
# stay balanced. The runner stops by itself when the daily cap is hit and
# continues where it stopped the next night; finished units are never redone.
#
# Usage: scripts/nightly.sh            (STEP=5 SEEDS=11,22,33 MAX=100 by default)
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; [ -f .env ] && . ./.env; set +a
export UV_PROJECT_ENVIRONMENT=.venv-vendors
STEP=${STEP:-5}
SEEDS=${SEEDS:-11,22,33}
MAX=${MAX:-100}

echo "=== nightly start $(date) ==="
for seed in ${SEEDS//,/ }; do
  # The smallest number of questions any (track, system) has finished on this seed.
  done_min=$(uv run python - "$seed" <<'EOF'
import json, sys, yaml
from pathlib import Path
seed = int(sys.argv[1])
tracks = yaml.safe_load(open("configs/tracks.yaml"))["tracks"]
least = None
for t in tracks:
    systems = [s for s in (t.get("systems") or "").split(",") if s]
    path = Path(f"results/runs/track-{t['name']}.jsonl")
    last = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if int(r["seed"]) == seed:
                    last[(r["system"], r["question_id"])] = r
    for s in systems:
        n = sum(1 for (sys_, _), r in last.items() if sys_ == s and not r.get("error"))
        least = n if least is None else min(least, n)
print(least or 0)
EOF
)
  if [ "$done_min" -ge "$MAX" ]; then
    echo "seed $seed complete on every track and system"
    continue
  fi
  limit=$(( done_min + STEP )); [ "$limit" -gt "$MAX" ] && limit=$MAX
  echo "seed $seed: least finished $done_min, running up to question $limit"
  scripts/run_tracks.sh --limit "$limit" --seeds "$seed" --workers 2 --retry-errors
  echo "=== nightly end $(date), run exit $? ==="
  exit 0
done
echo "every seed complete; nothing to do"
