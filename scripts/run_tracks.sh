#!/usr/bin/env bash
# Run every track in configs/tracks.yaml under the shared judge.
# Usage: scripts/run_tracks.sh [extra membench.run args, e.g. --limit 5 --seeds 11 --systems oracle,window]
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; [ -f .env ] && . ./.env; set +a
export UV_PROJECT_ENVIRONMENT=.venv-vendors
judge=$(uv run python -c "import yaml; print(yaml.safe_load(open('configs/tracks.yaml'))['judge_model'])")
uv run python -c "import yaml; [print(t['name'], t['answer_model']) for t in yaml.safe_load(open('configs/tracks.yaml'))['tracks']]" | while read -r name model; do
  echo "=== track $name ($model), judge $judge ==="
  uv run python -m membench.run --answer-model "$model" --judge-model "$judge" --out "results/runs/track-$name.jsonl" "$@"
  uv run python -m membench.report --runs "results/runs/track-$name.jsonl" --out-dir "results/tracks/$name"
done
