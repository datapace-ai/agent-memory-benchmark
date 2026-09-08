#!/usr/bin/env bash
# Run every track in configs/tracks.yaml under the shared judge.
# Usage: scripts/run_tracks.sh [extra membench.run args, e.g. --limit 5 --seeds 11 --systems oracle,window]
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; [ -f .env ] && . ./.env; set +a
export UV_PROJECT_ENVIRONMENT=.venv-vendors
judge=$(uv run python -c "import yaml; print(yaml.safe_load(open('configs/tracks.yaml'))['judge_model'])")
# A track's own systems list applies unless the caller passes --systems.
extra_systems=()
case " $* " in *" --systems "*) ;; *) extra_systems=(use_track) ;; esac
uv run python -c "import yaml; [print(t['name'], t['answer_model'], t.get('systems', '')) for t in yaml.safe_load(open('configs/tracks.yaml'))['tracks']]" | while read -r name model systems; do
  echo "=== track $name ($model), judge $judge, systems ${systems:-default} ==="
  args=("$@")
  if [ "${#extra_systems[@]}" -gt 0 ] && [ -n "$systems" ]; then args+=(--systems "$systems"); fi
  uv run python -m membench.run --answer-model "$model" --judge-model "$judge" --out "results/runs/track-$name.jsonl" "${args[@]}"
  uv run python -m membench.report --runs "results/runs/track-$name.jsonl" --out-dir "results/tracks/$name"
done
