#!/usr/bin/env bash
# Vendor smoke: 2 questions, the three product adapters, one seed, live Ollama.
# Refuses to run while the phase 1 full run holds its lock.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -e results/runs/runs.jsonl.lock ]; then
  echo "phase 1 full run is in progress (results/runs/runs.jsonl.lock exists); not touching Ollama" >&2
  exit 2
fi
export UV_PROJECT_ENVIRONMENT=.venv-vendors
./scripts/create_nothink_model.sh
./scripts/letta_server.sh
uv run python -m membench.run \
  --limit 2 --seeds 11 --systems file,mem0,langmem,cognee,graphiti,letta \
  --out results/runs/vendor-smoke.jsonl
uv run python -m membench.report \
  --runs results/runs/vendor-smoke.jsonl \
  --out-dir results/vendor-smoke
