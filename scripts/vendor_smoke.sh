#!/usr/bin/env bash
# Vendor smoke: one question through each product on the configured free
# model, one seed, in the vendors environment. Cognee and Graphiti need a
# model that accepts JSON-schema output (see configs/tracks.yaml).
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; [ -f .env ] && . ./.env; set +a
export UV_PROJECT_ENVIRONMENT=.venv-vendors
uv run python -m membench.run \
  --limit 1 --seeds 11 --systems file,mem0,langmem,cognee,graphiti --workers 1 \
  --out results/runs/vendor-smoke.jsonl
uv run python -m membench.report \
  --runs results/runs/vendor-smoke.jsonl \
  --out-dir results/vendor-smoke
