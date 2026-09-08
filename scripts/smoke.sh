#!/usr/bin/env bash
# Smoke run: 5 questions, oracle and window, one seed, on the configured free
# model. The integration test and the gate before any full run.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; [ -f .env ] && . ./.env; set +a

uv run python -m membench.run \
  --limit 5 --seeds 11 --systems oracle,window \
  --out results/runs/smoke.jsonl

uv run python -m membench.report \
  --runs results/runs/smoke.jsonl \
  --out-dir results/smoke
