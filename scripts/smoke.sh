#!/usr/bin/env bash
# Smoke run: 5 questions, both phase 1 systems, one seed. The integration test
# and the gate before any full run.
set -euo pipefail
cd "$(dirname "$0")/.."

uv run python -m membench.run \
  --limit 5 --seeds 11 \
  --out results/runs/smoke.jsonl

uv run python -m membench.report \
  --runs results/runs/smoke.jsonl \
  --out-dir results/smoke
