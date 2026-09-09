#!/usr/bin/env bash
# Run nightly steps back to back until the daily cap stops one, or MAX_STEPS.
# Usage: scripts/nightly_loop.sh [max_steps]   (default 6)
set -uo pipefail
cd "$(dirname "$0")/.."
max=${1:-6}
for i in $(seq 1 "$max"); do
  log=$(mktemp -t nightly-step)
  scripts/nightly.sh 2>&1 | tee "$log"
  if grep -q "daily cap reached" "$log"; then echo "=== stopped by the daily cap after step $i ==="; exit 0; fi
  if grep -q "every seed complete" "$log"; then echo "=== all seeds complete ==="; exit 0; fi
done
echo "=== ran $max steps ==="
