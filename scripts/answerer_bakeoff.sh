#!/usr/bin/env bash
# Rank free answerers on this task: each candidate answers the same ten oracle
# questions (evidence only, seed 11) under the shared Ling judge, about 40
# calls per model, and is probed for tools, JSON-object and JSON-schema output.
# Usage: scripts/answerer_bakeoff.sh [model ...]
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; [ -f .env ] && . ./.env; set +a
export UV_PROJECT_ENVIRONMENT=.venv-vendors
JUDGE=inclusionai/ling-3.0-flash-fin:free
CANDIDATES=("$@")
[ ${#CANDIDATES[@]} -eq 0 ] && CANDIDATES=(
  dots-studio/dots-3-note-preview:free
  google/gemma-4-31b-it:free
  google/gemma-4-26b-a4b-it:free
  poolside/laguna-s-2.1:free
  cohere/north-mini-code:free
  inclusionai/ling-3.0-flash-sante:free
)
mkdir -p results/bakeoff
echo "=== bakeoff start $(date) ==="
for model in "${CANDIDATES[@]}"; do
  slug=$(echo "$model" | tr '/:' '__')
  echo "--- $model"
  uv run python - "$model" <<'PY'
import json, os, sys, time, urllib.request, urllib.error
model = sys.argv[1]; key = os.environ["OPENROUTER_API_KEY"]
schema = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
def call(extra, label):
    payload = {"model": model, "messages": [{"role": "user", "content": "Which city is the capital of France? Answer as JSON {\"city\": ...}."}],
               "max_tokens": 60, "reasoning": {"enabled": False}, **extra}
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=90) as r: d = json.load(r)
        if "error" in d: return f"{label}: error {d['error'].get('code')} {str(d['error'].get('message'))[:60]}"
        m = d["choices"][0]["message"]; c = (m.get("content") or "").strip().replace("\n", " ")
        ok = "tool_calls" in label and bool(m.get("tool_calls")) or ("Paris" in c)
        return f"{label}: {'ok' if ok else 'weak'} {time.time()-t0:.1f}s {c[:50]!r}"
    except urllib.error.HTTPError as e: return f"{label}: http {e.code} {e.read()[:80].decode(errors='replace')}"
    except Exception as e: return f"{label}: exc {str(e)[:80]}"
plain = call({}, "plain"); print("  ", plain); time.sleep(3.5)
if not plain.startswith("plain: ok"):
    print("   skipped: the plain call failed, no point spending the budget"); sys.exit(3)
print("  ", call({"response_format": {"type": "json_object"}}, "json_object")); time.sleep(3.5)
print("  ", call({"response_format": {"type": "json_schema", "json_schema": {"name": "c", "schema": schema}}}, "json_schema")); time.sleep(3.5)
print("  ", call({"tools": [{"type": "function", "function": {"name": "answer", "parameters": schema}}], "tool_choice": "auto"}, "tool_calls")); time.sleep(3.5)
PY
  [ $? -eq 3 ] && continue
  uv run python -m membench.run --limit 10 --seeds 11 --systems oracle --workers 2 --retry-errors \
    --answer-model "$model" --judge-model "$JUDGE" --out "results/bakeoff/$slug.jsonl" 2>&1 | grep -E '^\[|daily cap|failed after' | tail -n 3
  uv run python - "results/bakeoff/$slug.jsonl" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]); rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
last = {}
for r in rows: last[r["question_id"]] = r
ok = [r for r in last.values() if not r.get("error")]
acc = sum(bool(r["correct_longmemeval"]) for r in ok)
import statistics
lat = statistics.median(r["answer_seconds"] for r in ok) if ok else 0
print(f"   oracle: {acc}/{len(ok)} correct, {len(last)-len(ok)} errors, answer p50 {lat:.1f}s")
PY
done
echo "=== bakeoff end $(date) ==="
