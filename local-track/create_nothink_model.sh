#!/usr/bin/env bash
# Build qwen3-nothink:14b: the qwen3:14b template with thinking forced off,
# for libraries that use Ollama's OpenAI-compatible endpoint (no think flag).
# Pure manifest work: no model download, no inference.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models
ollama show qwen3:14b --modelfile > local-track/Modelfile.qwen3-14b.orig
python3 - <<'PY'
import re
src = open("local-track/Modelfile.qwen3-14b.orig").read()
src = re.sub(r"^FROM .*$", "FROM qwen3:14b", src, count=1, flags=re.M)
before = src
# 1. The /think or /no_think suffix on the last user turn: make it unconditional /no_think.
src = src.replace("{{- if and $.IsThinkSet (eq $i $lastUserIdx) }}", "{{- if eq $i $lastUserIdx }}", 1)
src = re.sub(r"\{\{-\s*if \$\.Think\s*-\}\}.*?\{\{-\s*end\s*-\}\}", '{{- " "}}/no_think', src, count=1, flags=re.S)
# 2. The empty <think></think> pre-fill before the answer: make it unconditional.
src = src.replace("{{ if and $.IsThinkSet (not $.Think) -}}", "{{ if true -}}", 1)
assert src != before, "template anchors not found; inspect local-track/Modelfile.qwen3-14b.orig"
assert "/no_think" in src and "/think\n" not in src.replace("/no_think", ""), "edit 1 did not apply cleanly"
assert "{{ if true -}}" in src, "edit 2 did not apply"
open("local-track/Modelfile.qwen3-nothink", "w").write(src)
print("Modelfile written")
PY
ollama create qwen3-nothink:14b -f local-track/Modelfile.qwen3-nothink >/dev/null 2>&1
echo "no_think occurrences in new template: $(ollama show qwen3-nothink:14b --template | grep -c no_think)"
echo "conditional think blocks left: $(ollama show qwen3-nothink:14b --template | grep -c 'if \$.Think')"
