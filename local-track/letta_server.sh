#!/usr/bin/env bash
# Self-hosted Letta server with Ollama as the model provider.
set -euo pipefail
docker rm -f membench-letta >/dev/null 2>&1 || true
docker run -d --name membench-letta -p 8283:8283 \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  letta/letta:latest >/dev/null
for _ in $(seq 1 90); do
  curl -sf http://localhost:8283/v1/health/ >/dev/null 2>&1 && break
  sleep 2
done
curl -sf http://localhost:8283/v1/health/ && echo && echo "letta up"
