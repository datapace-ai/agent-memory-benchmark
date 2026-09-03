# Agent Memory Benchmark

A vendor-neutral benchmark for AI agent memory systems, run by Datapace.

Every system does the same job: ingest a user's conversation history one session
at a time in chronological order, then answer a question using only what it
stored. Every system uses the same local model, the same answer prompt, and the
same judge. A score difference is the memory, not the model.

It runs on a laptop and costs nothing.

## What is measured

Accuracy under three published judge rules, tokens per answer, retrieval and
answer latency, ingestion cost, a forgetting curve by distance from the
evidence, and the rate at which a system serves a superseded value after a fact
has changed.

## Systems

| Name | What it is |
| --- | --- |
| `oracle` | A ceiling, not a competitor. Sees only the evidence sessions. |
| `window` | No product. Keeps the most recent 32k tokens of history. |

Mem0, Graphiti (Zep's engine), Letta, LangMem, Cognee, and a file-search
baseline arrive in phase 2.

## Reproduce

```bash
brew install ollama && ollama serve &
ollama pull qwen3:14b && ollama pull nomic-embed-text

uv sync --extra dev
uv run pytest -q
uv run python -m membench.data.download
./scripts/smoke.sh
```

A full run is `uv run python -m membench.run` followed by
`uv run python -m membench.report`. It is resumable: rerun the same command
after an interruption and it continues where it stopped.

## Cost

Zero. Everything runs on a local model.

See `METHODS.md` for the protocol, the dataset reduction, and the limits.
