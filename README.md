# Agent Memory Benchmark

A vendor-neutral benchmark for AI agent memory systems, run by Datapace.

Every system does the same job: ingest a user's conversation history one session
at a time in chronological order, then answer a question using only what it
stored. Every system uses the same local model, the same answer prompt, and the
same judge. A score difference is the memory, not the model.

It runs on a laptop and costs nothing.

## Results so far

The pilot run of 7 September 2026, ten questions and one seed on free models,
is written up with charts in [results/README.md](results/README.md). The short
version: memory products answer with 20 to 50 times fewer tokens than the
baselines and land one question lower; the answering model moves the ceiling
more than any memory system does; nobody beat file search.

![Accuracy against prompt tokens per answer](results/charts/accuracy-vs-tokens.svg)

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
| `mem0` | Mem0 open source: extracted memories in a local Qdrant store. |
| `langmem` | LangMem: a LangGraph memory store managed by its background extractor. |
| `cognee` | Cognee: a knowledge graph plus vector index built from the sessions. |
| `file` | No product. Dated session files the model lists, reads and greps with tools. |
| `graphiti` | Zep's open-source engine on an embedded FalkorDB, local BGE reranker. |
| `letta` | Letta's agent runtime on its self-hosted server; the agent answers itself. |
Product adapters live in a second environment until the phase 1 run finishes:
`UV_PROJECT_ENVIRONMENT=.venv-vendors uv sync --extra dev --extra vendors`.

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

## Model tracks

The same questions and the same judge can be run with different answerers,
which also become the model inside each product. `configs/tracks.yaml` lists
the tracks; `scripts/run_tracks.sh` runs them one after another into
`results/runs/track-<name>.jsonl` and `results/tracks/<name>/`. The judge is
one model for every track, so accuracy differences between tracks come from
the answerer, not the grader.

## Cost

Zero. Everything runs on a local model.

See `METHODS.md` for the protocol, the dataset reduction, and the limits.
