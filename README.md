# Agent Memory Benchmark

Built and maintained by [Datapace](https://datapace.ai) for the article [Mem0 vs Zep vs Letta vs LangMem vs Cognee: which to pick](https://datapace.ai/blog/ai-agent-memory-tools-2026), which uses these numbers and explains how to choose a memory tool from your own measurements.

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
more than any memory system does; nobody beat file search. Of the free
models, Ling 3.0 Flash did best on every system but cannot run Cognee or
Graphiti, which need Nemotron 3 Super.

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

Everything runs on free models through OpenRouter. Create a free account,
make a key, and put it in `.env` (never committed):

```bash
cp .env.example .env   # then paste your key after OPENROUTER_API_KEY=
```

Then:

```bash
uv sync --extra dev
uv run pytest -q
./scripts/smoke.sh                          # 5 questions, oracle and window, one seed
```

The products need their own environment, because their dependencies conflict
with each other and with the harness:

```bash
UV_PROJECT_ENVIRONMENT=.venv-vendors uv sync --extra dev --extra vendors
./scripts/vendor_smoke.sh                   # 1 question through each product
./scripts/run_tracks.sh --limit 10 --seeds 11 --workers 2   # the pilot
```

`scripts/run_tracks.sh` runs every track in `configs/tracks.yaml` with the
systems that track can carry and writes a report per track. Runs are
resumable: rerun the same command after an interruption or a rate-limit stop
and it continues where it stopped; add `--retry-errors` to redo units that
failed on the endpoint. The question set is committed, so no download is
needed; `python -m membench.data.download` rebuilds it from LongMemEval.

## Model tracks

The same questions and the same judge can be run with different answerers,
which also become the model inside each product. `configs/tracks.yaml` lists
the tracks; `scripts/run_tracks.sh` runs them one after another into
`results/runs/track-<name>.jsonl` and `results/tracks/<name>/`. The judge is
one model for every track, so accuracy differences between tracks come from
the answerer, not the grader.

`results/README.md`, section 3.5, states for each track and each system which
model extracted, embedded, reranked, answered and judged.

## Cost

Zero dollars. OpenRouter's free tier allows 20 requests per minute and 1,000
per day per account; a ten-question track for the two graph products uses
most of a day. A local setup from the first phase is kept under
`local-track/` for anyone who prefers to run offline; the published results
do not use it.

See `METHODS.md` for the protocol, the dataset reduction, and the limits.
