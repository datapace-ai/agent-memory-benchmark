# Methods

## Protocol

For each question the system under test is reset to empty, then fed the
question's sessions one at a time in chronological order. It never sees the
whole history at once. After the last session the question is asked with its
own date given as today. The system may only use what it stored.

## Dataset

LongMemEval, cleaned 2025 release, from the Hugging Face dataset
`xiaowu0162/longmemeval-cleaned` (`longmemeval_s_cleaned.json`). We select 100
questions, 20 for each of five abilities: information extraction, multi-session
reasoning, temporal reasoning, knowledge updates, and abstention. Each history
is reduced to 12 sessions by keeping every evidence session and sampling
distractors from that question's own haystack, preserving the original
chronological order. Median reduced history is about 33,000 tokens.

Per-question sampling seeds come from sha256, not Python's built-in `hash()`,
because `PYTHONHASHSEED` randomizes string hashing per process and would make
the committed question set irreproducible. The selected set is committed as
`data/questions_s12.jsonl`, so a rerun does not need the download.

## Model

One local model through Ollama answers, judges, and would drive any framework's
internal extraction, so no system gets a stronger model than another.
Temperature 0, explicit seed on every call. Token counts come from Ollama's
`prompt_eval_count` and `eval_count`, never estimated. Model tag, Ollama
version, and Python version are recorded in every run record.

Reasoning is disabled (`think: false`). On a reasoning model the default
`<think>` block costs roughly four times the wall clock and hundreds of tokens
without changing the answer. It is disabled identically for every system, so
the benchmark measures retrieval rather than reasoning, and the token metric
compares memory footprints rather than reasoning verbosity.

## Judge

The LongMemEval per-type prompts are used verbatim from `evaluate_qa.py` in
`xiaowu0162/LongMemEval`. Every answer is scored again under Zep's rule and
Mem0's partial-credit rule, taken verbatim from `verify.py` in
`ThinkfleetAI/memmesh-benchmarks`, which reproduces `graphiti_core/prompts/eval.py`
and Mem0's `_JUDGE_TEMPLATE`. The judge never sees which system produced an
answer. Where the ranking changes between rules, the summary says so.

Judge agreement with human labels on the smoke sample: TO BE RECORDED after the
first smoke run.

## Limits

Our absolute scores sit below the vendors' published numbers because they used
paid frontier models as answerer and judge and we use a local model. Rankings,
cost, and latency are the durable findings.

LongMemEval asks each question after its whole history, so in this release the
clock orders sessions and drives the memory metrics but does not interleave
questions between sessions.

## Credits

Dataset and judge prompts: `xiaowu0162/LongMemEval`. Competing judge rules:
`ThinkfleetAI/memmesh-benchmarks`. Adapter references for phase 2:
`mem0ai/memory-benchmarks`, `getzep/zep-papers`, `letta-ai/letta-leaderboard`,
`RudrenduPaul/memtrust`.

## What we would do differently

Written after each full run.
