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

## Products under test

Each product ingests a session as chat messages with the session date on the
first turn, because none of the open-source engines accept a timestamp at
ingestion. Retrieval returns the product's own context; the benchmark's client
then answers with the shared prompt, so the answer step and its token counts
are identical across systems. A product's internal extraction tokens are not
observable from outside; ingestion is reported as seconds per session and
store size.

| Product | Version | Ingest call | Retrieval call | How reasoning is turned off |
| --- | --- | --- | --- | --- |
| Mem0 open source | mem0ai 2.0.20 | `Memory.add(messages, user_id)` per session | `Memory.search(query, filters={user_id}, top_k)` | LangChain `ChatOllama(reasoning=False)` through Mem0's `langchain` provider |
| LangMem | langmem 0.0.30 | `MemoryStoreManager.invoke({messages})` per session | `MemoryStoreManager.search(query)` | `ChatOllama(reasoning=False)` |
| Cognee | cognee 1.5.4 | `cognee.add(text, dataset)` then `cognee.cognify([dataset])` per session | `cognee.search(GRAPH_COMPLETION, only_context=True)` | `LLM_ARGS={"think": false}` in Cognee's environment; verified on the vendor smoke by comparing ingestion time against the other products |

Reset semantics: Mem0 deletes the namespace's memories; LangMem builds a fresh
in-memory store; Cognee uses one dataset per namespace and prunes the whole
system once at the start of a run.

Vendor smoke results (ingestion seconds per session per product, whether
Cognee's reasoning flag reached Ollama, adapter failures): TO BE RECORDED after
the first vendor smoke, which waits for the phase 1 full run to finish.

## Judge

The LongMemEval per-type prompts are used verbatim from `evaluate_qa.py` in
`xiaowu0162/LongMemEval`. Every answer is scored again under Zep's rule and
Mem0's partial-credit rule, taken verbatim from `verify.py` in
`ThinkfleetAI/memmesh-benchmarks`, which reproduces `graphiti_core/prompts/eval.py`
and Mem0's `_JUDGE_TEMPLATE`. The judge never sees which system produced an
answer. Where the ranking changes between rules, the summary says so.

Judge agreement with human labels on the smoke sample (2026-09-07, 10 judged
answers, 5 questions, oracle and window, seed 11): 10 of 10 under the LongMemEval
rule. The two disagreements between rules were both the Mem0 partial-credit rule
accepting "at least five" against a gold of 4, which is that rule's written
"how many" clause behaving as published, not a grading error. The 50-answer
sample the design calls for is completed on the first full run.

## Measured cost on an Apple M5, 24 GB, qwen3:14b, reasoning off

| Condition | Prompt tokens | Seconds per unit (answer plus four judge calls) |
| --- | ---: | ---: |
| oracle (evidence sessions only) | 800 to 12,500 | 7 to 73, median about 30 |
| window (last 32k tokens) | 20,700 to 28,700 | 143 to 245, median about 200 |

Prefill runs at roughly 130 tokens per second on this machine, so the window
condition is bounded by prompt length. One full pass of 100 questions over both
conditions is about 6.5 hours per seed.

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
