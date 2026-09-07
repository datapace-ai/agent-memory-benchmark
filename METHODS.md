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

Release one runs on OpenRouter's free models rather than a local model, so no
laptop is loaded and the model is one readers can name. Provider and model
tags, the judge, and the embedding model are recorded in every run record.

- Free models were chosen by a responsiveness probe on 2026-09-07 (13 models
  with at least 64k context and a seed parameter; 7 calls each: two small
  prompts, one 4,000-token prompt, one tool call, three in parallel):

  | Model | Calls answered | Median latency | Tool call |
  | --- | ---: | ---: | --- |
  | inclusionai/ling-3.0-flash-fin:free | 7 of 7 | 0.7 s | yes |
  | inclusionai/ling-3.0-flash-sante:free | 7 of 7 | 0.7 s | yes |
  | cohere/north-mini-code:free | 7 of 7 | 0.8 s | yes |
  | nvidia/nemotron-3-super-120b-a12b:free | 6 of 7 | 0.5 s | yes |
  | nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free | 6 of 7 | 0.5 s | yes |
  | nvidia/nemotron-3.5-lightning:free | 7 of 7 | 71 s | yes |
  | nvidia/nemotron-3-ultra-550b-a55b:free | 6 of 7 | 42 s | yes |
  | google/gemma-4-31b-it:free, gemma-4-26b-a4b-it:free | 0 of 7 | upstream pool saturated | |
  | thinkingmachines/inkling, inkling-small | 0 of 7 | refused | |
  | liquid/lfm-2.5-2.6b:free | 0 of 7 | rejected | |

- Answerers, as tracks: `inclusionai/ling-3.0-flash-fin:free` first, then
  `nvidia/nemotron-3-super-120b-a12b:free` and `cohere/north-mini-code:free`.
  The track's answerer is also the model inside each product.
- Judge: `inclusionai/ling-3.0-flash-fin:free` for every system and every
  track, so a difference between tracks is the answerer, not the grader. On the
  Ling track the judge grades its own answers; that is a known leniency risk,
  stated here, and it applies equally to every system inside the track.
- Reasoning off on every call through OpenRouter's `reasoning` parameter, and
  any residual think block is stripped before grading.
- OpenRouter caps free models at 20 requests per minute per account (the
  response body names the limit `free-models-per-min`). Requests are paced
  at one every 3.5 seconds across all worker threads and retried with backoff
  on rate-limit responses. The products' own calls to the model go through
  their libraries' clients with their own retries and are not paced by the
  harness. Latency figures on a shared free tier are therefore noisier than on
  a dedicated endpoint; token counts are unaffected.
- Embeddings for the products run on the CPU with `BAAI/bge-small-en-v1.5`
  (384 dimensions), because no free API serves embeddings.

The original local track (Ollama, qwen3:14b, reasoning off) is kept in
`configs/models.ollama.yaml` for anyone who wants to reproduce on their own
machine; its measured cost per unit is below.

### Local track, kept for reference

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
| Mem0 open source | mem0ai 2.0.20 | `Memory.add(messages, user_id)` per session | `Memory.search(query, filters={user_id}, top_k)` | LangChain chat model through Mem0's `langchain` provider: `ChatOpenAI` at the API provider with `reasoning: {enabled: false}`, or `ChatOllama(reasoning=False)` locally. Runs one unit at a time: each instance also opens a global store under `~/.mem0`. |
| LangMem | langmem 0.0.30 | `MemoryStoreManager.invoke({messages})` per session | `MemoryStoreManager.search(query)` | same LangChain chat model as Mem0 |
| File search (baseline) | none | one dated text file per session | the model's own `list_files`, `read_file`, `grep` calls, at most 8, then it must answer | `think: false` on every call |
| Graphiti (Zep's engine) | graphiti-core 0.30.1, embedded FalkorDB | `add_episode` per turn with the session date as reference time, one group per namespace | `Graphiti.search(query, group_ids)` edge facts | the `qwen3-nothink:14b` variant through the OpenAI-compatible endpoint; BGE reranker runs locally |
| Letta | letta-client 1.12.1 against the `letta/letta` Docker server | two user messages per session, the dated first turn then the full transcript, at most 6 agent steps | none: the agent answers inside Letta; tokens are Letta's reported usage | `reasoning=False`, `enable_reasoner=False` on the agent |
| Cognee | cognee 1.5.4 | `cognee.add(text, dataset)` then `cognee.cognify([dataset])` per session | `cognee.search(GRAPH_COMPLETION, only_context=True)` | the `qwen3-nothink:14b` variant through the OpenAI-compatible endpoint, plus `LLM_ARGS={"think": false}` |

Reset semantics: Mem0 deletes the namespace's memories; LangMem builds a fresh
in-memory store; Cognee uses one dataset per namespace and prunes the whole
system once at the start of a run; file search empties the namespace folder;
Graphiti opens a fresh embedded database; Letta deletes and recreates the agent.

The `qwen3-nothink:14b` variant is `qwen3:14b` with two template edits that
force the `/no_think` switch and an empty think block on every turn, for
libraries that reach Ollama through its OpenAI-compatible endpoint, which has
no reasoning flag. `scripts/create_nothink_model.sh` builds it and
`models/Modelfile.qwen3-nothink` is committed.

Embeddings for the products on an API track are `BAAI/bge-small-en-v1.5`
(384 dimensions) on the CPU: LangChain `HuggingFaceEmbeddings` for Mem0 and
LangMem, `fastembed` for Cognee, a sentence-transformers wrapper for Graphiti.

Vendor smoke results per product (ingestion seconds per session, retrieved
context shape, failures): recorded below as each product completes its first
ten-question pass.

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

Written after each full run. Findings while building are kept here too.

- 2026-09-07, judge truncation. On the first API track, five of twenty
  hand-checked LongMemEval-rule verdicts were wrong for one reason: the judge
  model began a numbered analysis, the 64-token output limit cut it off before
  the verdict, and the parser scored the truncation as "no". The JSON-style
  Zep and Mem0 rules were unaffected because their prompts ask for a JSON
  object the model produces first. Fix: the grader is told to reply with
  exactly one word, the limit is 256 tokens, a verdict that never says yes or
  no is flagged and counted in the summary, and `python -m membench.rejudge`
  re-grades an existing run without re-running its answers. Every published
  table carries the count of unparsed verdicts, and any run graded before this
  fix is re-judged before it is reported.
- 2026-09-07, Mem0 concurrency. Two Mem0 instances in one process collide on a
  global local store under `~/.mem0`, whatever path each instance is given.
  Mem0 units therefore run one at a time. Cognee's configuration is
  process-global for the same reason.
- 2026-09-07, tool-call markup. Withheld tools on the file baseline's final
  turn made one model write tool-call markup as its answer. The final turn now
  says the tools are gone and asks for the answer in prose.
