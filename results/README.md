# Pilot results, 7 September 2026

Ten LongMemEval questions, two per ability, one seed. Every system answered by
the same free model and graded by the same judge under three published rules.
Read the numbers as directions: at this size one question is ten points, and
every 95% interval overlaps every other.

| | |
| --- | --- |
| Questions | 10 from LongMemEval S, 12 sessions and about 127 turns of history each |
| Answerer, track A | Ling 3.0 Flash (free, via OpenRouter) |
| Answerer, track B | Nemotron 3 Super 120B (free, via OpenRouter) |
| Judge, both tracks | Ling 3.0 Flash, LongMemEval, Zep and Mem0 rules |
| Cost | 0 dollars; the free tier's 1,000 requests per day was the budget |

Charts are drawn from the committed summaries by `scripts/charts.py`.

## 1. Cheaper, not more accurate

Mem0 and LangMem answer with 20 to 50 times fewer tokens than the two
baselines and land one question lower. The dashed line is the oracle, which
sees only the sessions that hold the evidence: the best this answerer and judge
can do on these ten questions. Nobody beat the plain file-search baseline.

![Accuracy against prompt tokens per answer, Ling track](charts/accuracy-vs-tokens.svg)

## 2. The ceiling moves with the model, not the memory

Switching the answering model from Ling to Nemotron moved the oracle from 90 to
60 before any memory system was involved. A table that quotes a different model
quotes a different ceiling: compare products only under one model and print the
oracle next to them. Cognee and Graphiti need JSON-schema output, which Ling's
endpoint refuses, so they exist only on the Nemotron track, where they are
still partial.

![Accuracy per system on the Ling and Nemotron tracks](charts/tracks.svg)

## 3. Nobody knows what they don't know

Two questions per ability, so each cell is 0, 50 or 100. Every system except
the raw window scored 50 on abstention, and the window's 100 is the judge
accepting a long refusal. Memory layers do not fix an answerer's habit of
inventing a fish count for a tank the user never mentioned.

![Accuracy by ability and system, Ling track](charts/abilities.svg)

## 4. Knowledge graphs are paid for at ingest

Answer latency, the number vendors show, is under two seconds for every
product. Ingest, the number they do not show, is six to eight minutes per
question for Cognee and Graphiti: twelve sessions turned into a graph with on
the order of a hundred model calls, against one call for the baselines.
Graphiti's bar is one finished question and Cognee's four; the endpoint's own
slowness is included.

![Median ingest and answer seconds per question, Nemotron track](charts/ingest-vs-answer.svg)

## 5. The judge is a tenth of the score

The same answers under three published grading rules. Ten points of daylight
between rules is common. Before the grader was told to answer in one word, a
truncated chain of thought had flipped five of twenty verdicts.

| System | Ling, LongMemEval | Ling, Zep | Ling, Mem0 | Nemotron, LongMemEval | Nemotron, Zep | Nemotron, Mem0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Oracle ceiling | 90 | 90 | 90 | 60 | 60 | 70 |
| Window, 32k tokens | 80 | 80 | 80 | 50 | 50 | 70 |
| File search, 8 tool calls | 80 | 80 | 90 | 70 | 70 | 80 |
| Mem0 | 70 | 70 | 70 | 50 | 50 | 60 |
| LangMem | 70 | 70 | 70 | 70 | 70 | 70 |
| Cognee | not runnable | | | 25 (4 of 10 done) | 25 | 50 |
| Graphiti | not runnable | | | 0 (1 of 10 done) | 0 | 0 |

Nemotron's Mem0 row counts one unit that died on the endpoint as wrong; over
the nine finished units it is 56. Cognee and Graphiti rows are over finished
units only.

## 6. Free tiers benchmark ten questions a day

OpenRouter's free tier caps an account at 1,000 requests per day across all
free models, on top of 20 per minute. One day bought the whole Ling track, its
re-judge, the model probes and most of the Nemotron track. The two graph
products spend most of that budget on their own, so on this tier they advance
about ten questions a day.

## What to carry into the article

1. Name the answering model and print the oracle. The model moved the ceiling
   by thirty points; no product moved anything by that much.
2. Products buy context, not recall. Same neighbourhood as the baselines at a
   fraction of the tokens; nobody beat file search.
3. Abstention is unsolved by memory. Fifty across the board.
4. Report three judge rules and a hand-checked sample. The rule is worth ten
   points; a truncated grader was worth twenty-five.
5. Ask vendors for ingest cost, not answer latency. Minutes and a hundred
   calls per question for the graph products.
6. Budget in requests per day. A thousand free calls is ten product questions.

## Files

- `tracks/ling-3.0-flash-fin/summary.md` and `summary.json`: the complete Ling track.
- `tracks/nemotron-3-super-120b/summary.md` and `summary.json`: the Nemotron track, interim until Cognee and Graphiti finish.
- `runs/track-*.jsonl`: every answer, context, token count, timing and judge verdict, one line per unit.
- `charts/*.svg`: the figures above; regenerate with `uv run python scripts/charts.py`.
- `pilot-2026-09-07.html`: the same figures as a standalone page.
