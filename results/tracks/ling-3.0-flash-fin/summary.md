# Agent memory benchmark results

Answerer: inclusionai/ling-3.0-flash-fin:free. Judge: inclusionai/ling-3.0-flash-fin:free.

The oracle row is a ceiling, not a competitor. It sees only the evidence sessions, so it measures the best this answerer and judge can do on this set.

| System | LongMemEval rule | Zep rule | Mem0 rule | Share of ceiling | Gap to best baseline | Tokens per answer | Answer p50 | Answer p95 | Errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| oracle | 86.7 | 86.7 | 93.3 | 100.0 | +6.7 | 6295 | 6.5 | 7.4 | 0 |
| window | 80.0 | 80.0 | 80.0 | 92.3 | +0.0 | 27098 | 7.6 | 8.8 | 0 |
| file | 80.0 | 86.7 | 93.3 | 92.3 | +0.0 | 14997 | 24.9 | 48.5 | 0 |
| mem0 | 73.3 | 73.3 | 80.0 | 84.6 | -6.7 | 545 | 0.8 | 2.1 | 0 |
| langmem | 66.7 | 66.7 | 73.3 | 76.9 | -13.3 | 1430 | 0.9 | 10.8 | 0 |

Ranking is NOT stable across judge rules. Treat the order as unresolved.

## Accuracy by ability, LongMemEval rule

| System | abstention | extraction | knowledge_update | multi_session | temporal |
| --- | ---: | ---: | ---: | ---: | ---: |
| oracle | 66.7 | 66.7 | 100.0 | 100.0 | 100.0 |
| window | 100.0 | 66.7 | 100.0 | 66.7 | 66.7 |
| file | 66.7 | 33.3 | 100.0 | 100.0 | 100.0 |
| mem0 | 33.3 | 100.0 | 100.0 | 66.7 | 66.7 |
| langmem | 33.3 | 100.0 | 100.0 | 66.7 | 33.3 |

## Forgetting curve, accuracy by sessions since the evidence

- oracle: 0: 100, 1: 100, 2: 100, 4: 100, 6: 100, 7: 50, 8: 50
- window: 0: 100, 1: 67, 2: 100, 4: 100, 6: 67, 7: 100, 8: 50
- file: 0: 100, 1: 67, 2: 100, 4: 100, 6: 67, 7: 100, 8: 50
- mem0: 0: 100, 1: 0, 2: 100, 4: 100, 6: 67, 7: 100, 8: 100
- langmem: 0: 100, 1: 33, 2: 100, 4: 50, 6: 33, 7: 100, 8: 100

## Stale answers on knowledge updates

- oracle: 0.0 percent of wrong answers gave a superseded value
- window: 0.0 percent of wrong answers gave a superseded value
- file: 0.0 percent of wrong answers gave a superseded value
- mem0: 0.0 percent of wrong answers gave a superseded value
- langmem: 0.0 percent of wrong answers gave a superseded value
