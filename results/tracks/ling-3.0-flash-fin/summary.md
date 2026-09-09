# Agent memory benchmark results

Answerer: inclusionai/ling-3.0-flash-fin:free. Judge: inclusionai/ling-3.0-flash-fin:free.

The oracle row is a ceiling, not a competitor. It sees only the evidence sessions, so it measures the best this answerer and judge can do on this set.

| System | LongMemEval rule | Zep rule | Mem0 rule | Share of ceiling | Gap to best baseline | Tokens per answer | Answer p50 | Answer p95 | Errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| oracle | 80.0 | 88.0 | 92.0 | 100.0 | +8.0 | 6260 | 6.7 | 7.8 | 0 |
| mem0 | 76.0 | 80.0 | 84.0 | 95.0 | +4.0 | 559 | 0.8 | 1.2 | 0 |
| window | 72.0 | 80.0 | 80.0 | 90.0 | +0.0 | 26702 | 8.2 | 9.2 | 0 |
| file | 72.0 | 76.0 | 80.0 | 90.0 | +0.0 | 14092 | 24.9 | 48.7 | 0 |
| langmem | 56.0 | 60.0 | 64.0 | 70.0 | -16.0 | 1389 | 0.9 | 6.7 | 0 |

Ranking is NOT stable across judge rules. Treat the order as unresolved.

## Accuracy by ability, LongMemEval rule

| System | abstention | extraction | knowledge_update | multi_session | temporal |
| --- | ---: | ---: | ---: | ---: | ---: |
| oracle | 60.0 | 60.0 | 100.0 | 80.0 | 100.0 |
| mem0 | 60.0 | 100.0 | 100.0 | 60.0 | 60.0 |
| window | 80.0 | 60.0 | 100.0 | 60.0 | 60.0 |
| file | 60.0 | 40.0 | 80.0 | 80.0 | 100.0 |
| langmem | 60.0 | 80.0 | 80.0 | 40.0 | 20.0 |

## Forgetting curve, accuracy by sessions since the evidence

- oracle: 0: 75, 1: 100, 2: 67, 3: 100, 4: 75, 5: 100, 6: 100, 7: 50, 8: 50
- mem0: 0: 75, 1: 40, 2: 100, 3: 0, 4: 100, 5: 100, 6: 67, 7: 100, 8: 100
- window: 0: 75, 1: 80, 2: 67, 3: 0, 4: 75, 5: 100, 6: 67, 7: 100, 8: 50
- file: 0: 100, 1: 60, 2: 33, 3: 100, 4: 75, 5: 100, 6: 67, 7: 100, 8: 50
- langmem: 0: 50, 1: 60, 2: 33, 3: 0, 4: 50, 5: 100, 6: 33, 7: 100, 8: 100

## Stale answers on knowledge updates

- oracle: 0.0 percent of wrong answers gave a superseded value
- mem0: 0.0 percent of wrong answers gave a superseded value
- window: 0.0 percent of wrong answers gave a superseded value
- file: 0.0 percent of wrong answers gave a superseded value
- langmem: 0.0 percent of wrong answers gave a superseded value
