# Agent memory benchmark results

Answerer: inclusionai/ling-3.0-flash-fin:free. Judge: inclusionai/ling-3.0-flash-fin:free.

The oracle row is a ceiling, not a competitor. It sees only the evidence sessions, so it measures the best this answerer and judge can do on this set.

| System | LongMemEval rule | Zep rule | Mem0 rule | Share of ceiling | Gap to best baseline | Tokens per answer | Answer p50 | Answer p95 | Errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| oracle | 85.0 | 90.0 | 95.0 | 100.0 | +5.0 | 6180 | 6.6 | 7.8 | 0 |
| window | 80.0 | 85.0 | 85.0 | 94.1 | +0.0 | 26770 | 7.9 | 9.5 | 0 |
| mem0 | 80.0 | 80.0 | 85.0 | 94.1 | +0.0 | 541 | 0.8 | 1.4 | 0 |
| file | 75.0 | 80.0 | 85.0 | 88.2 | -5.0 | 13009 | 24.9 | 48.5 | 0 |
| langmem | 60.0 | 60.0 | 65.0 | 70.6 | -20.0 | 1404 | 0.9 | 7.4 | 0 |

Ranking is NOT stable across judge rules. Treat the order as unresolved.

## Accuracy by ability, LongMemEval rule

| System | abstention | extraction | knowledge_update | multi_session | temporal |
| --- | ---: | ---: | ---: | ---: | ---: |
| oracle | 50.0 | 75.0 | 100.0 | 100.0 | 100.0 |
| window | 75.0 | 75.0 | 100.0 | 75.0 | 75.0 |
| mem0 | 50.0 | 100.0 | 100.0 | 75.0 | 75.0 |
| file | 50.0 | 50.0 | 100.0 | 75.0 | 100.0 |
| langmem | 50.0 | 100.0 | 75.0 | 50.0 | 25.0 |

## Forgetting curve, accuracy by sessions since the evidence

- oracle: 0: 100, 1: 100, 2: 100, 4: 75, 5: 100, 6: 100, 7: 50, 8: 50
- window: 0: 100, 1: 67, 2: 100, 4: 75, 5: 100, 6: 67, 7: 100, 8: 50
- mem0: 0: 100, 1: 0, 2: 100, 4: 100, 5: 100, 6: 67, 7: 100, 8: 100
- file: 0: 100, 1: 67, 2: 50, 4: 75, 5: 100, 6: 67, 7: 100, 8: 50
- langmem: 0: 67, 1: 33, 2: 50, 4: 50, 5: 100, 6: 33, 7: 100, 8: 100

## Stale answers on knowledge updates

- oracle: 0.0 percent of wrong answers gave a superseded value
- window: 0.0 percent of wrong answers gave a superseded value
- mem0: 0.0 percent of wrong answers gave a superseded value
- file: 0.0 percent of wrong answers gave a superseded value
- langmem: 0.0 percent of wrong answers gave a superseded value
