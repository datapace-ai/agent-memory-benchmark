# Agent memory benchmark results

Answerer: inclusionai/ling-3.0-flash-fin:free. Judge: inclusionai/ling-3.0-flash-fin:free.

The oracle row is a ceiling, not a competitor. It sees only the evidence sessions, so it measures the best this answerer and judge can do on this set.

| System | LongMemEval rule | Zep rule | Mem0 rule | Share of ceiling | Gap to best baseline | Tokens per answer | Answer p50 | Answer p95 | Errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| oracle | 90.0 | 90.0 | 90.0 | 100.0 | +10.0 | 6585 | 6.5 | 7.1 | 0 |
| window | 80.0 | 80.0 | 80.0 | 88.9 | +0.0 | 26828 | 7.5 | 8.1 | 0 |
| file | 80.0 | 80.0 | 90.0 | 88.9 | +0.0 | 15604 | 24.9 | 48.6 | 0 |
| mem0 | 70.0 | 70.0 | 70.0 | 77.8 | -10.0 | 539 | 0.8 | 2.8 | 0 |
| langmem | 70.0 | 70.0 | 70.0 | 77.8 | -10.0 | 1632 | 1.2 | 14.1 | 0 |

Ranking is NOT stable across judge rules. Treat the order as unresolved.

## Accuracy by ability, LongMemEval rule

| System | abstention | extraction | knowledge_update | multi_session | temporal |
| --- | ---: | ---: | ---: | ---: | ---: |
| oracle | 50.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| window | 100.0 | 100.0 | 100.0 | 50.0 | 50.0 |
| file | 50.0 | 50.0 | 100.0 | 100.0 | 100.0 |
| mem0 | 50.0 | 100.0 | 100.0 | 50.0 | 50.0 |
| langmem | 50.0 | 100.0 | 100.0 | 50.0 | 50.0 |

## Forgetting curve, accuracy by sessions since the evidence

- oracle: 0: 100, 1: 100, 4: 100, 6: 100, 7: 0, 8: 100
- window: 0: 100, 1: 50, 4: 100, 6: 50, 7: 100, 8: 100
- file: 0: 100, 1: 50, 4: 100, 6: 50, 7: 100, 8: 100
- mem0: 0: 100, 1: 0, 4: 100, 6: 50, 7: 100, 8: 100
- langmem: 0: 100, 1: 50, 4: 50, 6: 50, 7: 100, 8: 100

## Stale answers on knowledge updates

- oracle: 0.0 percent of wrong answers gave a superseded value
- window: 0.0 percent of wrong answers gave a superseded value
- file: 0.0 percent of wrong answers gave a superseded value
- mem0: 0.0 percent of wrong answers gave a superseded value
- langmem: 0.0 percent of wrong answers gave a superseded value
