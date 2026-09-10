# Agent memory benchmark results

Answerer: nvidia/nemotron-3-super-120b-a12b:free. Judge: inclusionai/ling-3.0-flash-fin:free.

The oracle row is a ceiling, not a competitor. It sees only the evidence sessions, so it measures the best this answerer and judge can do on this set.

| System | LongMemEval rule | Zep rule | Mem0 rule | Share of ceiling | Gap to best baseline | Tokens per answer | Answer p50 | Answer p95 | Errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| file | 70.0 | 70.0 | 80.0 | 116.7 | +0.0 | 23999 | 52.2 | 76.1 | 0 |
| langmem | 70.0 | 70.0 | 70.0 | 116.7 | +0.0 | 595 | 1.4 | 15.3 | 0 |
| oracle | 60.0 | 60.0 | 70.0 | 100.0 | -10.0 | 6431 | 7.1 | 29.4 | 0 |
| window | 50.0 | 50.0 | 70.0 | 83.3 | -20.0 | 26321 | 9.8 | 16.3 | 0 |
| mem0 | 50.0 | 50.0 | 60.0 | 83.3 | -20.0 | 490 | 1.1 | 12.6 | 1 |
| cognee | 12.5 | 12.5 | 25.0 | 20.8 | -57.5 | 3492 | 0.4 | 6.2 | 4 |
| graphiti | 0.0 | 0.0 | 0.0 | 0.0 | -70.0 | 341 | 0.8 | 0.8 | 0 |

Ranking is NOT stable across judge rules. Treat the order as unresolved.

## Accuracy by ability, LongMemEval rule

| System | abstention | extraction | knowledge_update | multi_session | temporal |
| --- | ---: | ---: | ---: | ---: | ---: |
| file | 50.0 | 100.0 | 50.0 | 50.0 | 100.0 |
| langmem | 50.0 | 100.0 | 100.0 | 50.0 | 50.0 |
| oracle | 50.0 | 50.0 | 50.0 | 100.0 | 50.0 |
| window | 50.0 | 50.0 | 50.0 | 100.0 | 0.0 |
| mem0 | 50.0 | 100.0 | 50.0 | 50.0 | 0.0 |
| cognee | 0.0 | 50.0 | 0.0 | 0.0 | 0.0 |
| graphiti | 0.0 | n/a | n/a | n/a | n/a |

## Forgetting curve, accuracy by sessions since the evidence

- file: 0: 50, 1: 0, 4: 100, 6: 100, 7: 100, 8: 100
- langmem: 0: 100, 1: 0, 4: 100, 6: 50, 7: 100, 8: 100
- oracle: 0: 50, 1: 50, 4: 50, 6: 50, 7: 100, 8: 100
- window: 0: 50, 1: 50, 4: 50, 6: 0, 7: 100, 8: 100
- mem0: 0: 50, 1: 0, 4: 50, 6: 50, 7: 100, 8: 100
- cognee: 0: 0, 1: 0, 6: 0, 7: 0, 8: 100
- graphiti: 1: 0

## Stale answers on knowledge updates

- file: 0.0 percent of wrong answers gave a superseded value
- langmem: 0.0 percent of wrong answers gave a superseded value
- oracle: 0.0 percent of wrong answers gave a superseded value
- window: 0.0 percent of wrong answers gave a superseded value
- mem0: 0.0 percent of wrong answers gave a superseded value
- cognee: 0.0 percent of wrong answers gave a superseded value
- graphiti: 0.0 percent of wrong answers gave a superseded value
