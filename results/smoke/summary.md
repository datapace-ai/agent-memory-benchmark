# Agent memory benchmark results

The oracle row is a ceiling, not a competitor. It sees only the evidence sessions, so it measures the best this answerer and judge can do on this set.

| System | LongMemEval rule | Zep rule | Mem0 rule | Share of ceiling | Gap to best baseline | Tokens per answer | Answer p50 | Answer p95 | Errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| oracle | 60.0 | 60.0 | 80.0 | 100.0 | +20.0 | 6387 | 27.3 | 63.9 | 0 |
| window | 40.0 | 40.0 | 60.0 | 66.7 | +0.0 | 24623 | 200.2 | 234.1 | 0 |

Ranking is stable across all three judge rules.

## Accuracy by ability, LongMemEval rule

| System | abstention | extraction | knowledge_update | multi_session | temporal |
| --- | ---: | ---: | ---: | ---: | ---: |
| oracle | 100.0 | 100.0 | 0.0 | 0.0 | 100.0 |
| window | 100.0 | 100.0 | 0.0 | 0.0 | 0.0 |

## Forgetting curve, accuracy by sessions since the evidence

- oracle: 0: 0, 1: 50, 6: 100, 8: 100
- window: 0: 0, 1: 50, 6: 0, 8: 100

## Stale answers on knowledge updates

- oracle: 0.0 percent of wrong answers gave a superseded value
- window: 0.0 percent of wrong answers gave a superseded value
