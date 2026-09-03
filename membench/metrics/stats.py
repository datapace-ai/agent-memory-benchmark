"""Small statistics helpers. No third-party numerics, so the harness stays light."""

from __future__ import annotations

import random


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (p / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return float(ordered[low] * (1 - weight) + ordered[high] * weight)


def bootstrap_ci(
    values: list[float], iterations: int = 2000, seed: int = 7, level: float = 95.0
) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(iterations):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    tail = (100.0 - level) / 2.0
    return (percentile(means, tail), percentile(means, 100.0 - tail))
