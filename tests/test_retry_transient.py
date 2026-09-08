import pytest

from membench.systems.shared import retry_transient


def test_returns_after_transient_failures_with_growing_delays():
    calls = {"n": 0}
    slept = []

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError(f"502 overloaded {calls['n']}")
        return "done"

    assert retry_transient(flaky, what="x", attempts=3, base_delay=2.0, sleep=slept.append) == "done"
    assert calls["n"] == 3 and slept == [2.0, 6.0]


def test_raises_the_last_error_after_the_attempts_are_used():
    slept = []

    def always():
        raise ValueError("404 not found")

    with pytest.raises(ValueError, match="404"):
        retry_transient(always, what="x", attempts=2, base_delay=1.0, sleep=slept.append)
    assert slept == [1.0]
