"""Scheduler tests — bounded concurrency, retry/backoff, order, timing (R5).

No network: units are plain callables (sleeps / counters / a flaky closure) so the
concurrency bound, retry policy, and per-call timing are exercised deterministically.
"""
import threading
import time

from polis.scheduler import (
    Scheduler,
    SchedulerConfig,
    estimate_cost_usd,
    run_with_retry,
)

# backoff_base=0 keeps retry tests instant while still exercising the retry loop.
_FAST = SchedulerConfig(max_concurrency=4, max_retries=2, backoff_base=0.0)


def test_map_preserves_input_order_not_completion_order():
    # Later units finish first (descending sleeps); results must still be in order.
    units = [(i, (lambda d=(0.05 * (4 - i)), r=i: (time.sleep(d), r)[1])) for i in range(4)]
    out = Scheduler(_FAST).map(units)
    assert [key for key, _, _ in out] == [0, 1, 2, 3]
    assert [res for _, res, _ in out] == [0, 1, 2, 3]


def test_map_runs_concurrently():
    # 4 units each sleeping 0.1s: concurrent wall time is well under the 0.4s serial sum.
    units = [(i, (lambda: time.sleep(0.1))) for i in range(4)]
    start = time.perf_counter()
    Scheduler(SchedulerConfig(max_concurrency=4, backoff_base=0.0)).map(units)
    assert time.perf_counter() - start < 0.3


def test_concurrency_is_bounded():
    """max_concurrency caps simultaneous in-flight units (R5: a cap, not a pool)."""
    live = 0
    peak = 0
    lock = threading.Lock()

    def work():
        nonlocal live, peak
        with lock:
            live += 1
            peak = max(peak, live)
        time.sleep(0.05)
        with lock:
            live -= 1

    units = [(i, work) for i in range(8)]
    Scheduler(SchedulerConfig(max_concurrency=2, backoff_base=0.0)).map(units)
    assert peak <= 2


def test_retries_transient_failure_then_succeeds():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("429 rate limited")
        return "ok"

    result, timing = run_with_retry(flaky, _FAST)
    assert result == "ok"
    assert timing.ok and timing.attempts == 3


def test_exhausted_retries_report_failure_not_raise():
    def always_fail():
        raise RuntimeError("boom")

    result, timing = run_with_retry(always_fail, _FAST)
    assert result is None
    assert not timing.ok
    assert timing.attempts == 3  # 1 initial + 2 retries
    assert "boom" in timing.error


def test_backoff_grows_between_attempts():
    # base=0.01 → sleeps ~0.01 + 0.02 before the 3rd (successful) attempt.
    cfg = SchedulerConfig(max_retries=2, backoff_base=0.01, backoff_max=8.0)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("retry me")
        return "ok"

    _, timing = run_with_retry(flaky, cfg)
    assert timing.latency_s >= 0.03


def test_map_captures_timing_per_unit():
    out = Scheduler(_FAST).map([("a", lambda: 1), ("b", lambda: 2)])
    assert all(t.ok and t.latency_s >= 0 for _, _, t in out)


def test_empty_units():
    assert Scheduler(_FAST).map([]) == []


def test_cost_estimate_none_without_rate():
    assert estimate_cost_usd(None, 1000, 2000) is None


def test_cost_estimate_from_rate():
    # $0.10 / 1M prompt, $0.30 / 1M completion.
    cost = estimate_cost_usd((0.10, 0.30), 1_000_000, 1_000_000)
    assert cost == 0.40


def test_config_rejects_unimplemented_executor():
    try:
        SchedulerConfig(executor="batch")
    except ValueError as e:
        assert "P5" in str(e)
    else:
        raise AssertionError("expected ValueError for the deferred batch executor")


def test_config_rejects_bad_concurrency():
    for bad in (0, -1):
        try:
            SchedulerConfig(max_concurrency=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for max_concurrency={bad}")
