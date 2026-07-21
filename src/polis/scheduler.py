"""Concurrent per-agent-per-tick scheduler (Layer 1 — Engine; R5, R6).

Phase 2's tick loop called the persona endpoint once per agent per tick, serially.
This module fans those independent calls out concurrently so 20–30 agents are
sustainable, while keeping the scheduling scoped **strictly per-agent-per-tick**
(R5): one unit = one agent's decision for one tick. The scheduler runs units as
independent calls and *never* merges prompts or shares reasoning/cache across them
— bounding concurrency caps the number of in-flight requests, it does not pool
their compute. On the managed OpenRouter backend (P0–4, ADR 0002/0009) R5 is a
prohibition satisfied structurally; the vLLM continuous-batching executor is a
deferred seam (``executor="batch"``) whose benchmarking lands at P5 with the
self-hosted GPU.

The scheduler is deliberately LLM-agnostic: it times, retries, and gathers *any*
callable in order. Token accounting stays with the caller (which owns the result
type); the scheduler owns latency + retry resilience. That keeps it reusable for
the P5 batch executor and for any other bounded fan-out of blocking calls.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TypeVar

K = TypeVar("K")
R = TypeVar("R")

_EXECUTORS = ("concurrent",)  # "batch" (vLLM continuous batching) is a P5 seam


@dataclass(frozen=True)
class SchedulerConfig:
    """Throughput knobs, recorded in the run config (R17).

    ``max_concurrency`` bounds in-flight requests (R5 — a cap on parallel calls,
    not a shared-compute pool). ``price_per_mtok`` is an optional ``(prompt,
    completion)`` USD-per-million-token rate for the cost estimate; left ``None``
    so we never hard-code a guessed price — token counts are always logged as the
    real figure, USD is an estimate only when a rate is supplied.
    """

    max_concurrency: int = 8
    max_retries: int = 2
    backoff_base: float = 0.5  # seconds; exponential, capped at backoff_max
    backoff_max: float = 8.0
    executor: str = "concurrent"
    price_per_mtok: tuple[float, float] | None = None

    def __post_init__(self):
        if self.max_concurrency < 1:
            raise ValueError(f"max_concurrency must be >= 1, got {self.max_concurrency}")
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries}")
        if self.executor not in _EXECUTORS:
            raise ValueError(
                f"executor {self.executor!r} not implemented; "
                f"'batch' (vLLM continuous batching) is a P5 seam. Use one of {_EXECUTORS}."
            )


@dataclass(frozen=True)
class Timing:
    """Per-call resilience/latency record — the scheduler's half of a call's cost.
    Token usage is merged in by the caller, which owns the result type."""

    latency_s: float
    attempts: int
    ok: bool
    error: str | None = None


def run_with_retry(fn: Callable[[], R], config: SchedulerConfig) -> tuple[R | None, Timing]:
    """Call ``fn`` with capped exponential backoff on any exception; return
    ``(result, Timing)``. On exhaustion ``result`` is ``None`` and ``Timing.ok`` is
    ``False`` with the last error — the caller decides whether that aborts the run.

    Shared by the concurrent fan-out and the sequential (single-call) path so both
    time and retry identically; only concurrency differs between R28 schemes.
    """
    start = time.perf_counter()
    last_error: BaseException | None = None
    for attempt in range(1, config.max_retries + 2):
        try:
            result = fn()
            return result, Timing(latency_s=time.perf_counter() - start, attempts=attempt, ok=True)
        except Exception as exc:  # noqa: BLE001 - a throughput layer retries broadly
            last_error = exc
            if attempt <= config.max_retries:
                delay = min(config.backoff_max, config.backoff_base * 2 ** (attempt - 1))
                if delay > 0:
                    time.sleep(delay)
    return (
        None,
        Timing(
            latency_s=time.perf_counter() - start,
            attempts=config.max_retries + 1,
            ok=False,
            error=repr(last_error),
        ),
    )


class Scheduler:
    """Runs per-agent-per-tick units concurrently, bounded and retried (R5)."""

    def __init__(self, config: SchedulerConfig | None = None, *, on_progress=None):
        self.config = config or SchedulerConfig()
        # Optional no-arg callback fired once as each unit completes (progress/UI hook).
        # Called on the gathering thread, after the result is stored; keep it cheap.
        self.on_progress = on_progress

    def map(
        self, units: Sequence[tuple[K, Callable[[], R]]]
    ) -> list[tuple[K, R | None, Timing]]:
        """Run each unit's callable concurrently — bounded by ``max_concurrency``,
        each retried with backoff — and return ``(key, result, timing)`` in the
        input order (not completion order), so downstream logging stays
        deterministic per agent.

        R5: units are independent; the scheduler shares no state, prompt, or cached
        reasoning across them. It caps how many run at once; it does not merge them.
        """
        if not units:
            return []
        cfg = self.config
        results: list[tuple[K, R | None, Timing] | None] = [None] * len(units)
        with ThreadPoolExecutor(max_workers=cfg.max_concurrency) as pool:
            index_of = {
                pool.submit(run_with_retry, fn, cfg): i for i, (_, fn) in enumerate(units)
            }
            for fut in as_completed(index_of):
                i = index_of[fut]
                result, timing = fut.result()
                results[i] = (units[i][0], result, timing)
                if self.on_progress is not None:
                    self.on_progress()
        return results  # type: ignore[return-value]  # every slot filled above


def estimate_cost_usd(
    price_per_mtok: tuple[float, float] | None,
    prompt_tokens: int,
    completion_tokens: int,
) -> float | None:
    """USD estimate from token counts and a ``(prompt, completion)`` per-million
    rate. ``None`` rate → ``None`` (tokens are the logged truth; USD is an estimate
    layered on only when a rate is supplied)."""
    if price_per_mtok is None:
        return None
    prompt_rate, completion_rate = price_per_mtok
    return (prompt_tokens * prompt_rate + completion_tokens * completion_rate) / 1_000_000
