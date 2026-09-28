# Proactive rate limiting for ingest

## Problem

Alchemy throttles by requests over time (compute units per second), not by
how many connections are open at once. The current
`AdaptiveConcurrencyLimiter` (`src/latentedge/ingest/concurrency.py`) only
caps how many chunk requests may be *in flight simultaneously*, and only
reacts *after* a 429 has already happened, by halving that cap. Two gaps
follow from that shape:

1. **It's reactive, not proactive.** The limiter never tries to stay under a
   safe req/s budget — it just keeps hammering at whatever concurrency level
   it currently allows until the provider says no, then backs off.
2. **Gating happens at the wrong granularity.** `chunked.py` acquires the
   gate once per *chunk*, but a single chunk fetch (`fetch_swaps`) can issue
   several outbound HTTP calls internally (multiple `eth_getLogs` sub-range
   calls, plus a batched `eth_getBlockByNumber` call). Once a worker clears
   the per-chunk gate, those internal calls fire back-to-back with zero
   pacing between them — this is called out directly in a `chunked.py`
   comment as the reason `DEFAULT_CHUNK_SIZE` is pinned to the provider's
   tiny per-call range cap (10 blocks) instead of being allowed to grow.

Repeated real-world runs against Alchemy's free tier have hit sustained
429s under this scheme, risking an IP/API-key ban if it keeps recurring.

## Goal

Replace concurrency-slot gating with a **rate limiter that paces actual
requests/sec**, gates at the level of individual outbound HTTP calls (not
once per chunk), and still reacts to real 429s as the ground-truth signal —
but recovers more carefully than a naive "halve then double back up".

## Non-goals

- No latency-based early-warning heuristics (429s remain the only signal
  driving adjustments, per user decision — added complexity of trend
  detection isn't justified yet).
- No change to `DEFAULT_CHUNK_SIZE` in this change. Per-call gating removes
  the *reason* it's pinned to the provider's range cap, but growing it is a
  separate, later change.
- No migration of old `.concurrency.json` files — a fresh rate limiter just
  starts at its ceiling, or a floor-clamped value from a `.rate.json` file
  if one exists from a prior run of the new code.

## Design

### `RateLimiter` (new module: `src/latentedge/ingest/rate_limiter.py`)

Replaces `AdaptiveConcurrencyLimiter` (`concurrency.py`, deleted). Built
around **strict interval pacing**, not a bursty token bucket: a bucket that
lets a full second's worth of requests fire the instant it's refilled would
reproduce the exact burst-then-throttle pattern this change exists to fix.

```python
class RateLimiter:
    def __init__(
        self,
        ceiling: float,               # max requests/sec
        on_change: Callable[[float], None] | None = None,
        successes_before_increase: int = DEFAULT_SUCCESSES_BEFORE_INCREASE,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        start_rate: float | None = None,
    ) -> None: ...

    @property
    def rate(self) -> float: ...

    def acquire(self, cancel_event: threading.Event | None = None) -> None:
        """Block until the next scheduled slot arrives (now >=
        next_allowed_time) and any post-rate-limit cooldown has elapsed.
        Advances next_allowed_time by 1/rate on every successful acquire.
        Raises Cancelled if cancel_event is set while waiting, polling
        cancel_event every _CANCEL_POLL_SECONDS like the limiter it
        replaces."""

    def release(self, outcome: ReleaseOutcome) -> None:
        """Record an attempt's outcome and adjust rate accordingly.
        Invokes on_change(new_rate) itself when rate actually changes —
        unlike the old limiter, callers don't need to propagate a return
        value, since gating now happens several call-layers away from
        where a caller could consume it."""
```

Adjustment rules (AIMD on `rate`, not a slot count):

- **On `"rate_limited"`:** halve `rate`, floored at `DEFAULT_MIN_RPS` (0.5).
  Trigger the same shared cooldown as today (`cooldown_seconds`) — pauses
  every `acquire()` call, not just the caller that got limited, since other
  threads' calls may already be scheduled to fire imminently when a 429
  lands. Also multiply the *current* `successes_before_increase` threshold
  by `RECOVERY_THRESHOLD_BACKOFF` (1.5), capped at
  `MAX_SUCCESSES_BEFORE_INCREASE`, so a provider that just throttled us
  needs more consecutive proof of health before we trust it again.
- **On `"failed"`:** interrupt the success streak, same as today. No rate
  change — a non-429 failure isn't evidence of throttling.
- **On `"success"`:** after `successes_before_increase` (the current,
  possibly-inflated value) consecutive successes, multiply `rate` by
  `RATE_CLIMB_FACTOR` (1.1) toward `ceiling`, reset the streak, and decay
  the success-threshold back toward its base value by
  `RECOVERY_THRESHOLD_DECAY` (e.g. subtract 2, floored at the base) — so a
  provider we haven't touched in a while doesn't stay permanently harder to
  climb back to just because of one 429 several thousand requests ago.

`RATE_CLIMB_FACTOR`, `RECOVERY_THRESHOLD_BACKOFF`, `RECOVERY_THRESHOLD_DECAY`,
`DEFAULT_MIN_RPS`, and `DEFAULT_MAX_RPS` are starting constants, tunable
later without a design change.

### Gating moves to individual HTTP calls (`rpc_logs.py`)

- `_rpc_call(client, rpc_url, method, params, rate_limiter, cancel_event=None)`
  calls `rate_limiter.acquire(cancel_event)` before `client.post(...)`, and
  `rate_limiter.release(outcome)` right after — `"rate_limited"` for a 429
  (HTTP or JSON-RPC `{"code": 429}`), `"failed"` for any other error,
  `"success"` otherwise.
- `_batch_fetch_blocks(..., rate_limiter, cancel_event=None)` gates the same
  way around its single batched POST. A batch with any per-entry 429 still
  raises `RateLimitError` on the first offending entry (existing behavior)
  and reports `"rate_limited"` once for that call — a throttled batch is one
  throttled HTTP call, not N.
- `fetch_swaps(..., rate_limiter, cancel_event=None)` threads `rate_limiter`
  and `cancel_event` into every sub-call it makes: each `eth_getLogs`
  sub-range call in its loop, and the `_batch_fetch_blocks` call. This is
  what actually paces the previously-unpaced burst of sub-calls per chunk.

### `chunked.py` changes

- `_fetch_chunk_with_retries` stops wrapping `limiter.acquire()`/`release()`
  around the whole `fetch_fn` call — that responsibility now lives inside
  `fetch_swaps`/`rpc_logs`. It calls
  `fetch_fn(pool_address, from_block, to_block, client, rpc_url, rate_limiter, cancel_event)`
  inside its existing retry loop. The retry/backoff logic itself (retry
  `RateLimitError` indefinitely with capped exponential backoff, other
  errors up to `max_retries`) is unchanged — `fetch_swaps` still raises the
  same exception types on failure, just after already pacing/reporting
  every sub-call it made along the way.
- `FetchFn` type signature gains `rate_limiter: RateLimiter` and
  `cancel_event: threading.Event | None` params.
- `ingest_range` constructs one `RateLimiter` shared across the whole run
  (same lifetime/scope as today's `AdaptiveConcurrencyLimiter`), replacing
  `on_concurrency_change` with `on_rate_change` throughout.
- `max_workers` / `DEFAULT_MAX_WORKERS` stay as the `ThreadPoolExecutor`
  size (bounding simultaneous connections/threads) — now fully decoupled
  from the pacing rate, which is a separate ceiling.

### Persistence (`progress.py`)

- `read_concurrency_limit`/`write_concurrency_limit` →
  `read_rate_limit`/`write_rate_limit`, storing a float rate in
  `<out_path>.rate.json` instead of an int limit in
  `<out_path>.concurrency.json`.
- Same floor-not-fixed-point resume rule as today: never resume below half
  the ceiling, so a run that bottomed out early in a prior session still
  gets a fair re-test of current conditions.

### CLI / TUI

- New env var `LATENTEDGE_INGEST_MAX_RPS` (default `DEFAULT_MAX_RPS`) sets
  the rate ceiling, following this repo's env-var-over-flag convention.
  `--max-workers` / `LATENTEDGE_MAX_WORKERS` remains, now purely a thread
  pool size bound.
- `ingest_screen.py`: `on_concurrency_change`/`_handle_concurrency_change`
  → `on_rate_change`/`_handle_rate_change`; the `Concurrency` status row
  becomes `Rate`, displaying e.g. `3.2/8.0 req/s` instead of `4/8`.
- `cli.py`: wiring for the new env var and renamed parameters through the
  ingest command's plain (non-TUI) path, matching the TUI path.

### Cancellation

`RateLimiter.acquire` supports `cancel_event` exactly like the
`AdaptiveConcurrencyLimiter.acquire` it replaces (raises `Cancelled`,
polling every `_CANCEL_POLL_SECONDS`), so it plugs directly into the
ctrl+q cancellation support already shipped in `chunked.py`/`concurrency.py`.

## Testing

- `tests/ingest/test_rate_limiter.py` (replaces `test_concurrency.py`):
  interval pacing correctness, halve-on-429 with floor, gentle climb-on-
  success, temporary success-threshold backoff-and-decay, shared cooldown
  blocking all callers, cancellation.
- `tests/ingest/test_rpc_logs.py`: `rate_limiter.acquire`/`release` called
  once per actual HTTP call (each `eth_getLogs` sub-range call, and the
  block-timestamp batch call), not once per `fetch_swaps` invocation;
  partial-batch 429 still reports exactly one `"rate_limited"` outcome.
- `tests/ingest/test_chunked.py`: updated for the new `fetch_fn` signature;
  existing retry/backoff/cancellation behavior should need minimal changes
  since that logic doesn't move.
- `tests/tui/test_ingest_screen.py`, `tests/test_cli.py`: updated for the
  `Rate` row and `LATENTEDGE_INGEST_MAX_RPS` env var.
