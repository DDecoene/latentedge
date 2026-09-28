# Proactive Rate Limiting for Ingest Implementation Plan

**Goal:** Replace the ingest pipeline's reactive, concurrency-slot throttle with a proactive requests/sec rate limiter that paces every individual outbound RPC call, learns its safe rate from real 429s, and recovers from a throttle-down more carefully than a naive halve-then-double-back-up.

**Architecture:** A new `RateLimiter` (strict interval pacing, AIMD-adjusted on `rate_limited`/`success`/`failed` outcomes) replaces `AdaptiveConcurrencyLimiter`. Gating moves from once-per-chunk (`chunked.py`) to once-per-HTTP-call (`rpc_logs.py`'s `_rpc_call` and `_batch_fetch_blocks`, both called from `fetch_swaps`), fixing the burst-then-throttle pattern a single chunk's several internal sub-calls produce today. `chunked.py`, `progress.py` (persistence), `cli.py`, and `tui/ingest_screen.py` are updated to construct, persist, and display the new rate instead of a concurrency count.

**Tech Stack:** Python, httpx, threading (Condition-based limiter), click, textual (TUI), pytest.

**Spec:** `docs/specs/2026-09-28-proactive-rate-limiting-design.md`

## Global Constraints

- Gating happens per outbound HTTP call (inside `rpc_logs.py`), not per chunk — this is the core fix, not optional.
- `RateLimiter.acquire`/`release` and `Cancelled` keep the exact same external shape as `AdaptiveConcurrencyLimiter`'s (same cancel-polling behavior) so ctrl+q cancellation isn't regressed.
- Starting constants (tunable later without a design change): `DEFAULT_MAX_RPS = 5.0`, `DEFAULT_MIN_RPS = 0.5`, `RATE_CLIMB_FACTOR = 1.1`, `RECOVERY_THRESHOLD_BACKOFF = 1.5`, `RECOVERY_THRESHOLD_DECAY = 2.0`, `MAX_SUCCESSES_BEFORE_INCREASE = 100`, `DEFAULT_SUCCESSES_BEFORE_INCREASE = 20` (kept), `DEFAULT_COOLDOWN_SECONDS = 2.0` (kept).
- The new rate ceiling is configured via `LATENTEDGE_INGEST_MAX_RPS` only — no `--max-rps` CLI flag (this repo's convention for new run options, per project preference).
- `--max-workers`/`LATENTEDGE_MAX_WORKERS` is kept, unchanged in name, now purely a `ThreadPoolExecutor` size bound, decoupled from pacing.
- `concurrency_cooldown_seconds`/`LATENTEDGE_CONCURRENCY_COOLDOWN_SECONDS` keeps its existing name (still describes a post-429 cooldown; renaming a shipped, user-facing flag isn't required by this change and would needlessly break existing scripts/.env files).
- Persistence moves from `<out_path>.concurrency.json` (int `limit`) to `<out_path>.rate.json` (float `rate`) — no migration of old files; a fresh run just starts at its ceiling.
- No latency-based heuristics, no `DEFAULT_CHUNK_SIZE` change — out of scope per the spec's non-goals.

## Review Focus

- A rate-limited halving must never reach zero or a negative rate, however many consecutive 429s occur — floored at `DEFAULT_MIN_RPS`. (Task 1)
- Many worker threads calling `acquire()`/`release()` concurrently must not corrupt `RateLimiter`'s internal float state or double-apply a single adjustment — verified under real concurrent load, not just single-threaded calls. (Task 1)
- A `fetch_fn` that raises `RateLimitError` without ever touching the shared `rate_limiter` (a hand-written test double, or a future third-party plugin) must still retry correctly via `chunked.py`'s existing backoff — the rate limiter's own state simply won't reflect that failure, which is an accepted, explicit trade-off of moving gating inside `fetch_swaps`. (Task 4)
- Cancellation must still surface within `_CANCEL_POLL_SECONDS` even though `acquire()` now happens several call-frames deeper (inside `fetch_swaps`'s sub-call loop) than it used to — this must not regress the just-shipped ctrl+q behavior. (Task 2 at the `rpc_logs` layer; Task 4 end-to-end through `ingest_range`)
- The TUI's rate display must render sensibly at fractional/very small rates (e.g. `0.5/5.0 req/s` after repeated halvings) — the old integer-based `Concurrency` row could never show this, so it's an easy new formatting mistake. (Task 6)

---

## Task 1: `RateLimiter` (replaces `AdaptiveConcurrencyLimiter`)

**Files:**
- Create: `src/latentedge/ingest/rate_limiter.py`
- Delete: `src/latentedge/ingest/concurrency.py`
- Create: `tests/ingest/test_rate_limiter.py`
- Delete: `tests/ingest/test_concurrency.py`

**Interfaces:**
- Produces: `RateLimiter(ceiling: float, successes_before_increase: int = DEFAULT_SUCCESSES_BEFORE_INCREASE, cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS, start_rate: float | None = None, on_change: Callable[[float], None] | None = None)`, with `.rate` (property, float), `.acquire(cancel_event: threading.Event | None = None) -> None`, `.release(outcome: ReleaseOutcome) -> None`. `ReleaseOutcome = Literal["success", "failed", "rate_limited"]`. `Cancelled` exception (same semantics as before).

- [ ] **Step 1: Write the failing tests**

Create `tests/ingest/test_rate_limiter.py`:

```python
import threading
import time

import pytest

from latentedge.ingest.rate_limiter import RateLimiter, Cancelled, DEFAULT_MIN_RPS


def test_limiter_starts_at_the_ceiling():
    limiter = RateLimiter(ceiling=4.0)
    assert limiter.rate == 4.0


def test_limiter_halves_on_rate_limited_release():
    changes: list[float] = []
    limiter = RateLimiter(ceiling=4.0, cooldown_seconds=0, on_change=changes.append)
    limiter.release("rate_limited")
    assert limiter.rate == 2.0
    assert changes == [2.0]


def test_limiter_never_drops_below_the_floor():
    limiter = RateLimiter(ceiling=4.0, cooldown_seconds=0)
    for _ in range(10):
        limiter.release("rate_limited")
    assert limiter.rate == DEFAULT_MIN_RPS


def test_limiter_does_not_invoke_on_change_when_already_at_the_floor():
    changes: list[float] = []
    limiter = RateLimiter(ceiling=DEFAULT_MIN_RPS, cooldown_seconds=0, on_change=changes.append)
    limiter.release("rate_limited")
    assert limiter.rate == DEFAULT_MIN_RPS
    assert changes == []


def test_limiter_climbs_gently_after_enough_consecutive_successes():
    changes: list[float] = []
    limiter = RateLimiter(ceiling=4.0, successes_before_increase=3, cooldown_seconds=0, on_change=changes.append)
    limiter.release("rate_limited")  # 4.0 -> 2.0, threshold 3 -> 4.5
    changes.clear()
    for _ in range(4):
        limiter.release("success")
    assert changes == []  # 4 successes still short of the backed-off 4.5 threshold
    limiter.release("success")  # 5th consecutive success clears it
    assert changes == [pytest.approx(2.2)]  # 2.0 * RATE_CLIMB_FACTOR (1.1)
    assert limiter.rate == pytest.approx(2.2)


def test_limiter_never_exceeds_its_ceiling():
    limiter = RateLimiter(ceiling=2.0, successes_before_increase=1, cooldown_seconds=0)
    for _ in range(20):
        limiter.release("success")
    assert limiter.rate == pytest.approx(2.0)


def test_limiter_plain_failure_resets_the_success_streak_without_shrinking_rate():
    changes: list[float] = []
    limiter = RateLimiter(ceiling=4.0, successes_before_increase=2, cooldown_seconds=0, on_change=changes.append)
    limiter.release("rate_limited")  # 4.0 -> 2.0
    changes.clear()
    limiter.release("success")  # 1 consecutive success
    limiter.release("failed")  # streak reset, no shrink
    assert changes == []
    for _ in range(3):  # threshold is now 3 (2 * 1.5), needs 3 consecutive
        limiter.release("success")
    assert changes == [pytest.approx(2.2)]


def test_limiter_gradually_decays_the_backed_off_success_threshold_across_climbs():
    # Regression test for "smarter recovery": the success-streak
    # requirement inflated by a rate limit must come back down in steps
    # as the limiter proves it's healthy again, not snap straight back to
    # its pre-throttle value after a single climb.
    changes: list[float] = []
    limiter = RateLimiter(ceiling=1000.0, successes_before_increase=10, cooldown_seconds=0, on_change=changes.append)
    limiter.release("rate_limited")  # rate 1000.0 -> 500.0, threshold 10 -> 15.0
    changes.clear()

    for _ in range(14):
        limiter.release("success")
    assert changes == []  # 14 < 15
    limiter.release("success")  # 15th clears it
    assert changes == [pytest.approx(550.0)]  # 500.0 * RATE_CLIMB_FACTOR (1.1)
    changes.clear()  # threshold decays 15.0 -> 13.0 (still above base 10)

    for _ in range(12):
        limiter.release("success")
    assert changes == []  # 12 < 13
    limiter.release("success")  # 13th clears it
    assert changes == [pytest.approx(605.0)]  # 550.0 * 1.1


def test_limiter_starts_from_a_persisted_rate_instead_of_the_ceiling():
    limiter = RateLimiter(ceiling=8.0, start_rate=2.0)
    assert limiter.rate == 2.0


def test_limiter_clamps_a_persisted_rate_that_exceeds_the_current_ceiling():
    limiter = RateLimiter(ceiling=2.0, start_rate=8.0)
    assert limiter.rate == 2.0


def test_limiter_paces_successive_acquires_to_the_configured_interval():
    limiter = RateLimiter(ceiling=20.0, cooldown_seconds=0)  # interval = 0.05s
    start = time.monotonic()
    for _ in range(4):
        limiter.acquire()
    elapsed = time.monotonic() - start
    assert elapsed >= 0.14  # 3 gaps of ~0.05s, with a little scheduling slack


def test_limiter_does_not_delay_acquire_when_calls_are_already_spaced_out():
    limiter = RateLimiter(ceiling=1.0, cooldown_seconds=0)  # interval = 1s
    limiter.acquire()
    time.sleep(1.05)
    start = time.monotonic()
    limiter.acquire()
    assert time.monotonic() - start < 0.1


def test_limiter_paces_concurrent_acquires_from_multiple_threads():
    limiter = RateLimiter(ceiling=50.0, cooldown_seconds=0)  # interval = 0.02s
    timestamps: list[float] = []
    lock = threading.Lock()

    def worker() -> None:
        limiter.acquire()
        with lock:
            timestamps.append(time.monotonic())

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=2.0)

    timestamps.sort()
    gaps = [b - a for a, b in zip(timestamps, timestamps[1:])]
    assert len(gaps) == 9
    assert all(gap >= 0.018 for gap in gaps)  # allow a little scheduling slack below 0.02s


def test_limiter_pauses_every_acquire_for_a_cooldown_after_a_rate_limit():
    limiter = RateLimiter(ceiling=1000.0, cooldown_seconds=0.15)
    limiter.acquire()
    limiter.release("rate_limited")

    resumed = threading.Event()

    def acquire_after_cooldown() -> None:
        limiter.acquire()
        resumed.set()

    thread = threading.Thread(target=acquire_after_cooldown)
    thread.start()
    try:
        time.sleep(0.05)
        assert not resumed.is_set()
        thread.join(timeout=1.0)
        assert resumed.is_set()
    finally:
        thread.join(timeout=1.0)


def test_limiter_does_not_reset_the_cooldown_on_a_plain_failure():
    limiter = RateLimiter(ceiling=1000.0, cooldown_seconds=0)
    limiter.acquire()
    limiter.release("failed")
    start = time.monotonic()
    limiter.acquire()
    assert time.monotonic() - start < 0.1


def test_limiter_acquire_raises_cancelled_immediately_if_already_set():
    limiter = RateLimiter(ceiling=1000.0, cooldown_seconds=0)
    cancel_event = threading.Event()
    cancel_event.set()
    with pytest.raises(Cancelled):
        limiter.acquire(cancel_event)


def test_limiter_acquire_wakes_promptly_on_cancel_event_instead_of_waiting_for_the_interval():
    limiter = RateLimiter(ceiling=0.1, cooldown_seconds=0)  # ceiling floors to DEFAULT_MIN_RPS (0.5) -> interval = 2s
    limiter.acquire()  # schedules the next slot ~2s out

    cancel_event = threading.Event()
    raised = threading.Event()

    def acquire_blocked() -> None:
        try:
            limiter.acquire(cancel_event)
        except Cancelled:
            raised.set()

    thread = threading.Thread(target=acquire_blocked)
    thread.start()
    try:
        time.sleep(0.05)
        assert not raised.is_set()  # genuinely waiting on the interval, not a fluke pass
        cancel_event.set()
        thread.join(timeout=1.0)
        assert raised.is_set()
    finally:
        thread.join(timeout=1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/ingest/test_rate_limiter.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'latentedge.ingest.rate_limiter'`

- [ ] **Step 3: Implement `RateLimiter`**

Create `src/latentedge/ingest/rate_limiter.py`:

```python
"""A proactive, learning rate limiter for ingest's outbound RPC calls.

Alchemy (and similar providers) throttle by requests over time, not by how
many connections are open at once — a fixed concurrency cap can still
burst well past a safe req/s the instant several permitted callers each
fire in the same instant. This paces individual acquire() calls to at most
one every 1/rate seconds (a strict schedule, not a bursty token bucket
that would let a full second's worth of calls fire the moment it refills),
and adjusts rate itself from real 429s: halved immediately on a rate
limit, climbed back gently on sustained success. See
docs/specs/2026-09-28-proactive-rate-limiting-design.md.
"""

import threading
import time
from collections.abc import Callable
from typing import Literal

DEFAULT_SUCCESSES_BEFORE_INCREASE = 20
DEFAULT_COOLDOWN_SECONDS = 2.0
DEFAULT_MIN_RPS = 0.5
# Gentle, not a snap back to the pre-throttle rate — a provider that just
# rate-limited us shouldn't be re-approached at full speed the moment a
# few requests succeed.
RATE_CLIMB_FACTOR = 1.1
# Each rate limit makes the next climb require more proof of health;
# RECOVERY_THRESHOLD_DECAY unwinds that extra caution gradually as climbs
# actually happen, rather than either resetting it instantly or leaving
# it permanently inflated by one incident hours ago.
RECOVERY_THRESHOLD_BACKOFF = 1.5
RECOVERY_THRESHOLD_DECAY = 2.0
MAX_SUCCESSES_BEFORE_INCREASE = 100

ReleaseOutcome = Literal["success", "failed", "rate_limited"]

# How often acquire()'s wait loop wakes on its own to recheck cancel_event,
# even with nothing to notify it — the only way a blocked worker notices a
# user-requested stop without waiting out a full cooldown or pacing
# interval first.
_CANCEL_POLL_SECONDS = 0.5


class Cancelled(Exception):
    """A cancel_event was set while a worker was blocked in acquire()."""


class RateLimiter:
    def __init__(
        self,
        ceiling: float,
        successes_before_increase: int = DEFAULT_SUCCESSES_BEFORE_INCREASE,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        start_rate: float | None = None,
        on_change: Callable[[float], None] | None = None,
    ) -> None:
        self._ceiling = max(DEFAULT_MIN_RPS, ceiling)
        self._base_successes_before_increase = successes_before_increase
        self._successes_before_increase: float = successes_before_increase
        self._cooldown_seconds = cooldown_seconds
        self._on_change = on_change
        self._rate = max(DEFAULT_MIN_RPS, min(start_rate, self._ceiling)) if start_rate else self._ceiling
        self._consecutive_successes = 0
        self._cooldown_until = 0.0
        self._next_allowed_time = time.monotonic()
        self._cond = threading.Condition()

    @property
    def rate(self) -> float:
        with self._cond:
            return self._rate

    def acquire(self, cancel_event: threading.Event | None = None) -> None:
        """Block until this call's scheduled slot arrives and any
        post-rate-limit cooldown has elapsed, then claim the next slot.

        Uses a strict schedule (one slot every 1/rate seconds) rather
        than a bursty token bucket — a bucket that lets a full second's
        worth of calls fire the moment it refills would reproduce the
        exact burst-then-throttle pattern this limiter exists to avoid.
        Slot-claiming happens while holding _cond's lock, so concurrent
        callers can never claim the same slot.
        """
        with self._cond:
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    raise Cancelled()
                now = time.monotonic()
                remaining_cooldown = self._cooldown_until - now
                if remaining_cooldown > 0:
                    self._cond.wait(timeout=min(remaining_cooldown, _CANCEL_POLL_SECONDS))
                    continue
                wait_seconds = self._next_allowed_time - now
                if wait_seconds > 0:
                    self._cond.wait(timeout=min(wait_seconds, _CANCEL_POLL_SECONDS))
                    continue
                self._next_allowed_time = max(now, self._next_allowed_time) + (1.0 / self._rate)
                return

    def release(self, outcome: ReleaseOutcome) -> None:
        """Record an attempt's outcome and adjust rate accordingly,
        invoking on_change(new_rate) itself when rate actually changes —
        callers don't need to propagate a return value, since gating now
        happens several call-layers away (inside rpc_logs.py) from
        anywhere that could consume one.
        """
        new_rate: float | None = None
        with self._cond:
            if outcome == "rate_limited":
                candidate = max(DEFAULT_MIN_RPS, self._rate / 2)
                if candidate != self._rate:
                    self._rate = candidate
                    new_rate = candidate
                self._consecutive_successes = 0
                self._successes_before_increase = min(
                    self._successes_before_increase * RECOVERY_THRESHOLD_BACKOFF,
                    MAX_SUCCESSES_BEFORE_INCREASE,
                )
                self._cooldown_until = time.monotonic() + self._cooldown_seconds
            elif outcome == "failed":
                self._consecutive_successes = 0
            else:
                self._consecutive_successes += 1
                if self._consecutive_successes >= self._successes_before_increase and self._rate < self._ceiling:
                    self._rate = min(self._rate * RATE_CLIMB_FACTOR, self._ceiling)
                    new_rate = self._rate
                    self._consecutive_successes = 0
                    self._successes_before_increase = max(
                        self._successes_before_increase - RECOVERY_THRESHOLD_DECAY,
                        self._base_successes_before_increase,
                    )
            self._cond.notify_all()
        if new_rate is not None and self._on_change is not None:
            self._on_change(new_rate)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/ingest/test_rate_limiter.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Delete the old module and its tests**

```bash
git rm src/latentedge/ingest/concurrency.py tests/ingest/test_concurrency.py
```

- [ ] **Step 6: Confirm nothing else references the deleted module**

Run: `grep -rn "ingest.concurrency\|AdaptiveConcurrencyLimiter" src/ tests/`
Expected: no output (Tasks 2–6 below still reference `AdaptiveConcurrencyLimiter`/`concurrency` in a few places — if this greps clean before those tasks are done, re-run it again after Task 4 to confirm)

- [ ] **Step 7: Commit**

```bash
git add src/latentedge/ingest/rate_limiter.py tests/ingest/test_rate_limiter.py
git commit -m "Replace concurrency-slot limiter with a proactive req/s rate limiter"
```

---

## Task 2: Gate every outbound RPC call in `rpc_logs.py`

**Files:**
- Modify: `src/latentedge/ingest/rpc_logs.py`
- Test: `tests/ingest/test_rpc_logs.py`

**Interfaces:**
- Consumes: `RateLimiter.acquire(cancel_event)`, `RateLimiter.release(outcome)`, `Cancelled` from Task 1's `latentedge.ingest.rate_limiter`.
- Produces: `_rpc_call(client, rpc_url, method, params, rate_limiter=None, cancel_event=None)`, `_batch_fetch_blocks(block_numbers, client, rpc_url, rate_limiter=None, cancel_event=None)`, `fetch_swaps(pool_address, from_block, to_block, client, rpc_url, eth_getlogs_range_cap=ETH_GETLOGS_RANGE_CAP, rate_limiter=None, cancel_event=None)`. All default to `None` (no gating) so every existing caller that doesn't pass one keeps working unchanged.

- [ ] **Step 1: Write the failing tests**

Add to `tests/ingest/test_rpc_logs.py` (add `import threading` near the top alongside the existing `httpx`/`pytest` imports):

```python
class _RecordingRateLimiter:
    def __init__(self) -> None:
        self.acquired = 0
        self.released: list[str] = []

    def acquire(self, cancel_event=None) -> None:
        self.acquired += 1

    def release(self, outcome: str) -> None:
        self.released.append(outcome)


def test_rpc_call_acquires_and_releases_success_through_the_rate_limiter():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": "0x1"})

    rate_limiter = _RecordingRateLimiter()
    with _mock_client(handler) as client:
        _rpc_call(client, RPC_URL, "eth_blockNumber", [], rate_limiter=rate_limiter)

    assert rate_limiter.acquired == 1
    assert rate_limiter.released == ["success"]


def test_rpc_call_releases_rate_limited_on_http_429():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="rate limited")

    rate_limiter = _RecordingRateLimiter()
    with _mock_client(handler) as client:
        with pytest.raises(RateLimitError):
            _rpc_call(client, RPC_URL, "eth_blockNumber", [], rate_limiter=rate_limiter)

    assert rate_limiter.released == ["rate_limited"]


def test_rpc_call_propagates_cancelled_from_the_rate_limiter():
    from latentedge.ingest.rate_limiter import Cancelled, RateLimiter

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("must not fire an HTTP request once cancelled")

    rate_limiter = RateLimiter(ceiling=1.0)
    cancel_event = threading.Event()
    cancel_event.set()

    with _mock_client(handler) as client:
        with pytest.raises(Cancelled):
            _rpc_call(client, RPC_URL, "eth_blockNumber", [], rate_limiter=rate_limiter, cancel_event=cancel_event)


def test_batch_fetch_blocks_reports_exactly_one_rate_limited_outcome_for_an_embedded_429():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[{"jsonrpc": "2.0", "id": 1, "error": {"code": 429, "message": "compute units exceeded"}}],
        )

    rate_limiter = _RecordingRateLimiter()
    with _mock_client(handler) as client:
        with pytest.raises(RateLimitError):
            _batch_fetch_blocks([1], client, RPC_URL, rate_limiter=rate_limiter)

    assert rate_limiter.released == ["rate_limited"]


def test_fetch_swaps_gates_every_sub_call_through_the_rate_limiter():
    # Reuses the 3-getLogs-calls + 1-batched-block-call shape proven by
    # test_fetch_swaps_sub_chunks_eth_getlogs_but_batches_blocks_in_one_call.
    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        body = _json.loads(request.content)
        if isinstance(body, list):
            return httpx.Response(
                200,
                json=[
                    {"jsonrpc": "2.0", "id": entry["id"], "result": {"timestamp": hex(entry["id"] * 12), "baseFeePerGas": "0x1"}}
                    for entry in body
                ],
            )
        from_block = int(body["params"][0]["fromBlock"], 16)
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "result": [
                    {
                        "address": "0xpool",
                        "blockNumber": hex(from_block),
                        "transactionHash": "0x" + "1" * 64,
                        "logIndex": "0x0",
                        "data": "0x" + "0" * 64 * 5,
                    }
                ],
            },
        )

    rate_limiter = _RecordingRateLimiter()
    with _mock_client(handler) as client:
        fetch_swaps("0xpool", from_block=0, to_block=29, client=client, rpc_url=RPC_URL, rate_limiter=rate_limiter)

    # 3 eth_getLogs sub-range calls + 1 batched eth_getBlockByNumber call.
    assert rate_limiter.acquired == 4
    assert rate_limiter.released == ["success", "success", "success", "success"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/ingest/test_rpc_logs.py -k "rate_limiter or gates_every" -v`
Expected: FAIL with `TypeError: _rpc_call() got an unexpected keyword argument 'rate_limiter'` (and similar for the other two functions)

- [ ] **Step 3: Thread `rate_limiter`/`cancel_event` through every outbound call**

In `src/latentedge/ingest/rpc_logs.py`, update `_rpc_call`:

```python
def _rpc_call(
    client: httpx.Client,
    rpc_url: str,
    method: str,
    params: list[Any],
    rate_limiter: "RateLimiter | None" = None,
    cancel_event: threading.Event | None = None,
) -> Any:
    if rate_limiter is not None:
        rate_limiter.acquire(cancel_event)
    response = client.post(rpc_url, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
    if response.status_code == 429:
        if rate_limiter is not None:
            rate_limiter.release("rate_limited")
        raise RateLimitError(f"RPC rate limited (HTTP 429): {response.text}")
    if response.status_code != 200:
        if rate_limiter is not None:
            rate_limiter.release("failed")
        raise RpcLogsError(f"RPC returned HTTP {response.status_code}: {response.text}")
    payload = response.json()
    if "error" in payload:
        if rate_limiter is not None:
            rate_limiter.release("failed")
        raise RpcLogsError(f"RPC error: {payload['error']}")
    if rate_limiter is not None:
        rate_limiter.release("success")
    return payload["result"]
```

Add `import threading` and `from typing import TYPE_CHECKING` at the top of the file, and (to avoid a real import cycle — `rate_limiter.py` never imports `rpc_logs.py`, so this is just for a clean type-only reference):

```python
if TYPE_CHECKING:
    from latentedge.ingest.rate_limiter import RateLimiter
```

Update `get_latest_block` to accept and forward the same two optional params (unused by any current caller, but consistent with every other RPC entry point):

```python
def get_latest_block(
    client: httpx.Client,
    rpc_url: str,
    rate_limiter: "RateLimiter | None" = None,
    cancel_event: threading.Event | None = None,
) -> int:
    result: str = _rpc_call(client, rpc_url, "eth_blockNumber", [], rate_limiter=rate_limiter, cancel_event=cancel_event)
    return int(result, 16)
```

Update `_batch_fetch_blocks`:

```python
def _batch_fetch_blocks(
    block_numbers: list[int],
    client: httpx.Client,
    rpc_url: str,
    rate_limiter: "RateLimiter | None" = None,
    cancel_event: threading.Event | None = None,
) -> dict[int, tuple[int, int]]:
    result: dict[int, tuple[int, int]] = {}
    unique_blocks = list(dict.fromkeys(block_numbers))

    for i in range(0, len(unique_blocks), BLOCK_BATCH_SIZE):
        batch = unique_blocks[i : i + BLOCK_BATCH_SIZE]
        payload = [
            {"jsonrpc": "2.0", "id": block_number, "method": "eth_getBlockByNumber", "params": [hex(block_number), False]}
            for block_number in batch
        ]
        if rate_limiter is not None:
            rate_limiter.acquire(cancel_event)
        response = client.post(rpc_url, json=payload)
        if response.status_code == 429:
            if rate_limiter is not None:
                rate_limiter.release("rate_limited")
            raise RateLimitError(f"RPC rate limited (HTTP 429): {response.text}")
        if response.status_code != 200:
            if rate_limiter is not None:
                rate_limiter.release("failed")
            raise RpcLogsError(f"RPC returned HTTP {response.status_code}: {response.text}")

        responses = response.json()
        by_id = {entry["id"]: entry for entry in responses}

        for block_number in batch:
            entry = by_id.get(block_number)
            if entry is None:
                if rate_limiter is not None:
                    rate_limiter.release("failed")
                raise RpcLogsError(f"batch response missing block {block_number}")
            if "error" in entry:
                if entry["error"].get("code") == 429:
                    if rate_limiter is not None:
                        rate_limiter.release("rate_limited")
                    raise RateLimitError(f"RPC rate limited for block {block_number}: {entry['error']}")
                if rate_limiter is not None:
                    rate_limiter.release("failed")
                raise RpcLogsError(f"RPC error for block {block_number}: {entry['error']}")

            block = entry["result"]
            timestamp = int(block["timestamp"], 16)
            base_fee_wei = int(block["baseFeePerGas"], 16) if "baseFeePerGas" in block else 0
            result[block_number] = (timestamp, base_fee_wei)

        if rate_limiter is not None:
            rate_limiter.release("success")

    return result
```

Update `fetch_swaps` to thread both params into every sub-call:

```python
def fetch_swaps(
    pool_address: str,
    from_block: int,
    to_block: int,
    client: httpx.Client,
    rpc_url: str,
    eth_getlogs_range_cap: int = ETH_GETLOGS_RANGE_CAP,
    rate_limiter: "RateLimiter | None" = None,
    cancel_event: threading.Event | None = None,
) -> list[SwapRecord]:
    logs: list[dict[str, Any]] = []
    for sub_from in range(from_block, to_block + 1, eth_getlogs_range_cap):
        sub_to = min(sub_from + eth_getlogs_range_cap - 1, to_block)
        logs.extend(
            _rpc_call(
                client,
                rpc_url,
                "eth_getLogs",
                [{"address": pool_address, "topics": [SWAP_TOPIC], "fromBlock": hex(sub_from), "toBlock": hex(sub_to)}],
                rate_limiter=rate_limiter,
                cancel_event=cancel_event,
            )
        )

    unique_block_numbers = [int(log["blockNumber"], 16) for log in logs]
    block_cache = _batch_fetch_blocks(unique_block_numbers, client, rpc_url, rate_limiter=rate_limiter, cancel_event=cancel_event)

    records: list[SwapRecord] = []
    for log in logs:
        block_number = int(log["blockNumber"], 16)
        timestamp, base_fee_wei = block_cache[block_number]
        amount0, amount1, sqrt_price_x96, liquidity, tick = _decode_swap_data(log["data"])
        records.append(
            SwapRecord(
                block_number=block_number,
                timestamp=timestamp,
                tx_hash=log["transactionHash"],
                log_index=int(log["logIndex"], 16),
                sqrt_price_x96=sqrt_price_x96,
                tick=tick,
                liquidity=liquidity,
                amount0=amount0,
                amount1=amount1,
                base_fee_wei=base_fee_wei,
            )
        )
    return records
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/ingest/test_rpc_logs.py -v`
Expected: PASS (all tests, including the pre-existing real-network ones — they call every function with no `rate_limiter`, which defaults to `None` and skips gating exactly as before)

- [ ] **Step 5: Commit**

```bash
git add src/latentedge/ingest/rpc_logs.py tests/ingest/test_rpc_logs.py
git commit -m "Gate every outbound RPC call through the rate limiter, not once per chunk"
```

---

## Task 3: Persist the rate instead of a concurrency limit

**Files:**
- Modify: `src/latentedge/ingest/progress.py`
- Test: `tests/ingest/test_progress.py`

**Interfaces:**
- Produces: `read_rate_limit(out_path: Path) -> float | None`, `write_rate_limit(out_path: Path, rate: float) -> None` (replacing `read_concurrency_limit`/`write_concurrency_limit`, deleted).

- [ ] **Step 1: Write the failing tests**

In `tests/ingest/test_progress.py`, replace the import of `read_concurrency_limit`/`write_concurrency_limit` with `read_rate_limit`/`write_rate_limit`, and replace these two tests:

```python
def test_read_rate_limit_returns_none_when_no_file_exists(tmp_path: Path):
    assert read_rate_limit(tmp_path / "swaps.parquet") is None


def test_write_then_read_rate_limit_round_trips(tmp_path: Path):
    out_path = tmp_path / "swaps.parquet"
    write_rate_limit(out_path, 3.5)
    assert read_rate_limit(out_path) == 3.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/ingest/test_progress.py -k rate_limit -v`
Expected: FAIL with `ImportError: cannot import name 'read_rate_limit'`

- [ ] **Step 3: Rename the persistence functions**

In `src/latentedge/ingest/progress.py`, replace `_concurrency_path`/`read_concurrency_limit`/`write_concurrency_limit`:

```python
def _rate_path(out_path: Path) -> Path:
    return Path(str(out_path) + ".rate.json")


def read_rate_limit(out_path: Path) -> float | None:
    """The req/s rate a prior ingest_range run settled on for this output
    file, if any — lets a resumed run start from a rate already known to
    avoid rate limiting instead of the full ceiling (which just re-earns
    the same throttle-down again).
    """
    path = _rate_path(out_path)
    if not path.exists():
        return None
    return float(json.loads(path.read_text())["rate"])


def write_rate_limit(out_path: Path, rate: float) -> None:
    _rate_path(out_path).write_text(json.dumps({"rate": rate}))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/ingest/test_progress.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/latentedge/ingest/progress.py tests/ingest/test_progress.py
git commit -m "Persist ingest's learned rate instead of a concurrency limit"
```

---

## Task 4: Wire `RateLimiter` into `chunked.py`

**Files:**
- Modify: `src/latentedge/ingest/chunked.py`
- Modify: `tests/ingest/test_chunked.py`

**Interfaces:**
- Consumes: `RateLimiter`, `Cancelled` (Task 1); `read_rate_limit`, `write_rate_limit` (Task 3); `fetch_swaps(..., rate_limiter=..., cancel_event=...)` (Task 2).
- Produces: `ingest_range(..., max_rps: float = DEFAULT_MAX_RPS, on_rate_change: Callable[[float], None] | None = None, ...)` (replacing `on_concurrency_change`; `max_workers` stays, now purely the thread pool size). `FetchFn = Callable[..., list[SwapRecord]]` — every `FetchFn` implementation must accept `rate_limiter` and `cancel_event` as keyword arguments (ignoring them is fine for fakes that don't make real HTTP calls).

- [ ] **Step 1: Update the mechanical test fixtures first**

Every fake `fetch_fn` in `tests/ingest/test_chunked.py` needs to tolerate the new keyword arguments `chunked.py` will now pass. Run:

```bash
sed -i '' 's/client: httpx\.Client, rpc_url: str) -> list\[SwapRecord\]:/client: httpx.Client, rpc_url: str, **kwargs) -> list[SwapRecord]:/' tests/ingest/test_chunked.py
```

Then update the module-level import at the top of the file:

```python
from latentedge.ingest.progress import read_progress, read_rate_limit, write_progress, write_rate_limit
```

- [ ] **Step 2: Delete the test for a mechanism that no longer exists**

Remove `test_fetch_chunk_with_retries_reports_waiting_while_blocked_on_the_concurrency_gate` entirely from `tests/ingest/test_chunked.py` — gating no longer happens once-per-chunk in `_fetch_chunk_with_retries`, so there's no discrete "blocked on the gate" moment for it to observe.

- [ ] **Step 3: Rewrite the worker-status test to drop the "waiting" phase**

Replace `test_ingest_range_reports_worker_status_during_retries` with:

```python
def test_ingest_range_reports_worker_status_during_retries(tmp_path: Path):
    attempts = {"count": 0}

    def flaky_fetch(pool_address, from_block, to_block, client, rpc_url, **kwargs):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RpcLogsError("transient failure")
        return [_record(from_block, 0)]

    statuses: list[tuple[int, int, int, str]] = []

    def on_worker_status(slot, chunk_start, chunk_end, status) -> None:
        statuses.append((slot, chunk_start, chunk_end, status))

    out_path = tmp_path / "swaps.parquet"
    with httpx.Client() as client:
        ingest_range(
            pool_address="0xpool", from_block=0, to_block=99, out_path=out_path,
            client=client, rpc_url="http://fake", chunk_size=100, max_workers=1,
            fetch_fn=flaky_fetch, max_retries=5, retry_backoff_seconds=0.001,
            on_worker_status=on_worker_status,
        )

    retry_statuses = [s[3] for s in statuses if s[3].startswith("retry")]
    assert retry_statuses == ["retry 1/5, waiting 0.0s", "retry 2/5, waiting 0.0s"]
    # No separate "waiting" phase anymore — pacing now happens inside
    # fetch_fn's own per-call gating (see rpc_logs.py), not as a discrete
    # per-chunk gate chunked.py can observe and report on its own.
    assert statuses[0] == (0, 0, 99, "fetching")
    assert statuses[-1] == (0, 0, 99, "idle")
```

- [ ] **Step 4: Rewrite the three rate-persistence/adjustment tests**

Replace `test_ingest_range_throttles_down_worker_concurrency_after_a_rate_limit`:

```python
def test_ingest_range_throttles_down_the_pacing_rate_after_a_rate_limit(tmp_path: Path):
    attempts = {"count": 0}
    lock = threading.Lock()

    def one_rate_limit_then_fine(pool_address, from_block, to_block, client, rpc_url, rate_limiter=None, **kwargs):
        with lock:
            attempts["count"] += 1
            first = attempts["count"] == 1
        if first:
            if rate_limiter is not None:
                rate_limiter.release("rate_limited")
            raise RateLimitError("simulated rate limit")
        if rate_limiter is not None:
            rate_limiter.release("success")
        return [_record(from_block, 0)]

    rate_changes: list[float] = []
    changes_lock = threading.Lock()

    def on_rate_change(new_rate: float) -> None:
        with changes_lock:
            rate_changes.append(new_rate)

    out_path = tmp_path / "swaps.parquet"
    with httpx.Client() as client:
        total = ingest_range(
            pool_address="0xpool", from_block=0, to_block=99, out_path=out_path,
            client=client, rpc_url="http://fake", chunk_size=10, max_workers=4,
            max_rps=4.0, fetch_fn=one_rate_limit_then_fine, max_retries=3, retry_backoff_seconds=0.001,
            concurrency_cooldown_seconds=0,
            on_rate_change=on_rate_change,
        )

    assert total == 10
    assert 2.0 in rate_changes  # halved from the ceiling of 4.0 after the one rate limit
```

Replace `test_ingest_range_persists_the_settled_concurrency_limit_for_the_next_run`:

```python
def test_ingest_range_persists_the_settled_rate_for_the_next_run(tmp_path: Path):
    attempts = {"count": 0}
    lock = threading.Lock()

    def one_rate_limit_then_fine(pool_address, from_block, to_block, client, rpc_url, rate_limiter=None, **kwargs):
        with lock:
            attempts["count"] += 1
            first = attempts["count"] == 1
        if first:
            if rate_limiter is not None:
                rate_limiter.release("rate_limited")
            raise RateLimitError("simulated rate limit")
        if rate_limiter is not None:
            rate_limiter.release("success")
        return [_record(from_block, 0)]

    out_path = tmp_path / "swaps.parquet"
    with httpx.Client() as client:
        ingest_range(
            pool_address="0xpool", from_block=0, to_block=99, out_path=out_path,
            client=client, rpc_url="http://fake", chunk_size=10, max_workers=4,
            max_rps=4.0, fetch_fn=one_rate_limit_then_fine, max_retries=3, retry_backoff_seconds=0.001,
            concurrency_cooldown_seconds=0,
        )

    # Halved from a ceiling of 4.0 down to 2.0 and never climbed back
    # (successes_before_increase defaults to 20, far more than the
    # handful of chunks here) — a resumed run must start from that 2.0,
    # not silently reset to the ceiling and re-earn the same throttle.
    assert read_rate_limit(out_path) == 2.0

    seen_rates: list[float] = []

    def record_rate_seen(pool_address, from_block, to_block, client, rpc_url, rate_limiter=None, **kwargs):
        if rate_limiter is not None:
            rate_limiter.release("success")
        return [_record(from_block, 0)]

    def on_rate_change(new_rate: float) -> None:
        seen_rates.append(new_rate)

    with httpx.Client() as client:
        ingest_range(
            pool_address="0xpool", from_block=100, to_block=109, out_path=out_path,
            client=client, rpc_url="http://fake", chunk_size=10, max_workers=4,
            max_rps=4.0, fetch_fn=record_rate_seen, concurrency_cooldown_seconds=0,
            on_rate_change=on_rate_change,
        )

    # Nothing rate-limited this time, so the rate should only ever have
    # been read starting from 2.0 (the persisted value), never jumped
    # straight back to the ceiling of 4.0 from a single chunk's success.
    assert 4.0 not in seen_rates
    assert read_rate_limit(out_path) == 2.0
```

Replace `test_ingest_range_never_resumes_below_half_the_ceiling_even_if_a_prior_run_bottomed_out`:

```python
def test_ingest_range_never_resumes_below_half_the_ceiling_even_if_a_prior_run_bottomed_out(tmp_path: Path):
    out_path = tmp_path / "swaps.parquet"
    write_rate_limit(out_path, 0.5)

    def fake_fetch(pool_address, from_block, to_block, client, rpc_url, **kwargs):
        return [_record(from_block, 0)]

    with httpx.Client() as client:
        ingest_range(
            pool_address="0xpool", from_block=0, to_block=9, out_path=out_path,
            client=client, rpc_url="http://fake", chunk_size=10, max_workers=4,
            max_rps=4.0, fetch_fn=fake_fetch, concurrency_cooldown_seconds=0,
        )

    # Nothing rate-limited or climbed in this single-chunk run, so
    # whatever it ends on is exactly what it started from — must be
    # half the ceiling (2.0), not the persisted floor of 0.5.
    assert read_rate_limit(out_path) == 2.0
```

Add one more new test — gating now happens several call-frames below `chunked.py`'s own top-of-loop cancel check (inside whatever `fetch_fn` does with `rate_limiter.acquire()`), so this must be proven directly rather than assumed from Task 2's narrower `rpc_logs`-level test:

```python
def test_ingest_range_translates_cancelled_from_a_blocked_rate_limiter_acquire(tmp_path: Path):
    # Regression test: gating now happens inside fetch_fn (deep inside
    # fetch_swaps in production), several call-frames below chunked.py's
    # own cancel_event check at the top of the retry loop. A cancel_event
    # set while a worker is genuinely blocked in rate_limiter.acquire()
    # must still surface as IngestCancelled within a cancel-poll interval,
    # not only when the outer loop happens to re-check between attempts.
    cancel_event = threading.Event()
    entered_second_acquire = threading.Event()

    def blocked_fetch(pool_address, from_block, to_block, client, rpc_url, rate_limiter=None, cancel_event=None, **kwargs):
        rate_limiter.acquire(cancel_event)  # first call returns immediately (fresh limiter)
        entered_second_acquire.set()
        rate_limiter.acquire(cancel_event)  # blocks ~2s (RateLimiter floors max_rps=0.1 to DEFAULT_MIN_RPS=0.5) — long enough to cancel mid-wait
        return [_record(from_block, 0)]

    out_path = tmp_path / "swaps.parquet"

    def set_cancel_once_blocked() -> None:
        entered_second_acquire.wait(timeout=1.0)
        time.sleep(0.05)
        cancel_event.set()

    setter = threading.Thread(target=set_cancel_once_blocked)
    setter.start()
    try:
        with httpx.Client() as client:
            with pytest.raises(IngestCancelled):
                ingest_range(
                    pool_address="0xpool", from_block=0, to_block=9, out_path=out_path,
                    client=client, rpc_url="http://fake", chunk_size=10, max_workers=1,
                    max_rps=0.1, fetch_fn=blocked_fetch, cancel_event=cancel_event,
                )
    finally:
        setter.join(timeout=1.0)
```

- [ ] **Step 5: Run the whole test_chunked.py to verify the new/changed tests fail for the right reason**

Run: `uv run pytest tests/ingest/test_chunked.py -v`
Expected: several FAILs — `TypeError` for the `max_rps`/`on_rate_change` kwargs `ingest_range` doesn't accept yet, and `ImportError` if `read_rate_limit`/`write_rate_limit` aren't wired into this file's imports yet.

- [ ] **Step 6: Rewrite `chunked.py`**

In `src/latentedge/ingest/chunked.py`:

Replace the concurrency import with:

```python
from latentedge.ingest import rate_limiter as _rate_limiter
from latentedge.ingest.rate_limiter import Cancelled, RateLimiter
from latentedge.ingest.progress import (
    Interval,
    add_interval,
    read_progress,
    read_rate_limit,
    uncovered_gaps,
    write_progress,
    write_rate_limit,
)
```

Update the module-level constants (replace the `DEFAULT_CONCURRENCY_COOLDOWN_SECONDS` line and the long comment above `DEFAULT_CHUNK_SIZE` describing why chunking can't grow — that limitation is exactly what Task 2 removed):

```python
# Kept equal to rpc_logs.ETH_GETLOGS_RANGE_CAP (the provider's per-call
# eth_getLogs limit) for now. fetch_swaps sub-chunks internally and paces
# every one of those sub-calls through the shared RateLimiter (see
# rpc_logs.py), so a larger chunk no longer bursts unpaced requests the
# way it used to — growing this is a separate, later change, not blocked
# by pacing anymore.
DEFAULT_CHUNK_SIZE = 10
DEFAULT_MAX_RETRIES = 5
DEFAULT_RETRY_BACKOFF_SECONDS = 2.0
RATE_LIMIT_BACKOFF_MULTIPLIER = 5.0
DEFAULT_MAX_WORKERS = 8
DEFAULT_FLUSH_EVERY_N_CHUNKS = 50
DEFAULT_CONCURRENCY_COOLDOWN_SECONDS = _rate_limiter.DEFAULT_COOLDOWN_SECONDS
# A conservative starting ceiling on real requests/sec against the
# provider — proactive pacing, not just a reaction to 429s already
# happening. Tunable per provider tier via LATENTEDGE_INGEST_MAX_RPS.
DEFAULT_MAX_RPS = 5.0
DEFAULT_MAX_RATE_LIMIT_BACKOFF_SECONDS = 120.0

FetchFn = Callable[..., list[SwapRecord]]
```

Rewrite `_fetch_chunk_with_retries` (removing the `on_concurrency_change` param and the acquire/release wrapping — both moved inside `fetch_fn` via Task 2):

```python
def _fetch_chunk_with_retries(
    fetch_fn: FetchFn,
    pool_address: str,
    from_block: int,
    to_block: int,
    client: httpx.Client,
    rpc_url: str,
    max_retries: int,
    backoff_seconds: float,
    rate_limiter: RateLimiter,
    on_retry: OnRetryFn | None = None,
    on_status: Callable[[str], None] | None = None,
    max_rate_limit_backoff_seconds: float = DEFAULT_MAX_RATE_LIMIT_BACKOFF_SECONDS,
    cancel_event: threading.Event | None = None,
) -> list[SwapRecord]:
    failure_attempt = 0
    rate_limit_attempt = 0
    while True:
        if cancel_event is not None and cancel_event.is_set():
            raise IngestCancelled()
        if on_status is not None:
            on_status("fetching")
        try:
            result = fetch_fn(
                pool_address, from_block, to_block, client, rpc_url,
                rate_limiter=rate_limiter, cancel_event=cancel_event,
            )
        except Cancelled:
            raise IngestCancelled() from None
        except Exception as exc:  # RpcLogsError et al — real transient failures
            is_rate_limited = isinstance(exc, RateLimitError)
            if is_rate_limited:
                rate_limit_attempt += 1
                sleep_seconds = min(
                    backoff_seconds * (2 ** min(rate_limit_attempt - 1, 20)) * RATE_LIMIT_BACKOFF_MULTIPLIER,
                    max_rate_limit_backoff_seconds,
                )
                if on_retry is not None:
                    on_retry(from_block, to_block, rate_limit_attempt, None, sleep_seconds, str(exc))
                _interruptible_sleep(sleep_seconds, cancel_event)
                continue

            failure_attempt += 1
            if failure_attempt >= max_retries:
                raise
            sleep_seconds = backoff_seconds * (2 ** (failure_attempt - 1))
            if on_retry is not None:
                on_retry(from_block, to_block, failure_attempt, max_retries, sleep_seconds, str(exc))
            _interruptible_sleep(sleep_seconds, cancel_event)
        else:
            return result
```

In `ingest_range`'s signature, replace `on_concurrency_change` with `on_rate_change` and add `max_rps`:

```python
def ingest_range(
    pool_address: str,
    from_block: int,
    to_block: int,
    out_path: Path,
    client: httpx.Client,
    rpc_url: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
    max_workers: int = DEFAULT_MAX_WORKERS,
    max_rps: float = DEFAULT_MAX_RPS,
    flush_every_n_chunks: int = DEFAULT_FLUSH_EVERY_N_CHUNKS,
    concurrency_cooldown_seconds: float = DEFAULT_CONCURRENCY_COOLDOWN_SECONDS,
    max_rate_limit_backoff_seconds: float = DEFAULT_MAX_RATE_LIMIT_BACKOFF_SECONDS,
    on_progress: Callable[[int, int, int], None] | None = None,
    on_retry: OnRetryFn | None = None,
    on_queue_status: Callable[[int, int | None], None] | None = None,
    on_worker_status: Callable[[int, int, int, str], None] | None = None,
    on_rate_change: Callable[[float], None] | None = None,
    fetch_fn: FetchFn = fetch_swaps,
    cancel_event: threading.Event | None = None,
) -> int:
```

Replace the limiter construction block:

```python
    persisted_rate = read_rate_limit(out_path)
    # A floor, not a fixed resume point — see RateLimiter's docstring and
    # AdaptiveConcurrencyLimiter's history for why: never resume below
    # half the ceiling regardless of how low a prior run bottomed out.
    start_rate = max(persisted_rate, max_rps / 2) if persisted_rate is not None else None
    rate_limiter = RateLimiter(
        ceiling=max_rps,
        cooldown_seconds=concurrency_cooldown_seconds,
        start_rate=start_rate,
        on_change=on_rate_change,
    )
```

Update `process_chunk` to drop the `on_concurrency_change` pass-through and use `rate_limiter`:

```python
    def process_chunk(chunk_start: int) -> tuple[int, int, list[SwapRecord]]:
        chunk_end = min(chunk_start + chunk_size - 1, chunk_ceiling[chunk_start])

        slot = worker_slot() if on_worker_status is not None else -1

        def report_retry(cs: int, ce: int, attempt: int, retries: int | None, sleep_seconds: float, error_message: str) -> None:
            if on_retry is not None:
                on_retry(cs, ce, attempt, retries, sleep_seconds, error_message)
            if on_worker_status is not None:
                label = f"retry {attempt}/{retries}" if retries is not None else f"rate limited, retry {attempt}"
                on_worker_status(slot, cs, ce, f"{label}, waiting {sleep_seconds:.1f}s")

        def report_status(status: str) -> None:
            if on_worker_status is not None:
                on_worker_status(slot, chunk_start, chunk_end, status)

        retry_hook = report_retry if (on_retry is not None or on_worker_status is not None) else None
        status_hook = report_status if on_worker_status is not None else None
        records = _fetch_chunk_with_retries(
            fetch_fn, pool_address, chunk_start, chunk_end, client, rpc_url,
            max_retries, retry_backoff_seconds, rate_limiter, retry_hook, status_hook,
            max_rate_limit_backoff_seconds, cancel_event,
        )

        if on_worker_status is not None:
            on_worker_status(slot, chunk_start, chunk_end, "idle")
        return chunk_start, chunk_end, records
```

And in the `finally` block, replace `write_concurrency_limit(out_path, limiter.limit)` with:

```python
            write_rate_limit(out_path, rate_limiter.rate)
```

- [ ] **Step 7: Run the full test suite to verify everything passes**

Run: `uv run pytest tests/ingest/test_chunked.py tests/ingest/test_rpc_logs.py tests/ingest/test_rate_limiter.py tests/ingest/test_progress.py -v`
Expected: PASS (all tests)

- [ ] **Step 8: Confirm no leftover references to the deleted concurrency module**

Run: `grep -rn "ingest.concurrency\|AdaptiveConcurrencyLimiter\|on_concurrency_change\|read_concurrency_limit\|write_concurrency_limit" src/ tests/`
Expected: no output

- [ ] **Step 9: Commit**

```bash
git add src/latentedge/ingest/chunked.py tests/ingest/test_chunked.py
git commit -m "Wire the rate limiter into ingest_range, replacing concurrency-slot state"
```

---

## Task 5: CLI wiring — `LATENTEDGE_INGEST_MAX_RPS` (env-var only)

**Files:**
- Modify: `src/latentedge/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `DEFAULT_MAX_RPS` from `latentedge.ingest.chunked` (Task 4).
- Produces: `ingest()` now passes `max_rps=<float>` to `IngestScreen(...)` and every `ingest_range(...)` call, sourced from `os.environ.get("LATENTEDGE_INGEST_MAX_RPS", DEFAULT_MAX_RPS)` — deliberately not a `--max-rps` click option (this repo's convention: env vars for new run options, not new flags).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_cli.py`:

```python
def test_ingest_max_rps_falls_back_to_env_var_with_no_cli_flag(monkeypatch: pytest.MonkeyPatch):
    # LATENTEDGE_INGEST_MAX_RPS is deliberately env-var-only — no --max-rps
    # flag — matching this repo's preference for env vars over new flags
    # for run options that aren't part of every invocation's everyday use.
    captured: dict[str, object] = {}

    def fake_ingest_range(pool_address, from_block, to_block, out, client, rpc_url, **kwargs):
        captured["max_rps"] = kwargs["max_rps"]
        return 0

    monkeypatch.setattr("latentedge.cli.ingest_range", fake_ingest_range)

    runner = CliRunner()
    runner.invoke(
        cli,
        ["ingest", "--from-block", "1", "--to-block", "2"],
        env={"LATENTEDGE_INGEST_MAX_RPS": "2.5"},
    )

    assert captured["max_rps"] == 2.5


def test_ingest_max_rps_defaults_when_env_var_omitted(monkeypatch: pytest.MonkeyPatch):
    from latentedge.ingest.chunked import DEFAULT_MAX_RPS

    captured: dict[str, object] = {}

    def fake_ingest_range(pool_address, from_block, to_block, out, client, rpc_url, **kwargs):
        captured["max_rps"] = kwargs["max_rps"]
        return 0

    monkeypatch.setattr("latentedge.cli.ingest_range", fake_ingest_range)

    runner = CliRunner()
    runner.invoke(cli, ["ingest", "--from-block", "1", "--to-block", "2"])

    assert captured["max_rps"] == DEFAULT_MAX_RPS
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -k max_rps -v`
Expected: FAIL — `captured` stays empty (`KeyError: 'max_rps'`) since `ingest_range` isn't called with that kwarg yet.

- [ ] **Step 3: Wire the env var into `cli.py`**

Add `import os` to the top-level imports, and add `DEFAULT_MAX_RPS` to the existing `from latentedge.ingest.chunked import (...)` block.

Update the `--max-workers` option's help text (it no longer describes the actual throttle):

```python
@click.option(
    "--max-workers", type=int, default=DEFAULT_MAX_WORKERS, envvar="LATENTEDGE_MAX_WORKERS",
    help="Thread pool size (simultaneous connections) — the actual request pace against the provider is a separate ceiling, set via the LATENTEDGE_INGEST_MAX_RPS env var. Falls back to the LATENTEDGE_MAX_WORKERS env var (or a .env file).",
)
```

Inside `ingest()`, right after the existing `from_block`/`to_block` XOR check, read the env var directly (no click option):

```python
    max_rps = float(os.environ.get("LATENTEDGE_INGEST_MAX_RPS", DEFAULT_MAX_RPS))
```

Pass it to the `IngestScreen(...)` construction:

```python
        screen = IngestScreen(
            pool_address=config.POOL_ADDRESS, from_block=first_from, to_block=first_to,
            out_path=out, client_factory=lambda: httpx.Client(timeout=30.0), rpc_url=rpc_url,
            chunk_size=chunk_size, max_workers=max_workers, max_rps=max_rps,
            flush_every_n_chunks=flush_every_n_chunks, max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            concurrency_cooldown_seconds=concurrency_cooldown_seconds,
            max_rate_limit_backoff_seconds=max_rate_limit_backoff_seconds, ingest_fn=ingest_range,
            train_assemble_fn=_assemble_train_data,
            model_out_path=train_out, train_epochs=train_epochs, train_after_ingest=train_after_ingest,
            remaining_ranges=remaining_ranges,
        )
```

And to both `ingest_range(...)` calls in the non-TTY path (backfill loop and the requested-range call):

```python
                ingest_range(
                    config.POOL_ADDRESS, gap_start, gap_end, out, client, rpc_url,
                    chunk_size=chunk_size, max_retries=max_retries, retry_backoff_seconds=retry_backoff_seconds,
                    max_workers=max_workers, max_rps=max_rps, flush_every_n_chunks=flush_every_n_chunks,
                    concurrency_cooldown_seconds=concurrency_cooldown_seconds,
                    max_rate_limit_backoff_seconds=max_rate_limit_backoff_seconds,
                    on_progress=report,
                )
            if gaps:
                click.echo(f"backfill complete — {sum(e - s + 1 for s, e in gaps)} previously-skipped blocks recovered")
            total = ingest_range(
                config.POOL_ADDRESS, from_block, to_block, out, client, rpc_url,
                chunk_size=chunk_size, max_retries=max_retries, retry_backoff_seconds=retry_backoff_seconds,
                max_workers=max_workers, max_rps=max_rps, flush_every_n_chunks=flush_every_n_chunks,
                concurrency_cooldown_seconds=concurrency_cooldown_seconds,
                max_rate_limit_backoff_seconds=max_rate_limit_backoff_seconds,
                on_progress=report,
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/latentedge/cli.py tests/test_cli.py
git commit -m "Add LATENTEDGE_INGEST_MAX_RPS as the ingest rate ceiling"
```

---

## Task 6: TUI wiring — show the pacing rate, not a concurrency count

**Files:**
- Modify: `src/latentedge/tui/ingest_screen.py`
- Modify: `tests/tui/test_ingest_screen.py`

**Interfaces:**
- Consumes: `DEFAULT_MAX_RPS` from `latentedge.ingest.chunked` (Task 4); `ingest_range(..., max_rps=..., on_rate_change=...)` (Task 4).
- Produces: `IngestScreen(..., max_rps: float = DEFAULT_MAX_RPS, ...)`; stats panel shows a `Rate` row (`"X.X/Y.Y req/s"`) instead of `Concurrency` (`"X/Y"`).

- [ ] **Step 1: Update the mechanical test fixtures first**

Every fake fetch function in `tests/tui/test_ingest_screen.py` needs to tolerate the new keyword arguments. Run:

```bash
sed -i '' 's/(pool_address, from_block, to_block, client, rpc_url):/(pool_address, from_block, to_block, client, rpc_url, **kwargs):/' tests/tui/test_ingest_screen.py
```

- [ ] **Step 2: Delete the test for a rendering state that no longer exists**

Remove `test_ingest_screen_renders_a_worker_waiting_for_a_free_concurrency_slot` entirely — `_handle_worker_status` will no longer have a distinct "waiting for a free slot" rendering, since `chunked.py` never emits that status once gating moves inside `fetch_swaps` (Task 4, Step 3's rewritten `_fetch_chunk_with_retries` only ever reports `"fetching"`).

- [ ] **Step 3: Rewrite the two rate-driven tests**

Replace `test_ingest_screen_writes_retries_and_concurrency_changes_to_the_log_file`:

```python
@pytest.mark.asyncio
async def test_ingest_screen_writes_retries_and_rate_changes_to_the_log_file(tmp_path: Path):
    attempts = {"count": 0}

    def one_rate_limit_then_fine(pool_address, from_block, to_block, client, rpc_url, rate_limiter=None, **kwargs):
        attempts["count"] += 1
        if attempts["count"] == 1:
            if rate_limiter is not None:
                rate_limiter.release("rate_limited")
            raise RateLimitError("simulated rate limit")
        if rate_limiter is not None:
            rate_limiter.release("success")
        return [_record(from_block)]

    out_path = tmp_path / "swaps.parquet"
    screen = IngestScreen(
        pool_address="0xpool", from_block=0, to_block=39, out_path=out_path,
        client_factory=lambda: httpx.Client(), rpc_url="http://fake",
        chunk_size=10, max_workers=4, max_rps=4.0, flush_every_n_chunks=1,
        max_retries=3, retry_backoff_seconds=0.001, concurrency_cooldown_seconds=0,
        fetch_fn=one_rate_limit_then_fine,
        train_assemble_fn=_placeholder_assemble,
    )
    app = LatentEdgeApp(start_screen=screen)

    async with app.run_test() as pilot:
        for _ in range(50):
            await pilot.pause(0.01)
            if screen.is_complete:
                break

    assert screen.is_complete
    log_text = screen.log_path.read_text()
    assert "rate limited, retry 1" in log_text
    assert "simulated rate limit" in log_text
    assert "throttled down to 2.0/4.0 req/s" in log_text
```

Replace `test_ingest_screen_shows_concurrency_limit_after_a_throttle_down`:

```python
@pytest.mark.asyncio
async def test_ingest_screen_shows_the_pacing_rate_after_a_throttle_down(tmp_path: Path):
    attempts = {"count": 0}

    def one_rate_limit_then_fine(pool_address, from_block, to_block, client, rpc_url, rate_limiter=None, **kwargs):
        attempts["count"] += 1
        if attempts["count"] == 1:
            if rate_limiter is not None:
                rate_limiter.release("rate_limited")
            raise RateLimitError("simulated rate limit")
        if rate_limiter is not None:
            rate_limiter.release("success")
        return [_record(from_block)]

    out_path = tmp_path / "swaps.parquet"
    screen = IngestScreen(
        pool_address="0xpool", from_block=0, to_block=39, out_path=out_path,
        client_factory=lambda: httpx.Client(), rpc_url="http://fake",
        chunk_size=10, max_workers=4, max_rps=4.0, flush_every_n_chunks=1,
        max_retries=3, retry_backoff_seconds=0.001, concurrency_cooldown_seconds=0,
        fetch_fn=one_rate_limit_then_fine,
        train_assemble_fn=_placeholder_assemble,
    )
    app = LatentEdgeApp(start_screen=screen)

    async with app.run_test() as pilot:
        for _ in range(50):
            await pilot.pause(0.01)
            if screen.is_complete:
                break
        stats_text = str(app.screen.query_one("#ingest-stats-body").content)

    assert screen.is_complete
    assert "Rate" in stats_text
    assert "2.0/4.0" in stats_text
```

- [ ] **Step 4: Add a test for the fractional-rate formatting edge case**

```python
@pytest.mark.asyncio
async def test_ingest_screen_renders_a_fractional_rate_without_truncating_it(tmp_path: Path):
    # Regression guard: the old Concurrency row was int-based and could
    # never show a fractional value. A rate limiter can settle anywhere
    # (e.g. 0.5 req/s after repeated halvings) — the display must show
    # that precisely, not round it down to something misleading like 0.
    out_path = tmp_path / "swaps.parquet"
    screen = IngestScreen(
        pool_address="0xpool", from_block=0, to_block=9, out_path=out_path,
        client_factory=lambda: httpx.Client(), rpc_url="http://fake",
        chunk_size=10, max_workers=4, max_rps=4.0, flush_every_n_chunks=1,
        max_retries=1, retry_backoff_seconds=0.001, fetch_fn=_fake_fetch,
        train_assemble_fn=_placeholder_assemble,
    )
    app = LatentEdgeApp(start_screen=screen)

    async with app.run_test() as pilot:
        screen._handle_rate_change(0.5)
        await pilot.pause()
        stats_text = str(app.screen.query_one("#ingest-stats-body").content)

    assert "0.5/4.0" in stats_text
```

- [ ] **Step 5: Run tests to verify they fail**

Run: `uv run pytest tests/tui/test_ingest_screen.py -v`
Expected: FAIL — `TypeError: IngestScreen.__init__() got an unexpected keyword argument 'max_rps'`, and `AttributeError` for `_handle_rate_change`.

- [ ] **Step 6: Update `ingest_screen.py`**

Add `DEFAULT_MAX_RPS` to the existing `from latentedge.ingest.chunked import (...)` block.

Add `max_rps: float = DEFAULT_MAX_RPS` to `IngestScreen.__init__`'s parameters (after `max_workers`), and in the body:

```python
        self.max_workers = max_workers
        self.max_rps = max_rps
```

Replace the `self._concurrency_limit = max_workers` line with:

```python
        self._rate_limit: float = max_rps
```

Update the `on_mount` starting log line:

```python
        self._log(
            f"starting: blocks {self.from_block}-{self.to_block}{backfill_note}, chunk_size={self.chunk_size}, "
            f"max_workers={self.max_workers}, max_rps={self.max_rps}, concurrency_cooldown_seconds={self.concurrency_cooldown_seconds} "
            f"— full log at {self.log_path}"
        )
```

In `_run_ingest`, rename the callback and its registration:

```python
        def on_rate_change(new_rate: float) -> None:
            self.app.call_from_thread(self._handle_rate_change, new_rate)

        try:
            with self.client_factory() as client:
                total = self.ingest_fn(
                    self.pool_address, self.from_block, self.to_block, self.out_path,
                    client, self.rpc_url,
                    chunk_size=self.chunk_size, max_retries=self.max_retries,
                    retry_backoff_seconds=self.retry_backoff_seconds,
                    max_workers=self.max_workers, max_rps=self.max_rps,
                    flush_every_n_chunks=self.flush_every_n_chunks,
                    concurrency_cooldown_seconds=self.concurrency_cooldown_seconds,
                    max_rate_limit_backoff_seconds=self.max_rate_limit_backoff_seconds,
                    on_progress=on_progress, on_retry=on_retry,
                    on_queue_status=on_queue_status, on_worker_status=on_worker_status,
                    on_rate_change=on_rate_change,
                    fetch_fn=self.fetch_fn,
                    cancel_event=self._cancel_event,
                )
```

Replace `_handle_concurrency_change`:

```python
    def _handle_rate_change(self, new_rate: float) -> None:
        direction = "throttled down to" if new_rate < self._rate_limit else "raised to"
        self._rate_limit = new_rate
        self._log(f"rate {direction} {new_rate:.1f}/{self.max_rps:.1f} req/s")
        self._refresh_disk_stats()
```

In `_handle_worker_status`, remove the now-unreachable `elif status == "waiting":` branch:

```python
    def _handle_worker_status(self, slot: int, chunk_start: int, chunk_end: int, status: str) -> None:
        if status == "idle":
            detail = "[dim]○ idle[/dim]"
        elif status == "fetching":
            detail = f"[green]● blocks {chunk_start}-{chunk_end} — fetching[/green]"
        else:
            detail = f"[yellow]● blocks {chunk_start}-{chunk_end} — {status}[/yellow]"
        self.query_one("#ingest-threads", ThreadPanel).update_worker(slot, detail)
```

In `_refresh_disk_stats`, replace the `concurrency` row:

```python
        rate = (
            f"[yellow]{self._rate_limit:.1f}/{self.max_rps:.1f} req/s[/yellow]"
            if self._rate_limit < self.max_rps
            else f"{self._rate_limit:.1f}/{self.max_rps:.1f} req/s"
        )
        self.query_one("#ingest-stats", StatsPanel).update_stats([
            ("File size", f"{file_size / 1_048_576:.1f} MB"),
            ("Est. final size", self._format_estimated_final_size(file_size)),
            ("Free disk", f"{free_bytes / 1_073_741_824:.1f} GB"),
            ("Retries", retries),
            ("Stalled", stalled),
            ("Buffered", buffered),
            ("Rate", rate),
        ])
```

In `_build_next_screen`, pass `max_rps` through:

```python
    def _build_next_screen(self, from_block: int, to_block: int, remaining_ranges: list[tuple[int, int]]) -> "IngestScreen":
        return IngestScreen(
            pool_address=self.pool_address, from_block=from_block, to_block=to_block,
            out_path=self.out_path, client_factory=self.client_factory, rpc_url=self.rpc_url,
            chunk_size=self.chunk_size, max_workers=self.max_workers, max_rps=self.max_rps,
            flush_every_n_chunks=self.flush_every_n_chunks, max_retries=self.max_retries,
            retry_backoff_seconds=self.retry_backoff_seconds, train_assemble_fn=self.train_assemble_fn,
            concurrency_cooldown_seconds=self.concurrency_cooldown_seconds,
            max_rate_limit_backoff_seconds=self.max_rate_limit_backoff_seconds,
            model_out_path=self.model_out_path, train_epochs=self.train_epochs,
            train_after_ingest=self.train_after_ingest, ingest_fn=self.ingest_fn,
            fetch_fn=self.fetch_fn, time_fn=self.time_fn, remaining_ranges=remaining_ranges,
        )
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/tui/test_ingest_screen.py -v`
Expected: PASS (all tests)

- [ ] **Step 8: Run the entire suite**

Run: `uv run pytest -q`
Expected: PASS (all tests, no leftover references to the deleted concurrency module or old parameter names anywhere in the repo)

- [ ] **Step 9: Commit**

```bash
git add src/latentedge/tui/ingest_screen.py tests/tui/test_ingest_screen.py
git commit -m "Show the ingest TUI's pacing rate instead of a concurrency count"
```
