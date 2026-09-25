# No Duplicate Block Ingest Implementation Plan

> **For implementers:** work through tasks in order; each ends with a
> commit. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Guarantee that a block already recorded as ingested for a
given output file is never requested again by any caller — the `ingest`
CLI command, its TUI screen, or a test — and that an open (`--days`)
window that turns out to be already covered walks further back toward
the pool's deployment block instead of silently doing nothing.

**Architecture:** Progress tracking moves from a single forward
watermark to a sorted, merged list of disjoint block intervals, owned
by a new pure module (`latentedge/ingest/progress.py`, no I/O side
effects beyond reading/writing the interval list itself). `ingest_range`
consumes this to compute the uncovered sub-ranges ("gaps") of whatever
`[from_block, to_block]` it's asked for, and only ever fetches those
gaps — this is the single mechanism both the strict-bounds case and the
open `--days` case rely on. The CLI's `--days` branch additionally calls
a backward-extension function before invoking `ingest_range`, to widen
an already-covered naive window toward the pool's creation block.

**Tech Stack:** Python 3.12, `pytest`, `mypy --strict` — no new
dependencies.

**Spec:** `docs/specs/2026-09-25-no-duplicate-block-ingest-design.md`

## Global Constraints

- `progress.json`'s shape changes from `{"last_completed_block": N}` to
  `{"ingested": [[from, to], ...]}` — no migration path; there is no
  real progress data on disk to preserve.
- The pool's deployment block is `12_376_729` (WETH/USDC 0.05%, found by
  binary-searching `eth_getCode` against an archive RPC endpoint) — this
  is the floor past which backward extension must stop.
- `ingest_range`'s public signature (parameter names, types, return
  value) does not change.
- `mypy --strict` and the full test suite must stay clean at the end of
  every task.

## Review Focus

- A fresh output path with no progress file yet must behave exactly as
  before this change (fetch the whole requested range, no regression).
- `from_block > to_block` (an empty range) must return `0` immediately,
  not crash or fetch anything.
- A chunk that exhausts its retries must not cause progress to be
  recorded past the last real flush, even when the run had already
  crossed one or more gap boundaries since that flush.
- A requested range that's already entirely covered by progress
  (whether given explicitly or reached via backward extension all the
  way to the pool's creation block) must return `0` new records as a
  normal outcome, not an error.
- A single flush batch that spans more than one previously-covered gap
  (i.e. two or more disjoint newly-fetched stretches complete before
  the next flush) must record all of them as separate intervals, not
  collapse or drop any.

---

### Task 1: Pure interval-tracking module

**Files:**
- Create: `src/latentedge/ingest/progress.py`
- Test: `tests/ingest/test_progress.py`

**Interfaces:**
- Produces: `Interval = tuple[int, int]`; `read_progress(out_path: Path) -> list[Interval]`; `write_progress(out_path: Path, intervals: list[Interval]) -> None`; `add_interval(intervals: list[Interval], new_from: int, new_to: int) -> list[Interval]`; `uncovered_gaps(intervals: list[Interval], from_block: int, to_block: int) -> list[Interval]`; `extend_window_for_new_blocks(intervals: list[Interval], naive_from: int, naive_to: int, desired_new_blocks: int, floor_block: int) -> int`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/ingest/test_progress.py
from pathlib import Path

from latentedge.ingest.progress import (
    add_interval,
    extend_window_for_new_blocks,
    read_progress,
    uncovered_gaps,
    write_progress,
)


def test_read_progress_returns_empty_list_when_no_file_exists(tmp_path: Path):
    assert read_progress(tmp_path / "swaps.parquet") == []


def test_write_then_read_progress_round_trips(tmp_path: Path):
    out_path = tmp_path / "swaps.parquet"
    write_progress(out_path, [(0, 99), (200, 299)])
    assert read_progress(out_path) == [(0, 99), (200, 299)]


def test_add_interval_merges_overlapping_ranges():
    assert add_interval([(0, 99)], 50, 149) == [(0, 149)]


def test_add_interval_merges_adjacent_ranges():
    # 100 immediately follows 99 — must merge into one interval instead
    # of leaving two touching-but-separate entries.
    assert add_interval([(0, 99)], 100, 149) == [(0, 149)]


def test_add_interval_keeps_disjoint_ranges_separate():
    assert add_interval([(0, 99)], 200, 299) == [(0, 99), (200, 299)]


def test_add_interval_bridges_a_gap_between_two_existing_ranges():
    assert add_interval([(0, 99), (200, 299)], 100, 199) == [(0, 299)]


def test_add_interval_into_empty_list():
    assert add_interval([], 10, 20) == [(10, 20)]


def test_uncovered_gaps_with_no_existing_progress():
    assert uncovered_gaps([], 0, 99) == [(0, 99)]


def test_uncovered_gaps_fully_covered_returns_nothing():
    assert uncovered_gaps([(0, 99)], 0, 99) == []


def test_uncovered_gaps_covered_in_the_middle():
    assert uncovered_gaps([(40, 59)], 0, 99) == [(0, 39), (60, 99)]


def test_uncovered_gaps_covered_at_the_edges():
    assert uncovered_gaps([(0, 19), (80, 99)], 0, 99) == [(20, 79)]


def test_uncovered_gaps_ignores_intervals_outside_the_requested_range():
    assert uncovered_gaps([(200, 299)], 0, 99) == [(0, 99)]


def test_uncovered_gaps_returns_nothing_for_an_empty_range():
    assert uncovered_gaps([], 10, 5) == []


def test_extend_window_returns_naive_from_when_already_enough_new_blocks():
    result = extend_window_for_new_blocks([], naive_from=100, naive_to=199, desired_new_blocks=100, floor_block=0)
    assert result == 100


def test_extend_window_walks_back_when_naive_window_fully_covered():
    intervals = [(100, 199)]
    result = extend_window_for_new_blocks(intervals, naive_from=100, naive_to=199, desired_new_blocks=100, floor_block=0)
    assert result == 0
    assert sum(e - s + 1 for s, e in uncovered_gaps(intervals, result, 199)) == 100


def test_extend_window_walks_past_a_covered_stretch_in_the_extension_zone():
    intervals = [(80, 99)]  # covered stretch sits below naive_from
    result = extend_window_for_new_blocks(intervals, naive_from=100, naive_to=199, desired_new_blocks=150, floor_block=0)
    assert result == 30
    assert sum(e - s + 1 for s, e in uncovered_gaps(intervals, result, 199)) == 150


def test_extend_window_clamps_at_floor_block_when_not_enough_new_blocks_exist():
    result = extend_window_for_new_blocks([], naive_from=100, naive_to=199, desired_new_blocks=1000, floor_block=50)
    assert result == 50
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/ingest/test_progress.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'latentedge.ingest.progress'`

- [ ] **Step 3: Implement the module**

```python
# src/latentedge/ingest/progress.py
"""Pure interval arithmetic for tracking which block ranges have
already been ingested for a given output file. No I/O beyond reading
and writing the interval list itself, no threading — this is the
single source of truth `ingest_range` and the CLI both consult so a
block already recorded here is never requested again.
"""

import json
from pathlib import Path

Interval = tuple[int, int]


def _progress_path(out_path: Path) -> Path:
    return Path(str(out_path) + ".progress.json")


def read_progress(out_path: Path) -> list[Interval]:
    path = _progress_path(out_path)
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return [(pair[0], pair[1]) for pair in data["ingested"]]


def write_progress(out_path: Path, intervals: list[Interval]) -> None:
    _progress_path(out_path).write_text(json.dumps({"ingested": [list(pair) for pair in intervals]}))


def add_interval(intervals: list[Interval], new_from: int, new_to: int) -> list[Interval]:
    merged: list[Interval] = []
    placed = False
    cur_from, cur_to = new_from, new_to
    for start, end in sorted(intervals):
        if end < cur_from - 1:
            merged.append((start, end))
        elif start > cur_to + 1:
            if not placed:
                merged.append((cur_from, cur_to))
                placed = True
            merged.append((start, end))
        else:
            cur_from = min(cur_from, start)
            cur_to = max(cur_to, end)
    if not placed:
        merged.append((cur_from, cur_to))
    return merged


def uncovered_gaps(intervals: list[Interval], from_block: int, to_block: int) -> list[Interval]:
    if from_block > to_block:
        return []
    gaps: list[Interval] = []
    cursor = from_block
    for start, end in sorted(intervals):
        if end < from_block or start > to_block:
            continue
        clipped_start = max(start, from_block)
        clipped_end = min(end, to_block)
        if clipped_start > cursor:
            gaps.append((cursor, clipped_start - 1))
        cursor = max(cursor, clipped_end + 1)
    if cursor <= to_block:
        gaps.append((cursor, to_block))
    return gaps


def extend_window_for_new_blocks(
    intervals: list[Interval],
    naive_from: int,
    naive_to: int,
    desired_new_blocks: int,
    floor_block: int,
) -> int:
    """The smallest from_block <= naive_from (never below floor_block)
    such that [from_block, naive_to] contains at least
    desired_new_blocks blocks not already covered by `intervals` — or
    floor_block if even the full [floor_block, naive_to] range doesn't
    have that many.
    """
    new_in_naive = sum(end - start + 1 for start, end in uncovered_gaps(intervals, naive_from, naive_to))
    remaining_needed = desired_new_blocks - new_in_naive
    if remaining_needed <= 0:
        return naive_from
    if naive_from <= floor_block:
        return floor_block

    gaps_below = uncovered_gaps(intervals, floor_block, naive_from - 1)
    from_block = floor_block
    for start, end in reversed(gaps_below):
        gap_size = end - start + 1
        if gap_size >= remaining_needed:
            from_block = end - remaining_needed + 1
            remaining_needed = 0
            break
        remaining_needed -= gap_size
    return from_block
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/ingest/test_progress.py -v`
Expected: all PASS

- [ ] **Step 5: Type-check**

Run: `mypy --strict src/latentedge/ingest/progress.py`
Expected: no errors

- [ ] **Step 6: Commit**

```bash
git add src/latentedge/ingest/progress.py tests/ingest/test_progress.py
git commit -m "Add disjoint-interval progress tracking for ingested block ranges"
```

---

### Task 2: `ingest_range` gap-fill

**Files:**
- Modify: `src/latentedge/ingest/chunked.py`
- Modify: `tests/ingest/test_chunked.py`

**Interfaces:**
- Consumes: `read_progress`, `write_progress`, `add_interval`, `uncovered_gaps` from Task 1's `latentedge.ingest.progress`.
- Produces: `ingest_range(...)` unchanged in signature and return type; `read_progress`/`write_progress` are no longer defined in `chunked.py` — callers import them from `latentedge.ingest.progress` instead (Task 3 updates the one other caller).

- [ ] **Step 1: Write the failing tests (new + updated assertions)**

Replace the whole contents of `tests/ingest/test_chunked.py` with:

```python
import threading
import time
from pathlib import Path

import httpx
import pytest

from latentedge.ingest.chunked import ingest_range
from latentedge.ingest.progress import read_progress, write_progress
from latentedge.ingest.rpc_logs import RateLimitError, RpcLogsError
from latentedge.schema import SwapRecord
from latentedge.store import read_swaps


def _record(block_number: int, log_index: int) -> SwapRecord:
    return SwapRecord(
        block_number=block_number,
        timestamp=block_number * 12,
        tx_hash=f"0x{block_number:064x}",
        log_index=log_index,
        sqrt_price_x96=1 << 96,
        tick=0,
        liquidity=10**18,
        amount0=1000.0,
        amount1=-0.3,
        base_fee_wei=20_000_000_000,
    )


def test_ingest_range_splits_into_chunks_and_writes_incrementally(tmp_path: Path):
    calls: list[tuple[int, int]] = []
    lock = threading.Lock()

    def fake_fetch(pool_address: str, from_block: int, to_block: int, client: httpx.Client, rpc_url: str) -> list[SwapRecord]:
        with lock:
            calls.append((from_block, to_block))
        return [_record(from_block, 0)]

    out_path = tmp_path / "swaps.parquet"
    with httpx.Client() as client:
        total = ingest_range(
            pool_address="0xpool",
            from_block=0,
            to_block=249,
            out_path=out_path,
            client=client,
            rpc_url="http://fake",
            chunk_size=100,
            max_workers=1,  # deterministic call order for this assertion
            fetch_fn=fake_fetch,
        )

    # [0,99], [100,199], [200,249] — three chunks, one record each.
    assert sorted(calls) == [(0, 99), (100, 199), (200, 249)]
    assert total == 3
    assert len(read_swaps(out_path)) == 3
    assert read_progress(out_path) == [(0, 249)]


def test_ingest_range_resumes_from_progress_file_and_skips_completed_chunks(tmp_path: Path):
    calls: list[tuple[int, int]] = []
    lock = threading.Lock()

    def fake_fetch(pool_address: str, from_block: int, to_block: int, client: httpx.Client, rpc_url: str) -> list[SwapRecord]:
        with lock:
            calls.append((from_block, to_block))
        return [_record(from_block, 0)]

    out_path = tmp_path / "swaps.parquet"
    with httpx.Client() as client:
        # First run covers [0, 99] only.
        ingest_range(
            pool_address="0xpool", from_block=0, to_block=99, out_path=out_path,
            client=client, rpc_url="http://fake", chunk_size=100, max_workers=1, fetch_fn=fake_fetch,
        )
        calls.clear()

        # Second run asks for the full [0, 199] range again — must skip
        # the already-completed [0, 99], not re-fetch it.
        ingest_range(
            pool_address="0xpool", from_block=0, to_block=199, out_path=out_path,
            client=client, rpc_url="http://fake", chunk_size=100, max_workers=1, fetch_fn=fake_fetch,
        )

    assert calls == [(100, 199)]
    assert len(read_swaps(out_path)) == 2
    assert read_progress(out_path) == [(0, 199)]


def test_ingest_range_returns_zero_when_the_whole_requested_range_is_already_covered(tmp_path: Path):
    def unexpected_fetch(pool_address: str, from_block: int, to_block: int, client: httpx.Client, rpc_url: str) -> list[SwapRecord]:
        raise AssertionError("fetch must not be called for an already-covered range")

    out_path = tmp_path / "swaps.parquet"
    write_progress(out_path, [(0, 99)])

    with httpx.Client() as client:
        total = ingest_range(
            pool_address="0xpool", from_block=0, to_block=99, out_path=out_path,
            client=client, rpc_url="http://fake", chunk_size=10, max_workers=1, fetch_fn=unexpected_fetch,
        )

    assert total == 0


def test_ingest_range_skips_a_covered_stretch_in_the_middle_of_the_requested_range(tmp_path: Path):
    calls: list[tuple[int, int]] = []
    lock = threading.Lock()

    def fake_fetch(pool_address: str, from_block: int, to_block: int, client: httpx.Client, rpc_url: str) -> list[SwapRecord]:
        with lock:
            calls.append((from_block, to_block))
        return [_record(from_block, 0)]

    out_path = tmp_path / "swaps.parquet"
    write_progress(out_path, [(20, 29)])  # already ingested, sits inside [0, 49]

    with httpx.Client() as client:
        total = ingest_range(
            pool_address="0xpool", from_block=0, to_block=49, out_path=out_path,
            client=client, rpc_url="http://fake", chunk_size=10, max_workers=1, fetch_fn=fake_fetch,
        )

    assert sorted(calls) == [(0, 9), (10, 19), (30, 39), (40, 49)]
    assert total == 4
    # The pre-existing [20,29] plus the four newly-fetched stretches
    # merge back into one contiguous interval.
    assert read_progress(out_path) == [(0, 49)]


def test_ingest_range_fetches_around_two_disjoint_pre_existing_intervals(tmp_path: Path):
    calls: list[tuple[int, int]] = []
    lock = threading.Lock()

    def fake_fetch(pool_address: str, from_block: int, to_block: int, client: httpx.Client, rpc_url: str) -> list[SwapRecord]:
        with lock:
            calls.append((from_block, to_block))
        return [_record(from_block, 0)]

    out_path = tmp_path / "swaps.parquet"
    write_progress(out_path, [(10, 19), (30, 39)])

    with httpx.Client() as client:
        total = ingest_range(
            pool_address="0xpool", from_block=0, to_block=49, out_path=out_path,
            client=client, rpc_url="http://fake", chunk_size=10, max_workers=1,
            flush_every_n_chunks=50,  # all three new stretches land in one flush
            fetch_fn=fake_fetch,
        )

    assert sorted(calls) == [(0, 9), (20, 29), (40, 49)]
    assert total == 3
    assert read_progress(out_path) == [(0, 49)]


def test_ingest_range_flushes_correctly_when_failure_happens_after_crossing_a_gap_boundary(tmp_path: Path):
    # A pre-existing covered interval sits between two uncovered
    # stretches. The first stretch's chunk succeeds (crossing into the
    # second gap advances the fold past the covered middle); the second
    # stretch's chunk then fails permanently. The already-succeeded
    # chunk must still be flushed and merged with the pre-existing
    # interval, not discarded.
    call_count = {"n": 0}

    def succeed_first_gap_then_fail(pool_address: str, from_block: int, to_block: int, client: httpx.Client, rpc_url: str) -> list[SwapRecord]:
        call_count["n"] += 1
        if call_count["n"] > 1:
            raise RpcLogsError("simulated failure in the second gap")
        return [_record(from_block, 0)]

    out_path = tmp_path / "swaps.parquet"
    write_progress(out_path, [(10, 19)])

    with httpx.Client() as client:
        with pytest.raises(RpcLogsError):
            ingest_range(
                pool_address="0xpool", from_block=0, to_block=29, out_path=out_path,
                client=client, rpc_url="http://fake", chunk_size=10, max_workers=1,
                flush_every_n_chunks=100,  # never reached — only the failure-path flush matters
                max_retries=1, fetch_fn=succeed_first_gap_then_fail,
            )

    assert read_progress(out_path) == [(0, 19)]
    assert len(read_swaps(out_path)) == 1


def test_ingest_range_retries_transient_failures_then_succeeds(tmp_path: Path):
    attempts = {"count": 0}
    lock = threading.Lock()

    def flaky_fetch(pool_address: str, from_block: int, to_block: int, client: httpx.Client, rpc_url: str) -> list[SwapRecord]:
        with lock:
            attempts["count"] += 1
            count = attempts["count"]
        if count < 3:
            raise RpcLogsError("transient failure")
        return [_record(from_block, 0)]

    out_path = tmp_path / "swaps.parquet"
    with httpx.Client() as client:
        total = ingest_range(
            pool_address="0xpool", from_block=0, to_block=99, out_path=out_path,
            client=client, rpc_url="http://fake", chunk_size=100, max_workers=1,
            fetch_fn=flaky_fetch, max_retries=5, retry_backoff_seconds=0.001,
        )

    assert attempts["count"] == 3
    assert total == 1


def test_ingest_range_gives_up_after_max_retries(tmp_path: Path):
    def always_fails(pool_address: str, from_block: int, to_block: int, client: httpx.Client, rpc_url: str) -> list[SwapRecord]:
        raise RpcLogsError("permanent failure")

    out_path = tmp_path / "swaps.parquet"
    with httpx.Client() as client:
        with pytest.raises(RpcLogsError):
            ingest_range(
                pool_address="0xpool", from_block=0, to_block=99, out_path=out_path,
                client=client, rpc_url="http://fake", chunk_size=100, max_workers=1,
                fetch_fn=always_fails, max_retries=3, retry_backoff_seconds=0.001,
            )
    # No progress recorded for a chunk that never succeeded.
    assert read_progress(out_path) == []


def test_ingest_range_handles_empty_chunk_without_crashing(tmp_path: Path):
    def fake_fetch(pool_address: str, from_block: int, to_block: int, client: httpx.Client, rpc_url: str) -> list[SwapRecord]:
        return []  # a quiet period with no swaps at all

    out_path = tmp_path / "swaps.parquet"
    with httpx.Client() as client:
        total = ingest_range(
            pool_address="0xpool", from_block=0, to_block=99, out_path=out_path,
            client=client, rpc_url="http://fake", chunk_size=100, max_workers=1, fetch_fn=fake_fetch,
        )

    assert total == 0
    assert read_progress(out_path) == [(0, 99)]  # progress still advances past an empty chunk


def test_ingest_range_runs_chunks_concurrently(tmp_path: Path):
    # Regression test: at Alchemy's real free-tier eth_getLogs cap (10
    # blocks/call), a year of history is ~263,000 sequential requests —
    # concurrency is what makes that tractable. 20 chunks that each take
    # ~50ms must complete in well under 20*50ms if genuinely parallel.
    def slow_fetch(pool_address: str, from_block: int, to_block: int, client: httpx.Client, rpc_url: str) -> list[SwapRecord]:
        time.sleep(0.05)
        return [_record(from_block, 0)]

    out_path = tmp_path / "swaps.parquet"
    with httpx.Client() as client:
        start = time.monotonic()
        total = ingest_range(
            pool_address="0xpool", from_block=0, to_block=199, out_path=out_path,
            client=client, rpc_url="http://fake", chunk_size=10, max_workers=10, fetch_fn=slow_fetch,
        )
        elapsed = time.monotonic() - start

    assert total == 20
    assert elapsed < 0.5  # would be ~1.0s if processed strictly sequentially


def test_ingest_range_stays_correct_with_out_of_order_chunk_completion(tmp_path: Path):
    # Chunks complete in whatever order their network calls happen to
    # finish, not necessarily the order they were requested in. Progress
    # and written data must still end up complete and correct regardless.
    def variable_speed_fetch(pool_address: str, from_block: int, to_block: int, client: httpx.Client, rpc_url: str) -> list[SwapRecord]:
        # Make earlier chunks artificially slower so later ones finish first.
        time.sleep(0.03 if from_block < 50 else 0.001)
        return [_record(from_block, 0)]

    out_path = tmp_path / "swaps.parquet"
    with httpx.Client() as client:
        total = ingest_range(
            pool_address="0xpool", from_block=0, to_block=99, out_path=out_path,
            client=client, rpc_url="http://fake", chunk_size=10, max_workers=10, fetch_fn=variable_speed_fetch,
        )

    assert total == 10
    assert read_progress(out_path) == [(0, 99)]  # fully contiguous despite out-of-order completion
    df = read_swaps(out_path)
    assert sorted(df["block_number"]) == [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]


def test_ingest_range_batches_writes_instead_of_one_per_chunk(tmp_path: Path):
    # At a 10-block chunk size, a year is ~263,000 chunks — writing (a
    # full read-modify-write of the Parquet file) after every single one
    # would be prohibitively slow as the file grows. Writes must batch.
    write_calls = {"count": 0}
    original_write_swaps = __import__("latentedge.store", fromlist=["write_swaps"]).write_swaps

    def counting_write_swaps(records, path):
        write_calls["count"] += 1
        return original_write_swaps(records, path)

    def fake_fetch(pool_address: str, from_block: int, to_block: int, client: httpx.Client, rpc_url: str) -> list[SwapRecord]:
        return [_record(from_block, 0)]

    out_path = tmp_path / "swaps.parquet"
    with httpx.Client() as client:
        import latentedge.ingest.chunked as chunked_module

        original = chunked_module.write_swaps
        chunked_module.write_swaps = counting_write_swaps
        try:
            ingest_range(
                pool_address="0xpool", from_block=0, to_block=999, out_path=out_path,
                client=client, rpc_url="http://fake", chunk_size=10, max_workers=4,
                flush_every_n_chunks=20, fetch_fn=fake_fetch,
            )
        finally:
            chunked_module.write_swaps = original

    # 100 chunks / flush_every_n_chunks=20 -> 5 flushes, not 100 writes.
    assert write_calls["count"] == 5
    assert len(read_swaps(out_path)) == 100


def test_ingest_range_progress_never_exceeds_what_was_actually_flushed(tmp_path: Path):
    # Critical correctness property: if the process dies after some
    # chunks completed but before a flush, progress must NOT have
    # advanced past the last real flush — otherwise a resume would skip
    # blocks whose data was never actually written to disk.
    call_count = {"n": 0}

    def fetch_then_die(pool_address: str, from_block: int, to_block: int, client: httpx.Client, rpc_url: str) -> list[SwapRecord]:
        call_count["n"] += 1
        if call_count["n"] > 15:
            raise RpcLogsError("simulated crash")
        return [_record(from_block, 0)]

    out_path = tmp_path / "swaps.parquet"
    with httpx.Client() as client:
        with pytest.raises(RpcLogsError):
            ingest_range(
                pool_address="0xpool", from_block=0, to_block=999, out_path=out_path,
                client=client, rpc_url="http://fake", chunk_size=10, max_workers=1,
                flush_every_n_chunks=20, max_retries=1, fetch_fn=fetch_then_die,
            )

    # 15 chunks succeeded (blocks 0-149) but the flush threshold (20)
    # was never reached — the existing flush-before-raising safeguard
    # still flushes them on the way out, so progress lands exactly at
    # the end of the 15th chunk, never further.
    assert read_progress(out_path) == [(0, 149)]


def test_ingest_range_backs_off_longer_for_rate_limit_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # A rate-limited request retried on the same short schedule as a
    # generic transient error just re-triggers the same limit — this
    # was observed for real against a live provider. Rate limits must
    # back off noticeably longer.
    sleeps: list[float] = []
    monkeypatch.setattr("latentedge.ingest.chunked.time.sleep", lambda seconds: sleeps.append(seconds))

    attempts = {"count": 0}

    def rate_limited_fetch(pool_address: str, from_block: int, to_block: int, client: httpx.Client, rpc_url: str) -> list[SwapRecord]:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RateLimitError("simulated rate limit")
        return [_record(from_block, 0)]

    out_path = tmp_path / "swaps.parquet"
    with httpx.Client() as client:
        ingest_range(
            pool_address="0xpool", from_block=0, to_block=99, out_path=out_path,
            client=client, rpc_url="http://fake", chunk_size=100, max_workers=1,
            fetch_fn=rate_limited_fetch, max_retries=5, retry_backoff_seconds=1.0,
        )

    # Two retries happened (attempts 1 and 2 failed); both backoffs must
    # be well beyond the plain (non-rate-limited) schedule of 1s, 2s.
    assert len(sleeps) == 2
    assert all(s >= 5.0 for s in sleeps)


def test_ingest_range_calls_on_retry_for_each_failed_attempt(tmp_path: Path):
    attempts = {"count": 0}
    lock = threading.Lock()
    retry_calls = {"count": 0}
    retry_lock = threading.Lock()

    def flaky_fetch(pool_address: str, from_block: int, to_block: int, client: httpx.Client, rpc_url: str) -> list[SwapRecord]:
        with lock:
            attempts["count"] += 1
            count = attempts["count"]
        if count < 3:
            raise RpcLogsError("transient failure")
        return [_record(from_block, 0)]

    def on_retry() -> None:
        with retry_lock:
            retry_calls["count"] += 1

    out_path = tmp_path / "swaps.parquet"
    with httpx.Client() as client:
        ingest_range(
            pool_address="0xpool", from_block=0, to_block=99, out_path=out_path,
            client=client, rpc_url="http://fake", chunk_size=100, max_workers=1,
            fetch_fn=flaky_fetch, max_retries=5, retry_backoff_seconds=0.001,
            on_retry=on_retry,
        )

    # 3 attempts total means 2 failed-then-retried attempts.
    assert retry_calls["count"] == 2


def test_ingest_range_flushes_completed_chunks_before_raising_on_a_later_failure(tmp_path: Path):
    # Regression test: a single chunk exhausting its retries must not
    # discard every chunk that already succeeded since the last flush —
    # otherwise a real run that fetches hundreds of chunks successfully
    # loses all of them the moment one chunk hits a rate limit it can't
    # recover from in time.
    call_count = {"n": 0}

    def succeed_then_fail(pool_address: str, from_block: int, to_block: int, client: httpx.Client, rpc_url: str) -> list[SwapRecord]:
        call_count["n"] += 1
        if call_count["n"] > 5:
            raise RpcLogsError("simulated persistent rate limit")
        return [_record(from_block, 0)]

    out_path = tmp_path / "swaps.parquet"
    with httpx.Client() as client:
        with pytest.raises(RpcLogsError):
            ingest_range(
                pool_address="0xpool", from_block=0, to_block=99, out_path=out_path,
                client=client, rpc_url="http://fake", chunk_size=10, max_workers=1,
                flush_every_n_chunks=100,  # never reached — only the failure-path flush matters here
                max_retries=1, fetch_fn=succeed_then_fail,
            )

    # The 5 chunks that succeeded before the 6th chunk's failure must
    # have been written to disk and their watermark recorded, even
    # though the whole call ultimately raised.
    assert read_progress(out_path) == [(0, 49)]  # end of the 5th chunk (blocks 0-49)
    assert len(read_swaps(out_path)) == 5
```

- [ ] **Step 2: Run tests to verify the new/changed ones fail**

Run: `pytest tests/ingest/test_chunked.py -v`
Expected: FAIL — `ImportError` (progress module has no such name yet in
`chunked.py`'s namespace) and assertion failures on the `read_progress(...)`
format changes.

- [ ] **Step 3: Rewrite `ingest_range`**

Replace `src/latentedge/ingest/chunked.py`'s contents with:

```python
"""Chunked, resumable, concurrent ingestion over a large block range.

Real archive RPC providers cap eth_getLogs to a limited block range per
call — Alchemy's free tier caps at 10 blocks — so a year of history means
on the order of a quarter-million chunk requests. Sequentially that's
days; concurrency brings it down to hours. At that chunk count, writing
the Parquet file (a full read-modify-write) after every single chunk is
also prohibitively slow as the file grows, so writes are batched. Progress
is only ever recorded for data that has actually been flushed to disk —
a crash between flushes means re-fetching (cheap, deduped) some already-
fetched chunks on resume, never silently skipping unwritten ones.

Progress is tracked as a set of disjoint block intervals (see
latentedge.ingest.progress), not a single forward watermark — a request
only ever fetches the sub-ranges of [from_block, to_block] not already
covered by a prior run, regardless of how those prior runs were shaped.
"""

import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

from latentedge.ingest.progress import Interval, add_interval, read_progress, uncovered_gaps, write_progress
from latentedge.ingest.rpc_logs import RateLimitError, fetch_swaps
from latentedge.schema import SwapRecord
from latentedge.store import write_swaps

DEFAULT_CHUNK_SIZE = 10  # Alchemy's free-tier eth_getLogs cap; verified against the real service
DEFAULT_MAX_RETRIES = 5
DEFAULT_RETRY_BACKOFF_SECONDS = 2.0
# A rate limit retried on the same short schedule as a generic transient
# error just re-triggers the same limit (observed for real against
# Alchemy's free tier) — back off substantially longer for it.
RATE_LIMIT_BACKOFF_MULTIPLIER = 5.0
# Lower than earlier default (8): fewer concurrent workers means fewer
# simultaneous requests competing for the same per-second compute-unit
# budget, observed for real to matter more than backoff tuning alone.
DEFAULT_MAX_WORKERS = 4
# Lower than earlier default (200): a chunk that ultimately can't
# recover from a rate limit no longer loses unflushed progress (see
# ingest_range's finally-block flush), but flushing more often still
# bounds how much work a mid-run interruption could repeat on resume.
DEFAULT_FLUSH_EVERY_N_CHUNKS = 50

FetchFn = Callable[[str, int, int, httpx.Client, str], list[SwapRecord]]


def _fetch_chunk_with_retries(
    fetch_fn: FetchFn,
    pool_address: str,
    from_block: int,
    to_block: int,
    client: httpx.Client,
    rpc_url: str,
    max_retries: int,
    backoff_seconds: float,
    on_retry: Callable[[], None] | None = None,
) -> list[SwapRecord]:
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            return fetch_fn(pool_address, from_block, to_block, client, rpc_url)
        except Exception as exc:  # RpcLogsError et al — real transient failures
            last_error = exc
            if attempt < max_retries - 1:
                if on_retry is not None:
                    on_retry()
                sleep_seconds = backoff_seconds * (2**attempt)
                if isinstance(exc, RateLimitError):
                    sleep_seconds *= RATE_LIMIT_BACKOFF_MULTIPLIER
                time.sleep(sleep_seconds)
    assert last_error is not None
    raise last_error


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
    flush_every_n_chunks: int = DEFAULT_FLUSH_EVERY_N_CHUNKS,
    on_progress: Callable[[int, int, int], None] | None = None,
    on_retry: Callable[[], None] | None = None,
    fetch_fn: FetchFn = fetch_swaps,
) -> int:
    """Ingest [from_block, to_block] concurrently, in chunks, flushing to
    disk (and recording newly-covered intervals in the resumable
    progress file) every flush_every_n_chunks chunks — never per-chunk,
    which would mean rewriting the whole output file hundreds of
    thousands of times over a real year-long pull. Returns the total
    number of swap records written in this call. Any sub-range of
    [from_block, to_block] already present in a prior run's progress is
    skipped, never re-fetched.
    """
    progress_intervals = read_progress(out_path)
    gaps = uncovered_gaps(progress_intervals, from_block, to_block)
    if not gaps:
        return 0

    # Chunk each gap independently and concatenate in order — a chunk
    # never straddles a gap boundary, so an already-covered stretch is
    # never touched even at its edges.
    chunk_order: list[int] = []
    chunk_ceiling: dict[int, int] = {}
    for gap_start, gap_end in gaps:
        for chunk_start in range(gap_start, gap_end + 1, chunk_size):
            chunk_order.append(chunk_start)
            chunk_ceiling[chunk_start] = gap_end

    lock = threading.Lock()
    completed: dict[int, tuple[int, list[SwapRecord]]] = {}
    expected_index = 0

    pending_records: list[SwapRecord] = []
    # Contiguous runs completed since the last flush — normally one, but
    # can be several if a flush batch spans more than one previously-
    # covered gap.
    pending_intervals: list[Interval] = []
    chunks_since_flush = 0
    total_written = 0

    def flush() -> None:
        nonlocal pending_records, pending_intervals, chunks_since_flush, total_written, progress_intervals
        if not pending_intervals:
            return
        if pending_records:
            write_swaps(pending_records, out_path)
            total_written += len(pending_records)
        for start, end in pending_intervals:
            progress_intervals = add_interval(progress_intervals, start, end)
        write_progress(out_path, progress_intervals)
        pending_records = []
        pending_intervals = []
        chunks_since_flush = 0

    def process_chunk(chunk_start: int) -> tuple[int, int, list[SwapRecord]]:
        chunk_end = min(chunk_start + chunk_size - 1, chunk_ceiling[chunk_start])
        records = _fetch_chunk_with_retries(
            fetch_fn, pool_address, chunk_start, chunk_end, client, rpc_url,
            max_retries, retry_backoff_seconds, on_retry,
        )
        return chunk_start, chunk_end, records

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_chunk, cs): cs for cs in chunk_order}
        try:
            for future in as_completed(futures):
                chunk_start, chunk_end, records = future.result()

                with lock:
                    completed[chunk_start] = (chunk_end, records)

                    # Fold in whatever prefix of the ordered chunk plan
                    # is now available. Chunks can finish out of order,
                    # but this only ever advances through chunk_order in
                    # sequence — crossing from one gap's last chunk to
                    # the next gap's first chunk is just the next index,
                    # not an arithmetic +chunk_size step.
                    while expected_index < len(chunk_order) and chunk_order[expected_index] in completed:
                        cs = chunk_order[expected_index]
                        c_end, recs = completed.pop(cs)
                        pending_records.extend(recs)

                        if pending_intervals and pending_intervals[-1][1] + 1 == cs:
                            last_start, _ = pending_intervals[-1]
                            pending_intervals[-1] = (last_start, c_end)
                        else:
                            pending_intervals.append((cs, c_end))

                        if on_progress is not None:
                            on_progress(cs, c_end, len(recs))

                        expected_index += 1
                        chunks_since_flush += 1

                        if chunks_since_flush >= flush_every_n_chunks:
                            flush()
        finally:
            # A chunk that exhausted its retries raises out of
            # future.result() above; cancel whatever hasn't started yet
            # rather than let the pool keep firing more requests. Flush
            # here too (not only on the success path above) — an
            # exception propagating past this block must never discard
            # chunks that already succeeded since the last flush.
            for f in futures:
                f.cancel()
            with lock:
                flush()

    return total_written
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/ingest/test_chunked.py -v`
Expected: all PASS

- [ ] **Step 5: Type-check**

Run: `mypy --strict src/latentedge/ingest/chunked.py`
Expected: no errors

- [ ] **Step 6: Commit**

```bash
git add src/latentedge/ingest/chunked.py tests/ingest/test_chunked.py
git commit -m "Gap-fill ingest_range against disjoint progress instead of a single watermark"
```

---

### Task 3: TUI ingest screen — resumed-progress display

**Files:**
- Modify: `src/latentedge/tui/ingest_screen.py:1-20,82-98`
- Modify: `tests/tui/test_ingest_screen.py`

**Interfaces:**
- Consumes: `read_progress`, `uncovered_gaps` from `latentedge.ingest.progress` (Task 1).

- [ ] **Step 1: Update the failing test assertions**

In `tests/tui/test_ingest_screen.py`, change the import block at the top
from:

```python
from latentedge.ingest.chunked import ingest_range, read_progress
```

to:

```python
from latentedge.ingest.chunked import ingest_range
from latentedge.ingest.progress import read_progress
```

Then in `test_ingest_screen_reaches_complete_state`, change:

```python
    assert read_progress(out_path) == 29
```

to:

```python
    assert read_progress(out_path) == [(0, 29)]
```

- [ ] **Step 2: Run the affected tests to verify they fail**

Run: `pytest tests/tui/test_ingest_screen.py -v`
Expected: FAIL — `ImportError` (`read_progress` is no longer exported
from `latentedge.ingest.chunked`)

- [ ] **Step 3: Update `ingest_screen.py`**

Change the import block (`src/latentedge/tui/ingest_screen.py:1-19`) from:

```python
from latentedge.ingest.chunked import FetchFn, read_progress
from latentedge.ingest.chunked import ingest_range as default_ingest_range
from latentedge.ingest.rpc_logs import describe_error, fetch_swaps
```

to:

```python
from latentedge.ingest.chunked import FetchFn
from latentedge.ingest.chunked import ingest_range as default_ingest_range
from latentedge.ingest.progress import read_progress, uncovered_gaps
from latentedge.ingest.rpc_logs import describe_error, fetch_swaps
```

Change `on_mount` (`src/latentedge/tui/ingest_screen.py:82-98`) from:

```python
    def on_mount(self) -> None:
        total = max(self.to_block - self.from_block + 1, 0)

        # A resumed run's already-fetched blocks count toward completed so
        # the bar doesn't restart at 0% — mirrors ingest_range's own
        # resume-from-watermark logic in ingest.chunked.
        resume_from = read_progress(self.out_path)
        start_block = self.from_block
        if resume_from is not None and resume_from + 1 > start_block:
            start_block = min(resume_from + 1, self.to_block + 1)
        completed = max(start_block - self.from_block, 0)
        unit_label = f"resuming from block {start_block}" if completed > 0 else "starting..."

        self.query_one("#ingest-progress", ProgressPanel).update_progress(
            completed=completed, total=total, unit_label=unit_label,
            rate_per_sec=0.0, rate_unit="blocks/sec",
        )
        self.set_interval(STATS_REFRESH_INTERVAL_SECONDS, self._refresh_disk_stats)
        self.run_worker(self._run_ingest, thread=True, exclusive=True)
```

to:

```python
    def on_mount(self) -> None:
        total = max(self.to_block - self.from_block + 1, 0)

        # Already-ingested blocks within the requested range count
        # toward completed so the bar doesn't restart at 0% on a
        # resumed run — mirrors ingest_range's own gap-fill dedup logic
        # in ingest.chunked.
        intervals = read_progress(self.out_path)
        uncovered = sum(
            end - start + 1 for start, end in uncovered_gaps(intervals, self.from_block, self.to_block)
        )
        completed = max(total - uncovered, 0)
        unit_label = f"resuming ({completed} blocks already ingested)" if completed > 0 else "starting..."

        self.query_one("#ingest-progress", ProgressPanel).update_progress(
            completed=completed, total=total, unit_label=unit_label,
            rate_per_sec=0.0, rate_unit="blocks/sec",
        )
        self.set_interval(STATS_REFRESH_INTERVAL_SECONDS, self._refresh_disk_stats)
        self.run_worker(self._run_ingest, thread=True, exclusive=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/tui/test_ingest_screen.py -v`
Expected: all PASS

- [ ] **Step 5: Type-check**

Run: `mypy --strict src/latentedge/tui/ingest_screen.py`
Expected: no errors

- [ ] **Step 6: Commit**

```bash
git add src/latentedge/tui/ingest_screen.py tests/tui/test_ingest_screen.py
git commit -m "Adapt ingest screen's resume display to interval-based progress"
```

---

### Task 4: CLI backward extension for the `--days` window

**Files:**
- Modify: `src/latentedge/config.py`
- Modify: `src/latentedge/cli.py:1-101`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes: `extend_window_for_new_blocks`, `read_progress` from `latentedge.ingest.progress` (Task 1); `config.POOL_CREATION_BLOCK`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_cli.py`:

```python
def test_ingest_days_window_walks_back_past_already_ingested_blocks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    # A chain head high enough to stay well above the pool's real
    # deployment block (config.POOL_CREATION_BLOCK) so this test
    # exercises the backward walk itself, not the floor clamp.
    monkeypatch.setattr("latentedge.cli.get_latest_block", lambda client, rpc_url: 13_000_000)

    out_path = tmp_path / "swaps.parquet"
    naive_to = 13_000_000 - 5
    blocks_in_range = 7200  # 1 day at 12s/block
    naive_from = naive_to - blocks_in_range + 1

    from latentedge.ingest.progress import write_progress

    write_progress(out_path, [(naive_from, naive_to)])  # the whole naive window is already ingested

    captured: dict[str, int] = {}

    def fake_ingest_range(pool_address, from_block, to_block, out, client, rpc_url, **kwargs):
        captured["from_block"] = from_block
        captured["to_block"] = to_block
        return 0

    monkeypatch.setattr("latentedge.cli.ingest_range", fake_ingest_range)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["ingest", "--days", "1", "--out", str(out_path)],
    )

    assert result.exit_code == 0, result.output
    assert captured["to_block"] == naive_to
    # The whole naive window was already covered, so the request must
    # walk back to an earlier, equally-sized uncovered window instead of
    # silently doing nothing.
    assert captured["from_block"] == naive_from - blocks_in_range
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py::test_ingest_days_window_walks_back_past_already_ingested_blocks -v`
Expected: FAIL — `captured["from_block"]` equals `naive_from`, not
`naive_from - blocks_in_range` (no backward-extension logic exists yet)

- [ ] **Step 3: Add the pool creation block constant**

Add to `src/latentedge/config.py` (after `HEAD_BLOCK_SAFETY_BUFFER`):

```python

# WETH/USDC 0.05% pool deployment block — no swap history is possible
# before it, so a --days window that turns out to already be fully
# ingested stops walking backward here rather than requesting
# pre-deployment blocks. Found by binary-searching eth_getCode against
# an archive RPC endpoint.
POOL_CREATION_BLOCK = 12_376_729
```

- [ ] **Step 4: Wire the backward extension into the `ingest` command**

In `src/latentedge/cli.py`, add to the imports:

```python
from latentedge.ingest.progress import extend_window_for_new_blocks, read_progress
```

Change the range-derivation block (`src/latentedge/cli.py:91-99`) from:

```python
    if from_block is None:
        with httpx.Client(timeout=30.0) as client:
            try:
                head = get_latest_block(client, rpc_url)
            except (httpx.HTTPError, RpcLogsError) as exc:
                raise click.ClickException(describe_error(rpc_url, exc)) from None
        to_block = head - config.HEAD_BLOCK_SAFETY_BUFFER
        blocks_in_range = max(int(days * 86400 / config.AVG_BLOCK_SECONDS), 1)
        from_block = to_block - blocks_in_range + 1
    assert to_block is not None  # guaranteed by the from_block/to_block XOR check above
```

to:

```python
    if from_block is None:
        with httpx.Client(timeout=30.0) as client:
            try:
                head = get_latest_block(client, rpc_url)
            except (httpx.HTTPError, RpcLogsError) as exc:
                raise click.ClickException(describe_error(rpc_url, exc)) from None
        naive_to = head - config.HEAD_BLOCK_SAFETY_BUFFER
        blocks_in_range = max(int(days * 86400 / config.AVG_BLOCK_SECONDS), 1)
        naive_from = naive_to - blocks_in_range + 1
        # If the naive most-recent-N-days window is already (partly or
        # fully) ingested, walk further back toward the pool's
        # deployment block until a day's worth of genuinely new blocks
        # is found — never re-request what's already on disk.
        from_block = extend_window_for_new_blocks(
            read_progress(out), naive_from, naive_to, blocks_in_range, config.POOL_CREATION_BLOCK,
        )
        to_block = naive_to
    assert to_block is not None  # guaranteed by the from_block/to_block XOR check above
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_cli.py -v`
Expected: all PASS, including the new test and every pre-existing
`--days` test (they use a fresh `tmp_path` with no progress file, so
`extend_window_for_new_blocks` returns `naive_from` unchanged for them)

- [ ] **Step 6: Type-check and run the full suite**

Run: `mypy --strict src/latentedge/cli.py src/latentedge/config.py && pytest`
Expected: no mypy errors; full suite green

- [ ] **Step 7: Commit**

```bash
git add src/latentedge/config.py src/latentedge/cli.py tests/test_cli.py
git commit -m "Walk --days ingest window back past already-ingested blocks"
```
