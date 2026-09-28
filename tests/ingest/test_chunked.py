import threading
import time
from pathlib import Path

import httpx
import pytest

from latentedge.ingest.chunked import IngestCancelled, ingest_range
from latentedge.ingest.progress import read_concurrency_limit, read_progress, write_progress
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


def test_ingest_range_records_each_disjoint_new_stretch_as_its_own_interval_before_merging(tmp_path: Path):
    # The final progress file [(0,49)] alone doesn't prove the three new
    # stretches were tracked separately — a buggy implementation that
    # collapsed them into one span before merging would produce the same
    # final answer here. Spy on add_interval to prove pending_intervals
    # really held three separate (start, end) pairs, not one.
    import latentedge.ingest.chunked as chunked_module
    from latentedge.ingest.progress import add_interval as real_add_interval

    calls: list[tuple[int, int]] = []

    def spy_add_interval(intervals: list[tuple[int, int]], new_from: int, new_to: int) -> list[tuple[int, int]]:
        calls.append((new_from, new_to))
        return real_add_interval(intervals, new_from, new_to)

    original = chunked_module.add_interval
    chunked_module.add_interval = spy_add_interval
    try:
        def fake_fetch(pool_address: str, from_block: int, to_block: int, client: httpx.Client, rpc_url: str) -> list[SwapRecord]:
            return [_record(from_block, 0)]

        out_path = tmp_path / "swaps.parquet"
        write_progress(out_path, [(10, 19), (30, 39)])

        with httpx.Client() as client:
            ingest_range(
                pool_address="0xpool", from_block=0, to_block=49, out_path=out_path,
                client=client, rpc_url="http://fake", chunk_size=10, max_workers=1,
                flush_every_n_chunks=50,  # all three new stretches land in one flush
                fetch_fn=fake_fetch,
            )
    finally:
        chunked_module.add_interval = original

    assert calls == [(0, 9), (20, 29), (40, 49)]


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
            concurrency_cooldown_seconds=0,
        )

    # Two retries happened (attempts 1 and 2 failed); both backoffs must
    # be well beyond the plain (non-rate-limited) schedule of 1s, 2s.
    assert len(sleeps) == 2
    assert all(s >= 5.0 for s in sleeps)


def test_ingest_range_never_gives_up_on_a_rate_limited_chunk_even_past_max_retries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # A rate limit is expected, temporary provider behavior, not a bug —
    # an unattended, hours-long run must never quit over it, however many
    # times it recurs on one chunk. max_retries=2 would exhaust a
    # generic error in 2 attempts; this chunk fails 5 times and still
    # must succeed rather than raise.
    monkeypatch.setattr("latentedge.ingest.chunked.time.sleep", lambda seconds: None)
    attempts = {"count": 0}

    def rate_limited_many_times(pool_address: str, from_block: int, to_block: int, client: httpx.Client, rpc_url: str) -> list[SwapRecord]:
        attempts["count"] += 1
        if attempts["count"] < 6:
            raise RateLimitError("simulated rate limit")
        return [_record(from_block, 0)]

    out_path = tmp_path / "swaps.parquet"
    with httpx.Client() as client:
        total = ingest_range(
            pool_address="0xpool", from_block=0, to_block=99, out_path=out_path,
            client=client, rpc_url="http://fake", chunk_size=100, max_workers=1,
            fetch_fn=rate_limited_many_times, max_retries=2, retry_backoff_seconds=0.001,
            concurrency_cooldown_seconds=0,
        )

    assert total == 1
    assert attempts["count"] == 6


def test_ingest_range_caps_rate_limit_backoff_instead_of_growing_unbounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    sleeps: list[float] = []
    monkeypatch.setattr("latentedge.ingest.chunked.time.sleep", lambda seconds: sleeps.append(seconds))
    attempts = {"count": 0}

    def rate_limited_many_times(pool_address: str, from_block: int, to_block: int, client: httpx.Client, rpc_url: str) -> list[SwapRecord]:
        attempts["count"] += 1
        if attempts["count"] < 8:
            raise RateLimitError("simulated rate limit")
        return [_record(from_block, 0)]

    out_path = tmp_path / "swaps.parquet"
    with httpx.Client() as client:
        ingest_range(
            pool_address="0xpool", from_block=0, to_block=99, out_path=out_path,
            client=client, rpc_url="http://fake", chunk_size=100, max_workers=1,
            fetch_fn=rate_limited_many_times, max_retries=2, retry_backoff_seconds=2.0,
            concurrency_cooldown_seconds=0,
        )

    # Uncapped exponential growth (2 * 2^6 * 5 = 640s) would blow way past
    # any reasonable ceiling by the 7th rate-limited attempt.
    assert all(s <= 120.0 for s in sleeps)
    assert max(sleeps) == 120.0


def test_ingest_range_reports_no_retry_ceiling_for_a_rate_limited_retry(tmp_path: Path):
    attempts = {"count": 0}

    def rate_limited_then_fine(pool_address: str, from_block: int, to_block: int, client: httpx.Client, rpc_url: str) -> list[SwapRecord]:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RateLimitError("simulated rate limit")
        return [_record(from_block, 0)]

    retry_calls: list[tuple[int, int, int, int | None, float, str]] = []

    out_path = tmp_path / "swaps.parquet"
    with httpx.Client() as client:
        ingest_range(
            pool_address="0xpool", from_block=0, to_block=99, out_path=out_path,
            client=client, rpc_url="http://fake", chunk_size=100, max_workers=1,
            fetch_fn=rate_limited_then_fine, max_retries=2, retry_backoff_seconds=0.001,
            concurrency_cooldown_seconds=0, on_retry=lambda *args: retry_calls.append(args),
        )

    assert len(retry_calls) == 1
    assert retry_calls[0][3] is None  # no ceiling reported for a rate limit


def test_ingest_range_calls_on_retry_for_each_failed_attempt(tmp_path: Path):
    attempts = {"count": 0}
    lock = threading.Lock()
    retry_calls: list[tuple[int, int, int, int, float, str]] = []
    retry_lock = threading.Lock()

    def flaky_fetch(pool_address: str, from_block: int, to_block: int, client: httpx.Client, rpc_url: str) -> list[SwapRecord]:
        with lock:
            attempts["count"] += 1
            count = attempts["count"]
        if count < 3:
            raise RpcLogsError("transient failure")
        return [_record(from_block, 0)]

    def on_retry(chunk_start: int, chunk_end: int, attempt: int, max_retries: int, sleep_seconds: float, error_message: str) -> None:
        with retry_lock:
            retry_calls.append((chunk_start, chunk_end, attempt, max_retries, sleep_seconds, error_message))

    out_path = tmp_path / "swaps.parquet"
    with httpx.Client() as client:
        ingest_range(
            pool_address="0xpool", from_block=0, to_block=99, out_path=out_path,
            client=client, rpc_url="http://fake", chunk_size=100, max_workers=1,
            fetch_fn=flaky_fetch, max_retries=5, retry_backoff_seconds=0.001,
            on_retry=on_retry,
        )

    # 3 attempts total means 2 failed-then-retried attempts, with
    # increasing attempt numbers and the chunk's real block range and
    # error text carried through.
    assert [c[:4] for c in retry_calls] == [(0, 99, 1, 5), (0, 99, 2, 5)]
    assert all(c[4] > 0 for c in retry_calls)
    assert all("transient failure" in c[5] for c in retry_calls)


def test_ingest_range_reports_queue_status_when_a_chunk_buffers_behind_a_straggler(tmp_path: Path):
    # Chunk [0,9] is slow (simulates a retrying straggler); chunk [10,19]
    # finishes first and must buffer behind it rather than being folded
    # into progress immediately. on_queue_status reports that buildup.
    release_first_chunk = threading.Event()

    def gated_fetch(pool_address: str, from_block: int, to_block: int, client: httpx.Client, rpc_url: str) -> list[SwapRecord]:
        if from_block == 0:
            release_first_chunk.wait()
        return [_record(from_block, 0)]

    statuses: list[tuple[int, int | None]] = []
    status_lock = threading.Lock()
    saw_second_chunk_done = threading.Event()

    def on_queue_status(buffered_count: int, blocking_chunk_start: int | None) -> None:
        with status_lock:
            statuses.append((buffered_count, blocking_chunk_start))
        if buffered_count >= 1:
            saw_second_chunk_done.set()
            release_first_chunk.set()

    out_path = tmp_path / "swaps.parquet"
    with httpx.Client() as client:
        ingest_range(
            pool_address="0xpool", from_block=0, to_block=19, out_path=out_path,
            client=client, rpc_url="http://fake", chunk_size=10, max_workers=2,
            fetch_fn=gated_fetch, on_queue_status=on_queue_status,
        )

    assert saw_second_chunk_done.is_set()
    assert (1, 0) in statuses  # one chunk buffered, blocked on chunk starting at block 0


def test_ingest_range_reports_worker_status_while_fetching_and_when_idle(tmp_path: Path):
    release_chunks = threading.Event()

    def gated_fetch(pool_address: str, from_block: int, to_block: int, client: httpx.Client, rpc_url: str) -> list[SwapRecord]:
        release_chunks.wait()
        return [_record(from_block, 0)]

    statuses: list[tuple[int, int, int, str]] = []
    status_lock = threading.Lock()
    saw_two_fetching = threading.Event()

    def on_worker_status(slot: int, chunk_start: int, chunk_end: int, status: str) -> None:
        with status_lock:
            statuses.append((slot, chunk_start, chunk_end, status))
            fetching_slots = {s for s, _, _, st in statuses if st == "fetching"}
            if len(fetching_slots) >= 2:
                saw_two_fetching.set()
                release_chunks.set()

    out_path = tmp_path / "swaps.parquet"
    with httpx.Client() as client:
        ingest_range(
            pool_address="0xpool", from_block=0, to_block=19, out_path=out_path,
            client=client, rpc_url="http://fake", chunk_size=10, max_workers=2,
            fetch_fn=gated_fetch, on_worker_status=on_worker_status,
        )

    assert saw_two_fetching.is_set()
    fetching_slots = sorted({s for s, _, _, st in statuses if st == "fetching"})
    assert fetching_slots == [0, 1]
    idle_statuses = [s for s in statuses if s[3] == "idle"]
    assert len(idle_statuses) == 2


def test_ingest_range_reports_worker_status_during_retries(tmp_path: Path):
    attempts = {"count": 0}

    def flaky_fetch(pool_address: str, from_block: int, to_block: int, client: httpx.Client, rpc_url: str) -> list[SwapRecord]:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RpcLogsError("transient failure")
        return [_record(from_block, 0)]

    statuses: list[tuple[int, int, int, str]] = []

    def on_worker_status(slot: int, chunk_start: int, chunk_end: int, status: str) -> None:
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
    # Each attempt reports "waiting" (for a free concurrency slot) before
    # "fetching" (once it actually has one) — with only one worker and no
    # rate limiting, the slot is always immediately free.
    assert statuses[0] == (0, 0, 99, "waiting")
    assert statuses[1] == (0, 0, 99, "fetching")
    assert statuses[-1] == (0, 0, 99, "idle")


def test_fetch_chunk_with_retries_reports_waiting_while_blocked_on_the_concurrency_gate(tmp_path: Path):
    # Regression test: a throttled-down worker was reported as "fetching"
    # the whole time it sat blocked on the concurrency gate, because the
    # status was announced before limiter.acquire() — the one moment the
    # throttle is actually visible was invisible in the UI.
    from latentedge.ingest.chunked import _fetch_chunk_with_retries
    from latentedge.ingest.concurrency import AdaptiveConcurrencyLimiter

    limiter = AdaptiveConcurrencyLimiter(ceiling=1)
    limiter.acquire()  # hold the only permit so the call under test must wait for it

    statuses: list[str] = []
    status_before_release: list[str] = []

    def on_status(status: str) -> None:
        statuses.append(status)

    def fetch_fn(pool_address: str, from_block: int, to_block: int, client: httpx.Client, rpc_url: str) -> list[SwapRecord]:
        return [_record(from_block, 0)]

    def release_after_delay() -> None:
        time.sleep(0.05)
        status_before_release.extend(statuses)
        limiter.release("success")

    releaser = threading.Thread(target=release_after_delay)
    releaser.start()
    try:
        _fetch_chunk_with_retries(
            fetch_fn, "0xpool", 0, 9, None, "http://fake",  # type: ignore[arg-type]
            max_retries=1, backoff_seconds=0.001, limiter=limiter, on_status=on_status,
        )
    finally:
        releaser.join(timeout=1.0)

    assert status_before_release == ["waiting"]  # blocked on the gate, not "fetching"
    assert statuses == ["waiting", "fetching"]


def test_ingest_range_throttles_down_worker_concurrency_after_a_rate_limit(tmp_path: Path):
    attempts = {"count": 0}
    lock = threading.Lock()

    def one_rate_limit_then_fine(pool_address: str, from_block: int, to_block: int, client: httpx.Client, rpc_url: str) -> list[SwapRecord]:
        with lock:
            attempts["count"] += 1
            first = attempts["count"] == 1
        if first:
            raise RateLimitError("simulated rate limit")
        return [_record(from_block, 0)]

    concurrency_changes: list[int] = []
    changes_lock = threading.Lock()

    def on_concurrency_change(new_limit: int) -> None:
        with changes_lock:
            concurrency_changes.append(new_limit)

    out_path = tmp_path / "swaps.parquet"
    with httpx.Client() as client:
        total = ingest_range(
            pool_address="0xpool", from_block=0, to_block=99, out_path=out_path,
            client=client, rpc_url="http://fake", chunk_size=10, max_workers=4,
            fetch_fn=one_rate_limit_then_fine, max_retries=3, retry_backoff_seconds=0.001,
            concurrency_cooldown_seconds=0,
            on_concurrency_change=on_concurrency_change,
        )

    assert total == 10
    assert 2 in concurrency_changes  # halved from the ceiling of 4 after the one rate limit


def test_ingest_range_persists_the_settled_concurrency_limit_for_the_next_run(tmp_path: Path):
    attempts = {"count": 0}
    lock = threading.Lock()

    def one_rate_limit_then_fine(pool_address: str, from_block: int, to_block: int, client: httpx.Client, rpc_url: str) -> list[SwapRecord]:
        with lock:
            attempts["count"] += 1
            first = attempts["count"] == 1
        if first:
            raise RateLimitError("simulated rate limit")
        return [_record(from_block, 0)]

    out_path = tmp_path / "swaps.parquet"
    with httpx.Client() as client:
        ingest_range(
            pool_address="0xpool", from_block=0, to_block=99, out_path=out_path,
            client=client, rpc_url="http://fake", chunk_size=10, max_workers=4,
            fetch_fn=one_rate_limit_then_fine, max_retries=3, retry_backoff_seconds=0.001,
            concurrency_cooldown_seconds=0,
        )

    # The run above halved from a ceiling of 4 down to 2 and never grew
    # back (successes_before_increase defaults to 20, far more than the
    # handful of chunks here) — a resumed run must start from that 2,
    # not silently reset to the ceiling and re-earn the same throttle.
    assert read_concurrency_limit(out_path) == 2

    seen_limits: list[int] = []

    def record_limit_seen_on_acquire(pool_address: str, from_block: int, to_block: int, client: httpx.Client, rpc_url: str) -> list[SwapRecord]:
        return [_record(from_block, 0)]

    def on_concurrency_change(new_limit: int) -> None:
        seen_limits.append(new_limit)

    with httpx.Client() as client:
        ingest_range(
            pool_address="0xpool", from_block=100, to_block=109, out_path=out_path,
            client=client, rpc_url="http://fake", chunk_size=10, max_workers=4,
            fetch_fn=record_limit_seen_on_acquire, concurrency_cooldown_seconds=0,
            on_concurrency_change=on_concurrency_change,
        )

    # Nothing rate-limited this time, so the limit should only ever have
    # been read as 2 (the persisted value), never reported back up to
    # the ceiling of 4 from a single chunk's worth of successes.
    assert 4 not in seen_limits
    assert read_concurrency_limit(out_path) == 2


def test_ingest_range_never_resumes_below_half_the_ceiling_even_if_a_prior_run_bottomed_out(tmp_path: Path):
    # A prior run that bottomed all the way out to 1 (the AIMD floor)
    # must not permanently pin every future run to 1 — recovering from 1
    # needs successes_before_increase consecutive successes, which a
    # flaky provider may never string together, so a bare persisted
    # floor would ratchet throughput down forever with no way back up.
    from latentedge.ingest.progress import write_concurrency_limit

    out_path = tmp_path / "swaps.parquet"
    write_concurrency_limit(out_path, 1)

    def fake_fetch(pool_address: str, from_block: int, to_block: int, client: httpx.Client, rpc_url: str) -> list[SwapRecord]:
        return [_record(from_block, 0)]

    with httpx.Client() as client:
        ingest_range(
            pool_address="0xpool", from_block=0, to_block=9, out_path=out_path,
            client=client, rpc_url="http://fake", chunk_size=10, max_workers=4,
            fetch_fn=fake_fetch, concurrency_cooldown_seconds=0,
        )

    # Nothing rate-limited or grew the limit in this single-chunk run, so
    # whatever it ends on is exactly what it started from — must be half
    # the ceiling (2), not the persisted floor of 1.
    assert read_concurrency_limit(out_path) == 2


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


def test_ingest_range_cancel_event_stops_cleanly_and_flushes_completed_chunks(tmp_path: Path):
    # Regression test for the ctrl+q TUI hang: a cancel_event set mid-run
    # must interrupt outstanding chunks quickly (not after a full
    # rate-limit backoff), flush whatever already completed exactly like
    # a normal run, and surface IngestCancelled — never silently hang
    # and never lose or corrupt already-fetched data.
    cancel_event = threading.Event()
    call_count = {"n": 0}

    def fake_fetch(pool_address: str, from_block: int, to_block: int, client: httpx.Client, rpc_url: str) -> list[SwapRecord]:
        with threading.Lock():
            call_count["n"] += 1
        if call_count["n"] == 3:
            cancel_event.set()
        return [_record(from_block, 0)]

    out_path = tmp_path / "swaps.parquet"
    with httpx.Client() as client:
        with pytest.raises(IngestCancelled) as exc_info:
            ingest_range(
                pool_address="0xpool", from_block=0, to_block=99, out_path=out_path,
                client=client, rpc_url="http://fake", chunk_size=10, max_workers=1,
                fetch_fn=fake_fetch, cancel_event=cancel_event,
            )

    # Exactly the chunks fetched before cancellation was noticed are
    # flushed — nothing beyond that, nothing lost from before it.
    written = exc_info.value.total_written
    assert written == len(read_swaps(out_path))
    assert written >= 3
    progress = read_progress(out_path)
    assert progress and progress[0][0] == 0
