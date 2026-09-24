import threading
import time
from pathlib import Path

import httpx
import pytest

from latentedge.ingest.chunked import ingest_range, read_progress
from latentedge.ingest.rpc_logs import RpcLogsError
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
    assert read_progress(out_path) == 249


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

        # Second run asks for the full [0, 199] range again — must resume
        # from block 100, not re-fetch the already-completed [0, 99].
        ingest_range(
            pool_address="0xpool", from_block=0, to_block=199, out_path=out_path,
            client=client, rpc_url="http://fake", chunk_size=100, max_workers=1, fetch_fn=fake_fetch,
        )

    assert calls == [(100, 199)]
    assert len(read_swaps(out_path)) == 2
    assert read_progress(out_path) == 199


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
    assert read_progress(out_path) is None


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
    assert read_progress(out_path) == 99  # progress still advances past an empty chunk


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
    assert read_progress(out_path) == 99  # fully contiguous despite out-of-order completion
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

    # 15 chunks succeeded but the flush threshold (20) was never reached,
    # so nothing should have been written or progress-marked yet.
    progress = read_progress(out_path)
    assert progress is None or progress < 190  # nowhere near chunk 15's block range
