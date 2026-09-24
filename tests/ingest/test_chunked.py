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

    def fake_fetch(pool_address: str, from_block: int, to_block: int, client: httpx.Client, rpc_url: str) -> list[SwapRecord]:
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
            fetch_fn=fake_fetch,
        )

    # [0,99], [100,199], [200,249] — three chunks, one record each.
    assert calls == [(0, 99), (100, 199), (200, 249)]
    assert total == 3
    assert len(read_swaps(out_path)) == 3
    assert read_progress(out_path) == 249


def test_ingest_range_resumes_from_progress_file_and_skips_completed_chunks(tmp_path: Path):
    calls: list[tuple[int, int]] = []

    def fake_fetch(pool_address: str, from_block: int, to_block: int, client: httpx.Client, rpc_url: str) -> list[SwapRecord]:
        calls.append((from_block, to_block))
        return [_record(from_block, 0)]

    out_path = tmp_path / "swaps.parquet"
    with httpx.Client() as client:
        # First run covers [0, 99] only.
        ingest_range(
            pool_address="0xpool", from_block=0, to_block=99, out_path=out_path,
            client=client, rpc_url="http://fake", chunk_size=100, fetch_fn=fake_fetch,
        )
        calls.clear()

        # Second run asks for the full [0, 199] range again — must resume
        # from block 100, not re-fetch the already-completed [0, 99].
        ingest_range(
            pool_address="0xpool", from_block=0, to_block=199, out_path=out_path,
            client=client, rpc_url="http://fake", chunk_size=100, fetch_fn=fake_fetch,
        )

    assert calls == [(100, 199)]
    assert len(read_swaps(out_path)) == 2
    assert read_progress(out_path) == 199


def test_ingest_range_retries_transient_failures_then_succeeds(tmp_path: Path):
    attempts = {"count": 0}

    def flaky_fetch(pool_address: str, from_block: int, to_block: int, client: httpx.Client, rpc_url: str) -> list[SwapRecord]:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RpcLogsError("transient failure")
        return [_record(from_block, 0)]

    out_path = tmp_path / "swaps.parquet"
    with httpx.Client() as client:
        total = ingest_range(
            pool_address="0xpool", from_block=0, to_block=99, out_path=out_path,
            client=client, rpc_url="http://fake", chunk_size=100,
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
                client=client, rpc_url="http://fake", chunk_size=100,
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
            client=client, rpc_url="http://fake", chunk_size=100, fetch_fn=fake_fetch,
        )

    assert total == 0
    assert read_progress(out_path) == 99  # progress still advances past an empty chunk
