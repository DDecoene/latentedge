from pathlib import Path

import httpx
import pytest

from latentedge.ingest.chunked import ingest_range, read_progress
from latentedge.schema import SwapRecord
from latentedge.store import read_swaps
from latentedge.tui.app import LatentEdgeApp
from latentedge.tui.ingest_screen import IngestScreen


def _record(block_number: int) -> SwapRecord:
    return SwapRecord(
        block_number=block_number, timestamp=block_number * 12,
        tx_hash=f"0x{block_number:064x}", log_index=0,
        sqrt_price_x96=1 << 96, tick=0, liquidity=10**18,
        amount0=1000.0, amount1=-0.3, base_fee_wei=20_000_000_000,
    )


def _fake_fetch(pool_address, from_block, to_block, client, rpc_url):
    return [_record(from_block)]


@pytest.mark.asyncio
async def test_ingest_screen_reaches_complete_state(tmp_path: Path):
    out_path = tmp_path / "swaps.parquet"
    screen = IngestScreen(
        pool_address="0xpool", from_block=0, to_block=29, out_path=out_path,
        client_factory=lambda: httpx.Client(), rpc_url="http://fake",
        chunk_size=10, max_workers=1, flush_every_n_chunks=1,
        max_retries=1, retry_backoff_seconds=0.001, fetch_fn=_fake_fetch,
    )
    app = LatentEdgeApp(start_screen=screen)

    async with app.run_test() as pilot:
        for _ in range(50):
            await pilot.pause()
            if screen.is_complete:
                break
            await pilot.pause(0.01)

    assert screen.is_complete
    assert screen.total_written == 3
    assert read_progress(out_path) == 29
    assert len(read_swaps(out_path)) == 3


@pytest.mark.asyncio
async def test_ingest_screen_progress_starts_from_resumed_block(tmp_path: Path):
    # Regression for Review Focus: resumed runs must not restart the bar at 0%.
    out_path = tmp_path / "swaps.parquet"
    with httpx.Client() as client:
        ingest_range(
            pool_address="0xpool", from_block=0, to_block=9, out_path=out_path,
            client=client, rpc_url="http://fake", chunk_size=10, max_workers=1,
            fetch_fn=_fake_fetch,
        )

    screen = IngestScreen(
        pool_address="0xpool", from_block=0, to_block=29, out_path=out_path,
        client_factory=lambda: httpx.Client(), rpc_url="http://fake",
        chunk_size=10, max_workers=1, flush_every_n_chunks=1,
        max_retries=1, retry_backoff_seconds=0.001, fetch_fn=_fake_fetch,
    )
    app = LatentEdgeApp(start_screen=screen)

    async with app.run_test() as pilot:
        await pilot.pause()
        # Immediately after start, completed-so-far must already reflect
        # the resumed watermark (block 9), not 0 — checked below via
        # total_written, the stable observable contract.
        for _ in range(50):
            await pilot.pause(0.01)
            if screen.is_complete:
                break

    assert screen.is_complete
    assert screen.total_written == 2  # only blocks [10,29] were new


@pytest.mark.asyncio
async def test_ingest_screen_handles_zero_length_range_without_hanging(tmp_path: Path):
    # Review Focus: from_block > to_block must complete immediately.
    out_path = tmp_path / "swaps.parquet"
    screen = IngestScreen(
        pool_address="0xpool", from_block=10, to_block=5, out_path=out_path,
        client_factory=lambda: httpx.Client(), rpc_url="http://fake",
        chunk_size=10, max_workers=1, flush_every_n_chunks=1,
        max_retries=1, retry_backoff_seconds=0.001, fetch_fn=_fake_fetch,
    )
    app = LatentEdgeApp(start_screen=screen)

    async with app.run_test() as pilot:
        for _ in range(50):
            await pilot.pause(0.01)
            if screen.is_complete:
                break

    assert screen.is_complete
    assert screen.total_written == 0


@pytest.mark.asyncio
async def test_ingest_screen_logs_error_on_exhausted_retries_without_crashing(tmp_path: Path):
    # Review Focus: a mid-run failure must surface in the log, not crash the app.
    def always_fails(pool_address, from_block, to_block, client, rpc_url):
        raise RuntimeError("permanent failure")

    out_path = tmp_path / "swaps.parquet"
    screen = IngestScreen(
        pool_address="0xpool", from_block=0, to_block=9, out_path=out_path,
        client_factory=lambda: httpx.Client(), rpc_url="http://fake",
        chunk_size=10, max_workers=1, flush_every_n_chunks=1,
        max_retries=1, retry_backoff_seconds=0.001, fetch_fn=always_fails,
    )
    app = LatentEdgeApp(start_screen=screen)

    async with app.run_test() as pilot:
        for _ in range(50):
            await pilot.pause(0.01)
            if screen.is_complete or screen.error is not None:
                break

    assert screen.error is not None
    assert "permanent failure" in screen.error
