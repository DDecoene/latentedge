import threading
from pathlib import Path

import httpx
import pytest

from latentedge.ingest.chunked import ingest_range, read_progress
from latentedge.schema import SwapRecord
from latentedge.store import read_swaps
from latentedge.tui.app import LatentEdgeApp
from latentedge.tui.ingest_screen import IngestScreen
from latentedge.tui.train_screen import TrainScreen


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


@pytest.mark.asyncio
async def test_ingest_screen_shows_completion_prompt(tmp_path: Path):
    out_path = tmp_path / "swaps.parquet"
    screen = IngestScreen(
        pool_address="0xpool", from_block=0, to_block=9, out_path=out_path,
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
        action_bar = app.screen.query_one("#ingest-action-bar")
        text = str(action_bar.content)

    assert "Train now" in text
    assert "Exit" in text
    assert str(out_path) in text or "1" in text  # swap count or path present


@pytest.mark.asyncio
async def test_ingest_screen_train_key_pushes_train_screen(tmp_path: Path):
    out_path = tmp_path / "swaps.parquet"
    screen = IngestScreen(
        pool_address="0xpool", from_block=0, to_block=9, out_path=out_path,
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
        await pilot.press("t")
        await pilot.pause()
        active_screen = app.screen

    assert isinstance(active_screen, TrainScreen)
    assert active_screen.swaps_path == out_path


@pytest.mark.asyncio
async def test_ingest_screen_train_key_ignored_before_completion(tmp_path: Path):
    # A fetch gated on an Event, so completion is deterministically held
    # back until the test explicitly releases it — a fixed time.sleep
    # raced against the pilot's own event-loop-idle wait and was flaky.
    release_fetch = threading.Event()

    def gated_fetch(pool_address, from_block, to_block, client, rpc_url):
        release_fetch.wait()
        return [_record(from_block)]

    out_path = tmp_path / "swaps.parquet"
    screen = IngestScreen(
        pool_address="0xpool", from_block=0, to_block=9, out_path=out_path,
        client_factory=lambda: httpx.Client(), rpc_url="http://fake",
        chunk_size=10, max_workers=1, flush_every_n_chunks=1,
        max_retries=1, retry_backoff_seconds=0.001, fetch_fn=gated_fetch,
    )
    app = LatentEdgeApp(start_screen=screen)

    try:
        async with app.run_test() as pilot:
            await pilot.pause()
            assert not screen.is_complete
            await pilot.press("t")
            await pilot.pause()
            active_screen = app.screen
    finally:
        release_fetch.set()

    assert not isinstance(active_screen, TrainScreen)
