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
        train_assemble_fn=lambda p: (None, None, 0, {}),
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
    # Regression test: resumed runs must not restart the progress bar at 0%.
    out_path = tmp_path / "swaps.parquet"
    with httpx.Client() as client:
        ingest_range(
            pool_address="0xpool", from_block=0, to_block=9, out_path=out_path,
            client=client, rpc_url="http://fake", chunk_size=10, max_workers=1,
            fetch_fn=_fake_fetch,
        )

    release_fetch = threading.Event()

    def gated_fetch(pool_address, from_block, to_block, client, rpc_url):
        release_fetch.wait()
        return [_record(from_block)]

    screen = IngestScreen(
        pool_address="0xpool", from_block=0, to_block=29, out_path=out_path,
        client_factory=lambda: httpx.Client(), rpc_url="http://fake",
        chunk_size=10, max_workers=1, flush_every_n_chunks=1,
        max_retries=1, retry_backoff_seconds=0.001, fetch_fn=gated_fetch,
        train_assemble_fn=lambda p: (None, None, 0, {}),
    )
    app = LatentEdgeApp(start_screen=screen)

    try:
        async with app.run_test() as pilot:
            await pilot.pause()
            # Before any new chunk has been fetched, the bar must already
            # reflect the resumed watermark (10 of 30 blocks), not 0%.
            detail_text = str(app.screen.query_one("#ingest-progress-detail").content)
            assert "33%" in detail_text

            release_fetch.set()
            for _ in range(50):
                await pilot.pause(0.01)
                if screen.is_complete:
                    break
    finally:
        release_fetch.set()

    assert screen.is_complete
    assert screen.total_written == 2  # only blocks [10,29] were new


@pytest.mark.asyncio
async def test_ingest_screen_handles_zero_length_range_without_hanging(tmp_path: Path):
    # An empty range (from_block > to_block) must complete immediately, not hang.
    out_path = tmp_path / "swaps.parquet"
    screen = IngestScreen(
        pool_address="0xpool", from_block=10, to_block=5, out_path=out_path,
        client_factory=lambda: httpx.Client(), rpc_url="http://fake",
        chunk_size=10, max_workers=1, flush_every_n_chunks=1,
        max_retries=1, retry_backoff_seconds=0.001, fetch_fn=_fake_fetch,
        train_assemble_fn=lambda p: (None, None, 0, {}),
    )
    app = LatentEdgeApp(start_screen=screen)

    async with app.run_test() as pilot:
        for _ in range(50):
            await pilot.pause(0.01)
            if screen.is_complete:
                break
        detail_text = str(app.screen.query_one("#ingest-progress-detail").content)

    assert screen.is_complete
    assert screen.total_written == 0
    assert "complete" in detail_text


@pytest.mark.asyncio
async def test_ingest_screen_resumed_run_with_nothing_left_reaches_full_bar(tmp_path: Path):
    # A resumed range where every block is already ingested must reach
    # 100% immediately, not sit at 0% while the footer claims completion.
    out_path = tmp_path / "swaps.parquet"
    with httpx.Client() as client:
        ingest_range(
            pool_address="0xpool", from_block=0, to_block=19, out_path=out_path,
            client=client, rpc_url="http://fake", chunk_size=10, max_workers=1,
            fetch_fn=_fake_fetch,
        )

    screen = IngestScreen(
        pool_address="0xpool", from_block=0, to_block=19, out_path=out_path,
        client_factory=lambda: httpx.Client(), rpc_url="http://fake",
        chunk_size=10, max_workers=1, flush_every_n_chunks=1,
        max_retries=1, retry_backoff_seconds=0.001, fetch_fn=_fake_fetch,
        train_assemble_fn=lambda p: (None, None, 0, {}),
    )
    app = LatentEdgeApp(start_screen=screen)

    async with app.run_test() as pilot:
        for _ in range(50):
            await pilot.pause(0.01)
            if screen.is_complete:
                break
        detail_text = str(app.screen.query_one("#ingest-progress-detail").content)

    assert screen.is_complete
    assert screen.total_written == 0
    assert "100%" in detail_text


@pytest.mark.asyncio
async def test_ingest_screen_logs_error_on_exhausted_retries_without_crashing(tmp_path: Path):
    # A mid-run failure must surface in the log, not crash the app.
    def always_fails(pool_address, from_block, to_block, client, rpc_url):
        raise RuntimeError("permanent failure")

    out_path = tmp_path / "swaps.parquet"
    screen = IngestScreen(
        pool_address="0xpool", from_block=0, to_block=9, out_path=out_path,
        client_factory=lambda: httpx.Client(), rpc_url="http://fake",
        chunk_size=10, max_workers=1, flush_every_n_chunks=1,
        max_retries=1, retry_backoff_seconds=0.001, fetch_fn=always_fails,
        train_assemble_fn=lambda p: (None, None, 0, {}),
    )
    app = LatentEdgeApp(start_screen=screen)

    async with app.run_test() as pilot:
        for _ in range(50):
            await pilot.pause(0.01)
            if screen.is_complete or screen.error is not None:
                break
        action_bar_text = str(app.screen.query_one("#ingest-action-bar").content)

        # T must stay disabled after a failure — there's no completed
        # ingest to train on.
        await pilot.press("t")
        await pilot.pause()
        screen_after_t = app.screen

        # Q must exit the app after a failure, so the user isn't stuck
        # on a half-drawn screen with no way out.
        await pilot.press("q")
        await pilot.pause()
        still_running = app.is_running

    assert screen.error is not None
    assert "permanent failure" in screen.error
    assert "Exit" in action_bar_text
    assert not isinstance(screen_after_t, TrainScreen)
    assert not still_running


@pytest.mark.asyncio
async def test_ingest_screen_shows_a_plain_language_message_for_a_connection_failure(tmp_path: Path):
    # Regression test: a raw httpx exception (errno numbers, internal
    # jargon) must not reach the screen verbatim — it should read like
    # something a person can act on.
    def connection_refused(pool_address, from_block, to_block, client, rpc_url):
        raise httpx.ConnectError("[Errno 8] nodename nor servname provided, or not known")

    out_path = tmp_path / "swaps.parquet"
    screen = IngestScreen(
        pool_address="0xpool", from_block=0, to_block=9, out_path=out_path,
        client_factory=lambda: httpx.Client(), rpc_url="http://fake-rpc.invalid",
        chunk_size=10, max_workers=1, flush_every_n_chunks=1,
        max_retries=1, retry_backoff_seconds=0.001, fetch_fn=connection_refused,
        train_assemble_fn=lambda p: (None, None, 0, {}),
    )
    app = LatentEdgeApp(start_screen=screen)

    async with app.run_test() as pilot:
        for _ in range(50):
            await pilot.pause(0.01)
            if screen.error is not None:
                break

    assert screen.error is not None
    assert "internet connection" in screen.error
    assert "Errno" not in screen.error


@pytest.mark.asyncio
async def test_ingest_screen_shows_completion_prompt(tmp_path: Path):
    out_path = tmp_path / "swaps.parquet"
    screen = IngestScreen(
        pool_address="0xpool", from_block=0, to_block=9, out_path=out_path,
        client_factory=lambda: httpx.Client(), rpc_url="http://fake",
        chunk_size=10, max_workers=1, flush_every_n_chunks=1,
        max_retries=1, retry_backoff_seconds=0.001, fetch_fn=_fake_fetch,
        train_assemble_fn=lambda p: (None, None, 0, {}),
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


def _fake_assemble(swaps_path: Path):
    return [[0.0]], [0.0], 1, {}


@pytest.mark.asyncio
async def test_ingest_screen_train_key_pushes_train_screen(tmp_path: Path):
    out_path = tmp_path / "swaps.parquet"
    model_out_path = tmp_path / "model.safetensors"
    screen = IngestScreen(
        pool_address="0xpool", from_block=0, to_block=9, out_path=out_path,
        client_factory=lambda: httpx.Client(), rpc_url="http://fake",
        chunk_size=10, max_workers=1, flush_every_n_chunks=1,
        max_retries=1, retry_backoff_seconds=0.001, fetch_fn=_fake_fetch,
        train_assemble_fn=_fake_assemble, model_out_path=model_out_path,
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
    # Must use the configured model path, not TrainScreen's own default —
    # otherwise pushing T here would write over the real
    # data/model.safetensors on disk.
    assert active_screen.out_path == model_out_path


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
        train_assemble_fn=lambda p: (None, None, 0, {}),
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
