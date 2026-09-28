import threading
import time
from pathlib import Path

import httpx
import numpy as np
import pytest
from textual.css.query import NoMatches

from latentedge.ingest.chunked import ingest_range
from latentedge.ingest.progress import read_progress, write_progress
from latentedge.ingest.rpc_logs import RateLimitError, RpcLogsError
from latentedge.schema import SwapRecord
from latentedge.store import read_swaps
from latentedge.training_data import AssembledTrainingData, SplitArrays
from latentedge.tui.app import LatentEdgeApp
from latentedge.tui.ingest_screen import IngestScreen
from latentedge.tui.train_screen import TrainScreen
from latentedge.tui.widgets import ProgressPanel


def _placeholder_assemble(swaps_path: Path) -> AssembledTrainingData:
    empty = SplitArrays(x=np.zeros((1, 1), dtype="float32"), y=np.zeros(1, dtype="float32"))
    return AssembledTrainingData(
        train=empty, validate=empty, test=empty, input_dim=1, stats={"net_return": (0.0, 1.0)}
    )


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
        train_assemble_fn=_placeholder_assemble,
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
    assert read_progress(out_path) == [(0, 29)]
    assert len(read_swaps(out_path)) == 3

    # Everything shown in the on-screen log panel must also land on disk
    # — the panel only keeps its last MAX_LOG_LINES, but the file is
    # where retry/timing history survives after the run ends.
    log_text = screen.log_path.read_text()
    assert "starting:" in log_text
    assert "blocks 0-9: 1 swaps" in log_text
    assert "complete: wrote 3 swaps" in log_text


@pytest.mark.asyncio
async def test_ingest_screen_chains_through_remaining_ranges_inside_the_tui(tmp_path: Path):
    # Regression test: a backfill must run entirely inside the TUI
    # dashboard, not as plain click.echo lines ahead of it — the CLI
    # hands a backfill gap plus the requested range to a single
    # IngestScreen via remaining_ranges, and the screen itself chains
    # from one to the next by switching to a fresh IngestScreen on
    # completion, never dropping back to plain terminal output.
    def slow_fetch(pool_address, from_block, to_block, client, rpc_url):
        time.sleep(0.05)
        return [_record(from_block)]

    out_path = tmp_path / "swaps.parquet"
    screen = IngestScreen(
        pool_address="0xpool", from_block=0, to_block=9, out_path=out_path,
        client_factory=lambda: httpx.Client(), rpc_url="http://fake",
        chunk_size=10, max_workers=1, flush_every_n_chunks=1,
        max_retries=1, retry_backoff_seconds=0.001, fetch_fn=slow_fetch,
        train_assemble_fn=_placeholder_assemble,
        remaining_ranges=[(100, 109)],
    )
    app = LatentEdgeApp(start_screen=screen)

    async with app.run_test() as pilot:
        # Wait for the first (backfill) screen to actually mount its
        # widgets — its one slow chunk (50ms) hasn't finished yet, so
        # this settles well before it could chain away to the next range.
        for _ in range(50):
            await pilot.pause(0.005)
            try:
                app.screen.query_one("#ingest-action-bar")
                break
            except NoMatches:
                continue
        assert not screen.is_complete  # still mid-flight — otherwise this assertion race is silently passing on luck
        backfill_action_bar_text = str(app.screen.query_one("#ingest-action-bar").content)
        backfill_border_title = app.screen.query_one("#ingest-progress", ProgressPanel).border_title

        for _ in range(50):
            await pilot.pause(0.01)
            active_screen = app.screen
            if isinstance(active_screen, IngestScreen) and active_screen.is_complete and not active_screen.remaining_ranges:
                break
        final_screen = app.screen

    assert isinstance(final_screen, IngestScreen)
    # The first leg (a backfill leg, since something was queued after
    # it) never shows the T/Q completion prompt — it auto-continues.
    assert screen.is_backfill_leg
    assert screen.remaining_ranges == [(100, 109)]
    # A backfill leg must stay visibly distinct on screen for its whole
    # run (border + action bar), not just as a log line that scrolls
    # away — the user reported not being able to tell from the TUI that
    # a backfill was even happening.
    assert "backfilling" in backfill_action_bar_text.lower()
    assert "backfilling" in backfill_border_title.lower()
    # The final leg is the originally requested range, run for real.
    assert final_screen.from_block == 100
    assert final_screen.to_block == 109
    assert not final_screen.is_backfill_leg
    assert final_screen.is_complete
    assert read_progress(out_path) == [(0, 9), (100, 109)]
    assert len(read_swaps(out_path)) == 2

    log_text = screen.log_path.read_text()
    assert "backfilling previously-skipped history" in log_text
    assert "continuing: 0 range(s) still queued after this one" in log_text


@pytest.mark.asyncio
async def test_ingest_screen_ctrl_q_stops_cleanly_and_exits_instead_of_hanging(tmp_path: Path):
    # Regression test: ctrl+q used to call app.exit() immediately while
    # the background ingest thread's worker pool was still alive. Those
    # workers reported progress via call_from_thread against an event
    # loop that had just stopped, which blocked forever — the app never
    # actually exited, and the process had to be killed externally.
    def slow_fetch(pool_address, from_block, to_block, client, rpc_url):
        time.sleep(0.03)
        return [_record(from_block)]

    out_path = tmp_path / "swaps.parquet"
    screen = IngestScreen(
        pool_address="0xpool", from_block=0, to_block=999, out_path=out_path,
        client_factory=lambda: httpx.Client(), rpc_url="http://fake",
        chunk_size=10, max_workers=1, flush_every_n_chunks=1,
        max_retries=1, retry_backoff_seconds=0.001, fetch_fn=slow_fetch,
        train_assemble_fn=_placeholder_assemble,
    )
    app = LatentEdgeApp(start_screen=screen)

    async with app.run_test() as pilot:
        await pilot.pause(0.1)  # let a few chunks complete first
        assert not screen.is_complete  # still running — 1000 blocks at 0.03s/chunk won't finish yet
        await pilot.press("ctrl+q")
        for _ in range(200):
            await pilot.pause(0.01)
            if not app.is_running:
                break

    assert not app.is_running  # actually exited — the whole point of this test
    assert not screen.is_complete
    log_text = screen.log_path.read_text()
    assert "stopping: user requested termination" in log_text
    assert "terminated: stopped by user" in log_text
    # Whatever completed before the stop must be safely on disk, not
    # discarded just because the run was cut short.
    assert read_progress(out_path)
    assert len(read_swaps(out_path)) >= 1


@pytest.mark.asyncio
async def test_ingest_screen_writes_retries_and_concurrency_changes_to_the_log_file(tmp_path: Path):
    attempts = {"count": 0}

    def one_rate_limit_then_fine(pool_address, from_block, to_block, client, rpc_url):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RateLimitError("simulated rate limit")
        return [_record(from_block)]

    out_path = tmp_path / "swaps.parquet"
    screen = IngestScreen(
        pool_address="0xpool", from_block=0, to_block=39, out_path=out_path,
        client_factory=lambda: httpx.Client(), rpc_url="http://fake",
        chunk_size=10, max_workers=4, flush_every_n_chunks=1,
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
    assert "throttled down to 2/4" in log_text


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
        train_assemble_fn=_placeholder_assemble,
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
async def test_ingest_screen_rate_is_a_cumulative_average_not_a_recent_window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Regression test: a rate computed from only the most recent progress
    # events swings wildly with normal throttle behavior — a burst of
    # concurrent fetches right after a cooldown looks fast, then a
    # rate-limit stall looks like a near-zero rate, even though both are
    # expected. Anchoring on the first real fetch and averaging
    # cumulatively since then stays stable across exactly that pattern.
    rates: list[float] = []
    original_update = ProgressPanel.update_progress

    def spy_update(self, completed, total, unit_label, rate_per_sec, rate_unit):  # type: ignore[no-untyped-def]
        rates.append(rate_per_sec)
        return original_update(self, completed, total, unit_label, rate_per_sec, rate_unit)

    monkeypatch.setattr(ProgressPanel, "update_progress", spy_update)

    clock = {"t": 0.0}
    screen = IngestScreen(
        pool_address="0xpool", from_block=0, to_block=99, out_path=tmp_path / "swaps.parquet",
        client_factory=lambda: httpx.Client(), rpc_url="http://fake",
        chunk_size=10, max_workers=1, flush_every_n_chunks=1,
        max_retries=1, retry_backoff_seconds=0.001, fetch_fn=_fake_fetch,
        train_assemble_fn=_placeholder_assemble, time_fn=lambda: clock["t"],
    )
    app = LatentEdgeApp(start_screen=screen)

    async with app.run_test():
        # A fast burst: three 10-block chunks land close together.
        clock["t"] = 1.0
        screen._handle_progress(0, 9, 1)
        clock["t"] = 3.0
        screen._handle_progress(10, 19, 1)
        clock["t"] = 5.0
        screen._handle_progress(20, 29, 1)

        # A rate-limit stall: the next chunk takes 20s longer to land.
        clock["t"] = 25.0
        screen._handle_progress(30, 39, 1)

        # Another fast chunk right after the stall clears.
        clock["t"] = 25.2
        screen._handle_progress(40, 49, 1)

    # rates[0] is on_mount's own initial (pre-fetch) update; the five
    # _handle_progress calls above are rates[1:].
    #
    # Cumulative average since the first real fetch at t=1.0, not a
    # recent-window delta — a windowed calc would show the stall as a
    # near-zero rate and the chunk right after it as an implausible spike.
    assert rates[1] == pytest.approx(0.0)  # no elapsed time yet on the very first real event
    assert rates[2] == pytest.approx(10 / 2.0)
    assert rates[3] == pytest.approx(20 / 4.0)
    assert rates[4] == pytest.approx(30 / 24.0)
    assert rates[5] == pytest.approx(40 / 24.2)


@pytest.mark.asyncio
async def test_ingest_screen_progress_accounts_for_a_covered_gap_crossed_mid_run(tmp_path: Path):
    # Regression test: once progress can have gaps, a chunk's position
    # (chunk_end) no longer maps directly to "blocks completed since
    # from_block" — the bar must track blocks actually fetched (plus
    # what was already covered), not the chunk's raw position.
    out_path = tmp_path / "swaps.parquet"
    write_progress(out_path, [(20, 29)])  # a gap sits in the middle of [0, 49]

    release_second_chunk = threading.Event()
    call_count = {"n": 0}

    def gated_fetch(pool_address, from_block, to_block, client, rpc_url):
        call_count["n"] += 1
        if call_count["n"] > 1:
            release_second_chunk.wait()
        return [_record(from_block)]

    screen = IngestScreen(
        pool_address="0xpool", from_block=0, to_block=49, out_path=out_path,
        client_factory=lambda: httpx.Client(), rpc_url="http://fake",
        chunk_size=10, max_workers=1, flush_every_n_chunks=1,
        max_retries=1, retry_backoff_seconds=0.001, fetch_fn=gated_fetch,
        train_assemble_fn=_placeholder_assemble,
    )
    app = LatentEdgeApp(start_screen=screen)

    try:
        async with app.run_test() as pilot:
            detail_text = ""
            for _ in range(50):
                await pilot.pause(0.01)
                detail_text = str(app.screen.query_one("#ingest-progress-detail").content)
                if "block 9" in detail_text:
                    break

            # Already-covered [20,29] (10 blocks) plus the just-fetched
            # [0,9] (10 blocks) = 20 of 50 total = 40%. The old
            # position-based formula would report chunk_end - from_block
            # + 1 = 10 -> 20%, understating real progress.
            assert "40%" in detail_text

            release_second_chunk.set()
            for _ in range(50):
                await pilot.pause(0.01)
                if screen.is_complete:
                    break
    finally:
        release_second_chunk.set()

    assert screen.is_complete


@pytest.mark.asyncio
async def test_ingest_screen_all_panels_are_visible_within_the_viewport(tmp_path: Path):
    # Regression test: panels with no explicit height default to filling
    # the whole screen, so each one stacks at a *virtual* position that
    # pushes everything before the last panel off-screen (negative y) and
    # the action bar off the bottom — only one panel's content is ever
    # actually visible even though compose() yields all of them.
    out_path = tmp_path / "swaps.parquet"
    screen = IngestScreen(
        pool_address="0xpool", from_block=0, to_block=9, out_path=out_path,
        client_factory=lambda: httpx.Client(), rpc_url="http://fake",
        chunk_size=10, max_workers=1, flush_every_n_chunks=1,
        max_retries=1, retry_backoff_seconds=0.001, fetch_fn=_fake_fetch,
        train_assemble_fn=_placeholder_assemble,
    )
    app = LatentEdgeApp(start_screen=screen)

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        regions = {
            widget_id: app.screen.query_one(f"#{widget_id}").region
            for widget_id in (
                "ingest-progress", "ingest-stats", "ingest-threads",
                "ingest-log", "ingest-action-bar",
            )
        }

    for widget_id, region in regions.items():
        assert region.y >= 0, f"{widget_id} is pushed above the viewport (y={region.y})"
        assert region.y + region.height <= 40, f"{widget_id} extends past the viewport"

    # The summary panels are compact (a handful of lines), not full-screen.
    assert regions["ingest-progress"].height <= 5
    assert regions["ingest-stats"].height <= 9
    assert regions["ingest-threads"].height <= 5


@pytest.mark.asyncio
async def test_ingest_screen_shows_retry_detail_in_log_and_stalled_stat(tmp_path: Path):
    attempts = {"count": 0}

    def flaky_fetch(pool_address, from_block, to_block, client, rpc_url):
        attempts["count"] += 1
        if attempts["count"] < 2:
            raise RpcLogsError("simulated rate limit")
        return [_record(from_block)]

    out_path = tmp_path / "swaps.parquet"
    screen = IngestScreen(
        pool_address="0xpool", from_block=0, to_block=9, out_path=out_path,
        client_factory=lambda: httpx.Client(), rpc_url="http://fake",
        chunk_size=10, max_workers=1, flush_every_n_chunks=1,
        max_retries=3, retry_backoff_seconds=0.001, fetch_fn=flaky_fetch,
        train_assemble_fn=_placeholder_assemble,
    )
    app = LatentEdgeApp(start_screen=screen)

    async with app.run_test() as pilot:
        for _ in range(50):
            await pilot.pause(0.01)
            if screen.is_complete:
                break
        log_text = "\n".join(
            str(line) for line in app.screen.query_one("#ingest-log-body").lines
        )
        stats_text = str(app.screen.query_one("#ingest-stats-body").content)

    assert screen.is_complete
    assert "retry 1/3" in log_text
    assert "simulated rate limit" in log_text
    assert "Stalled" in stats_text


@pytest.mark.asyncio
async def test_ingest_screen_shows_per_worker_status(tmp_path: Path):
    release_chunks = threading.Event()

    def gated_fetch(pool_address, from_block, to_block, client, rpc_url):
        release_chunks.wait()
        return [_record(from_block)]

    out_path = tmp_path / "swaps.parquet"
    screen = IngestScreen(
        pool_address="0xpool", from_block=0, to_block=19, out_path=out_path,
        client_factory=lambda: httpx.Client(), rpc_url="http://fake",
        chunk_size=10, max_workers=2, flush_every_n_chunks=1,
        max_retries=1, retry_backoff_seconds=0.001, fetch_fn=gated_fetch,
        train_assemble_fn=_placeholder_assemble,
    )
    app = LatentEdgeApp(start_screen=screen)

    try:
        async with app.run_test() as pilot:
            threads_text = ""
            for _ in range(50):
                await pilot.pause(0.01)
                threads_text = str(app.screen.query_one("#ingest-threads-body").content)
                if "Worker 0" in threads_text and "Worker 1" in threads_text:
                    break
            release_chunks.set()
            for _ in range(50):
                await pilot.pause(0.01)
                if screen.is_complete:
                    break
    finally:
        release_chunks.set()

    assert "Worker 0" in threads_text
    assert "Worker 1" in threads_text
    assert "fetching" in threads_text
    assert screen.is_complete


@pytest.mark.asyncio
async def test_ingest_screen_renders_a_worker_waiting_for_a_free_concurrency_slot(tmp_path: Path):
    # Regression test: a worker blocked on the concurrency gate must
    # render distinctly from one actively fetching — otherwise the
    # auto-throttle has no visible effect in the UI at all. Drives the
    # screen's own handler directly rather than racing real threads,
    # since forcing a real gate-block deterministically through
    # ingest_range's timing would be flaky.
    out_path = tmp_path / "swaps.parquet"
    screen = IngestScreen(
        pool_address="0xpool", from_block=0, to_block=9, out_path=out_path,
        client_factory=lambda: httpx.Client(), rpc_url="http://fake",
        chunk_size=10, max_workers=1, flush_every_n_chunks=1,
        max_retries=1, retry_backoff_seconds=0.001, fetch_fn=_fake_fetch,
        train_assemble_fn=_placeholder_assemble,
    )
    app = LatentEdgeApp(start_screen=screen)

    async with app.run_test() as pilot:
        for _ in range(50):
            await pilot.pause(0.01)
            if screen.is_complete:
                break
        screen._handle_worker_status(0, 0, 9, "waiting")
        await pilot.pause()
        threads_text = str(app.screen.query_one("#ingest-threads-body").content)
        fetching_text = threads_text
        screen._handle_worker_status(0, 0, 9, "fetching")
        await pilot.pause()
        fetching_text = str(app.screen.query_one("#ingest-threads-body").content)

    assert "waiting for a free slot" in threads_text
    assert "waiting for a free slot" not in fetching_text
    assert "fetching" in fetching_text


@pytest.mark.asyncio
async def test_ingest_screen_shows_concurrency_limit_after_a_throttle_down(tmp_path: Path):
    attempts = {"count": 0}

    def one_rate_limit_then_fine(pool_address, from_block, to_block, client, rpc_url):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RateLimitError("simulated rate limit")
        return [_record(from_block)]

    out_path = tmp_path / "swaps.parquet"
    screen = IngestScreen(
        pool_address="0xpool", from_block=0, to_block=39, out_path=out_path,
        client_factory=lambda: httpx.Client(), rpc_url="http://fake",
        chunk_size=10, max_workers=4, flush_every_n_chunks=1,
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
    assert "Concurrency" in stats_text
    assert "2/4" in stats_text


@pytest.mark.asyncio
async def test_ingest_screen_shows_estimated_final_file_size(tmp_path: Path):
    # Regression test: the stats panel should extrapolate the file's
    # final size from bytes written so far vs. blocks remaining, not
    # just show the current on-disk size.
    release_second_chunk = threading.Event()
    call_count = {"n": 0}

    def gated_fetch(pool_address, from_block, to_block, client, rpc_url):
        call_count["n"] += 1
        if call_count["n"] > 1:
            release_second_chunk.wait()
        return [_record(from_block)]

    out_path = tmp_path / "swaps.parquet"
    screen = IngestScreen(
        pool_address="0xpool", from_block=0, to_block=99, out_path=out_path,
        client_factory=lambda: httpx.Client(), rpc_url="http://fake",
        chunk_size=10, max_workers=1, flush_every_n_chunks=1,
        max_retries=1, retry_backoff_seconds=0.001, fetch_fn=gated_fetch,
        train_assemble_fn=_placeholder_assemble,
    )
    app = LatentEdgeApp(start_screen=screen)

    try:
        async with app.run_test() as pilot:
            stats_text = ""
            for _ in range(50):
                await pilot.pause(0.01)
                stats_text = str(app.screen.query_one("#ingest-stats-body").content)
                if "block 9" in str(app.screen.query_one("#ingest-progress-detail").content):
                    break
            # Only one chunk has landed so far — not yet enough to derive
            # a bytes-per-block rate.
            assert "estimating" in stats_text.lower()

            release_second_chunk.set()
            for _ in range(50):
                await pilot.pause(0.01)
                if screen.is_complete:
                    break
            stats_text = str(app.screen.query_one("#ingest-stats-body").content)
    finally:
        release_second_chunk.set()

    assert screen.is_complete
    assert "Est. final size" in stats_text
    assert "estimating" not in stats_text.lower()


@pytest.mark.asyncio
async def test_ingest_screen_handles_zero_length_range_without_hanging(tmp_path: Path):
    # An empty range (from_block > to_block) must complete immediately, not hang.
    out_path = tmp_path / "swaps.parquet"
    screen = IngestScreen(
        pool_address="0xpool", from_block=10, to_block=5, out_path=out_path,
        client_factory=lambda: httpx.Client(), rpc_url="http://fake",
        chunk_size=10, max_workers=1, flush_every_n_chunks=1,
        max_retries=1, retry_backoff_seconds=0.001, fetch_fn=_fake_fetch,
        train_assemble_fn=_placeholder_assemble,
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
        train_assemble_fn=_placeholder_assemble,
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
        train_assemble_fn=_placeholder_assemble,
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
    assert "exit" in action_bar_text.lower()
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
        train_assemble_fn=_placeholder_assemble,
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
        train_assemble_fn=_placeholder_assemble,
    )
    app = LatentEdgeApp(start_screen=screen)

    async with app.run_test() as pilot:
        for _ in range(50):
            await pilot.pause(0.01)
            if screen.is_complete:
                break
        action_bar = app.screen.query_one("#ingest-action-bar")
        text = str(action_bar.content)

    assert "train now" in text.lower()
    assert "exit" in text.lower()
    assert str(out_path) in text or "1" in text  # swap count or path present


def _fake_assemble(swaps_path: Path) -> AssembledTrainingData:
    split = SplitArrays(x=np.array([[0.0]], dtype="float32"), y=np.array([0.0], dtype="float32"))
    return AssembledTrainingData(train=split, validate=split, test=split, input_dim=1, stats={"net_return": (0.0, 1.0)})


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
async def test_ingest_screen_auto_starts_training_when_train_after_ingest_is_on(tmp_path: Path):
    out_path = tmp_path / "swaps.parquet"
    model_out_path = tmp_path / "model.safetensors"
    screen = IngestScreen(
        pool_address="0xpool", from_block=0, to_block=9, out_path=out_path,
        client_factory=lambda: httpx.Client(), rpc_url="http://fake",
        chunk_size=10, max_workers=1, flush_every_n_chunks=1,
        max_retries=1, retry_backoff_seconds=0.001, fetch_fn=_fake_fetch,
        train_assemble_fn=_fake_assemble, model_out_path=model_out_path,
        train_after_ingest=True,
    )
    app = LatentEdgeApp(start_screen=screen)

    async with app.run_test() as pilot:
        for _ in range(50):
            await pilot.pause(0.01)
            if isinstance(app.screen, TrainScreen):
                break
        active_screen = app.screen
        assert isinstance(active_screen, TrainScreen)
        # Let the pushed screen's own background worker finish before the
        # context manager tears the app down — otherwise a still-running
        # worker's queued call_from_thread update can hit an
        # already-unmounted widget tree (a test-only race: the fake train
        # function here completes near-instantly).
        for _ in range(50):
            await pilot.pause(0.01)
            if active_screen.is_complete:
                break

    # No [T] keypress — completion alone must have pushed TrainScreen.
    assert isinstance(active_screen, TrainScreen)
    assert active_screen.swaps_path == out_path
    assert active_screen.out_path == model_out_path


@pytest.mark.asyncio
async def test_ingest_screen_pauses_for_prompt_when_train_after_ingest_is_off(tmp_path: Path):
    out_path = tmp_path / "swaps.parquet"
    screen = IngestScreen(
        pool_address="0xpool", from_block=0, to_block=9, out_path=out_path,
        client_factory=lambda: httpx.Client(), rpc_url="http://fake",
        chunk_size=10, max_workers=1, flush_every_n_chunks=1,
        max_retries=1, retry_backoff_seconds=0.001, fetch_fn=_fake_fetch,
        train_assemble_fn=_fake_assemble, train_after_ingest=False,
    )
    app = LatentEdgeApp(start_screen=screen)

    async with app.run_test() as pilot:
        for _ in range(50):
            await pilot.pause(0.01)
            if screen.is_complete:
                break
        await pilot.pause()
        active_screen = app.screen

    assert not isinstance(active_screen, TrainScreen)


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
        train_assemble_fn=_placeholder_assemble,
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
