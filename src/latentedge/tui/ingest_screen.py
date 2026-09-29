"""The ingest command's progress screen."""

import shutil
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import httpx
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Static
from textual_plotext import PlotextPlot

from latentedge.ingest.chunked import (
    DEFAULT_CONCURRENCY_COOLDOWN_SECONDS,
    DEFAULT_MAX_RATE_LIMIT_BACKOFF_SECONDS,
    DEFAULT_MAX_RPS,
    FetchFn,
    IngestCancelled,
)
from latentedge.ingest.chunked import ingest_range as default_ingest_range
from latentedge.env_file import update_env_value
from latentedge.ingest.progress import read_progress, uncovered_gaps
from latentedge.ingest.rate_limiter import RateLimiter
from latentedge.ingest.rpc_logs import describe_error, fetch_swaps
from latentedge.tui.backtest_screen import BacktestFn
from latentedge.tui.train_screen import DEFAULT_MODEL_OUT_PATH, TrainAssembleFn, TrainScreen
from latentedge.tui.widgets import LogPanel, ProgressPanel, StatsPanel, ThreadPanel

STATS_REFRESH_INTERVAL_SECONDS = 1.0
# Throughput is sampled once a second and each plotted point averages the
# last few samples: chunks land in bursts, so raw per-second deltas are
# too jagged to read.
RATE_HISTORY_SAMPLES = 300
RATE_SMOOTHING_SAMPLES = 5
# A run with nothing to report yet (no chunk has completed) isn't
# "stalled" — only flag it once enough time has passed that a healthy
# run would normally have made progress.
STALL_THRESHOLD_SECONDS = 5.0
BLOCKS_PER_DAY = 7_200  # 12-second blocks


def _format_blocks(blocks: int) -> str:
    """Thousands-separated block count, with the equivalent days once it is
    big enough for that to be meaningful."""
    text = f"{blocks:,}"
    if blocks >= BLOCKS_PER_DAY // 2:
        text += f" (~{blocks / BLOCKS_PER_DAY:.0f} days)"
    return text


class IngestScreen(Screen[None]):
    BINDINGS = [
        Binding("t", "train_now", "Train now", show=False),
        Binding("q", "exit_now", "Exit", show=False),
        Binding("up", "rate_up", "Rate +0.1", show=False, priority=True),
        Binding("down", "rate_down", "Rate -0.1", show=False, priority=True),
    ]
    CSS = """
    #ingest-top-row { height: auto; }
    #ingest-left-col { width: 1fr; height: auto; }
    #ingest-right-col { width: 1fr; height: auto; }
    #ingest-rate-plot { height: 10; }
    """

    def __init__(
        self,
        pool_address: str,
        from_block: int,
        to_block: int,
        out_path: Path,
        client_factory: Callable[[], httpx.Client],
        rpc_url: str,
        chunk_size: int,
        max_workers: int,
        flush_every_n_chunks: int,
        max_retries: int,
        retry_backoff_seconds: float,
        train_assemble_fn: TrainAssembleFn,
        max_rps: float = DEFAULT_MAX_RPS,
        fixed_rps: float | None = None,
        concurrency_cooldown_seconds: float = DEFAULT_CONCURRENCY_COOLDOWN_SECONDS,
        max_rate_limit_backoff_seconds: float = DEFAULT_MAX_RATE_LIMIT_BACKOFF_SECONDS,
        model_out_path: Path = DEFAULT_MODEL_OUT_PATH,
        train_epochs: int = 100,
        train_after_ingest: bool = False,
        backtest_fn: BacktestFn | None = None,
        backtest_after_train: bool = False,
        ingest_fn: Callable[..., int] = default_ingest_range,
        fetch_fn: FetchFn = fetch_swaps,
        time_fn: Callable[[], float] = time.monotonic,
        env_path: Path | None = None,
    ) -> None:
        super().__init__()
        self.pool_address = pool_address
        self.from_block = from_block
        self.to_block = to_block
        self.out_path = out_path
        # Everything that scrolls off the top of the on-screen log panel
        # is still available here afterward — the panel itself only ever
        # keeps its last MAX_LOG_LINES, but a long run's retry/timing
        # history is exactly what's needed to diagnose why it felt slow.
        self.log_path = Path(str(out_path) + ".ingest.log")
        self.client_factory = client_factory
        self.rpc_url = rpc_url
        self.chunk_size = chunk_size
        self.max_workers = max_workers
        self.max_rps = max_rps
        self.fixed_rps = fixed_rps
        self.flush_every_n_chunks = flush_every_n_chunks
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.concurrency_cooldown_seconds = concurrency_cooldown_seconds
        self.max_rate_limit_backoff_seconds = max_rate_limit_backoff_seconds
        self.train_assemble_fn = train_assemble_fn
        self.model_out_path = model_out_path
        self.train_epochs = train_epochs
        self.train_after_ingest = train_after_ingest
        self.backtest_fn = backtest_fn
        self.backtest_after_train = backtest_after_train
        self.ingest_fn = ingest_fn
        self.fetch_fn = fetch_fn
        self.time_fn = time_fn
        # Where a manually tuned fixed rate is written back, so the next
        # launch starts from it; None (tests) skips persisting.
        self.env_path = env_path
        self._rate_limiter: RateLimiter | None = None

        self.is_complete = False
        self.total_written: int | None = None
        self.error: str | None = None
        self.retry_count = 0
        # Cumulative blocks accounted for (already-covered + fetched),
        # tracked as a running count rather than derived from a chunk's
        # position — gap-fill means chunk_end no longer maps directly to
        # "blocks completed since from_block".
        self._completed = 0
        # Fixed startup snapshot for the stats panel — set once in
        # on_mount, never updated afterward (see on_mount's comment on
        # why "in file" is deliberately whole-file, not range-scoped).
        self._range_total_blocks = 0
        # Blocks of the *requested* range already on disk, kept live:
        # seeded from the progress file on mount, bumped per chunk.
        self._range_in_file = 0
        # Anchored on the *first real fetch* (set in _handle_progress),
        # not here at startup — a resumed run's already-on-disk blocks
        # are folded into self._completed before that first callback, so
        # _rate_start_completed's baseline already excludes them from
        # the rate/ETA math below; only genuinely fetched blocks and the
        # wall-clock time actually spent fetching them count.
        self._rate_start_time: float | None = None
        self._rate_start_completed: int | None = None
        # File size at the same anchor point as the rate/ETA baseline
        # above — lets the size estimate use a bytes-per-block figure
        # measured only over blocks actually fetched this run, rather
        # than being skewed by whatever the file already held (e.g. from
        # other block ranges) before this run started.
        self._size_rate_start_bytes: int | None = None
        self._last_progress_time = self.time_fn()
        self._buffered_count = 0
        self._blocking_chunk_start: int | None = None
        self._rate_limit: float = fixed_rps if fixed_rps is not None else max_rps
        # Tracks the limiter's own ceiling, which can self-raise above
        # max_rps (the config default) once it proves there's real
        # headroom — distinct from self.max_rps, which never changes and
        # is only ever the starting point passed into ingest_fn.
        self._rate_ceiling: float = fixed_rps if fixed_rps is not None else max_rps
        # Set by request_stop() (ctrl+q) to cooperatively unwind the
        # background ingest thread's worker pool instead of exiting the
        # app immediately — an immediate app.exit() would leave those
        # threads calling call_from_thread against an event loop that no
        # longer exists, which hangs them (and the process) forever.
        self._cancel_event = threading.Event()
        # Set by the T key while ingest is still running: the same
        # cooperative stop as ctrl+q, but once it has flushed, hand off
        # to training instead of exiting.
        self._train_after_stop = False

    def _log(self, message: str) -> None:
        self.query_one("#ingest-log", LogPanel).log_line(message)
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        # Opened, written, and closed per line rather than held open —
        # a run killed mid-flight (the same event the concurrency
        # limiter's own persistence is trying to survive) must never
        # cost the retry/timing history that explains what happened.
        with self.log_path.open("a") as f:
            f.write(f"{timestamp} {message}\n")

    def compose(self) -> ComposeResult:
        with Horizontal(id="ingest-top-row"):
            with Vertical(id="ingest-left-col"):
                yield ProgressPanel(id="ingest-progress")
                yield StatsPanel(id="ingest-stats")
            with Vertical(id="ingest-right-col"):
                yield ThreadPanel(id="ingest-threads")
                yield PlotextPlot(id="ingest-rate-plot")
        yield LogPanel(id="ingest-log")
        yield Static("", id="ingest-action-bar")

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
        self._completed = max(total - uncovered, 0)
        unit_label = f"resuming ({self._completed:,} blocks already ingested)" if self._completed > 0 else "starting..."

        # A fixed startup snapshot, distinct from the live progress bar:
        # "in file" counts everything ever ingested (any range, not just
        # this one) so a run can be judged against the dataset's real
        # size, not just this request's slice of it.
        self._range_total_blocks = total
        self._range_in_file = self._range_total_blocks - sum(
            end - start + 1 for start, end in uncovered_gaps(intervals, self.from_block, self.to_block)
        )

        progress_panel = self.query_one("#ingest-progress", ProgressPanel)
        progress_panel.update_progress(
            completed=self._completed, total=total, unit_label=unit_label,
            rate_per_sec=0.0, rate_unit="blocks/sec",
        )
        self._refresh_disk_stats()
        self.query_one("#ingest-action-bar", Static).update(
            "Press [b]T[/b] to stop ingesting and train on what's ingested so far, or [b]ctrl+q[/b] to stop and exit."
        )
        self.set_interval(STATS_REFRESH_INTERVAL_SECONDS, self._refresh_disk_stats)
        self._rate_plot_completed = self._completed
        self._rate_deltas: list[int] = []
        self._rate_history: list[float] = []
        # Whole-run extremes of the plotted (smoothed) rate, kept apart
        # from the 5-minute window so they survive it scrolling past.
        self._rate_min: float | None = None
        self._rate_max: float | None = None
        self.set_interval(STATS_REFRESH_INTERVAL_SECONDS, self._sample_rate)
        self._draw_rate_plot()

        progress_panel.border_title = "Progress — requested range"

        self._log(
            f"starting: blocks {self.from_block:,}-{self.to_block:,}, chunk_size={self.chunk_size}, "
            f"max_workers={self.max_workers}, rate={'fixed ' + str(self.fixed_rps) if self.fixed_rps is not None else 'auto, max_rps=' + str(self.max_rps)}, concurrency_cooldown_seconds={self.concurrency_cooldown_seconds} "
            f"— full log at {self.log_path}"
        )
        self.run_worker(self._run_ingest, thread=True, exclusive=True)

    def _run_ingest(self) -> None:
        def on_progress(chunk_start: int, chunk_end: int, count: int) -> None:
            self.app.call_from_thread(self._handle_progress, chunk_start, chunk_end, count)

        def on_retry(
            chunk_start: int, chunk_end: int, attempt: int, max_retries: int | None,
            sleep_seconds: float, error_message: str,
        ) -> None:
            self.app.call_from_thread(
                self._handle_retry, chunk_start, chunk_end, attempt, max_retries,
                sleep_seconds, error_message,
            )

        def on_queue_status(buffered_count: int, blocking_chunk_start: int | None) -> None:
            self.app.call_from_thread(self._handle_queue_status, buffered_count, blocking_chunk_start)

        def on_worker_status(slot: int, chunk_start: int, chunk_end: int, status: str) -> None:
            self.app.call_from_thread(self._handle_worker_status, slot, chunk_start, chunk_end, status)

        def on_rate_change(new_rate: float) -> None:
            self.app.call_from_thread(self._handle_rate_change, new_rate)

        def on_ceiling_change(new_ceiling: float) -> None:
            self.app.call_from_thread(self._handle_ceiling_change, new_ceiling)

        def on_rate_limiter(limiter: RateLimiter) -> None:
            self._rate_limiter = limiter

        try:
            with self.client_factory() as client:
                total = self.ingest_fn(
                    self.pool_address, self.from_block, self.to_block, self.out_path,
                    client, self.rpc_url,
                    chunk_size=self.chunk_size, max_retries=self.max_retries,
                    retry_backoff_seconds=self.retry_backoff_seconds,
                    max_workers=self.max_workers, max_rps=self.max_rps, fixed_rps=self.fixed_rps,
                    flush_every_n_chunks=self.flush_every_n_chunks,
                    concurrency_cooldown_seconds=self.concurrency_cooldown_seconds,
                    max_rate_limit_backoff_seconds=self.max_rate_limit_backoff_seconds,
                    on_progress=on_progress, on_retry=on_retry,
                    on_queue_status=on_queue_status, on_worker_status=on_worker_status,
                    on_rate_change=on_rate_change, on_ceiling_change=on_ceiling_change,
                    on_rate_limiter=on_rate_limiter,
                    fetch_fn=self.fetch_fn,
                    cancel_event=self._cancel_event,
                )
        except IngestCancelled as exc:
            self.app.call_from_thread(self._handle_stopped, exc.total_written)
            return
        except Exception as exc:
            self.app.call_from_thread(self._handle_error, describe_error(self.rpc_url, exc))
            return
        self.app.call_from_thread(self._handle_complete, total)

    def _handle_progress(self, chunk_start: int, chunk_end: int, count: int) -> None:
        total = max(self.to_block - self.from_block + 1, 0)
        self._completed += chunk_end - chunk_start + 1
        completed = self._completed
        now = self.time_fn()
        self._last_progress_time = now
        self._range_in_file += chunk_end - chunk_start + 1

        # Cumulative average since the first real fetch, not a recent
        # window — a windowed rate swings wildly with a concurrency
        # burst right after a cooldown followed by a rate-limit stall,
        # even though both are normal, expected throttle behavior. The
        # cumulative average smooths that out and gives a stable ETA
        # extrapolated from total elapsed time and percent complete.
        if self._rate_start_time is None:
            self._rate_start_time = now
            self._rate_start_completed = completed
            self._size_rate_start_bytes = self.out_path.stat().st_size if self.out_path.exists() else 0
            rate = 0.0
        else:
            assert self._rate_start_completed is not None
            elapsed = now - self._rate_start_time
            blocks_fetched = completed - self._rate_start_completed
            rate = blocks_fetched / elapsed if elapsed > 0 else 0.0

        self.query_one("#ingest-progress", ProgressPanel).update_progress(
            completed=completed, total=total, unit_label=f"block {chunk_end:,}",
            rate_per_sec=rate, rate_unit="blocks/sec",
        )
        self._log(f"blocks {chunk_start:,}-{chunk_end:,}: {count:,} swaps")
        self._refresh_disk_stats()

    def _sample_rate(self) -> None:
        if self.is_complete or self._rate_start_time is None:
            return
        self._rate_deltas.append(self._completed - self._rate_plot_completed)
        self._rate_plot_completed = self._completed
        window = self._rate_deltas[-RATE_SMOOTHING_SAMPLES:]
        smoothed = sum(window) / len(window) / STATS_REFRESH_INTERVAL_SECONDS
        self._rate_history.append(smoothed)
        self._rate_min = smoothed if self._rate_min is None else min(self._rate_min, smoothed)
        self._rate_max = smoothed if self._rate_max is None else max(self._rate_max, smoothed)
        del self._rate_history[:-RATE_HISTORY_SAMPLES]
        del self._rate_deltas[:-RATE_SMOOTHING_SAMPLES]
        self._draw_rate_plot()

    def _draw_rate_plot(self) -> None:
        plot = self.query_one("#ingest-rate-plot", PlotextPlot)
        plot.plt.clear_data()
        title = "Blocks/sec (last 5 min)"
        if self._rate_min is not None and self._rate_max is not None:
            title += f" — run min {self._rate_min:.1f} / max {self._rate_max:.1f}"
        plot.plt.title(title)
        if self._rate_history:
            plot.plt.plot(list(range(-len(self._rate_history) + 1, 1)), self._rate_history)
        plot.refresh()

    def _handle_retry(
        self, chunk_start: int, chunk_end: int, attempt: int, max_retries: int | None,
        sleep_seconds: float, error_message: str,
    ) -> None:
        self.retry_count += 1
        label = f"retry {attempt}/{max_retries}" if max_retries is not None else f"rate limited, retry {attempt} (retrying until it clears)"
        self._log(f"blocks {chunk_start:,}-{chunk_end:,}: {label} ({error_message}) — waiting {sleep_seconds:.1f}s")
        self._refresh_disk_stats()

    def _handle_queue_status(self, buffered_count: int, blocking_chunk_start: int | None) -> None:
        self._buffered_count = buffered_count
        self._blocking_chunk_start = blocking_chunk_start
        self._refresh_disk_stats()

    def _handle_rate_change(self, new_rate: float) -> None:
        direction = "throttled down to" if new_rate < self._rate_limit else "raised to"
        self._rate_limit = new_rate
        self._log(f"rate {direction} {new_rate:.1f}/{self._rate_ceiling:.1f} req/s")
        self._refresh_disk_stats()

    def _handle_ceiling_change(self, new_ceiling: float) -> None:
        self._rate_ceiling = new_ceiling
        self._log(f"ceiling raised to {new_ceiling:.1f} req/s — real headroom found above the starting default")
        self._refresh_disk_stats()

    def _handle_worker_status(self, slot: int, chunk_start: int, chunk_end: int, status: str) -> None:
        if status == "idle":
            detail = "[dim]○ idle[/dim]"
        elif status == "fetching":
            detail = f"[green]● blocks {chunk_start:,}-{chunk_end:,} — fetching[/green]"
        else:
            detail = f"[yellow]● blocks {chunk_start:,}-{chunk_end:,} — {status}[/yellow]"
        self.query_one("#ingest-threads", ThreadPanel).update_worker(slot, detail)

    def _refresh_disk_stats(self) -> None:
        file_size = self.out_path.stat().st_size if self.out_path.exists() else 0
        free_bytes = shutil.disk_usage(self.out_path.parent).free if self.out_path.parent.exists() else 0
        stalled_seconds = self.time_fn() - self._last_progress_time
        stalled = f"[red]{stalled_seconds:.0f}s[/red]" if stalled_seconds >= STALL_THRESHOLD_SECONDS else "[dim]no[/dim]"
        buffered = (
            f"[yellow]{self._buffered_count} (waiting on block {self._blocking_chunk_start:,})[/yellow]"
            if self._buffered_count > 0
            else "0"
        )
        retries = f"[yellow]{self.retry_count}[/yellow]" if self.retry_count > 0 else "0"
        if self.fixed_rps is not None:
            rate = f"{self._rate_limit:.1f} req/s (fixed) [dim]↑/↓ = ±0.1[/dim]"
        else:
            rate = (
                f"[yellow]{self._rate_limit:.1f}/{self._rate_ceiling:.1f} req/s[/yellow]"
                if self._rate_limit < self._rate_ceiling
                else f"{self._rate_limit:.1f}/{self._rate_ceiling:.1f} req/s"
            )
        self.query_one("#ingest-stats", StatsPanel).update_stats([
            ("Blocks in range", _format_blocks(self._range_total_blocks)),
            ("Blocks in file", _format_blocks(self._range_in_file)),
            ("Remaining in range", _format_blocks(max(self._range_total_blocks - self._range_in_file, 0))),
            ("File size", f"{file_size / 1_048_576:.1f} MB"),
            ("Est. final size", self._format_estimated_final_size(file_size)),
            ("Free disk", f"{free_bytes / 1_073_741_824:.1f} GB"),
            ("Retries", retries),
            ("Stalled", stalled),
            ("Buffered", buffered),
            ("Rate", rate),
        ])

    def _format_estimated_final_size(self, file_size: int) -> str:
        total = max(self.to_block - self.from_block + 1, 0)
        remaining = max(total - self._completed, 0)
        if remaining == 0:
            return f"{file_size / 1_048_576:.1f} MB"
        if (
            self._rate_start_completed is None
            or self._size_rate_start_bytes is None
            or self._completed <= self._rate_start_completed
        ):
            return "-- (estimating...)"
        bytes_per_block = (file_size - self._size_rate_start_bytes) / (self._completed - self._rate_start_completed)
        estimated_final = file_size + bytes_per_block * remaining
        return f"{estimated_final / 1_048_576:.1f} MB"

    def _handle_complete(self, total: int) -> None:
        self.is_complete = True
        self.total_written = total
        self._log(f"complete: wrote {total:,} swaps, {self.retry_count} retries total")
        self._refresh_disk_stats()
        range_total = max(self.to_block - self.from_block + 1, 0)
        self.query_one("#ingest-progress", ProgressPanel).update_progress(
            completed=range_total, total=range_total, unit_label="complete",
            rate_per_sec=0.0, rate_unit="blocks/sec",
        )

        if self.train_after_ingest:
            self._log("train_after_ingest is on — starting training now")
            self.query_one("#ingest-action-bar", Static).update(
                f"Ingestion complete — wrote {total:,} swaps to {self.out_path}. Starting training now."
            )
            self._push_train_screen()
            return

        self.query_one("#ingest-action-bar", Static).update(
            f"Ingestion complete — wrote {total:,} swaps to {self.out_path}. "
            "Press [b]T[/b] to train now, or [b]Q[/b] to exit."
        )

    def _handle_error(self, message: str) -> None:
        self.error = message
        self._log(f"ERROR: {message}")
        self.query_one("#ingest-action-bar", Static).update(f"Ingestion failed: {message}. Press [b]Q[/b] to exit.")

    def request_stop(self) -> bool:
        """Called by the app on ctrl+q instead of exiting immediately.

        Returns True to tell the app "I'm handling this, don't exit yet"
        — the background thread notices cancel_event, lets in-flight
        chunks finish or bail out quickly, flushes whatever completed,
        then reports back via _handle_stopped, which is what actually
        exits the app. Returns False once there's no run left to stop
        (already complete/errored, or a stop is already in flight), so
        ctrl+q falls through to the app's normal immediate exit.
        """
        if self.is_complete or self.error is not None:
            return False
        if self._cancel_event.is_set():
            # Already stopping — ignore a repeated press rather than
            # forcing app.exit() while worker threads may still be
            # unwinding, which would reintroduce the exact hang this
            # exists to prevent.
            return True
        self._cancel_event.set()
        self._log("stopping: user requested termination (ctrl+q) — waiting for in-flight requests to finish and flushing progress")
        self.query_one("#ingest-action-bar", Static).update(
            "Terminating — waiting for in-flight requests to finish and flushing progress..."
        )
        return True

    def _handle_stopped(self, total_written: int) -> None:
        self._log(f"terminated: stopped by user, {total_written:,} swaps written this run before stopping")
        if self._train_after_stop:
            self._log("progress flushed — starting training on what's ingested")
            self.query_one("#ingest-action-bar", Static).update(
                f"Ingestion stopped — {total_written:,} swaps written this run. Starting training on what's ingested."
            )
            self._push_train_screen()
            return
        self.app.exit()

    def _push_train_screen(self) -> None:
        self.app.push_screen(TrainScreen(
            swaps_path=self.out_path, out_path=self.model_out_path, epochs=self.train_epochs,
            assemble_fn=self.train_assemble_fn, backtest_fn=self.backtest_fn,
            backtest_after_train=self.backtest_after_train,
        ))

    def action_rate_up(self) -> None:
        self._nudge_fixed_rate(+0.1)

    def action_rate_down(self) -> None:
        self._nudge_fixed_rate(-0.1)

    def _nudge_fixed_rate(self, delta: float) -> None:
        """Manual fine-tuning, fixed-rate mode only: the auto-throttle
        owns the rate otherwise."""
        limiter = self._rate_limiter
        if self.fixed_rps is None or limiter is None or self.is_complete:
            return
        applied = limiter.set_fixed_rate(limiter.rate + delta)
        self._rate_limit = self._rate_ceiling = applied
        self._log(f"fixed rate set to {applied:.1f} req/s")
        if self.env_path is not None:
            try:
                update_env_value(self.env_path, "LATENTEDGE_INGEST_FIXED_RPS", f"{applied:.1f}")
            except OSError as exc:
                self._log(f"could not update {self.env_path}: {exc}")
        self._refresh_disk_stats()

    def action_train_now(self) -> None:
        if self.is_complete:
            self._push_train_screen()
            return
        # Mid-run: stop cooperatively (in-flight requests finish, completed
        # chunks flush), then _handle_stopped starts training. Ignored once
        # a stop is already underway (a repeated press, or ctrl+q first)
        # or after a failure — there is nothing left to stop.
        if self.error is not None or self._cancel_event.is_set():
            return
        self._train_after_stop = True
        self._cancel_event.set()
        self._log("stopping: user requested training with what's ingested — waiting for in-flight requests to finish and flushing progress")
        self.query_one("#ingest-action-bar", Static).update(
            "Stopping ingest, then training — waiting for in-flight requests to finish and flushing progress..."
        )

    def action_exit_now(self) -> None:
        if not (self.is_complete or self.error is not None):
            return
        self.app.exit()
