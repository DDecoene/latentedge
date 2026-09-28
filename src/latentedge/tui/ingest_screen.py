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

from latentedge.ingest.chunked import (
    DEFAULT_CONCURRENCY_COOLDOWN_SECONDS,
    DEFAULT_MAX_RATE_LIMIT_BACKOFF_SECONDS,
    FetchFn,
    IngestCancelled,
)
from latentedge.ingest.chunked import ingest_range as default_ingest_range
from latentedge.ingest.progress import read_progress, uncovered_gaps
from latentedge.ingest.rpc_logs import describe_error, fetch_swaps
from latentedge.tui.train_screen import DEFAULT_MODEL_OUT_PATH, TrainAssembleFn, TrainScreen
from latentedge.tui.widgets import LogPanel, ProgressPanel, StatsPanel, ThreadPanel

STATS_REFRESH_INTERVAL_SECONDS = 1.0
# A run with nothing to report yet (no chunk has completed) isn't
# "stalled" — only flag it once enough time has passed that a healthy
# run would normally have made progress.
STALL_THRESHOLD_SECONDS = 5.0


class IngestScreen(Screen[None]):
    BINDINGS = [
        Binding("t", "train_now", "Train now", show=False),
        Binding("q", "exit_now", "Exit", show=False),
    ]
    CSS = """
    #ingest-top-row { height: auto; }
    #ingest-left-col { width: 1fr; height: auto; }
    #ingest-threads { width: 1fr; }
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
        concurrency_cooldown_seconds: float = DEFAULT_CONCURRENCY_COOLDOWN_SECONDS,
        max_rate_limit_backoff_seconds: float = DEFAULT_MAX_RATE_LIMIT_BACKOFF_SECONDS,
        model_out_path: Path = DEFAULT_MODEL_OUT_PATH,
        train_epochs: int = 100,
        train_after_ingest: bool = False,
        ingest_fn: Callable[..., int] = default_ingest_range,
        fetch_fn: FetchFn = fetch_swaps,
        time_fn: Callable[[], float] = time.monotonic,
        remaining_ranges: list[tuple[int, int]] | None = None,
    ) -> None:
        super().__init__()
        self.pool_address = pool_address
        self.from_block = from_block
        self.to_block = to_block
        self.out_path = out_path
        # Ranges still queued to run after this one, in order — set when
        # the CLI found previously-skipped history to backfill ahead of
        # the range actually requested. By convention the requested range
        # is always last, so "there's something queued after this one"
        # is exactly what makes this particular leg a backfill leg,
        # rather than needing a separate flag the two could drift out of
        # sync with.
        self.remaining_ranges = list(remaining_ranges) if remaining_ranges else []
        self.is_backfill_leg = bool(self.remaining_ranges)
        # Everything that scrolls off the top of the on-screen log panel
        # is still available here afterward — the panel itself only ever
        # keeps its last MAX_LOG_LINES, but a long run's retry/timing
        # history is exactly what's needed to diagnose why it felt slow.
        self.log_path = Path(str(out_path) + ".ingest.log")
        self.client_factory = client_factory
        self.rpc_url = rpc_url
        self.chunk_size = chunk_size
        self.max_workers = max_workers
        self.flush_every_n_chunks = flush_every_n_chunks
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.concurrency_cooldown_seconds = concurrency_cooldown_seconds
        self.max_rate_limit_backoff_seconds = max_rate_limit_backoff_seconds
        self.train_assemble_fn = train_assemble_fn
        self.model_out_path = model_out_path
        self.train_epochs = train_epochs
        self.train_after_ingest = train_after_ingest
        self.ingest_fn = ingest_fn
        self.fetch_fn = fetch_fn
        self.time_fn = time_fn

        self.is_complete = False
        self.total_written: int | None = None
        self.error: str | None = None
        self.retry_count = 0
        # Cumulative blocks accounted for (already-covered + fetched),
        # tracked as a running count rather than derived from a chunk's
        # position — gap-fill means chunk_end no longer maps directly to
        # "blocks completed since from_block".
        self._completed = 0
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
        self._concurrency_limit = max_workers
        # Set by request_stop() (ctrl+q) to cooperatively unwind the
        # background ingest thread's worker pool instead of exiting the
        # app immediately — an immediate app.exit() would leave those
        # threads calling call_from_thread against an event loop that no
        # longer exists, which hangs them (and the process) forever.
        self._cancel_event = threading.Event()

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
            yield ThreadPanel(id="ingest-threads")
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
        unit_label = f"resuming ({self._completed} blocks already ingested)" if self._completed > 0 else "starting..."

        progress_panel = self.query_one("#ingest-progress", ProgressPanel)
        progress_panel.update_progress(
            completed=self._completed, total=total, unit_label=unit_label,
            rate_per_sec=0.0, rate_unit="blocks/sec",
        )
        self.set_interval(STATS_REFRESH_INTERVAL_SECONDS, self._refresh_disk_stats)

        # A backfill leg needs to stay visibly distinct for its whole
        # run, not just as a one-line log entry at the start — the panel
        # border and action bar are what's actually on screen the rest
        # of the time, and the log line scrolls out of view within
        # seconds on a busy run.
        if self.is_backfill_leg:
            backfill_note = f", backfilling previously-skipped history ({len(self.remaining_ranges)} range(s) queued after this) "
            queued_note = "range" if len(self.remaining_ranges) == 1 else "ranges"
            progress_panel.border_title = f"Progress — Backfilling skipped history ({len(self.remaining_ranges)} {queued_note} queued after this)"
            self.query_one("#ingest-action-bar", Static).update(
                "Backfilling previously-skipped history before the range you requested — this isn't the run you asked for yet."
            )
        else:
            backfill_note = ""
            progress_panel.border_title = "Progress — requested range"

        self._log(
            f"starting: blocks {self.from_block}-{self.to_block}{backfill_note}, chunk_size={self.chunk_size}, "
            f"max_workers={self.max_workers}, concurrency_cooldown_seconds={self.concurrency_cooldown_seconds} "
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

        def on_concurrency_change(new_limit: int) -> None:
            self.app.call_from_thread(self._handle_concurrency_change, new_limit)

        try:
            with self.client_factory() as client:
                total = self.ingest_fn(
                    self.pool_address, self.from_block, self.to_block, self.out_path,
                    client, self.rpc_url,
                    chunk_size=self.chunk_size, max_retries=self.max_retries,
                    retry_backoff_seconds=self.retry_backoff_seconds,
                    max_workers=self.max_workers,
                    flush_every_n_chunks=self.flush_every_n_chunks,
                    concurrency_cooldown_seconds=self.concurrency_cooldown_seconds,
                    max_rate_limit_backoff_seconds=self.max_rate_limit_backoff_seconds,
                    on_progress=on_progress, on_retry=on_retry,
                    on_queue_status=on_queue_status, on_worker_status=on_worker_status,
                    on_concurrency_change=on_concurrency_change,
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
            completed=completed, total=total, unit_label=f"block {chunk_end}",
            rate_per_sec=rate, rate_unit="blocks/sec",
        )
        self._log(f"blocks {chunk_start}-{chunk_end}: {count} swaps")

    def _handle_retry(
        self, chunk_start: int, chunk_end: int, attempt: int, max_retries: int | None,
        sleep_seconds: float, error_message: str,
    ) -> None:
        self.retry_count += 1
        label = f"retry {attempt}/{max_retries}" if max_retries is not None else f"rate limited, retry {attempt} (retrying until it clears)"
        self._log(f"blocks {chunk_start}-{chunk_end}: {label} ({error_message}) — waiting {sleep_seconds:.1f}s")
        self._refresh_disk_stats()

    def _handle_queue_status(self, buffered_count: int, blocking_chunk_start: int | None) -> None:
        self._buffered_count = buffered_count
        self._blocking_chunk_start = blocking_chunk_start
        self._refresh_disk_stats()

    def _handle_concurrency_change(self, new_limit: int) -> None:
        direction = "throttled down to" if new_limit < self._concurrency_limit else "raised to"
        self._concurrency_limit = new_limit
        self._log(f"concurrency {direction} {new_limit}/{self.max_workers}")
        self._refresh_disk_stats()

    def _handle_worker_status(self, slot: int, chunk_start: int, chunk_end: int, status: str) -> None:
        if status == "idle":
            detail = "[dim]○ idle[/dim]"
        elif status == "fetching":
            detail = f"[green]● blocks {chunk_start}-{chunk_end} — fetching[/green]"
        elif status == "waiting":
            # Blocked on the concurrency gate, not the network — the
            # visible sign the auto-throttle is actually doing something.
            detail = f"[cyan]◐ blocks {chunk_start}-{chunk_end} — waiting for a free slot[/cyan]"
        else:
            detail = f"[yellow]● blocks {chunk_start}-{chunk_end} — {status}[/yellow]"
        self.query_one("#ingest-threads", ThreadPanel).update_worker(slot, detail)

    def _refresh_disk_stats(self) -> None:
        file_size = self.out_path.stat().st_size if self.out_path.exists() else 0
        free_bytes = shutil.disk_usage(self.out_path.parent).free if self.out_path.parent.exists() else 0
        stalled_seconds = self.time_fn() - self._last_progress_time
        stalled = f"[red]{stalled_seconds:.0f}s[/red]" if stalled_seconds >= STALL_THRESHOLD_SECONDS else "[dim]no[/dim]"
        buffered = (
            f"[yellow]{self._buffered_count} (waiting on block {self._blocking_chunk_start})[/yellow]"
            if self._buffered_count > 0
            else "0"
        )
        retries = f"[yellow]{self.retry_count}[/yellow]" if self.retry_count > 0 else "0"
        concurrency = (
            f"[yellow]{self._concurrency_limit}/{self.max_workers}[/yellow]"
            if self._concurrency_limit < self.max_workers
            else f"{self._concurrency_limit}/{self.max_workers}"
        )
        self.query_one("#ingest-stats", StatsPanel).update_stats([
            ("File size", f"{file_size / 1_048_576:.1f} MB"),
            ("Est. final size", self._format_estimated_final_size(file_size)),
            ("Free disk", f"{free_bytes / 1_073_741_824:.1f} GB"),
            ("Retries", retries),
            ("Stalled", stalled),
            ("Buffered", buffered),
            ("Concurrency", concurrency),
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
        self._log(f"complete: wrote {total} swaps, {self.retry_count} retries total")
        self._refresh_disk_stats()
        range_total = max(self.to_block - self.from_block + 1, 0)
        self.query_one("#ingest-progress", ProgressPanel).update_progress(
            completed=range_total, total=range_total, unit_label="complete",
            rate_per_sec=0.0, rate_unit="blocks/sec",
        )

        if self.remaining_ranges:
            # A backfill leg auto-continues into the next queued range
            # (and eventually the range actually requested) without
            # stopping for the T/Q prompt below — the user asked for one
            # ingest run, not one prompt per previously-skipped gap.
            next_from, next_to = self.remaining_ranges[0]
            rest = self.remaining_ranges[1:]
            self._log(f"continuing: {len(rest)} range(s) still queued after this one")
            self.app.switch_screen(self._build_next_screen(next_from, next_to, rest))
            return

        if self.train_after_ingest:
            self._log("train_after_ingest is on — starting training now")
            self.query_one("#ingest-action-bar", Static).update(
                f"Ingestion complete — wrote {total} swaps to {self.out_path}. Starting training now."
            )
            self._push_train_screen()
            return

        self.query_one("#ingest-action-bar", Static).update(
            f"Ingestion complete — wrote {total} swaps to {self.out_path}. "
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
        self._log(f"terminated: stopped by user, {total_written} swaps written this run before stopping")
        self.app.exit()

    def _build_next_screen(self, from_block: int, to_block: int, remaining_ranges: list[tuple[int, int]]) -> "IngestScreen":
        return IngestScreen(
            pool_address=self.pool_address, from_block=from_block, to_block=to_block,
            out_path=self.out_path, client_factory=self.client_factory, rpc_url=self.rpc_url,
            chunk_size=self.chunk_size, max_workers=self.max_workers,
            flush_every_n_chunks=self.flush_every_n_chunks, max_retries=self.max_retries,
            retry_backoff_seconds=self.retry_backoff_seconds, train_assemble_fn=self.train_assemble_fn,
            concurrency_cooldown_seconds=self.concurrency_cooldown_seconds,
            max_rate_limit_backoff_seconds=self.max_rate_limit_backoff_seconds,
            model_out_path=self.model_out_path, train_epochs=self.train_epochs,
            train_after_ingest=self.train_after_ingest, ingest_fn=self.ingest_fn,
            fetch_fn=self.fetch_fn, time_fn=self.time_fn, remaining_ranges=remaining_ranges,
        )

    def _push_train_screen(self) -> None:
        self.app.push_screen(TrainScreen(
            swaps_path=self.out_path, out_path=self.model_out_path, epochs=self.train_epochs,
            assemble_fn=self.train_assemble_fn,
        ))

    def action_train_now(self) -> None:
        if not self.is_complete:
            return
        self._push_train_screen()

    def action_exit_now(self) -> None:
        if not (self.is_complete or self.error is not None):
            return
        self.app.exit()
