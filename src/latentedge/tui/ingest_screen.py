"""The ingest command's progress screen."""

import shutil
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

import httpx
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Static

from latentedge.ingest.chunked import FetchFn
from latentedge.ingest.chunked import ingest_range as default_ingest_range
from latentedge.ingest.progress import read_progress, uncovered_gaps
from latentedge.ingest.rpc_logs import describe_error, fetch_swaps
from latentedge.tui.train_screen import DEFAULT_MODEL_OUT_PATH, TrainAssembleFn, TrainScreen
from latentedge.tui.widgets import LogPanel, ProgressPanel, StatsPanel, ThreadPanel

RATE_WINDOW_SIZE = 20
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
        model_out_path: Path = DEFAULT_MODEL_OUT_PATH,
        train_epochs: int = 100,
        ingest_fn: Callable[..., int] = default_ingest_range,
        fetch_fn: FetchFn = fetch_swaps,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__()
        self.pool_address = pool_address
        self.from_block = from_block
        self.to_block = to_block
        self.out_path = out_path
        self.client_factory = client_factory
        self.rpc_url = rpc_url
        self.chunk_size = chunk_size
        self.max_workers = max_workers
        self.flush_every_n_chunks = flush_every_n_chunks
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.train_assemble_fn = train_assemble_fn
        self.model_out_path = model_out_path
        self.train_epochs = train_epochs
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
        self._rate_window: deque[tuple[float, int]] = deque(maxlen=RATE_WINDOW_SIZE)
        self._last_progress_time = self.time_fn()
        self._buffered_count = 0
        self._blocking_chunk_start: int | None = None

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

        self.query_one("#ingest-progress", ProgressPanel).update_progress(
            completed=self._completed, total=total, unit_label=unit_label,
            rate_per_sec=0.0, rate_unit="blocks/sec",
        )
        self.set_interval(STATS_REFRESH_INTERVAL_SECONDS, self._refresh_disk_stats)
        self.run_worker(self._run_ingest, thread=True, exclusive=True)

    def _run_ingest(self) -> None:
        def on_progress(chunk_start: int, chunk_end: int, count: int) -> None:
            self.app.call_from_thread(self._handle_progress, chunk_start, chunk_end, count)

        def on_retry(
            chunk_start: int, chunk_end: int, attempt: int, max_retries: int,
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

        try:
            with self.client_factory() as client:
                total = self.ingest_fn(
                    self.pool_address, self.from_block, self.to_block, self.out_path,
                    client, self.rpc_url,
                    chunk_size=self.chunk_size, max_retries=self.max_retries,
                    retry_backoff_seconds=self.retry_backoff_seconds,
                    max_workers=self.max_workers,
                    flush_every_n_chunks=self.flush_every_n_chunks,
                    on_progress=on_progress, on_retry=on_retry,
                    on_queue_status=on_queue_status, on_worker_status=on_worker_status,
                    fetch_fn=self.fetch_fn,
                )
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
        self._rate_window.append((now, completed))

        rate = 0.0
        if len(self._rate_window) >= 2:
            (t0, c0), (t1, c1) = self._rate_window[0], self._rate_window[-1]
            elapsed = t1 - t0
            if elapsed > 0:
                rate = (c1 - c0) / elapsed

        self.query_one("#ingest-progress", ProgressPanel).update_progress(
            completed=completed, total=total, unit_label=f"block {chunk_end}",
            rate_per_sec=rate, rate_unit="blocks/sec",
        )
        self.query_one("#ingest-log", LogPanel).log_line(
            f"blocks {chunk_start}-{chunk_end}: {count} swaps"
        )

    def _handle_retry(
        self, chunk_start: int, chunk_end: int, attempt: int, max_retries: int,
        sleep_seconds: float, error_message: str,
    ) -> None:
        self.retry_count += 1
        self.query_one("#ingest-log", LogPanel).log_line(
            f"blocks {chunk_start}-{chunk_end}: retry {attempt}/{max_retries} "
            f"({error_message}) — waiting {sleep_seconds:.1f}s"
        )
        self._refresh_disk_stats()

    def _handle_queue_status(self, buffered_count: int, blocking_chunk_start: int | None) -> None:
        self._buffered_count = buffered_count
        self._blocking_chunk_start = blocking_chunk_start
        self._refresh_disk_stats()

    def _handle_worker_status(self, slot: int, chunk_start: int, chunk_end: int, status: str) -> None:
        if status == "idle":
            detail = "[dim]○ idle[/dim]"
        elif status == "fetching":
            detail = f"[green]● blocks {chunk_start}-{chunk_end} — fetching[/green]"
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
        self.query_one("#ingest-stats", StatsPanel).update_stats([
            ("File size", f"{file_size / 1_048_576:.1f} MB"),
            ("Free disk", f"{free_bytes / 1_073_741_824:.1f} GB"),
            ("Retries", retries),
            ("Stalled", stalled),
            ("Buffered", buffered),
        ])

    def _handle_complete(self, total: int) -> None:
        self.is_complete = True
        self.total_written = total
        self._refresh_disk_stats()
        range_total = max(self.to_block - self.from_block + 1, 0)
        self.query_one("#ingest-progress", ProgressPanel).update_progress(
            completed=range_total, total=range_total, unit_label="complete",
            rate_per_sec=0.0, rate_unit="blocks/sec",
        )
        self.query_one("#ingest-action-bar", Static).update(
            f"Ingestion complete — wrote {total} swaps to {self.out_path}. "
            "[T] Train now   [Q] Exit"
        )

    def _handle_error(self, message: str) -> None:
        self.error = message
        self.query_one("#ingest-log", LogPanel).log_line(f"ERROR: {message}")
        self.query_one("#ingest-action-bar", Static).update(f"Ingestion failed: {message}. [Q] Exit")

    def action_train_now(self) -> None:
        if not self.is_complete:
            return
        self.app.push_screen(TrainScreen(
            swaps_path=self.out_path, out_path=self.model_out_path, epochs=self.train_epochs,
            assemble_fn=self.train_assemble_fn,
        ))

    def action_exit_now(self) -> None:
        if not (self.is_complete or self.error is not None):
            return
        self.app.exit()
