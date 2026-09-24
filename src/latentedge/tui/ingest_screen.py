"""The ingest command's progress screen."""

import shutil
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

import httpx
from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Static

from latentedge.ingest.chunked import ingest_range as default_ingest_range
from latentedge.ingest.rpc_logs import fetch_swaps
from latentedge.tui.train_screen import TrainScreen
from latentedge.tui.widgets import LogPanel, ProgressPanel, StatsPanel

RATE_WINDOW_SIZE = 20
STATS_REFRESH_INTERVAL_SECONDS = 1.0


class IngestScreen(Screen):
    BINDINGS = [
        Binding("t", "train_now", "Train now", show=False),
        Binding("q", "exit_now", "Exit", show=False),
    ]

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
        ingest_fn: Callable[..., int] = default_ingest_range,
        fetch_fn: Callable[..., list[object]] = fetch_swaps,
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
        self.ingest_fn = ingest_fn
        self.fetch_fn = fetch_fn
        self.time_fn = time_fn

        self.is_complete = False
        self.total_written: int | None = None
        self.error: str | None = None
        self.retry_count = 0
        self._rate_window: deque[tuple[float, int]] = deque(maxlen=RATE_WINDOW_SIZE)

    def compose(self) -> ComposeResult:
        yield ProgressPanel(id="ingest-progress")
        yield StatsPanel(id="ingest-stats")
        yield LogPanel(id="ingest-log")
        yield Static("", id="ingest-action-bar")

    def on_mount(self) -> None:
        total = max(self.to_block - self.from_block + 1, 0)
        self.query_one("#ingest-progress", ProgressPanel).update_progress(
            completed=0, total=total, unit_label="starting...",
            rate_per_sec=0.0, rate_unit="blocks/sec",
        )
        self.set_interval(STATS_REFRESH_INTERVAL_SECONDS, self._refresh_disk_stats)
        self.run_worker(self._run_ingest, thread=True, exclusive=True)

    def _run_ingest(self) -> None:
        def on_progress(chunk_start: int, chunk_end: int, count: int) -> None:
            self.app.call_from_thread(self._handle_progress, chunk_start, chunk_end, count)

        def on_retry() -> None:
            self.app.call_from_thread(self._handle_retry)

        try:
            with self.client_factory() as client:
                total = self.ingest_fn(
                    self.pool_address, self.from_block, self.to_block, self.out_path,
                    client, self.rpc_url,
                    chunk_size=self.chunk_size, max_retries=self.max_retries,
                    retry_backoff_seconds=self.retry_backoff_seconds,
                    max_workers=self.max_workers,
                    flush_every_n_chunks=self.flush_every_n_chunks,
                    on_progress=on_progress, on_retry=on_retry, fetch_fn=self.fetch_fn,
                )
        except Exception as exc:
            self.app.call_from_thread(self._handle_error, str(exc))
            return
        self.app.call_from_thread(self._handle_complete, total)

    def _handle_progress(self, chunk_start: int, chunk_end: int, count: int) -> None:
        total = max(self.to_block - self.from_block + 1, 0)
        completed = chunk_end - self.from_block + 1
        now = self.time_fn()
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

    def _handle_retry(self) -> None:
        self.retry_count += 1
        self._refresh_disk_stats()

    def _refresh_disk_stats(self) -> None:
        file_size = self.out_path.stat().st_size if self.out_path.exists() else 0
        free_bytes = shutil.disk_usage(self.out_path.parent).free if self.out_path.parent.exists() else 0
        self.query_one("#ingest-stats", StatsPanel).update_stats([
            ("File size", f"{file_size / 1_048_576:.1f} MB"),
            ("Free disk", f"{free_bytes / 1_073_741_824:.1f} GB"),
            ("Retries", str(self.retry_count)),
        ])

    def _handle_complete(self, total: int) -> None:
        self.is_complete = True
        self.total_written = total
        self._refresh_disk_stats()
        self.query_one("#ingest-action-bar", Static).update(
            f"Ingestion complete — wrote {total} swaps to {self.out_path}. "
            "[T] Train now   [Q] Exit"
        )

    def _handle_error(self, message: str) -> None:
        self.error = message
        self.query_one("#ingest-log", LogPanel).log_line(f"ERROR: {message}")

    def action_train_now(self) -> None:
        if not self.is_complete:
            return
        self.app.push_screen(TrainScreen(swaps_path=self.out_path))

    def action_exit_now(self) -> None:
        if not self.is_complete:
            return
        self.app.exit()
