"""The study command's live screen: every (horizon, seed, fold) run streams
into a table, and each horizon's pooled result lands in the log."""

import threading
from collections.abc import Callable
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import DataTable, Static

from latentedge.study import StudyObserver, describe_summary
from latentedge.tui.widgets import LogPanel, ProgressPanel

# Takes the observer to report to; returns {"path", "rows", "summaries"}.
StudyFn = Callable[[StudyObserver], dict]

COLUMNS = ("Horizon", "Seed", "Fold", "Bars", "Gross corr", "Cost corr")


class StudyCancelled(Exception):
    """Raised from an observer callback on the worker thread to unwind it after ctrl+q."""


def _cells(row: dict[str, Any]) -> tuple[str, ...]:
    return (
        f"{row['horizon_minutes']} min", str(row["seed"]), str(row["fold"]), f"{row['n']:,}",
        f"{row['gross_correlation']:+.3f}", f"{row['cost_correlation']:+.3f}",
    )


class StudyScreen(Screen[None]):
    BINDINGS = [Binding("q", "exit_now", "Exit", show=False)]
    CSS = """
    #study-table { height: 1fr; border: round $primary; }
    #study-log { height: 10; }
    """

    def __init__(self, study_fn: StudyFn) -> None:
        super().__init__()
        self.study_fn = study_fn
        self.is_complete = False
        self.result: dict | None = None
        self.error: str | None = None
        self._cancel_event = threading.Event()

    def compose(self) -> ComposeResult:
        yield ProgressPanel(id="study-progress")
        yield DataTable(id="study-table")
        yield LogPanel(id="study-log")
        yield Static("", id="study-action-bar")

    def on_mount(self) -> None:
        table = self.query_one("#study-table", DataTable)
        table.add_columns(*COLUMNS)
        table.border_title = "Runs"
        self.query_one("#study-progress", ProgressPanel).update_progress(
            completed=0, total=0, unit_label="preparing data...", rate_per_sec=0.0, rate_unit="runs/sec",
        )
        self.run_worker(self._run_study, thread=True, exclusive=True)

    def _run_study(self) -> None:
        def check_cancelled() -> None:
            if self._cancel_event.is_set():
                raise StudyCancelled

        def on_stage(label: str, done: int = 0, total: int = 0) -> None:
            check_cancelled()
            self.app.call_from_thread(self._handle_stage, label, done, total)

        def on_row(row: dict[str, Any]) -> None:
            check_cancelled()
            self.app.call_from_thread(self._handle_row, row)

        def on_summary(summary: dict[str, Any]) -> None:
            self.app.call_from_thread(self._handle_summary, summary)

        try:
            result = self.study_fn(StudyObserver(on_stage=on_stage, on_row=on_row, on_summary=on_summary))
        except StudyCancelled:
            self.app.call_from_thread(self.app.exit)
            return
        except Exception as exc:
            self.app.call_from_thread(self._handle_error, getattr(exc, "message", None) or str(exc))
            return
        self.app.call_from_thread(self._handle_complete, result)

    def _handle_stage(self, label: str, done: int, total: int) -> None:
        self.query_one("#study-progress", ProgressPanel).update_progress(
            completed=done, total=total, unit_label=label, rate_per_sec=0.0, rate_unit="runs/sec",
        )
        if total == 0:
            self.query_one("#study-log", LogPanel).log_line(label)

    def _handle_row(self, row: dict[str, Any]) -> None:
        self.query_one("#study-table", DataTable).add_row(*_cells(row))

    def _handle_summary(self, summary: dict[str, Any]) -> None:
        self.query_one("#study-log", LogPanel).log_line(describe_summary(summary))

    def _handle_complete(self, result: dict) -> None:
        self.is_complete = True
        self.result = result
        self.query_one("#study-progress", ProgressPanel).update_progress(
            completed=len(result["rows"]), total=len(result["rows"]), unit_label="complete",
            rate_per_sec=0.0, rate_unit="runs/sec",
        )
        self.query_one("#study-log", LogPanel).log_line(f"study complete — {result['path']}")
        self.query_one("#study-action-bar", Static).update(f"Study complete. Saved to {result['path']}. Press [b]Q[/b] to exit.")

    def _handle_error(self, message: str) -> None:
        self.error = message
        self.query_one("#study-log", LogPanel).log_line(f"ERROR: {message}")
        self.query_one("#study-action-bar", Static).update(f"Study failed: {message}. Press [b]Q[/b] to exit.")

    def request_stop(self) -> bool:
        if self.is_complete or self.error is not None:
            return False
        if not self._cancel_event.is_set():
            self._cancel_event.set()
            self.query_one("#study-action-bar", Static).update("Stopping — finishing the current run...")
        return True

    def action_exit_now(self) -> None:
        if not (self.is_complete or self.error is not None):
            return
        self.app.exit()
