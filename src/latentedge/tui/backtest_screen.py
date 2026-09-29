"""The backtest command's progress screen."""

import threading
from collections.abc import Callable

from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Static

from latentedge.tui.widgets import LogPanel, ProgressPanel

# report(stage, done, total) -> None, may raise to abort; returns the
# backtest's summary dict.
BacktestFn = Callable[[Callable[..., None]], dict]


class BacktestCancelled(Exception):
    """Raised from the progress callback on the worker thread to unwind it after ctrl+q."""


def describe_summary(summary: dict) -> str:
    return (
        f"{summary['bars']:,} test bars: total return ${summary['total_return_usd']:,.2f} "
        f"({summary['total_return_fraction']:.2%}), max drawdown ${summary['max_drawdown_usd']:,.2f}, "
        f"win rate {summary['win_rate']:.1%} over {summary['num_trades']:,} trades, sharpe {summary['sharpe']:.2f}"
    )


class BacktestScreen(Screen[None]):
    BINDINGS = [
        Binding("q", "exit_now", "Exit", show=False),
    ]

    def __init__(self, backtest_fn: BacktestFn) -> None:
        super().__init__()
        self.backtest_fn = backtest_fn
        self.is_complete = False
        self.summary: dict | None = None
        self.error: str | None = None
        self._cancel_event = threading.Event()

    def compose(self) -> ComposeResult:
        yield ProgressPanel(id="backtest-progress")
        yield LogPanel(id="backtest-log")
        yield Static("", id="backtest-action-bar")

    def on_mount(self) -> None:
        self.query_one("#backtest-progress", ProgressPanel).update_progress(
            completed=0, total=0, unit_label="preparing data...", rate_per_sec=0.0, rate_unit="bars/sec",
        )
        self.run_worker(self._run_backtest, thread=True, exclusive=True)

    def _run_backtest(self) -> None:
        def on_progress(stage: str, done: int = 0, total: int = 0) -> None:
            if self._cancel_event.is_set():
                raise BacktestCancelled
            self.app.call_from_thread(self._handle_progress, stage, done, total)

        try:
            summary = self.backtest_fn(on_progress)
        except BacktestCancelled:
            self.app.call_from_thread(self.app.exit)
            return
        except Exception as exc:
            self.app.call_from_thread(self._handle_error, getattr(exc, "message", None) or str(exc))
            return
        self.app.call_from_thread(self._handle_complete, summary)

    def _handle_progress(self, stage: str, done: int, total: int) -> None:
        self.query_one("#backtest-progress", ProgressPanel).update_progress(
            completed=done, total=total, unit_label=stage, rate_per_sec=0.0, rate_unit="bars/sec",
        )
        if total == 0:
            self.query_one("#backtest-log", LogPanel).log_line(stage)

    def _handle_complete(self, summary: dict) -> None:
        self.is_complete = True
        self.summary = summary
        text = describe_summary(summary)
        self.query_one("#backtest-log", LogPanel).log_line(f"backtest complete — {text}")
        self.query_one("#backtest-action-bar", Static).update(f"Backtest complete — {text}. Press [b]Q[/b] to exit.")

    def _handle_error(self, message: str) -> None:
        self.error = message
        self.query_one("#backtest-log", LogPanel).log_line(f"ERROR: {message}")
        self.query_one("#backtest-action-bar", Static).update(f"Backtest failed: {message}. Press [b]Q[/b] to exit.")

    def request_stop(self) -> bool:
        if self.is_complete or self.error is not None:
            return False
        if not self._cancel_event.is_set():
            self._cancel_event.set()
            self.query_one("#backtest-action-bar", Static).update("Stopping — finishing the current step...")
        return True

    def action_exit_now(self) -> None:
        if not (self.is_complete or self.error is not None):
            return
        self.app.exit()
