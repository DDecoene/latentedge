"""The backtest command's live screen."""

import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime

import numpy as np
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Static
from textual_plotext import PlotextPlot

from latentedge.backtest import BacktestObserver, BacktestProgress
from latentedge.tui.widgets import LogPanel, ProgressPanel, StatsPanel

# Takes the observer to report to; returns the backtest's summary dict.
BacktestFn = Callable[[BacktestObserver], dict]


class BacktestCancelled(Exception):
    """Raised from an observer callback on the worker thread to unwind it after ctrl+q."""


def describe_summary(summary: dict) -> str:
    return (
        f"{summary['bars']:,} test bars: total return ${summary['total_return_usd']:,.2f} "
        f"({summary['total_return_fraction']:.2%}), max drawdown ${summary['max_drawdown_usd']:,.2f}, "
        f"win rate {summary['win_rate']:.1%} over {summary['num_trades']:,} trades, sharpe {summary['sharpe']:.2f}"
    )


def _day(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp, UTC).strftime("%b %d %H:%M")


class BacktestScreen(Screen[None]):
    BINDINGS = [
        Binding("q", "exit_now", "Exit", show=False),
    ]
    CSS = """
    #backtest-top-row { height: auto; }
    #backtest-left-col { width: 1fr; height: auto; }
    #backtest-right-col { width: 1fr; height: auto; }
    #backtest-equity-plot { height: 12; }
    #backtest-predictions-plot { height: 10; }
    """

    def __init__(self, backtest_fn: BacktestFn, time_fn: Callable[[], float] = time.monotonic) -> None:
        super().__init__()
        self.backtest_fn = backtest_fn
        self.time_fn = time_fn
        self.is_complete = False
        self.summary: dict | None = None
        self.error: str | None = None
        self.last_progress: BacktestProgress | None = None
        self._predictions: np.ndarray | None = None
        self._start_timestamp: int | None = None
        self._initial_equity: float | None = None
        self._days: list[float] = []
        self._equity: list[float] = []
        self._rate_start_time: float | None = None
        self._rate_start_done = 0
        self._cancel_event = threading.Event()

    def compose(self) -> ComposeResult:
        with Horizontal(id="backtest-top-row"):
            with Vertical(id="backtest-left-col"):
                yield ProgressPanel(id="backtest-progress")
                yield StatsPanel(id="backtest-stats")
            with Vertical(id="backtest-right-col"):
                yield PlotextPlot(id="backtest-equity-plot")
                yield PlotextPlot(id="backtest-predictions-plot")
        yield LogPanel(id="backtest-log")
        yield Static("", id="backtest-action-bar")

    def on_mount(self) -> None:
        self.query_one("#backtest-progress", ProgressPanel).update_progress(
            completed=0, total=0, unit_label="preparing data...", rate_per_sec=0.0, rate_unit="bars/sec",
        )
        self.run_worker(self._run_backtest, thread=True, exclusive=True)

    def _run_backtest(self) -> None:
        def check_cancelled() -> None:
            if self._cancel_event.is_set():
                raise BacktestCancelled

        def on_stage(stage: str, done: int = 0, total: int = 0) -> None:
            check_cancelled()
            self.app.call_from_thread(self._handle_stage, stage, done, total)

        def on_predictions(predictions: np.ndarray) -> None:
            check_cancelled()
            self.app.call_from_thread(self._handle_predictions, predictions)

        def on_progress(progress: BacktestProgress) -> None:
            check_cancelled()
            self.app.call_from_thread(self._handle_progress, progress)

        observer = BacktestObserver(on_stage=on_stage, on_predictions=on_predictions, on_progress=on_progress)
        try:
            summary = self.backtest_fn(observer)
        except BacktestCancelled:
            self.app.call_from_thread(self.app.exit)
            return
        except Exception as exc:
            self.app.call_from_thread(self._handle_error, getattr(exc, "message", None) or str(exc))
            return
        self.app.call_from_thread(self._handle_complete, summary)

    def _handle_stage(self, stage: str, done: int, total: int) -> None:
        self.query_one("#backtest-progress", ProgressPanel).update_progress(
            completed=done, total=total, unit_label=stage, rate_per_sec=0.0, rate_unit="bars/sec",
        )
        if total == 0:
            self.query_one("#backtest-log", LogPanel).log_line(stage)

    def _handle_predictions(self, predictions: np.ndarray) -> None:
        self._predictions = predictions
        if len(predictions) == 0:
            self.query_one("#backtest-log", LogPanel).log_line("model predicted no bars")
            return
        share_positive = float((predictions > 0).mean())
        self.query_one("#backtest-log", LogPanel).log_line(
            f"model predicted {len(predictions):,} bars: mean {predictions.mean():+.4%}, "
            f"max {predictions.max():+.4%}, {share_positive:.1%} above zero"
        )
        plot = self.query_one("#backtest-predictions-plot", PlotextPlot)
        plot.plt.clear_figure()
        plot.plt.hist(list(predictions * 100), bins=40)
        plot.plt.vline(0)
        plot.plt.title("Predicted net return per bar (%)")
        plot.refresh()

    def _handle_progress(self, progress: BacktestProgress) -> None:
        self.last_progress = progress
        if self._start_timestamp is None:
            self._start_timestamp = progress.timestamp
            self._initial_equity = progress.equity_usd
            self._rate_start_time = self.time_fn()
            self._rate_start_done = progress.done
            rate = 0.0
        else:
            elapsed = self.time_fn() - (self._rate_start_time or 0.0)
            rate = (progress.done - self._rate_start_done) / elapsed if elapsed > 0 else 0.0

        self.query_one("#backtest-progress", ProgressPanel).update_progress(
            completed=progress.done, total=progress.total,
            unit_label=f"bar {progress.done:,} of {progress.total:,} ({_day(progress.timestamp)} UTC)",
            rate_per_sec=rate, rate_unit="bars/sec",
        )

        self._days.append((progress.timestamp - self._start_timestamp) / 86_400)
        self._equity.append(progress.equity_usd)
        plot = self.query_one("#backtest-equity-plot", PlotextPlot)
        plot.plt.clear_figure()
        plot.plt.plot(self._days, self._equity)
        plot.plt.hline(self._initial_equity)
        plot.plt.title("Equity (USD) by day")
        plot.refresh()

        self._update_stats(progress)
        log = self.query_one("#backtest-log", LogPanel)
        for event in progress.events:
            log.log_line(event)

    def _update_stats(self, progress: BacktestProgress) -> None:
        initial = self._initial_equity or progress.equity_usd
        total_return = progress.equity_usd - initial
        win_rate = progress.wins / progress.num_trades if progress.num_trades else 0.0
        avg_pnl = total_return / progress.num_trades if progress.num_trades else 0.0
        traded = progress.num_trades / progress.done if progress.done else 0.0
        rows = [
            ("Equity", f"${progress.equity_usd:,.2f} ({total_return / initial:+.2%})"),
            ("Peak equity", f"${progress.peak_equity_usd:,.2f}"),
            ("Max drawdown", f"${progress.max_drawdown_usd:,.2f}"),
            ("Trades", f"{progress.num_trades:,} ({traded:.1%} of bars)"),
            ("Win rate", f"{win_rate:.1%}"),
            ("Avg PnL / trade", f"${avg_pnl:+,.2f}"),
            ("Open positions", f"{progress.open_positions:,} (${progress.committed_usd:,.0f} committed)"),
            ("Lockout days", f"{progress.lockout_days:,}"),
        ]
        if self._predictions is not None and len(self._predictions):
            rows.append(("Bars predicted > 0", f"{float((self._predictions > 0).mean()):.1%}"))
        self.query_one("#backtest-stats", StatsPanel).update_stats(rows)

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
