"""The sweep command's live screen: scenarios stream into a results table."""

import threading
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import DataTable, Static

from latentedge.sweep import SweepObserver, describe_findings, describe_rule, describe_sweep
from latentedge.tui.backtest_screen import BacktestFn, BacktestScreen
from latentedge.tui.widgets import LogPanel, ProgressPanel

# Takes the observer to report to; returns {"path", "scenarios", "selection"}.
SweepFn = Callable[[SweepObserver], dict]

COLUMNS = ("Window", "Signal", "Rule", "Trades", "Return", "Max DD", "Win rate", "Avg PnL", "Sharpe")


class SweepCancelled(Exception):
    """Raised from an observer callback on the worker thread to unwind it after ctrl+q."""


def _cells(row: dict[str, Any]) -> tuple[str, ...]:
    return (
        row["window"], row["signal"], describe_rule(row),
        f"{row['num_trades']:,}", f"{row['total_return_fraction']:+.2%}", f"${row['max_drawdown_usd']:,.0f}",
        f"{row['win_rate']:.1%}", f"${row['avg_pnl_usd']:+,.2f}", f"{row['sharpe']:.1f}",
    )


class SweepScreen(Screen[None]):
    BINDINGS = [
        Binding("b", "backtest_now", "Backtest now", show=False),
        Binding("q", "exit_now", "Exit", show=False),
    ]
    CSS = """
    #sweep-findings { height: auto; border: round $success; padding: 0 1; display: none; }
    #sweep-table { height: 1fr; border: round $primary; }
    #sweep-log { height: 8; }
    """

    def __init__(
        self, sweep_fn: SweepFn, backtest_fn: BacktestFn | None = None, backtest_after_sweep: bool = False
    ) -> None:
        super().__init__()
        self.sweep_fn = sweep_fn
        self.backtest_fn = backtest_fn
        self.backtest_after_sweep = backtest_after_sweep
        self.is_complete = False
        self.result: dict | None = None
        self.error: str | None = None
        self.rows: list[dict[str, Any]] = []
        self._cancel_event = threading.Event()

    def compose(self) -> ComposeResult:
        yield ProgressPanel(id="sweep-progress")
        yield Static("", id="sweep-findings")
        yield DataTable(id="sweep-table")
        yield LogPanel(id="sweep-log")
        yield Static("", id="sweep-action-bar")

    def on_mount(self) -> None:
        self.query_one("#sweep-table", DataTable).add_columns(*COLUMNS)
        self.query_one("#sweep-table", DataTable).border_title = "Scenarios"
        self.query_one("#sweep-progress", ProgressPanel).update_progress(
            completed=0, total=0, unit_label="preparing data...", rate_per_sec=0.0, rate_unit="scenarios/sec",
        )
        self.run_worker(self._run_sweep, thread=True, exclusive=True)

    def _run_sweep(self) -> None:
        def check_cancelled() -> None:
            if self._cancel_event.is_set():
                raise SweepCancelled

        def on_stage(label: str, done: int = 0, total: int = 0) -> None:
            check_cancelled()
            self.app.call_from_thread(self._handle_stage, label, done, total)

        def on_scenario(row: dict[str, Any]) -> None:
            check_cancelled()
            self.app.call_from_thread(self._handle_scenario, row)

        try:
            result = self.sweep_fn(SweepObserver(on_stage=on_stage, on_scenario=on_scenario))
        except SweepCancelled:
            self.app.call_from_thread(self.app.exit)
            return
        except Exception as exc:
            self.app.call_from_thread(self._handle_error, getattr(exc, "message", None) or str(exc))
            return
        self.app.call_from_thread(self._handle_complete, result)

    def _handle_stage(self, label: str, done: int, total: int) -> None:
        self.query_one("#sweep-progress", ProgressPanel).update_progress(
            completed=done, total=total, unit_label=label, rate_per_sec=0.0, rate_unit="scenarios/sec",
        )
        if total == 0:
            self.query_one("#sweep-log", LogPanel).log_line(label)

    def _handle_scenario(self, row: dict[str, Any]) -> None:
        self.rows.append(row)
        self.query_one("#sweep-table", DataTable).add_row(*_cells(row))
        start = datetime.fromtimestamp(row["window_start"], UTC).strftime("%b %d")
        end = datetime.fromtimestamp(row["window_end"], UTC).strftime("%b %d")
        self.query_one("#sweep-log", LogPanel).log_line(
            f"{row['window']} ({start}-{end}) {row['signal']} {describe_rule(row)}: "
            f"{row['total_return_fraction']:+.2%} over {row['num_trades']:,} trades"
        )

    def _handle_complete(self, result: dict) -> None:
        self.is_complete = True
        self.result = result
        text = describe_sweep(result)
        self.query_one("#sweep-progress", ProgressPanel).update_progress(
            completed=len(result["scenarios"]), total=len(result["scenarios"]), unit_label="complete",
            rate_per_sec=0.0, rate_unit="scenarios/sec",
        )
        self.query_one("#sweep-log", LogPanel).log_line(f"sweep complete — {text}")
        if result.get("findings"):
            findings = self.query_one("#sweep-findings", Static)
            findings.border_title = "What it means"
            findings.update(describe_findings(result))
            findings.display = True
        summary = f"Sweep complete — {text}."
        if self.backtest_fn is not None and self.backtest_after_sweep:
            self.query_one("#sweep-log", LogPanel).log_line("backtest_after_train is on — starting backtest now")
            self.query_one("#sweep-action-bar", Static).update(f"{summary} Starting backtest now.")
            self._push_backtest_screen()
            return
        prompt = "Press [b]B[/b] to backtest now, or [b]Q[/b] to exit." if self.backtest_fn is not None else "Press [b]Q[/b] to exit."
        self.query_one("#sweep-action-bar", Static).update(f"{summary} {prompt}")

    def _push_backtest_screen(self) -> None:
        assert self.backtest_fn is not None
        self.app.push_screen(BacktestScreen(backtest_fn=self.backtest_fn))

    def action_backtest_now(self) -> None:
        if self.is_complete and self.backtest_fn is not None:
            self._push_backtest_screen()

    def _handle_error(self, message: str) -> None:
        self.error = message
        self.query_one("#sweep-log", LogPanel).log_line(f"ERROR: {message}")
        self.query_one("#sweep-action-bar", Static).update(f"Sweep failed: {message}. Press [b]Q[/b] to exit.")

    def request_stop(self) -> bool:
        if self.is_complete or self.error is not None:
            return False
        if not self._cancel_event.is_set():
            self._cancel_event.set()
            self.query_one("#sweep-action-bar", Static).update("Stopping — finishing the current scenario...")
        return True

    def action_exit_now(self) -> None:
        if not (self.is_complete or self.error is not None):
            return
        self.app.exit()
