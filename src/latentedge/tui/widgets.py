"""Shared dashboard widgets used by every long-running command's screen."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widget import Widget
from textual.widgets import ProgressBar, RichLog, Static

MAX_LOG_LINES = 2000


def format_eta(seconds_remaining: float | None) -> str:
    """Render a remaining-time estimate as mm:ss, or h:mm:ss past an hour."""
    if seconds_remaining is None or seconds_remaining < 0:
        return "--:--"
    total_seconds = int(seconds_remaining)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


class ProgressPanel(Widget):
    """A progress bar plus a detail line: percent, current unit, rate, ETA."""

    def compose(self) -> ComposeResult:
        with Vertical():
            yield ProgressBar(total=100, show_eta=False, id=f"{self.id}-bar")
            yield Static("", id=f"{self.id}-detail")

    def update_progress(
        self,
        completed: int,
        total: int,
        unit_label: str,
        rate_per_sec: float,
        rate_unit: str,
    ) -> None:
        total = max(total, 1)  # a zero-length range still renders a valid (complete) bar
        percent = min(100, round(completed / total * 100))
        bar = self.query_one(f"#{self.id}-bar", ProgressBar)
        bar.update(total=total, progress=min(completed, total))

        remaining = total - completed
        eta_seconds = remaining / rate_per_sec if rate_per_sec > 0 else None
        detail = self.query_one(f"#{self.id}-detail", Static)
        detail.update(
            f"{percent}% — {unit_label} — {rate_per_sec:.1f} {rate_unit} — ETA {format_eta(eta_seconds)}"
        )


class StatsPanel(Widget):
    """A small label/value table for point-in-time stats (file size, retries, ...)."""

    def compose(self) -> ComposeResult:
        yield Static("", id=f"{self.id}-body")

    def update_stats(self, rows: list[tuple[str, str]]) -> None:
        text = "\n".join(f"{label}: {value}" for label, value in rows)
        self.query_one(f"#{self.id}-body", Static).update(text)


class LogPanel(Widget):
    """A bounded scrolling feed of recent discrete events."""

    def compose(self) -> ComposeResult:
        yield RichLog(max_lines=MAX_LOG_LINES, id=f"{self.id}-body")

    def log_line(self, text: str) -> None:
        self.query_one(f"#{self.id}-body", RichLog).write(text)
