"""Shared dashboard widgets used by every long-running command's screen."""

from collections.abc import Callable
from datetime import datetime, timedelta

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


def format_landing_time(seconds_remaining: float | None, now: datetime) -> str:
    """Render when an ETA duration actually lands as a wall-clock
    date/time — a duration like "3:45:12" doesn't say whether that's
    "done before lunch" or "done at 3am"; this does. The date is omitted
    when it lands the same calendar day as `now`.
    """
    if seconds_remaining is None or seconds_remaining < 0:
        return "--"
    landing = now + timedelta(seconds=seconds_remaining)
    if landing.date() == now.date():
        return landing.strftime("%H:%M")
    return landing.strftime("%b %d %H:%M")


class ProgressPanel(Widget):
    """A progress bar plus a detail line: percent, current unit, rate, ETA."""

    # A plain Widget with no height rule defaults to filling whatever
    # space is available rather than sizing to its content — stacked in
    # a vertical screen, every such panel would independently claim the
    # full screen height, pushing everything after it off-screen. These
    # summary panels are a fixed handful of lines; only LogPanel (below)
    # should actually expand to fill the remaining space.
    DEFAULT_CSS = """
    ProgressPanel {
        height: auto;
        border: round $primary;
        border-title-color: $text;
        padding: 0 1;
    }
    ProgressPanel Vertical { height: auto; }
    """

    def on_mount(self) -> None:
        self.border_title = "Progress"

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
        now_fn: Callable[[], datetime] = datetime.now,
    ) -> None:
        total = max(total, 1)  # a zero-length range still renders a valid (complete) bar
        percent = min(100, round(completed / total * 100))
        bar = self.query_one(f"#{self.id}-bar", ProgressBar)
        bar.update(total=total, progress=min(completed, total))

        remaining = total - completed
        eta_seconds = remaining / rate_per_sec if rate_per_sec > 0 else None
        landing = format_landing_time(eta_seconds, now_fn())
        detail = self.query_one(f"#{self.id}-detail", Static)
        detail.update(
            f"{percent}% — {unit_label} — {rate_per_sec:.1f} {rate_unit} — "
            f"ETA {format_eta(eta_seconds)} (lands {landing})"
        )


class StatsPanel(Widget):
    """A small label/value table for point-in-time stats (file size, retries, ...)."""

    DEFAULT_CSS = """
    StatsPanel {
        height: auto;
        border: round $primary;
        border-title-color: $text;
        padding: 0 1;
    }
    """

    def on_mount(self) -> None:
        self.border_title = "Stats"

    def compose(self) -> ComposeResult:
        yield Static("", id=f"{self.id}-body")

    def update_stats(self, rows: list[tuple[str, str]]) -> None:
        text = "\n".join(f"{label}: {value}" for label, value in rows)
        self.query_one(f"#{self.id}-body", Static).update(text)


class ThreadPanel(Widget):
    """One line per worker thread: its current block range and status."""

    DEFAULT_CSS = """
    ThreadPanel {
        height: auto;
        border: round $primary;
        border-title-color: $text;
        padding: 0 1;
    }
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._lines: dict[int, str] = {}

    def on_mount(self) -> None:
        self.border_title = "Workers"

    def compose(self) -> ComposeResult:
        yield Static("", id=f"{self.id}-body")

    def update_worker(self, slot: int, text: str) -> None:
        self._lines[slot] = f"Worker {slot}: {text}"
        body = "\n".join(self._lines[slot] for slot in sorted(self._lines))
        self.query_one(f"#{self.id}-body", Static).update(body)


class LogPanel(Widget):
    """A bounded scrolling feed of recent discrete events."""

    # The one panel that should actually expand to fill whatever space
    # the fixed-height summary panels above it leave behind.
    DEFAULT_CSS = """
    LogPanel {
        height: 1fr;
        border: round $primary;
        border-title-color: $text;
        padding: 0 1;
    }
    """

    def on_mount(self) -> None:
        self.border_title = "Log"

    def compose(self) -> ComposeResult:
        yield RichLog(max_lines=MAX_LOG_LINES, id=f"{self.id}-body")

    def log_line(self, text: str) -> None:
        self.query_one(f"#{self.id}-body", RichLog).write(text)
