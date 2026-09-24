import pytest
from textual.app import App, ComposeResult

from textual.widgets import RichLog

from latentedge.tui.widgets import LogPanel, ProgressPanel, StatsPanel, format_eta


class _ProgressPanelHarness(App[None]):
    def compose(self) -> ComposeResult:
        yield ProgressPanel(id="panel")


@pytest.mark.asyncio
async def test_progress_panel_renders_percent_and_unit_label():
    app = _ProgressPanelHarness()
    async with app.run_test() as pilot:
        panel = app.query_one(ProgressPanel)
        panel.update_progress(
            completed=25, total=100, unit_label="block 1024",
            rate_per_sec=5.0, rate_unit="blocks/sec",
        )
        await pilot.pause()
        detail_text = str(app.query_one("#panel-detail").content)

    assert "25%" in detail_text
    assert "block 1024" in detail_text
    assert "5.0 blocks/sec" in detail_text


@pytest.mark.asyncio
async def test_progress_panel_shows_complete_state_at_total():
    app = _ProgressPanelHarness()
    async with app.run_test() as pilot:
        panel = app.query_one(ProgressPanel)
        panel.update_progress(
            completed=100, total=100, unit_label="done",
            rate_per_sec=0.0, rate_unit="blocks/sec",
        )
        await pilot.pause()
        detail_text = str(app.query_one("#panel-detail").content)

    assert "100%" in detail_text


def test_format_eta_handles_none_and_zero_rate():
    assert format_eta(None) == "--:--"


def test_format_eta_formats_minutes_and_seconds():
    assert format_eta(125) == "02:05"


def test_format_eta_formats_hours():
    assert format_eta(3725) == "1:02:05"


class _StatsPanelHarness(App[None]):
    def compose(self) -> ComposeResult:
        yield StatsPanel(id="stats")


@pytest.mark.asyncio
async def test_stats_panel_renders_label_value_rows():
    app = _StatsPanelHarness()
    async with app.run_test() as pilot:
        panel = app.query_one(StatsPanel)
        panel.update_stats([
            ("File size", "12.3 MB"),
            ("Free disk", "45.6 GB"),
            ("Retries", "2"),
        ])
        await pilot.pause()
        text = str(app.query_one("#stats-body").content)

    assert "File size: 12.3 MB" in text
    assert "Free disk: 45.6 GB" in text
    assert "Retries: 2" in text


class _LogPanelHarness(App[None]):
    def compose(self) -> ComposeResult:
        yield LogPanel(id="log")


@pytest.mark.asyncio
async def test_log_panel_appends_lines():
    app = _LogPanelHarness()
    async with app.run_test() as pilot:
        panel = app.query_one(LogPanel)
        panel.log_line("blocks 0-9: 3 swaps")
        panel.log_line("blocks 10-19: 0 swaps")
        await pilot.pause()
        rich_log = app.query_one(RichLog)

    assert len(rich_log.lines) == 2


@pytest.mark.asyncio
async def test_log_panel_caps_scrollback():
    app = _LogPanelHarness()
    async with app.run_test() as pilot:
        panel = app.query_one(LogPanel)
        for i in range(2500):
            panel.log_line(f"line {i}")
        await pilot.pause()
        rich_log = app.query_one(RichLog)

    assert len(rich_log.lines) <= 2000
