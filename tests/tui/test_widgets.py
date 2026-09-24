import pytest
from textual.app import App, ComposeResult

from latentedge.tui.widgets import ProgressPanel, format_eta


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
