import pytest
from textual.widgets import DataTable

from latentedge.sweep import SweepObserver
from latentedge.tui.app import LatentEdgeApp
from latentedge.tui.sweep_screen import SweepScreen

ROW = {
    "window": "test", "signal": "model", "min_edge": 0.001, "full_size_return": 0.002, "bars": 10,
    "window_start": 1_700_000_000, "window_end": 1_700_003_600, "num_trades": 7, "total_return_usd": -12.0,
    "total_return_fraction": -0.0012, "max_drawdown_usd": 20.0, "win_rate": 0.4, "avg_pnl_usd": -1.7, "sharpe": -2.0,
}


def _fake_sweep(observer: SweepObserver) -> dict:
    observer("preparing")
    for i in range(3):
        observer("scenario", i, 3)
        observer.on_scenario(ROW)
    return {"path": "data/sweeps/x.json", "scenarios": [ROW] * 3, "selection": (ROW, ROW)}


@pytest.mark.asyncio
async def test_sweep_screen_streams_scenarios_into_the_table():
    screen = SweepScreen(sweep_fn=_fake_sweep)
    async with LatentEdgeApp(start_screen=screen).run_test() as pilot:
        for _ in range(200):
            await pilot.pause(0.01)
            if screen.is_complete:
                break
        assert screen.query_one("#sweep-table", DataTable).row_count == 3
    assert screen.is_complete and len(screen.rows) == 3


@pytest.mark.asyncio
async def test_sweep_screen_shows_a_failure():
    def failing(observer):
        raise RuntimeError("model not trained")

    screen = SweepScreen(sweep_fn=failing)
    async with LatentEdgeApp(start_screen=screen).run_test() as pilot:
        for _ in range(100):
            await pilot.pause(0.01)
            if screen.error:
                break
    assert screen.error == "model not trained"
