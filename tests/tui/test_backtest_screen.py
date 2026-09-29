import numpy as np
import pytest

from latentedge.backtest import BacktestObserver, BacktestProgress
from latentedge.tui.app import LatentEdgeApp
from latentedge.tui.backtest_screen import BacktestScreen

SUMMARY = {
    "bars": 4, "total_return_usd": 12.5, "total_return_fraction": 0.00125,
    "max_drawdown_usd": 3.0, "win_rate": 0.6, "num_trades": 5, "sharpe": 1.1,
}


def _progress(done: int, equity: float, events: list[str] | None = None) -> BacktestProgress:
    return BacktestProgress(
        done=done, total=4, timestamp=1_700_000_000 + done * 3600, equity_usd=equity, peak_equity_usd=10_010.0,
        max_drawdown_usd=5.0, num_trades=done, wins=done // 2, open_positions=1, committed_usd=500.0,
        lockout_days=0, events=events or [],
    )


def _fake_backtest(observer: BacktestObserver) -> dict:
    observer("replaying test window")
    observer.on_predictions(np.array([-0.002, -0.001, 0.0005, 0.003]))
    for done, equity in [(1, 10_000.0), (2, 10_004.0), (3, 9_995.0), (4, 10_012.5)]:
        observer.on_progress(_progress(done, equity, [f"trade {done} closed"]))
    return SUMMARY


async def _wait(pilot, screen: BacktestScreen) -> None:
    for _ in range(200):
        await pilot.pause(0.01)
        if screen.is_complete or screen.error:
            return


@pytest.mark.asyncio
async def test_screen_shows_live_progress_and_reaches_complete():
    screen = BacktestScreen(backtest_fn=_fake_backtest)
    async with LatentEdgeApp(start_screen=screen).run_test() as pilot:
        await _wait(pilot, screen)

    assert screen.is_complete and screen.summary == SUMMARY
    assert screen.last_progress is not None and screen.last_progress.done == 4
    assert screen._equity == [10_000.0, 10_004.0, 9_995.0, 10_012.5]
    assert screen._days[0] == 0.0 and screen._days[-1] == pytest.approx(3 / 24)


@pytest.mark.asyncio
async def test_screen_shows_a_failure():
    def failing(observer):
        raise RuntimeError("no test window")

    screen = BacktestScreen(backtest_fn=failing)
    async with LatentEdgeApp(start_screen=screen).run_test() as pilot:
        await _wait(pilot, screen)
    assert screen.error == "no test window"


@pytest.mark.asyncio
async def test_screen_handles_a_run_with_no_bars():
    def empty(observer):
        observer.on_predictions(np.array([]))
        return SUMMARY

    screen = BacktestScreen(backtest_fn=empty)
    async with LatentEdgeApp(start_screen=screen).run_test() as pilot:
        await _wait(pilot, screen)
    assert screen.is_complete
