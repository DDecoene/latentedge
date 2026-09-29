"""Render the study and sweep dashboards from the recorded result files.

The recorded rows are replayed through the real screens, so nothing is
computed and no data or model is touched.

Run: uv run python docs/screenshots/make_screenshots.py
"""

import asyncio
import json
from pathlib import Path

from latentedge.study import StudyObserver
from latentedge.sweep import SweepObserver
from latentedge.tui.app import LatentEdgeApp
from latentedge.tui.study_screen import StudyScreen
from latentedge.tui.sweep_screen import SweepScreen

HERE = Path(__file__).parent
RESULTS = HERE.parent / "paper" / "results"
STUDY = RESULTS / "studies" / "20260929T192107Z.json"
SWEEP = RESULTS / "sweeps" / "20260929T185051Z.json"
SIZE = (140, 42)


async def render(screen, name: str, done) -> None:
    app = LatentEdgeApp(start_screen=screen)
    async with app.run_test(size=SIZE) as pilot:
        while not done(screen):
            await pilot.pause(0.05)
        await pilot.pause(0.2)
        (HERE / f"{name}.svg").write_text(app.export_screenshot(title=f"latentedge {name}"))


def study_screen() -> StudyScreen:
    data = json.loads(STUDY.read_text())

    def replay(observer: StudyObserver) -> dict:
        rows = data["rows"]
        for i, row in enumerate(rows, 1):
            observer.on_row(row)
            observer.on_stage(f"run {i}/{len(rows)}", i, len(rows))
        for summary in data["summaries"]:
            observer.on_summary(summary)
        return {"path": "data/studies/" + STUDY.name, "rows": rows, "summaries": data["summaries"]}

    return StudyScreen(study_fn=replay)


def sweep_screen() -> SweepScreen:
    data = json.loads(SWEEP.read_text())

    def replay(observer: SweepObserver) -> dict:
        scenarios = data["scenarios"]
        for i, row in enumerate(scenarios, 1):
            observer.on_scenario(row)
            observer(f"scenario {i}/{len(scenarios)}", i, len(scenarios))
        return {"path": "data/sweeps/" + SWEEP.name, "scenarios": scenarios, "selection": None}

    return SweepScreen(sweep_fn=replay)


async def main() -> None:
    await render(study_screen(), "study", lambda s: s.is_complete or s.error is not None)
    await render(sweep_screen(), "sweep", lambda s: s.is_complete or s.error is not None)


asyncio.run(main())
