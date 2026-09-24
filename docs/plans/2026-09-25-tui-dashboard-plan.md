# Interactive Progress Dashboard Implementation Plan

> **For implementers:** work through tasks in order; each ends with a
> commit. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `ingest`'s and `train`'s scrolling `click.echo` progress
output with a live, in-process terminal dashboard (progress bar, stats,
scrolling log), and let `ingest` chain directly into `train` on
completion.

**Architecture:** A new `latentedge.tui` package holds three shared
widgets (progress bar + ETA, stats table, scrolling log) and two
screens (`IngestScreen`, `TrainScreen`) built on top of them. Each
long-running CLI command runs its existing, unchanged blocking logic
(`ingest_range`, `model.train`) in a worker thread and feeds it back to
the screen via `call_from_thread`. When stdout isn't a real terminal
(CI, piping, `CliRunner`), the CLI falls back to today's plain
`click.echo` output instead of launching the TUI.

**Tech Stack:** Python 3.12, Textual (new dependency) for the TUI
framework, `textual-plotext` (new dependency) embedding the existing
`plotext` dependency for the training loss sparkline, `pytest` +
Textual's `App.run_test()` pilot for screen/widget tests, `mypy --strict`.

**Spec:** `docs/specs/2026-09-25-tui-dashboard-design.md`

## Global Constraints

- `ingest` and `train` keep their existing command names and all
  existing CLI options — no new entry point, no flag-based dispatch.
- `backtest` and live trading get no dashboard screens in this plan —
  `backtest` has no real implementation yet.
- No remote/cross-terminal monitoring — the dashboard runs in the same
  process and terminal as the command.
- `ingest_range`'s and `model.train`'s core algorithms are unchanged;
  only new optional callback parameters are added, so every existing
  caller and test keeps working unmodified.
- When stdout is not a TTY, both commands fall back to plain
  `click.echo`-based output identical in shape to today's — this is
  required for the existing `CliRunner`-based CLI tests to keep passing
  unmodified, and is generally correct behavior for CI/piped use.
- `mypy --strict` must pass on all new and modified files.
- No AI-tooling attribution or internal process vocabulary in any
  committed file (commit messages, code comments, docs) — plain
  engineering prose only.

## Review Focus

- **Non-interactive invocation** (CI, piped output, redirected stdout,
  `CliRunner`): must not hang or crash trying to draw a TUI — falls
  back to plain progress output.
- **Resumed ingestion run** (a `.progress.json` watermark already past
  `from_block`): the progress bar must start from the resumed block,
  not 0%.
- **Mid-run failure** (a chunk exhausts its retries and `ingest_range`
  raises): the dashboard must show the error in the log panel and stay
  usable enough to exit, not crash the whole process with an
  unformatted traceback over a half-drawn screen.
- **Zero-length or already-complete work** (`from_block > to_block`, or
  a resumed range with nothing left to fetch): progress/ETA math must
  not divide by zero, and the screen must reach its completion state
  immediately rather than hang waiting for a worker that has nothing to
  do.
- **High chunk-count runs** (a year of history is ~263,000 chunks): the
  scrolling log must not grow unbounded and slow the UI down — it caps
  its scrollback rather than keeping every line ever written.

---

## Task 1: `ProgressPanel` widget

**Files:**
- Create: `src/latentedge/tui/__init__.py`
- Create: `src/latentedge/tui/widgets.py`
- Test: `tests/tui/__init__.py`
- Test: `tests/tui/test_widgets.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: `ProgressPanel` (Textual `Widget` subclass) with method
  `update_progress(completed: int, total: int, unit_label: str, rate_per_sec: float, rate_unit: str) -> None`
  and a module-level helper `format_eta(seconds_remaining: float | None) -> str`.

- [ ] **Step 1: Add Textual dependencies to `pyproject.toml`**

Add to the `dependencies` list in `pyproject.toml` (alongside the
existing `plotext>=5.3`):

```toml
    "textual>=0.60",
    "textual-plotext>=0.2.1",
```

Then run `uv sync` (or `pip install -e .` if not using `uv`) so the
packages are installed.

- [ ] **Step 2: Create the empty package**

Create `src/latentedge/tui/__init__.py` (empty file) and
`tests/tui/__init__.py` (empty file).

- [ ] **Step 3: Write the failing test**

```python
# tests/tui/test_widgets.py
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
        detail_text = str(app.query_one("#panel-detail").renderable)

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
        detail_text = str(app.query_one("#panel-detail").renderable)

    assert "100%" in detail_text


def test_format_eta_handles_none_and_zero_rate():
    assert format_eta(None) == "--:--"


def test_format_eta_formats_minutes_and_seconds():
    assert format_eta(125) == "02:05"


def test_format_eta_formats_hours():
    assert format_eta(3725) == "1:02:05"
```

Add `pytest-asyncio` is not required — Textual's `run_test()` pilot
works with plain `pytest.mark.asyncio` only if `pytest-asyncio` is
installed. Instead, use Textual's own sync test support: replace
`@pytest.mark.asyncio` / `async def` with plain `async def` test
functions and configure `asyncio_mode` via `pytest.ini`/`pyproject.toml`.
Add to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

And add `pytest-asyncio` to the `dev` dependency group in
`pyproject.toml`:

```toml
    "pytest-asyncio>=0.24",
```

Re-run `uv sync` after this change.

- [ ] **Step 4: Run test to verify it fails**

Run: `pytest tests/tui/test_widgets.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'latentedge.tui.widgets'`

- [ ] **Step 5: Write the implementation**

```python
# src/latentedge/tui/widgets.py
"""Shared dashboard widgets used by every long-running command's screen."""

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widget import Widget
from textual.widgets import ProgressBar, Static


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
```

Note: the test above queries `#panel-detail` (the panel's `id` is
`"panel"`, so its detail child is `"panel-detail"`) — this only works
because Textual's `query_one` searches the whole screen, not just the
panel's own children, so the id needs to be unique per mounted
`ProgressPanel` instance. This is why `compose` builds ids from
`self.id` rather than hardcoding them.

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/tui/test_widgets.py -v`
Expected: PASS (all four tests)

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/latentedge/tui/ tests/tui/
git commit -m "Add ProgressPanel dashboard widget"
```

---

## Task 2: `StatsPanel` widget

**Files:**
- Modify: `src/latentedge/tui/widgets.py`
- Modify: `tests/tui/test_widgets.py`

**Interfaces:**
- Consumes: nothing from Task 1 beyond the file/module it extends.
- Produces: `StatsPanel` (Textual `Widget` subclass) with method
  `update_stats(rows: list[tuple[str, str]]) -> None`, where each tuple
  is a `(label, value)` pair rendered as a line.

- [ ] **Step 1: Write the failing test**

Append to `tests/tui/test_widgets.py`:

```python
from latentedge.tui.widgets import StatsPanel


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
        text = str(app.query_one("#stats-body").renderable)

    assert "File size: 12.3 MB" in text
    assert "Free disk: 45.6 GB" in text
    assert "Retries: 2" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/tui/test_widgets.py::test_stats_panel_renders_label_value_rows -v`
Expected: FAIL with `ImportError: cannot import name 'StatsPanel'`

- [ ] **Step 3: Write the implementation**

Append to `src/latentedge/tui/widgets.py`:

```python
class StatsPanel(Widget):
    """A small label/value table for point-in-time stats (file size, retries, ...)."""

    def compose(self) -> ComposeResult:
        yield Static("", id=f"{self.id}-body")

    def update_stats(self, rows: list[tuple[str, str]]) -> None:
        text = "\n".join(f"{label}: {value}" for label, value in rows)
        self.query_one(f"#{self.id}-body", Static).update(text)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/tui/test_widgets.py -v`
Expected: PASS (all tests, including Task 1's)

- [ ] **Step 5: Commit**

```bash
git add src/latentedge/tui/widgets.py tests/tui/test_widgets.py
git commit -m "Add StatsPanel dashboard widget"
```

---

## Task 3: `LogPanel` widget

**Files:**
- Modify: `src/latentedge/tui/widgets.py`
- Modify: `tests/tui/test_widgets.py`

**Interfaces:**
- Produces: `LogPanel` (Textual `Widget` subclass wrapping `RichLog`)
  with method `log_line(text: str) -> None`. Caps scrollback at
  `MAX_LOG_LINES = 2000` so a multi-hundred-thousand-chunk ingest run
  doesn't grow memory or rendering time unbounded (Review Focus: high
  chunk-count runs).

- [ ] **Step 1: Write the failing test**

Append to `tests/tui/test_widgets.py`:

```python
from textual.widgets import RichLog

from latentedge.tui.widgets import LogPanel


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

    assert rich_log.line_count == 2


@pytest.mark.asyncio
async def test_log_panel_caps_scrollback():
    app = _LogPanelHarness()
    async with app.run_test() as pilot:
        panel = app.query_one(LogPanel)
        for i in range(2500):
            panel.log_line(f"line {i}")
        await pilot.pause()
        rich_log = app.query_one(RichLog)

    assert rich_log.line_count <= 2000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/tui/test_widgets.py::test_log_panel_appends_lines -v`
Expected: FAIL with `ImportError: cannot import name 'LogPanel'`

- [ ] **Step 3: Write the implementation**

Append to `src/latentedge/tui/widgets.py`:

```python
from textual.widgets import RichLog

MAX_LOG_LINES = 2000


class LogPanel(Widget):
    """A bounded scrolling feed of recent discrete events."""

    def compose(self) -> ComposeResult:
        yield RichLog(max_lines=MAX_LOG_LINES, id=f"{self.id}-body")

    def log_line(self, text: str) -> None:
        self.query_one(f"#{self.id}-body", RichLog).write(text)
```

Add the `RichLog` import to the existing `from textual.widgets import
ProgressBar, Static` line at the top of the file instead of a second
import line:

```python
from textual.widgets import ProgressBar, RichLog, Static
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/tui/test_widgets.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/latentedge/tui/widgets.py tests/tui/test_widgets.py
git commit -m "Add LogPanel dashboard widget with bounded scrollback"
```

---

## Task 4: `on_retry` callback on `ingest_range`

**Files:**
- Modify: `src/latentedge/ingest/chunked.py`
- Modify: `tests/ingest/test_chunked.py`

**Interfaces:**
- Produces: `ingest_range(..., on_retry: Callable[[], None] | None = None)`
  — called once per failed attempt (i.e. every retry, not the final
  giving-up raise), from whichever worker thread hit the failure.

- [ ] **Step 1: Write the failing test**

Append to `tests/ingest/test_chunked.py`:

```python
def test_ingest_range_calls_on_retry_for_each_failed_attempt(tmp_path: Path):
    attempts = {"count": 0}
    lock = threading.Lock()
    retry_calls = {"count": 0}
    retry_lock = threading.Lock()

    def flaky_fetch(pool_address: str, from_block: int, to_block: int, client: httpx.Client, rpc_url: str) -> list[SwapRecord]:
        with lock:
            attempts["count"] += 1
            count = attempts["count"]
        if count < 3:
            raise RpcLogsError("transient failure")
        return [_record(from_block, 0)]

    def on_retry() -> None:
        with retry_lock:
            retry_calls["count"] += 1

    out_path = tmp_path / "swaps.parquet"
    with httpx.Client() as client:
        ingest_range(
            pool_address="0xpool", from_block=0, to_block=99, out_path=out_path,
            client=client, rpc_url="http://fake", chunk_size=100, max_workers=1,
            fetch_fn=flaky_fetch, max_retries=5, retry_backoff_seconds=0.001,
            on_retry=on_retry,
        )

    # 3 attempts total means 2 failed-then-retried attempts.
    assert retry_calls["count"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ingest/test_chunked.py::test_ingest_range_calls_on_retry_for_each_failed_attempt -v`
Expected: FAIL with `TypeError: ingest_range() got an unexpected keyword argument 'on_retry'`

- [ ] **Step 3: Write the implementation**

In `src/latentedge/ingest/chunked.py`, modify `_fetch_chunk_with_retries`
to accept and call the callback:

```python
def _fetch_chunk_with_retries(
    fetch_fn: FetchFn,
    pool_address: str,
    from_block: int,
    to_block: int,
    client: httpx.Client,
    rpc_url: str,
    max_retries: int,
    backoff_seconds: float,
    on_retry: Callable[[], None] | None = None,
) -> list[SwapRecord]:
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            return fetch_fn(pool_address, from_block, to_block, client, rpc_url)
        except Exception as exc:  # RpcLogsError et al — real transient failures
            last_error = exc
            if attempt < max_retries - 1:
                if on_retry is not None:
                    on_retry()
                time.sleep(backoff_seconds * (2**attempt))
    assert last_error is not None
    raise last_error
```

Modify `ingest_range`'s signature to accept `on_retry` and thread it
through to `process_chunk`'s call to `_fetch_chunk_with_retries`:

```python
def ingest_range(
    pool_address: str,
    from_block: int,
    to_block: int,
    out_path: Path,
    client: httpx.Client,
    rpc_url: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
    max_workers: int = DEFAULT_MAX_WORKERS,
    flush_every_n_chunks: int = DEFAULT_FLUSH_EVERY_N_CHUNKS,
    on_progress: Callable[[int, int, int], None] | None = None,
    on_retry: Callable[[], None] | None = None,
    fetch_fn: FetchFn = fetch_swaps,
) -> int:
```

(docstring unchanged) and inside, change `process_chunk`:

```python
    def process_chunk(chunk_start: int) -> tuple[int, int, list[SwapRecord]]:
        chunk_end = min(chunk_start + chunk_size - 1, to_block)
        records = _fetch_chunk_with_retries(
            fetch_fn, pool_address, chunk_start, chunk_end, client, rpc_url,
            max_retries, retry_backoff_seconds, on_retry,
        )
        return chunk_start, chunk_end, records
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/ingest/test_chunked.py -v`
Expected: PASS (all tests, including the new one and every pre-existing one)

- [ ] **Step 5: Commit**

```bash
git add src/latentedge/ingest/chunked.py tests/ingest/test_chunked.py
git commit -m "Add on_retry callback to ingest_range for dashboard retry counts"
```

---

## Task 5: `on_epoch` callback on `model.train`

**Files:**
- Modify: `src/latentedge/model.py`
- Modify: `tests/test_model.py`

**Interfaces:**
- Produces: `train(model, features, labels, epochs, learning_rate, on_epoch: Callable[[int, int, float], None] | None = None) -> list[float]`
  — `on_epoch` called after each epoch with `(epoch_index_1_based,
  total_epochs, loss)`.

- [ ] **Step 1: Read the existing test file to match its fixture style**

Run: `sed -n '1,40p' tests/test_model.py` and match whatever fixture
pattern it already uses for building a small model/dataset (existing
tests already exercise `train`; reuse the same small synthetic
`features`/`labels` arrays rather than inventing new ones).

- [ ] **Step 2: Write the failing test**

Add to `tests/test_model.py` (using the same small-model construction
pattern as the surrounding tests in that file):

```python
def test_train_calls_on_epoch_with_progress_and_loss():
    model = NetReturnRegressor(input_dim=3)
    features = np.random.RandomState(0).randn(10, 3).astype("float32")
    labels = np.random.RandomState(1).randn(10).astype("float32")

    calls: list[tuple[int, int, float]] = []

    def on_epoch(epoch: int, total_epochs: int, loss: float) -> None:
        calls.append((epoch, total_epochs, loss))

    train(model, features, labels, epochs=3, learning_rate=0.001, on_epoch=on_epoch)

    assert [c[:2] for c in calls] == [(1, 3), (2, 3), (3, 3)]
    assert all(isinstance(c[2], float) for c in calls)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_model.py::test_train_calls_on_epoch_with_progress_and_loss -v`
Expected: FAIL with `TypeError: train() got an unexpected keyword argument 'on_epoch'`

- [ ] **Step 4: Write the implementation**

In `src/latentedge/model.py`, modify `train`:

```python
from collections.abc import Callable


def train(
    model: NetReturnRegressor,
    features: np.ndarray,
    labels: np.ndarray,
    epochs: int,
    learning_rate: float,
    on_epoch: Callable[[int, int, float], None] | None = None,
) -> list[float]:
    x = mx.array(features)
    y = mx.array(labels)
    optimizer = optim.Adam(learning_rate=learning_rate)
    loss_and_grad = nn.value_and_grad(model, _loss_fn)

    losses: list[float] = []
    for epoch in range(epochs):
        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        loss_value = float(loss)
        losses.append(loss_value)
        if on_epoch is not None:
            on_epoch(epoch + 1, epochs, loss_value)
    return losses
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_model.py -v`
Expected: PASS (all tests)

- [ ] **Step 6: Commit**

```bash
git add src/latentedge/model.py tests/test_model.py
git commit -m "Add on_epoch callback to train() for dashboard progress"
```

---

## Task 6: `IngestScreen` — core wiring and progress

**Files:**
- Create: `src/latentedge/tui/ingest_screen.py`
- Create: `src/latentedge/tui/app.py`
- Test: `tests/tui/test_ingest_screen.py`

**Interfaces:**
- Consumes: `ProgressPanel.update_progress`, `StatsPanel.update_stats`,
  `LogPanel.log_line` (Tasks 1-3); `ingest_range(..., on_progress=...,
  on_retry=...)` (Task 4, existing signature otherwise).
- Produces: `LatentEdgeApp` (Textual `App` subclass, no state of its
  own beyond being a screen host); `IngestScreen` (Textual `Screen`
  subclass) constructor:
  `IngestScreen(pool_address: str, from_block: int, to_block: int, out_path: Path, client_factory: Callable[[], httpx.Client], rpc_url: str, chunk_size: int, max_workers: int, flush_every_n_chunks: int, max_retries: int, retry_backoff_seconds: float, ingest_fn: Callable[..., int] = ingest_range, fetch_fn: FetchFn = fetch_swaps, time_fn: Callable[[], float] = time.monotonic)`.
  Exposes `is_complete: bool` and `total_written: int | None` after
  completion, for Task 7 to build on.

- [ ] **Step 1: Write the failing test**

```python
# tests/tui/test_ingest_screen.py
import threading
import time
from pathlib import Path

import httpx
import pytest

from latentedge.ingest.chunked import read_progress
from latentedge.schema import SwapRecord
from latentedge.store import read_swaps
from latentedge.tui.app import LatentEdgeApp
from latentedge.tui.ingest_screen import IngestScreen
from latentedge.tui.widgets import LogPanel, ProgressPanel, StatsPanel


def _record(block_number: int) -> SwapRecord:
    return SwapRecord(
        block_number=block_number, timestamp=block_number * 12,
        tx_hash=f"0x{block_number:064x}", log_index=0,
        sqrt_price_x96=1 << 96, tick=0, liquidity=10**18,
        amount0=1000.0, amount1=-0.3, base_fee_wei=20_000_000_000,
    )


def _fake_fetch(pool_address, from_block, to_block, client, rpc_url):
    return [_record(from_block)]


@pytest.mark.asyncio
async def test_ingest_screen_reaches_complete_state(tmp_path: Path):
    out_path = tmp_path / "swaps.parquet"
    screen = IngestScreen(
        pool_address="0xpool", from_block=0, to_block=29, out_path=out_path,
        client_factory=lambda: httpx.Client(), rpc_url="http://fake",
        chunk_size=10, max_workers=1, flush_every_n_chunks=1,
        max_retries=1, retry_backoff_seconds=0.001, fetch_fn=_fake_fetch,
    )
    app = LatentEdgeApp(start_screen=screen)

    async with app.run_test() as pilot:
        for _ in range(50):
            await pilot.pause()
            if screen.is_complete:
                break
            await pilot.pause(0.01)

    assert screen.is_complete
    assert screen.total_written == 3
    assert read_progress(out_path) == 29
    assert len(read_swaps(out_path)) == 3


@pytest.mark.asyncio
async def test_ingest_screen_progress_starts_from_resumed_block(tmp_path: Path):
    # Regression for Review Focus: resumed runs must not restart the bar at 0%.
    out_path = tmp_path / "swaps.parquet"
    with httpx.Client() as client:
        from latentedge.ingest.chunked import ingest_range
        ingest_range(
            pool_address="0xpool", from_block=0, to_block=9, out_path=out_path,
            client=client, rpc_url="http://fake", chunk_size=10, max_workers=1,
            fetch_fn=_fake_fetch,
        )

    screen = IngestScreen(
        pool_address="0xpool", from_block=0, to_block=29, out_path=out_path,
        client_factory=lambda: httpx.Client(), rpc_url="http://fake",
        chunk_size=10, max_workers=1, flush_every_n_chunks=1,
        max_retries=1, retry_backoff_seconds=0.001, fetch_fn=_fake_fetch,
    )
    app = LatentEdgeApp(start_screen=screen)

    async with app.run_test() as pilot:
        await pilot.pause()
        # Immediately after start, completed-so-far must already reflect
        # the resumed watermark (block 9), not 0.
        panel = app.query_one(ProgressPanel)
        for _ in range(50):
            await pilot.pause(0.01)
            if screen.is_complete:
                break

    assert screen.is_complete
    assert screen.total_written == 2  # only blocks [10,29] were new


@pytest.mark.asyncio
async def test_ingest_screen_handles_zero_length_range_without_hanging(tmp_path: Path):
    # Review Focus: from_block > to_block must complete immediately.
    out_path = tmp_path / "swaps.parquet"
    screen = IngestScreen(
        pool_address="0xpool", from_block=10, to_block=5, out_path=out_path,
        client_factory=lambda: httpx.Client(), rpc_url="http://fake",
        chunk_size=10, max_workers=1, flush_every_n_chunks=1,
        max_retries=1, retry_backoff_seconds=0.001, fetch_fn=_fake_fetch,
    )
    app = LatentEdgeApp(start_screen=screen)

    async with app.run_test() as pilot:
        for _ in range(50):
            await pilot.pause(0.01)
            if screen.is_complete:
                break

    assert screen.is_complete
    assert screen.total_written == 0


@pytest.mark.asyncio
async def test_ingest_screen_logs_error_on_exhausted_retries_without_crashing(tmp_path: Path):
    # Review Focus: a mid-run failure must surface in the log, not crash the app.
    def always_fails(pool_address, from_block, to_block, client, rpc_url):
        raise RuntimeError("permanent failure")

    out_path = tmp_path / "swaps.parquet"
    screen = IngestScreen(
        pool_address="0xpool", from_block=0, to_block=9, out_path=out_path,
        client_factory=lambda: httpx.Client(), rpc_url="http://fake",
        chunk_size=10, max_workers=1, flush_every_n_chunks=1,
        max_retries=1, retry_backoff_seconds=0.001, fetch_fn=always_fails,
    )
    app = LatentEdgeApp(start_screen=screen)

    async with app.run_test() as pilot:
        for _ in range(50):
            await pilot.pause(0.01)
            if screen.is_complete or screen.error is not None:
                break

    assert screen.error is not None
    assert "permanent failure" in screen.error
```

Note: `test_ingest_screen_progress_starts_from_resumed_block` asserts
behavior via `total_written` (only new chunks fetched) rather than
reading the `ProgressPanel`'s rendered text, since that's the
observable, stable contract; a follow-up manual check (Step 6) confirms
the rendered percentage visually.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/tui/test_ingest_screen.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'latentedge.tui.app'`

- [ ] **Step 3: Write `LatentEdgeApp`**

```python
# src/latentedge/tui/app.py
"""The shared Textual application shell every dashboard command runs in."""

from textual.app import App
from textual.screen import Screen


class LatentEdgeApp(App[None]):
    """Hosts whichever screen a long-running command starts on."""

    def __init__(self, start_screen: Screen) -> None:
        super().__init__()
        self._start_screen = start_screen

    def on_mount(self) -> None:
        self.push_screen(self._start_screen)
```

- [ ] **Step 4: Write `IngestScreen`**

```python
# src/latentedge/tui/ingest_screen.py
"""The ingest command's progress screen."""

import shutil
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

import httpx
from textual.app import ComposeResult
from textual.screen import Screen

from latentedge.ingest.chunked import ingest_range as default_ingest_range
from latentedge.ingest.rpc_logs import fetch_swaps
from latentedge.tui.widgets import LogPanel, ProgressPanel, StatsPanel

RATE_WINDOW_SIZE = 20
STATS_REFRESH_INTERVAL_SECONDS = 1.0


class IngestScreen(Screen):
    def __init__(
        self,
        pool_address: str,
        from_block: int,
        to_block: int,
        out_path: Path,
        client_factory: Callable[[], httpx.Client],
        rpc_url: str,
        chunk_size: int,
        max_workers: int,
        flush_every_n_chunks: int,
        max_retries: int,
        retry_backoff_seconds: float,
        ingest_fn: Callable[..., int] = default_ingest_range,
        fetch_fn: Callable[..., list[object]] = fetch_swaps,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__()
        self.pool_address = pool_address
        self.from_block = from_block
        self.to_block = to_block
        self.out_path = out_path
        self.client_factory = client_factory
        self.rpc_url = rpc_url
        self.chunk_size = chunk_size
        self.max_workers = max_workers
        self.flush_every_n_chunks = flush_every_n_chunks
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.ingest_fn = ingest_fn
        self.fetch_fn = fetch_fn
        self.time_fn = time_fn

        self.is_complete = False
        self.total_written: int | None = None
        self.error: str | None = None
        self.retry_count = 0
        self._rate_window: deque[tuple[float, int]] = deque(maxlen=RATE_WINDOW_SIZE)

    def compose(self) -> ComposeResult:
        yield ProgressPanel(id="ingest-progress")
        yield StatsPanel(id="ingest-stats")
        yield LogPanel(id="ingest-log")

    def on_mount(self) -> None:
        total = max(self.to_block - self.from_block + 1, 0)
        self.query_one("#ingest-progress", ProgressPanel).update_progress(
            completed=0, total=total, unit_label="starting...",
            rate_per_sec=0.0, rate_unit="blocks/sec",
        )
        self.set_interval(STATS_REFRESH_INTERVAL_SECONDS, self._refresh_disk_stats)
        self.run_worker(self._run_ingest, thread=True, exclusive=True)

    def _run_ingest(self) -> None:
        def on_progress(chunk_start: int, chunk_end: int, count: int) -> None:
            self.call_from_thread(self._handle_progress, chunk_start, chunk_end, count)

        def on_retry() -> None:
            self.call_from_thread(self._handle_retry)

        try:
            with self.client_factory() as client:
                total = self.ingest_fn(
                    self.pool_address, self.from_block, self.to_block, self.out_path,
                    client, self.rpc_url,
                    chunk_size=self.chunk_size, max_retries=self.max_retries,
                    retry_backoff_seconds=self.retry_backoff_seconds,
                    max_workers=self.max_workers,
                    flush_every_n_chunks=self.flush_every_n_chunks,
                    on_progress=on_progress, on_retry=on_retry, fetch_fn=self.fetch_fn,
                )
        except Exception as exc:
            self.call_from_thread(self._handle_error, str(exc))
            return
        self.call_from_thread(self._handle_complete, total)

    def _handle_progress(self, chunk_start: int, chunk_end: int, count: int) -> None:
        total = max(self.to_block - self.from_block + 1, 0)
        completed = chunk_end - self.from_block + 1
        now = self.time_fn()
        self._rate_window.append((now, completed))

        rate = 0.0
        if len(self._rate_window) >= 2:
            (t0, c0), (t1, c1) = self._rate_window[0], self._rate_window[-1]
            elapsed = t1 - t0
            if elapsed > 0:
                rate = (c1 - c0) / elapsed

        self.query_one("#ingest-progress", ProgressPanel).update_progress(
            completed=completed, total=total, unit_label=f"block {chunk_end}",
            rate_per_sec=rate, rate_unit="blocks/sec",
        )
        self.query_one("#ingest-log", LogPanel).log_line(
            f"blocks {chunk_start}-{chunk_end}: {count} swaps"
        )

    def _handle_retry(self) -> None:
        self.retry_count += 1
        self._refresh_disk_stats()

    def _refresh_disk_stats(self) -> None:
        file_size = self.out_path.stat().st_size if self.out_path.exists() else 0
        free_bytes = shutil.disk_usage(self.out_path.parent).free if self.out_path.parent.exists() else 0
        self.query_one("#ingest-stats", StatsPanel).update_stats([
            ("File size", f"{file_size / 1_048_576:.1f} MB"),
            ("Free disk", f"{free_bytes / 1_073_741_824:.1f} GB"),
            ("Retries", str(self.retry_count)),
        ])

    def _handle_complete(self, total: int) -> None:
        self.is_complete = True
        self.total_written = total
        self._refresh_disk_stats()

    def _handle_error(self, message: str) -> None:
        self.error = message
        self.query_one("#ingest-log", LogPanel).log_line(f"ERROR: {message}")
```

`out_path.parent` always exists in practice because the CLI creates it
(`out.parent.mkdir(parents=True, exist_ok=True)`) before launching the
screen — Task 9 preserves that call.

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/tui/test_ingest_screen.py -v`
Expected: PASS (all four tests)

- [ ] **Step 6: Manual visual check**

Run a short real ingest against a small real block range with a real
RPC URL (or the default public endpoint) to confirm the panel actually
looks right, e.g.:

```bash
latentedge ingest --from-block <recent_block-20> --to-block <recent_block> --out /tmp/manual_check.parquet
```

Confirm the progress bar, block label, throughput, ETA, file size, and
free disk space all render sensibly and update live. This step has no
automated assertion — it's a one-time human check that the rendered
layout is legible, not part of the test suite.

- [ ] **Step 7: Commit**

```bash
git add src/latentedge/tui/app.py src/latentedge/tui/ingest_screen.py tests/tui/test_ingest_screen.py
git commit -m "Add IngestScreen with live progress, stats, and error handling"
```

---

## Task 7: `IngestScreen` completion prompt → `TrainScreen` handoff

**Files:**
- Modify: `src/latentedge/tui/ingest_screen.py`
- Modify: `tests/tui/test_ingest_screen.py`
- Create: `src/latentedge/tui/train_screen.py` (minimal stub sufficient
  for this task; fully built out in Task 8)

**Interfaces:**
- Consumes: `IngestScreen` from Task 6.
- Produces: `IngestScreen` gains key bindings `t` (train now) and `q`
  (exit), active only once `is_complete` is true, and an
  `#ingest-action-bar` `Static` showing the completion prompt text.
  `TrainScreen.__init__` gains a `swaps_path: Path` parameter (stub
  body for now — Task 8 fills in the rest without changing this
  signature).

- [ ] **Step 1: Write the minimal `TrainScreen` stub**

```python
# src/latentedge/tui/train_screen.py
"""The train command's progress screen."""

from pathlib import Path

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Static


class TrainScreen(Screen):
    def __init__(self, swaps_path: Path) -> None:
        super().__init__()
        self.swaps_path = swaps_path

    def compose(self) -> ComposeResult:
        yield Static(f"Training on {self.swaps_path}...", id="train-placeholder")
```

- [ ] **Step 2: Write the failing test**

Append to `tests/tui/test_ingest_screen.py`:

```python
from latentedge.tui.train_screen import TrainScreen


@pytest.mark.asyncio
async def test_ingest_screen_shows_completion_prompt(tmp_path: Path):
    out_path = tmp_path / "swaps.parquet"
    screen = IngestScreen(
        pool_address="0xpool", from_block=0, to_block=9, out_path=out_path,
        client_factory=lambda: httpx.Client(), rpc_url="http://fake",
        chunk_size=10, max_workers=1, flush_every_n_chunks=1,
        max_retries=1, retry_backoff_seconds=0.001, fetch_fn=_fake_fetch,
    )
    app = LatentEdgeApp(start_screen=screen)

    async with app.run_test() as pilot:
        for _ in range(50):
            await pilot.pause(0.01)
            if screen.is_complete:
                break
        action_bar = app.query_one("#ingest-action-bar")
        text = str(action_bar.renderable)

    assert "Train now" in text
    assert "Exit" in text
    assert str(out_path) in text or "1" in text  # swap count or path present


@pytest.mark.asyncio
async def test_ingest_screen_train_key_pushes_train_screen(tmp_path: Path):
    out_path = tmp_path / "swaps.parquet"
    screen = IngestScreen(
        pool_address="0xpool", from_block=0, to_block=9, out_path=out_path,
        client_factory=lambda: httpx.Client(), rpc_url="http://fake",
        chunk_size=10, max_workers=1, flush_every_n_chunks=1,
        max_retries=1, retry_backoff_seconds=0.001, fetch_fn=_fake_fetch,
    )
    app = LatentEdgeApp(start_screen=screen)

    async with app.run_test() as pilot:
        for _ in range(50):
            await pilot.pause(0.01)
            if screen.is_complete:
                break
        await pilot.press("t")
        await pilot.pause()

    assert isinstance(app.screen, TrainScreen)
    assert app.screen.swaps_path == out_path


@pytest.mark.asyncio
async def test_ingest_screen_train_key_ignored_before_completion(tmp_path: Path):
    # A slow-fetch screen where completion hasn't happened yet.
    def slow_fetch(pool_address, from_block, to_block, client, rpc_url):
        time.sleep(0.2)
        return [_record(from_block)]

    out_path = tmp_path / "swaps.parquet"
    screen = IngestScreen(
        pool_address="0xpool", from_block=0, to_block=9, out_path=out_path,
        client_factory=lambda: httpx.Client(), rpc_url="http://fake",
        chunk_size=10, max_workers=1, flush_every_n_chunks=1,
        max_retries=1, retry_backoff_seconds=0.001, fetch_fn=slow_fetch,
    )
    app = LatentEdgeApp(start_screen=screen)

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("t")
        await pilot.pause()

    assert not isinstance(app.screen, TrainScreen)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/tui/test_ingest_screen.py -k completion_prompt -v`
Expected: FAIL — no `#ingest-action-bar` widget exists yet.

- [ ] **Step 4: Wire the completion prompt into `IngestScreen`**

Modify `src/latentedge/tui/ingest_screen.py`:

```python
from textual.binding import Binding
from textual.widgets import Static

from latentedge.tui.train_screen import TrainScreen
```

```python
class IngestScreen(Screen):
    BINDINGS = [
        Binding("t", "train_now", "Train now", show=False),
        Binding("q", "exit_now", "Exit", show=False),
    ]

    # ... __init__ unchanged ...

    def compose(self) -> ComposeResult:
        yield ProgressPanel(id="ingest-progress")
        yield StatsPanel(id="ingest-stats")
        yield LogPanel(id="ingest-log")
        yield Static("", id="ingest-action-bar")

    # ... on_mount, _run_ingest, _handle_progress, _handle_retry,
    # _refresh_disk_stats unchanged ...

    def _handle_complete(self, total: int) -> None:
        self.is_complete = True
        self.total_written = total
        self._refresh_disk_stats()
        self.query_one("#ingest-action-bar", Static).update(
            f"Ingestion complete — wrote {total} swaps to {self.out_path}. "
            "[T] Train now   [Q] Exit"
        )

    def action_train_now(self) -> None:
        if not self.is_complete:
            return
        self.app.push_screen(TrainScreen(swaps_path=self.out_path))

    def action_exit_now(self) -> None:
        if not self.is_complete:
            return
        self.app.exit()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/tui/test_ingest_screen.py -v`
Expected: PASS (all tests, including Task 6's)

- [ ] **Step 6: Commit**

```bash
git add src/latentedge/tui/ingest_screen.py src/latentedge/tui/train_screen.py tests/tui/test_ingest_screen.py
git commit -m "Add ingest completion prompt that hands off to TrainScreen"
```

---

## Task 8: `TrainScreen` — progress, loss sparkline, completion

**Files:**
- Modify: `src/latentedge/tui/train_screen.py`
- Test: `tests/tui/test_train_screen.py`

**Interfaces:**
- Consumes: `train(..., on_epoch=...)` (Task 5); `ProgressPanel`,
  `StatsPanel`... only `ProgressPanel` and `LogPanel` are used here
  (no disk stats relevant to training); `PlotextPlot` from
  `textual_plotext`.
- Produces: `TrainScreen.__init__(swaps_path: Path, out_path: Path = Path("data/model.safetensors"), epochs: int = 100, assemble_fn: Callable[[Path], tuple[np.ndarray, np.ndarray, int]] | None = None, train_fn: Callable[..., list[float]] = model.train, time_fn: Callable[[], float] = time.monotonic)`.
  `is_complete: bool`, `error: str | None`, `final_loss: float | None`
  attributes after completion, mirroring `IngestScreen`'s pattern.

Note on `assemble_fn`: building `(features, labels, input_dim)` from a
swaps parquet file reuses `build_bars`, `assemble_training_data`,
`chronological_split`, and `compute_feature_stats`/`standardize_features`
exactly as `cli.py`'s existing `train` command body already does
(`cli.py:93-125`). Rather than duplicating that assembly logic inside
the screen, `TrainScreen` takes it as an injected callable so the CLI
task (Task 10) supplies the real pipeline and tests supply a fast fake.

- [ ] **Step 1: Write the failing test**

```python
# tests/tui/test_train_screen.py
from pathlib import Path

import numpy as np
import pytest

from latentedge.tui.app import LatentEdgeApp
from latentedge.tui.train_screen import TrainScreen
from latentedge.tui.widgets import ProgressPanel


def _fake_assemble(swaps_path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    features = np.random.RandomState(0).randn(10, 3).astype("float32")
    labels = np.random.RandomState(1).randn(10).astype("float32")
    return features, labels, 3


def _fake_train(model, features, labels, epochs, learning_rate, on_epoch=None):
    losses = []
    for epoch in range(epochs):
        loss = 1.0 / (epoch + 1)
        losses.append(loss)
        if on_epoch is not None:
            on_epoch(epoch + 1, epochs, loss)
    return losses


@pytest.mark.asyncio
async def test_train_screen_reaches_complete_state(tmp_path: Path):
    screen = TrainScreen(
        swaps_path=tmp_path / "swaps.parquet",
        out_path=tmp_path / "model.safetensors",
        epochs=5,
        assemble_fn=_fake_assemble,
        train_fn=_fake_train,
    )
    app = LatentEdgeApp(start_screen=screen)

    async with app.run_test() as pilot:
        for _ in range(50):
            await pilot.pause(0.01)
            if screen.is_complete:
                break

    assert screen.is_complete
    assert screen.final_loss == pytest.approx(0.2)
    assert (tmp_path / "model.safetensors").exists()


@pytest.mark.asyncio
async def test_train_screen_progress_reaches_full_bar(tmp_path: Path):
    screen = TrainScreen(
        swaps_path=tmp_path / "swaps.parquet",
        out_path=tmp_path / "model.safetensors",
        epochs=5,
        assemble_fn=_fake_assemble,
        train_fn=_fake_train,
    )
    app = LatentEdgeApp(start_screen=screen)

    async with app.run_test() as pilot:
        for _ in range(50):
            await pilot.pause(0.01)
            if screen.is_complete:
                break
        detail_text = str(app.query_one("#train-progress-detail").renderable)

    assert "100%" in detail_text


@pytest.mark.asyncio
async def test_train_screen_logs_error_without_crashing(tmp_path: Path):
    def failing_train(model, features, labels, epochs, learning_rate, on_epoch=None):
        raise RuntimeError("bad shapes")

    screen = TrainScreen(
        swaps_path=tmp_path / "swaps.parquet",
        out_path=tmp_path / "model.safetensors",
        epochs=5,
        assemble_fn=_fake_assemble,
        train_fn=failing_train,
    )
    app = LatentEdgeApp(start_screen=screen)

    async with app.run_test() as pilot:
        for _ in range(50):
            await pilot.pause(0.01)
            if screen.error is not None:
                break

    assert "bad shapes" in screen.error
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/tui/test_train_screen.py -v`
Expected: FAIL — `TrainScreen.__init__()` doesn't accept `out_path`,
`epochs`, `assemble_fn`, `train_fn` yet.

- [ ] **Step 3: Write the implementation**

```python
# src/latentedge/tui/train_screen.py
"""The train command's progress screen."""

import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Static
from textual_plotext import PlotextPlot

from latentedge.model import NetReturnRegressor, save
from latentedge.model import train as default_train
from latentedge.tui.widgets import LogPanel, ProgressPanel

DEFAULT_MODEL_OUT_PATH = Path("data/model.safetensors")


class TrainScreen(Screen):
    def __init__(
        self,
        swaps_path: Path,
        out_path: Path = DEFAULT_MODEL_OUT_PATH,
        epochs: int = 100,
        learning_rate: float = 0.001,
        assemble_fn: Callable[[Path], tuple[np.ndarray, np.ndarray, int]] | None = None,
        train_fn: Callable[..., list[float]] = default_train,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__()
        if assemble_fn is None:
            raise ValueError("assemble_fn is required (see Task 10 for the real pipeline implementation)")
        self.swaps_path = swaps_path
        self.out_path = out_path
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.assemble_fn = assemble_fn
        self.train_fn = train_fn
        self.time_fn = time_fn

        self.is_complete = False
        self.final_loss: float | None = None
        self.error: str | None = None
        self._losses: list[float] = []

    def compose(self) -> ComposeResult:
        yield ProgressPanel(id="train-progress")
        yield PlotextPlot(id="train-loss-plot")
        yield LogPanel(id="train-log")

    def on_mount(self) -> None:
        self.query_one("#train-progress", ProgressPanel).update_progress(
            completed=0, total=self.epochs, unit_label="assembling training data...",
            rate_per_sec=0.0, rate_unit="epochs/sec",
        )
        self.run_worker(self._run_train, thread=True, exclusive=True)

    def _run_train(self) -> None:
        def on_epoch(epoch: int, total_epochs: int, loss: float) -> None:
            self.call_from_thread(self._handle_epoch, epoch, total_epochs, loss)

        try:
            features, labels, input_dim = self.assemble_fn(self.swaps_path)
            model = NetReturnRegressor(input_dim=input_dim)
            losses = self.train_fn(
                model, features, labels, epochs=self.epochs,
                learning_rate=self.learning_rate, on_epoch=on_epoch,
            )
            self.out_path.parent.mkdir(parents=True, exist_ok=True)
            save(model, self.out_path)
        except Exception as exc:
            self.call_from_thread(self._handle_error, str(exc))
            return
        self.call_from_thread(self._handle_complete, losses[-1])

    def _handle_epoch(self, epoch: int, total_epochs: int, loss: float) -> None:
        self._losses.append(loss)
        start = self._rate_start_time if hasattr(self, "_rate_start_time") else None
        if start is None:
            self._rate_start_time = self.time_fn()
            rate = 0.0
        else:
            elapsed = self.time_fn() - start
            rate = epoch / elapsed if elapsed > 0 else 0.0

        self.query_one("#train-progress", ProgressPanel).update_progress(
            completed=epoch, total=total_epochs, unit_label=f"epoch {epoch}, loss {loss:.6f}",
            rate_per_sec=rate, rate_unit="epochs/sec",
        )
        self.query_one("#train-log", LogPanel).log_line(f"epoch {epoch}/{total_epochs}: loss {loss:.6f}")

        plot = self.query_one("#train-loss-plot", PlotextPlot)
        plot.plt.clear_data()
        plot.plt.plot(list(range(1, len(self._losses) + 1)), self._losses)
        plot.plt.title("Training loss")
        plot.refresh()

    def _handle_complete(self, final_loss: float) -> None:
        self.is_complete = True
        self.final_loss = final_loss
        self.query_one("#train-log", LogPanel).log_line(
            f"training complete — final loss {final_loss:.6f}, saved to {self.out_path}"
        )

    def _handle_error(self, message: str) -> None:
        self.error = message
        self.query_one("#train-log", LogPanel).log_line(f"ERROR: {message}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/tui/test_train_screen.py -v`
Expected: PASS (all three tests). If `PlotextPlot`'s constructor or
`.plt`/`.refresh()` calls differ from the above in the installed
`textual-plotext` version, adjust `_handle_epoch`'s plotting calls to
match — the rest of the screen's behavior (progress, completion,
error handling) doesn't depend on the plotting library's exact API.

- [ ] **Step 5: Re-run Task 7's tests to confirm `TrainScreen`'s real
  constructor still satisfies the earlier stub-based tests**

Run: `pytest tests/tui/test_ingest_screen.py -v`
Expected: PASS — `TrainScreen(swaps_path=...)` from Task 7's handoff
still works because `out_path`, `epochs`, `assemble_fn`, `train_fn` all
default (Task 10 will supply a real default for `assemble_fn` when
wiring the CLI's `t` handoff; until then the default raising a
`ValueError` only matters if `action_train_now` is reached without
Task 10's fix, which the CLI wiring in Task 10 addresses directly).

- [ ] **Step 6: Commit**

```bash
git add src/latentedge/tui/train_screen.py tests/tui/test_train_screen.py
git commit -m "Build out TrainScreen with epoch progress and loss sparkline"
```

---

## Task 9: Wire `latentedge ingest` to the dashboard, with non-TTY fallback

**Files:**
- Modify: `src/latentedge/cli.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes: `LatentEdgeApp`, `IngestScreen` (Task 6/7).
- Produces: no change to `ingest`'s CLI signature; behavior branches on
  `sys.stdout.isatty()`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cli.py`:

```python
def test_ingest_launches_dashboard_when_stdout_is_a_tty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    launched = {"called": False}

    class _FakeApp:
        def __init__(self, start_screen):
            launched["called"] = True
            self.start_screen = start_screen

        def run(self):
            pass

    monkeypatch.setattr("latentedge.cli.sys.stdout.isatty", lambda: True)
    monkeypatch.setattr("latentedge.cli.LatentEdgeApp", _FakeApp)

    runner = CliRunner()
    runner.invoke(
        cli,
        ["ingest", "--from-block", "1", "--to-block", "2", "--out", str(tmp_path / "swaps.parquet")],
    )

    assert launched["called"]


def test_ingest_falls_back_to_plain_output_when_stdout_is_not_a_tty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    # CliRunner's captured stdout is never a real TTY, so this is
    # actually today's default behavior for every other CLI test in
    # this file too — this test makes that fallback explicit.
    captured: dict[str, str] = {}

    def fake_ingest_range(pool_address, from_block, to_block, out, client, rpc_url, **kwargs):
        captured["rpc_url"] = rpc_url
        return 0

    monkeypatch.setattr("latentedge.cli.ingest_range", fake_ingest_range)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["ingest", "--from-block", "1", "--to-block", "2", "--out", str(tmp_path / "swaps.parquet")],
    )

    assert result.exit_code == 0
    assert captured["rpc_url"] == DEFAULT_RPC_URL
    assert "wrote 0 new swap records" in result.output
```

Add the import needed for the first test:

```python
from latentedge.cli import DEFAULT_RPC_URL
```

(`DEFAULT_RPC_URL` is already defined in `cli.py`; this just exposes it
to the test file.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py -k dashboard_when_stdout -v`
Expected: FAIL — `latentedge.cli` has no `LatentEdgeApp` attribute to
patch yet, and `isatty()`-based branching doesn't exist.

- [ ] **Step 3: Modify `cli.py`'s `ingest` command**

Add imports at the top of `src/latentedge/cli.py`:

```python
import sys

from latentedge.tui.app import LatentEdgeApp
from latentedge.tui.ingest_screen import IngestScreen
```

Replace the body of the `ingest` command (currently `cli.py:67-86`):

```python
def ingest(
    from_block: int,
    to_block: int,
    rpc_url: str,
    out: Path,
    chunk_size: int,
    max_workers: int,
    flush_every_n_chunks: int,
    max_retries: int,
    retry_backoff_seconds: float,
) -> None:
    # A large range (e.g. a year of history) needs chunking to respect
    # provider limits, concurrency to finish in a reasonable time, and
    # resumability to survive a multi-hour run being interrupted — see
    # ingest.chunked for the real logic; fetch_swaps already populates
    # base_fee_wei from the same eth_getBlockByNumber call it makes for
    # each block's timestamp, no separate backfill pass.
    out.parent.mkdir(parents=True, exist_ok=True)

    if sys.stdout.isatty():
        screen = IngestScreen(
            pool_address=config.POOL_ADDRESS, from_block=from_block, to_block=to_block,
            out_path=out, client_factory=lambda: httpx.Client(timeout=30.0), rpc_url=rpc_url,
            chunk_size=chunk_size, max_workers=max_workers,
            flush_every_n_chunks=flush_every_n_chunks, max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds, ingest_fn=ingest_range,
        )
        LatentEdgeApp(start_screen=screen).run()
        return

    def report(chunk_start: int, chunk_end: int, count: int) -> None:
        click.echo(f"  blocks {chunk_start}-{chunk_end}: {count} swaps")

    with httpx.Client(timeout=30.0) as client:
        total = ingest_range(
            config.POOL_ADDRESS, from_block, to_block, out, client, rpc_url,
            chunk_size=chunk_size, max_retries=max_retries, retry_backoff_seconds=retry_backoff_seconds,
            max_workers=max_workers, flush_every_n_chunks=flush_every_n_chunks,
            on_progress=report,
        )
    click.echo(f"wrote {total} new swap records to {out}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli.py -v`
Expected: PASS (all tests, including every pre-existing one — the
non-TTY fallback path is exactly what `CliRunner` exercises for every
test that doesn't explicitly force `isatty()` true).

- [ ] **Step 5: Commit**

```bash
git add src/latentedge/cli.py tests/test_cli.py
git commit -m "Launch the ingest dashboard when attached to a real terminal"
```

---

## Task 10: Wire `latentedge train` to the dashboard, with non-TTY fallback

**Files:**
- Modify: `src/latentedge/cli.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes: `TrainScreen` (Task 8), `IngestScreen.action_train_now`
  (Task 7 — updated here to supply the real `assemble_fn`).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cli.py`:

```python
def test_train_launches_dashboard_when_stdout_is_a_tty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    launched = {"called": False}

    class _FakeApp:
        def __init__(self, start_screen):
            launched["called"] = True

        def run(self):
            pass

    monkeypatch.setattr("latentedge.cli.sys.stdout.isatty", lambda: True)
    monkeypatch.setattr("latentedge.cli.LatentEdgeApp", _FakeApp)

    runner = CliRunner()
    runner.invoke(cli, ["train", "--swaps", str(tmp_path / "swaps.parquet")])

    assert launched["called"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py::test_train_launches_dashboard_when_stdout_is_a_tty -v`
Expected: FAIL — `train`'s body doesn't check `isatty()` yet.

- [ ] **Step 3: Extract the existing assembly logic into a reusable function**

`SignalClient` reads a `<model_path>.stats.json` file at inference time
(`signal_client.py:20`), so the extracted function must keep writing it
via `save_feature_stats`, exactly as `cli.py`'s current `train` body
does today (`cli.py:124`). It takes `out_path` for this reason — where
to write the stats file alongside the eventual model file.

In `src/latentedge/cli.py`, above the `train` command, add:

```python
def _assemble_train_data(swaps_path: Path, out_path: Path) -> tuple["np.ndarray", "np.ndarray", int]:
    swap_df = read_swaps(swaps_path)
    bar_df = build_bars(swap_df, config.BAR_INTERVAL_SECONDS)

    bar_return_std = bar_df["price_usdc_per_weth"].pct_change().std()
    tp_sl_fraction = max(bar_return_std * 2, 0.001) if pd.notna(bar_return_std) else 0.01

    assembled = assemble_training_data(
        bar_df, swap_df, return_windows=[5, 15, 30], volatility_window=15, tp_sl_fraction=tp_sl_fraction
    )
    train_split, _validate_split, _test_split = chronological_split(assembled, train_fraction=0.7, validate_fraction=0.15)

    stats = compute_feature_stats(train_split, FEATURE_COLUMNS)
    save_feature_stats(stats, Path(str(out_path) + ".stats.json"))
    train_split = standardize_features(train_split, FEATURE_COLUMNS, stats)

    x = train_split[FEATURE_COLUMNS].to_numpy(dtype="float32")
    y = train_split["net_return"].to_numpy(dtype="float32")
    return x, y, len(FEATURE_COLUMNS)
```

Add `import numpy as np` to the top of `cli.py` for the type hint.

- [ ] **Step 4: Rewrite the `train` command**

Replace the `train` command's body (currently `cli.py:93-125`):

```python
def _assemble_train_data(swaps_path: Path, out_path: Path) -> tuple["np.ndarray", "np.ndarray", int]:
    swap_df = read_swaps(swaps_path)
    bar_df = build_bars(swap_df, config.BAR_INTERVAL_SECONDS)

    bar_return_std = bar_df["price_usdc_per_weth"].pct_change().std()
    tp_sl_fraction = max(bar_return_std * 2, 0.001) if pd.notna(bar_return_std) else 0.01

    assembled = assemble_training_data(
        bar_df, swap_df, return_windows=[5, 15, 30], volatility_window=15, tp_sl_fraction=tp_sl_fraction
    )
    train_split, _validate_split, _test_split = chronological_split(assembled, train_fraction=0.7, validate_fraction=0.15)

    stats = compute_feature_stats(train_split, FEATURE_COLUMNS)
    save_feature_stats(stats, Path(str(out_path) + ".stats.json"))
    train_split = standardize_features(train_split, FEATURE_COLUMNS, stats)

    x = train_split[FEATURE_COLUMNS].to_numpy(dtype="float32")
    y = train_split["net_return"].to_numpy(dtype="float32")
    return x, y, len(FEATURE_COLUMNS)


@cli.command()
@click.option("--swaps", type=click.Path(path_type=Path), default=Path("data/swaps.parquet"))
@click.option("--out", type=click.Path(path_type=Path), default=Path("data/model.safetensors"))
@click.option("--epochs", type=int, default=100)
def train(swaps: Path, out: Path, epochs: int) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)

    if sys.stdout.isatty():
        screen = TrainScreen(
            swaps_path=swaps, out_path=out, epochs=epochs,
            assemble_fn=lambda p: _assemble_train_data(p, out),
        )
        LatentEdgeApp(start_screen=screen).run()
        return

    x, y, input_dim = _assemble_train_data(swaps, out)
    regressor = NetReturnRegressor(input_dim=input_dim)
    losses = train_model(regressor, x, y, epochs=epochs, learning_rate=0.001)
    save(regressor, out)
    click.echo(f"trained {epochs} epochs, final loss {losses[-1]:.6f}, saved to {out}")
```

(`TrainScreen`'s own `_run_train` in Task 8 already calls `save`
itself, using the same `out_path` — keep that as-is; the plain-fallback
branch above duplicates the `save` call intentionally, matching how
`ingest`'s two branches each own their full completion behavior rather
than sharing a post-hoc save step.)

Add the `TrainScreen` import alongside the others already added in
Task 9:

```python
from latentedge.tui.train_screen import TrainScreen
```

- [ ] **Step 5: Update `IngestScreen.action_train_now` to supply the real assembler**

In `src/latentedge/tui/ingest_screen.py`, `action_train_now` currently
calls `TrainScreen(swaps_path=self.out_path)`, which needs an
`assemble_fn` (Task 8 made it required). Since `ingest_screen.py`
importing from `cli.py` would create a circular import (`cli.py`
already imports from `ingest_screen.py`), pass `assemble_fn` into
`IngestScreen` itself as a constructor parameter instead, threaded from
the CLI:

```python
    def __init__(
        self,
        pool_address: str,
        from_block: int,
        to_block: int,
        out_path: Path,
        client_factory: Callable[[], httpx.Client],
        rpc_url: str,
        chunk_size: int,
        max_workers: int,
        flush_every_n_chunks: int,
        max_retries: int,
        retry_backoff_seconds: float,
        train_assemble_fn: Callable[[Path], tuple[object, object, int]],
        ingest_fn: Callable[..., int] = default_ingest_range,
        fetch_fn: Callable[..., list[object]] = fetch_swaps,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        # ... existing body ...
        self.train_assemble_fn = train_assemble_fn
```

```python
    def action_train_now(self) -> None:
        if not self.is_complete:
            return
        self.app.push_screen(TrainScreen(swaps_path=self.out_path, assemble_fn=self.train_assemble_fn))
```

Update `cli.py`'s `ingest` command to pass it:

```python
        screen = IngestScreen(
            pool_address=config.POOL_ADDRESS, from_block=from_block, to_block=to_block,
            out_path=out, client_factory=lambda: httpx.Client(timeout=30.0), rpc_url=rpc_url,
            chunk_size=chunk_size, max_workers=max_workers,
            flush_every_n_chunks=flush_every_n_chunks, max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds, ingest_fn=ingest_range,
            train_assemble_fn=lambda p: _assemble_train_data(p, Path("data/model.safetensors")),
        )
```

Update every `IngestScreen(...)` construction in `tests/tui/test_ingest_screen.py`
(Tasks 6 and 7's tests) to pass a `train_assemble_fn=lambda p: (None, None, 0)`
placeholder — those tests never reach `action_train_now`'s use of it
except `test_ingest_screen_train_key_pushes_train_screen`, which needs
a real-enough fake:

```python
def _fake_assemble(swaps_path: Path):
    return [[0.0]], [0.0], 1
```

and pass `train_assemble_fn=_fake_assemble` in that test's
`IngestScreen(...)` call, and `train_assemble_fn=lambda p: (None, None, 0)`
in every other test's call in that file.

- [ ] **Step 6: Run the full test suite**

Run: `pytest -v`
Expected: PASS — every test in `tests/`, including `test_cli.py`,
`tests/tui/`, and every pre-existing test file untouched by this plan
(`test_backtest.py`, `test_bars.py`, etc.).

- [ ] **Step 7: Run mypy**

Run: `mypy src/latentedge`
Expected: no errors. Fix any type mismatches surfaced by the new
callback parameters or Textual's stubs before proceeding.

- [ ] **Step 8: Commit**

```bash
git add src/latentedge/cli.py src/latentedge/tui/ingest_screen.py tests/test_cli.py tests/tui/test_ingest_screen.py
git commit -m "Launch the train dashboard when attached to a real terminal; wire ingest's train handoff to the real data assembly"
```

---

## Final check

- [ ] Run `pytest -v` — full suite passes.
- [ ] Run `mypy src/latentedge` — no errors.
- [ ] Manually run `latentedge ingest --from-block <n> --to-block <n+50> --out /tmp/final_check.parquet` in a real terminal, confirm the dashboard renders and updates live, let it complete, press `t`, confirm it transitions into the training dashboard against the just-ingested file, let training finish, confirm the final summary and saved model path are shown.
