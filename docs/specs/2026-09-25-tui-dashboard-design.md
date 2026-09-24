# Interactive progress dashboard — design spec

Status: approved 2026-09-25.

## 1. Goal and scope

Long-running commands (`ingest` today; `train`, `backtest`, and eventually
live trading later) currently report progress as scrolling `click.echo`
lines. There is no way to see overall percent complete, throughput, ETA,
or resource usage at a glance, and no natural place to decide "what next"
once a run finishes.

This spec covers a shared, reusable terminal dashboard, and its concrete
application to two commands:

- `ingest` — full dashboard, including an end-of-run action prompt.
- `train` — retrofit of its existing epoch loop onto the same scaffolding,
  to prove the framework is genuinely reusable rather than aspirational.

`backtest` and live trading are explicitly out of scope here. `backtest`
is currently an unimplemented stub; a dashboard screen for it would be
dead code until the underlying command exists. The scaffolding below is
built generically enough that adding those screens later is additive.

## 2. Stack

**Textual**, a Python TUI application framework, added as a new
dependency. Chosen over a lighter option (e.g. `rich.progress` +
`rich.Live`) because the dashboard needs to grow into multiple live
panels (progress, stats, scrolling log, charts) across several distinct
commands, and Textual's screen/widget model supports that without a
later rewrite. `plotext` — already a declared dependency, currently
unused — renders in-terminal charts (loss curve, later equity curves);
Textual has a maintained widget (`textual-plotext`) for embedding it.

## 3. Architecture

A single `LatentEdgeApp` (Textual `App` subclass) lives in a new
`latentedge/tui/` package. Each long-running command launches this app
into a specific starting screen rather than each command building its
own throwaway UI:

```
latentedge/tui/
  app.py            # LatentEdgeApp — thin shell, screen registry
  widgets.py         # shared widgets: ProgressPanel, StatsPanel, LogPanel
  ingest_screen.py    # IngestScreen
  train_screen.py     # TrainScreen
```

- `latentedge ingest ...` launches `LatentEdgeApp` starting on
  `IngestScreen`.
- On successful completion, `IngestScreen`'s footer becomes an action
  prompt. Choosing "train now" **pushes** `TrainScreen` onto the same
  running app (in-process transition, no new CLI invocation); choosing
  "exit" quits the app normally, leaving the ingested file on disk for a
  later `latentedge train` run.

### Shared widgets

- **ProgressPanel** — a bar, percent complete, current unit of work
  (e.g. block range, or epoch number), throughput, and ETA computed from
  a rolling rate.
- **StatsPanel** — small key/value table: output file size, free disk
  space, retry/error count.
- **LogPanel** — scrolling feed of recent discrete events (chunk
  completions, epoch summaries), replacing today's `click.echo` lines.

Each screen composes these three plus whatever is specific to it (e.g.
`TrainScreen` adds a loss sparkline via `plotext`).

### Bridging blocking work into the async UI

`ingest_range` and the training loop are synchronous, thread-pool-based
functions and stay that way — no changes to their core algorithms. Each
screen starts its work with `self.run_worker(fn, thread=True)` and the
existing (or lightly extended) progress callback calls back into the
screen with `self.call_from_thread(...)` to update reactive state safely
across the thread boundary. This is the standard Textual pattern for
wrapping blocking code and keeps `ingest_range`'s public contract stable
for anything else that calls it directly (tests, future scripts).

`ingest_range`'s `on_progress` callback signature gains two additional
values needed for the stats panel — current retry count and bytes
written since last flush — rather than the screen re-deriving them by
polling the filesystem on a timer, which would race with the writer
thread's own flush cadence.

## 4. `ingest` command changes

`latentedge ingest` keeps its existing command name and all existing
options (`--from-block`, `--to-block`, `--rpc-url`, `--out`,
`--chunk-size`, `--max-workers`, `--flush-every-n-chunks`,
`--max-retries`, `--retry-backoff-seconds`) — no new entry point, no
flag-based dispatch. Its body changes from a bare `ingest_range(...)`
call with a `click.echo`-based `on_progress` to launching
`LatentEdgeApp` on `IngestScreen`, which itself drives `ingest_range` in
a worker thread.

`IngestScreen` shows:

- Progress bar over `[from_block, to_block]`, percent complete.
- Block range currently being worked (the contiguous watermark, matching
  what's already tracked internally for resumability).
- Throughput: blocks/sec and swaps/sec, smoothed over a short rolling
  window.
- ETA, derived from current throughput and blocks remaining.
- Output file size and free disk space on the output's filesystem,
  refreshed periodically (not on every chunk).
- Retry/error counter, incremented from the existing retry logic in
  `_fetch_chunk_with_retries`.
- Scrolling log of recently completed chunk ranges (swap count per
  range) — same information as today's `click.echo` lines, in a bounded
  scrollback panel instead of unbounded terminal output.

On completion (or on an unhandled error, which is shown in the log panel
without crashing the dashboard), the footer becomes:

> Ingestion complete — wrote N swaps to `<path>`. [T] Train now [Q] Exit

`Q` (or closing the app) exits cleanly, same as today's behavior — the
ingested file and its `.progress.json` watermark are already durable on
disk regardless of what's chosen here. `T` pushes `TrainScreen`,
pre-filled with the just-written output path as its swaps input.

Resuming an interrupted run (the existing `read_progress`/watermark
behavior) is unchanged; the dashboard simply starts its progress bar
from the resumed block rather than 0.

## 5. `train` command changes

`latentedge train` keeps its existing name and options
(`--swaps`, `--out`, `--epochs`). Its body launches `LatentEdgeApp` on
`TrainScreen` (or, when reached via the ingest screen's "Train now"
prompt, that screen is pushed directly with its swaps path already
known).

`TrainScreen` shows:

- Progress bar over epochs (current epoch / total epochs).
- Current loss value, plus a `plotext`-rendered sparkline of loss over
  completed epochs.
- Elapsed time and estimated time remaining, from per-epoch timing.

The training loop (`latentedge.model.train`) gains an optional
per-epoch callback, mirroring `ingest_range`'s existing `on_progress`
pattern, so `TrainScreen` can update after each epoch without changing
the loop's core logic. On completion, the footer shows a simple summary
(final loss, saved path) and waits for a keypress to exit — no further
chained action, since there's nothing meaningful to chain to until
`backtest` is implemented.

## 6. Testing

- `ProgressPanel`/`StatsPanel`/`LogPanel` get direct unit tests against
  their reactive state (feed known values in, assert rendered content),
  independent of any running command.
- `IngestScreen` and `TrainScreen` are tested with Textual's `App.run_test()`
  pilot, driving a fake/fast `ingest_range` and training loop (small
  block ranges, one or two epochs) rather than hitting a real RPC
  endpoint, and asserting the footer prompt and screen-push transition
  behave as designed.
- The underlying `ingest_range` and training-loop unit tests are
  unaffected — their public behavior doesn't change, only the callbacks
  they're given.

## 7. Non-goals

- No dashboard for `backtest` or live trading in this pass — those
  commands don't have real logic yet ahead of this work.
- No remote/cross-terminal monitoring (e.g. watching a run from a
  second terminal window) — the dashboard runs in the same process and
  terminal as the command itself.
- No persistence of dashboard state across runs beyond what
  `ingest_range` already persists (the `.progress.json` watermark).
