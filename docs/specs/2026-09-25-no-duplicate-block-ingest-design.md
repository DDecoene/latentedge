# No duplicate block downloads — design spec

Status: draft, pending review.

## 1. Goal and scope

A block that has already been fetched and flushed to a given output
file must never be requested again, regardless of which code path
triggers the request — the `ingest` CLI command, its TUI screen, or a
test. Today `ingest_range` only tracks a single forward watermark
(`last_completed_block` in `progress.json`), so it correctly avoids
re-fetching on a simple resume, but it has no way to represent or
reason about non-contiguous ingested history.

This spec covers:

- A disjoint-interval progress model that can represent any set of
  already-ingested block ranges for a given output file, not just one
  contiguous prefix.
- `ingest_range` skipping every block already covered by progress,
  within whatever `[from_block, to_block]` it's called with — this is
  the strict-bounds case, used as-is when `--from-block`/`--to-block`
  are given explicitly.
- The `ingest` command's `--days` (open, head-relative) case: when the
  naive most-recent-N-days window is already fully or partially
  covered, the window's lower bound walks further back — toward the
  pool's deployment block — until a day-count's worth of genuinely new
  blocks is found, or the pool's deployment block is reached.

Out of scope: cross-output-file dedup (progress is per output path, as
today), and any change to how a single chunk's data is fetched
(`fetch_swaps`/`rpc_logs.py`) or to the retry/rate-limit/flush behavior
in `ingest_range` beyond generalizing it to gap-fill.

## 2. Data model: disjoint interval progress file

`progress.json` changes shape from a single watermark:

```json
{"last_completed_block": 1500}
```

to a sorted, merged, non-overlapping list of closed intervals:

```json
{"ingested": [[1000, 1500], [2000, 2600]]}
```

No migration path is needed — there is no real progress data on disk
yet.

New module `latentedge/ingest/progress.py` holds this as pure,
independently testable functions, with no I/O or threading:

- `read_progress(out_path) -> list[tuple[int, int]]` — sorted, merged
  intervals; `[]` if no progress file exists yet.
- `write_progress(out_path, intervals) -> None` — writes the given
  intervals as-is (callers pass already-merged intervals from
  `add_interval`).
- `add_interval(intervals, new_from, new_to) -> list[tuple[int, int]]`
  — pure insert-and-merge: returns a new sorted list with
  `[new_from, new_to]` folded in, merging with any interval it
  overlaps or touches (adjacent, i.e. `new_from == existing_to + 1`,
  also merges — this keeps the file from fragmenting into
  one-interval-per-flush over a long run).
- `uncovered_gaps(intervals, from_block, to_block) -> list[tuple[int, int]]`
  — the sub-ranges of `[from_block, to_block]` not covered by any
  interval, in ascending order. Empty if the whole range is covered.
- `extend_window_for_new_blocks(intervals, naive_from, naive_to, desired_new_blocks, floor_block) -> int`
  — returns the smallest `from_block <= naive_from` (i.e. walking
  backward) such that `uncovered_gaps(intervals, from_block, naive_to)`
  contains at least `desired_new_blocks` blocks in total, or
  `floor_block` if that's reached first without satisfying
  `desired_new_blocks`. Never returns a value below `floor_block`.

## 3. `ingest_range`: gap-fill instead of a single forward watermark

`ingest_range(pool_address, from_block, to_block, out_path, ...)`
keeps its existing signature and strict contract: it never requests a
block outside `[from_block, to_block]`, and after this change it also
never requests a block already present in `out_path`'s progress.

Internally:

1. `gaps = uncovered_gaps(read_progress(out_path), from_block, to_block)`.
   If `gaps` is empty, return `0` immediately — the whole requested
   range is already ingested.
2. Each gap is chunked independently at `chunk_size`, and the
   per-gap chunk lists are concatenated in ascending order into one
   flat task list, e.g. gaps `(1000, 1500)` and `(2000, 2600)` produce
   chunk starts covering `1000..1500` then `2000..2600`, never
   `1501..1999`.
3. The existing "fold completed chunks into a contiguous flush-eligible
   run" logic generalizes from a single arithmetic watermark
   (`next_watermark_start += chunk_size`) to walking a precomputed,
   ordered list of expected `(chunk_start, chunk_end)` pairs. Finishing
   one gap's chunks jumps the expected pointer directly to the next
   gap's first chunk, instead of requiring
   `chunk_end + 1 == next_chunk_start`.
4. On each flush (periodic, or on exception per the existing
   rate-limit-safe finally-block), the newly-completed contiguous
   stretch since the last flush is folded into progress via
   `add_interval` and written with `write_progress`, instead of
   overwriting a single `last_completed_block`.
5. Retry, backoff, and flush-before-raising behavior on exhausted
   retries is unchanged — it already flushes pending records before
   propagating; that now also means the pending stretch gets recorded
   as a progress interval before the exception surfaces.

This is the single path all callers (CLI, TUI, tests) go through, so
the no-duplicate guarantee holds structurally rather than needing to
be reimplemented per caller.

## 4. CLI: backward extension for the `--days` case

`config.py` gains `POOL_CREATION_BLOCK`, the WETH/USDC 0.05% pool's
deployment block — the floor past which there is no possible swap
history for this pool, so extending further back would be pure waste.
This value is looked up (via a block explorer or binary-searching
`eth_getCode` over RPC) during implementation and hardcoded as a
constant, the same way `POOL_ADDRESS` is today.

In the `ingest` command, when `--from-block`/`--to-block` are both
omitted:

1. Compute `naive_to` and `blocks_in_range` exactly as today (head
   minus the safety buffer, blocks-per-day times `--days`).
2. `naive_from = naive_to - blocks_in_range + 1`.
3. `from_block = extend_window_for_new_blocks(read_progress(out), naive_from, naive_to, blocks_in_range, config.POOL_CREATION_BLOCK)`.
4. `to_block = naive_to`, unchanged — "most recent N days" always
   means the same head-relative end; only the start walks backward.
5. Call `ingest_range(from_block, to_block, ...)` as today. Its own
   gap-fill (Section 3) then naturally skips any still-covered
   sub-ranges inside this extended window.

If the pool's entire history back to `POOL_CREATION_BLOCK` is already
ingested, `ingest_range` returns `0` new records — this is a normal,
expected outcome ("fully caught up"), not an error, and the CLI's
existing `"wrote {total} new swap records"` message reports it as-is.

The explicit `--from-block`/`--to-block` path is unchanged: it calls
`ingest_range` directly with the given bounds, which is the strict
case from Section 3 — gaps inside the given range are skipped, but the
range itself is never extended.

## 5. Testing

- `tests/ingest/test_progress.py` (new): pure unit tests for
  `add_interval` (merging overlapping, adjacent, and disjoint
  intervals), `uncovered_gaps` (edge overlap, middle overlap, full
  overlap, no overlap), and `extend_window_for_new_blocks` (simple
  backward walk, walking past a covered stretch in the middle,
  clamping at `floor_block` when there aren't enough new blocks left).
- `tests/ingest/test_chunked.py` (updated): a case that pre-seeds a
  progress file with a covered sub-range inside the requested
  `[from_block, to_block]` and asserts the mocked fetch function is
  never called for those blocks; a case with two disjoint pre-existing
  intervals inside the requested range, asserting both surrounding
  gaps are fetched and the covered middle is skipped.
- CLI test: a `--days` run where the naive window is fully pre-covered
  walks back and fetches an earlier, uncovered window instead of
  returning `0`.

## 6. Non-goals

- Reclaiming or compacting already-written Parquet data — progress
  tracking only prevents re-*fetching*; it never rewrites or dedupes
  the output file itself (existing `write_swaps` append behavior is
  unchanged).
- Cross-pool or cross-output-file progress sharing.
