"""Chunked, resumable, concurrent ingestion over a large block range.

Real archive RPC providers cap eth_getLogs to a limited block range per
call — Alchemy's free tier caps at 10 blocks — so a year of history means
on the order of a quarter-million chunk requests. Sequentially that's
days; concurrency brings it down to hours. At that chunk count, writing
the Parquet file (a full read-modify-write) after every single chunk is
also prohibitively slow as the file grows, so writes are batched. Progress
is only ever recorded for data that has actually been flushed to disk —
a crash between flushes means re-fetching (cheap, deduped) some already-
fetched chunks on resume, never silently skipping unwritten ones.

Progress is tracked as a set of disjoint block intervals (see
latentedge.ingest.progress), not a single forward watermark — a request
only ever fetches the sub-ranges of [from_block, to_block] not already
covered by a prior run, regardless of how those prior runs were shaped.
"""

import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

from latentedge.ingest import concurrency as _concurrency
from latentedge.ingest.concurrency import AdaptiveConcurrencyLimiter, ReleaseOutcome
from latentedge.ingest.progress import (
    Interval,
    add_interval,
    read_concurrency_limit,
    read_progress,
    uncovered_gaps,
    write_concurrency_limit,
    write_progress,
)
from latentedge.ingest.rpc_logs import RateLimitError, fetch_swaps
from latentedge.schema import SwapRecord
from latentedge.store import write_swaps

# Kept equal to rpc_logs.ETH_GETLOGS_RANGE_CAP (the provider's per-call
# eth_getLogs limit) on purpose. fetch_swaps CAN accept a larger chunk and
# sub-chunk eth_getLogs internally to batch block-timestamp lookups across
# more blocks per round-trip — tried that here, but it made real 429s
# *worse*: the concurrency limiter only gates once per chunk, so a larger
# chunk means one "permitted" worker fires many eth_getLogs calls back-to-
# back with no pacing between them, bursting far more requests per permit
# than the limiter's concurrency count suggests. Needs per-request pacing
# inside fetch_swaps (not just per-chunk gating) to be safe — not built
# yet, so keep this at the provider cap until that exists.
DEFAULT_CHUNK_SIZE = 10
DEFAULT_MAX_RETRIES = 5
DEFAULT_RETRY_BACKOFF_SECONDS = 2.0
# A rate limit retried on the same short schedule as a generic transient
# error just re-triggers the same limit (observed for real against
# Alchemy's free tier) — back off substantially longer for it.
RATE_LIMIT_BACKOFF_MULTIPLIER = 5.0
DEFAULT_MAX_WORKERS = 8
# A chunk that ultimately can't recover from a rate limit no longer loses
# unflushed progress (see ingest_range's finally-block flush), but
# flushing more often still bounds how much work a mid-run interruption
# could repeat on resume.
DEFAULT_FLUSH_EVERY_N_CHUNKS = 50
DEFAULT_CONCURRENCY_COOLDOWN_SECONDS = _concurrency.DEFAULT_COOLDOWN_SECONDS
# A rate limit is an expected, temporary condition on a free-tier provider
# — not evidence of a real bug — so it must never be the reason an
# unattended, hours-long ingest run gives up partway through. Rate-limited
# chunks retry indefinitely (unlike max_retries below, which still bounds
# genuine errors so a real bug fails fast instead of looping forever), with
# backoff capped at this ceiling rather than growing unboundedly.
DEFAULT_MAX_RATE_LIMIT_BACKOFF_SECONDS = 120.0

FetchFn = Callable[[str, int, int, httpx.Client, str], list[SwapRecord]]


# max_retries (5th positional) is None for a rate-limited retry — there is
# no ceiling to report since it never gives up.
OnRetryFn = Callable[[int, int, int, int | None, float, str], None]


def _fetch_chunk_with_retries(
    fetch_fn: FetchFn,
    pool_address: str,
    from_block: int,
    to_block: int,
    client: httpx.Client,
    rpc_url: str,
    max_retries: int,
    backoff_seconds: float,
    limiter: AdaptiveConcurrencyLimiter,
    on_retry: OnRetryFn | None = None,
    on_concurrency_change: Callable[[int], None] | None = None,
    on_status: Callable[[str], None] | None = None,
    max_rate_limit_backoff_seconds: float = DEFAULT_MAX_RATE_LIMIT_BACKOFF_SECONDS,
) -> list[SwapRecord]:
    # Two independent counters: a rate limit is expected, temporary
    # provider behavior and retries forever (capped backoff, uncapped
    # attempts) so an unattended run never quits over it; a genuine
    # error (bad params, a real RPC bug) still gives up after
    # max_retries, so an actually-broken run fails fast instead of
    # retrying something that can never succeed.
    failure_attempt = 0
    rate_limit_attempt = 0
    while True:
        # Blocking on the gate is the one moment the throttle is actually
        # visible — report it distinctly from "fetching" rather than
        # announcing "fetching" before the wait even starts.
        if on_status is not None:
            on_status("waiting")
        limiter.acquire()
        if on_status is not None:
            on_status("fetching")
        try:
            result = fetch_fn(pool_address, from_block, to_block, client, rpc_url)
        except Exception as exc:  # RpcLogsError et al — real transient failures
            is_rate_limited = isinstance(exc, RateLimitError)
            new_limit, changed = limiter.release("rate_limited" if is_rate_limited else "failed")
            if changed and on_concurrency_change is not None:
                on_concurrency_change(new_limit)

            if is_rate_limited:
                rate_limit_attempt += 1
                # Exponent capped so the count doesn't grow into a huge
                # integer over a run lasting hours — the min() below
                # already does the real capping of how long it sleeps.
                sleep_seconds = min(
                    backoff_seconds * (2 ** min(rate_limit_attempt - 1, 20)) * RATE_LIMIT_BACKOFF_MULTIPLIER,
                    max_rate_limit_backoff_seconds,
                )
                if on_retry is not None:
                    on_retry(from_block, to_block, rate_limit_attempt, None, sleep_seconds, str(exc))
                time.sleep(sleep_seconds)
                continue

            failure_attempt += 1
            if failure_attempt >= max_retries:
                raise
            sleep_seconds = backoff_seconds * (2 ** (failure_attempt - 1))
            if on_retry is not None:
                on_retry(from_block, to_block, failure_attempt, max_retries, sleep_seconds, str(exc))
            time.sleep(sleep_seconds)
        else:
            new_limit, changed = limiter.release("success")
            if changed and on_concurrency_change is not None:
                on_concurrency_change(new_limit)
            return result


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
    concurrency_cooldown_seconds: float = DEFAULT_CONCURRENCY_COOLDOWN_SECONDS,
    max_rate_limit_backoff_seconds: float = DEFAULT_MAX_RATE_LIMIT_BACKOFF_SECONDS,
    on_progress: Callable[[int, int, int], None] | None = None,
    on_retry: OnRetryFn | None = None,
    on_queue_status: Callable[[int, int | None], None] | None = None,
    on_worker_status: Callable[[int, int, int, str], None] | None = None,
    on_concurrency_change: Callable[[int], None] | None = None,
    fetch_fn: FetchFn = fetch_swaps,
) -> int:
    """Ingest [from_block, to_block] concurrently, in chunks, flushing to
    disk (and recording newly-covered intervals in the resumable
    progress file) every flush_every_n_chunks chunks — never per-chunk,
    which would mean rewriting the whole output file hundreds of
    thousands of times over a real year-long pull. Returns the total
    number of swap records written in this call. Any sub-range of
    [from_block, to_block] already present in a prior run's progress is
    skipped, never re-fetched.
    """
    progress_intervals = read_progress(out_path)
    gaps = uncovered_gaps(progress_intervals, from_block, to_block)
    if not gaps:
        return 0

    # Chunk each gap independently and concatenate in order — a chunk
    # never straddles a gap boundary, so an already-covered stretch is
    # never touched even at its edges.
    chunk_order: list[int] = []
    chunk_ceiling: dict[int, int] = {}
    for gap_start, gap_end in gaps:
        for chunk_start in range(gap_start, gap_end + 1, chunk_size):
            chunk_order.append(chunk_start)
            chunk_ceiling[chunk_start] = gap_end

    lock = threading.Lock()
    completed: dict[int, tuple[int, list[SwapRecord]]] = {}
    expected_index = 0

    pending_records: list[SwapRecord] = []
    # Contiguous runs completed since the last flush — normally one, but
    # can be several if a flush batch spans more than one previously-
    # covered gap.
    pending_intervals: list[Interval] = []
    chunks_since_flush = 0
    total_written = 0

    def flush() -> None:
        nonlocal pending_records, pending_intervals, chunks_since_flush, total_written, progress_intervals
        if not pending_intervals:
            return
        if pending_records:
            write_swaps(pending_records, out_path)
            total_written += len(pending_records)
        for start, end in pending_intervals:
            progress_intervals = add_interval(progress_intervals, start, end)
        write_progress(out_path, progress_intervals)
        pending_records = []
        pending_intervals = []
        chunks_since_flush = 0

    persisted_limit = read_concurrency_limit(out_path)
    # A floor, not a fixed resume point: recovering from a throttle-down
    # needs successes_before_increase consecutive successes, which a
    # genuinely flaky provider may rarely string together — resuming at
    # the exact worst level a prior run ever reached would otherwise
    # ratchet every future run down to that floor permanently, with no
    # chance to re-test whether conditions (or the provider's load) have
    # improved since. Never resume below half the ceiling regardless of
    # how low a prior run bottomed out.
    start_limit = max(persisted_limit, -(-max_workers // 2)) if persisted_limit is not None else None
    limiter = AdaptiveConcurrencyLimiter(
        ceiling=max_workers,
        cooldown_seconds=concurrency_cooldown_seconds,
        start_limit=start_limit,
    )

    worker_slots: dict[int, int] = {}
    slots_lock = threading.Lock()

    def worker_slot() -> int:
        ident = threading.get_ident()
        with slots_lock:
            if ident not in worker_slots:
                worker_slots[ident] = len(worker_slots)
            return worker_slots[ident]

    def process_chunk(chunk_start: int) -> tuple[int, int, list[SwapRecord]]:
        chunk_end = min(chunk_start + chunk_size - 1, chunk_ceiling[chunk_start])

        slot = worker_slot() if on_worker_status is not None else -1

        def report_retry(cs: int, ce: int, attempt: int, retries: int | None, sleep_seconds: float, error_message: str) -> None:
            if on_retry is not None:
                on_retry(cs, ce, attempt, retries, sleep_seconds, error_message)
            if on_worker_status is not None:
                label = f"retry {attempt}/{retries}" if retries is not None else f"rate limited, retry {attempt}"
                on_worker_status(slot, cs, ce, f"{label}, waiting {sleep_seconds:.1f}s")

        def report_status(status: str) -> None:
            if on_worker_status is not None:
                on_worker_status(slot, chunk_start, chunk_end, status)

        retry_hook = report_retry if (on_retry is not None or on_worker_status is not None) else None
        status_hook = report_status if on_worker_status is not None else None
        records = _fetch_chunk_with_retries(
            fetch_fn, pool_address, chunk_start, chunk_end, client, rpc_url,
            max_retries, retry_backoff_seconds, limiter, retry_hook, on_concurrency_change, status_hook,
            max_rate_limit_backoff_seconds,
        )

        if on_worker_status is not None:
            on_worker_status(slot, chunk_start, chunk_end, "idle")
        return chunk_start, chunk_end, records

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_chunk, cs): cs for cs in chunk_order}
        try:
            for future in as_completed(futures):
                chunk_start, chunk_end, records = future.result()

                with lock:
                    completed[chunk_start] = (chunk_end, records)

                    # Fold in whatever prefix of the ordered chunk plan
                    # is now available. Chunks can finish out of order,
                    # but this only ever advances through chunk_order in
                    # sequence — crossing from one gap's last chunk to
                    # the next gap's first chunk is just the next index,
                    # not an arithmetic +chunk_size step.
                    while expected_index < len(chunk_order) and chunk_order[expected_index] in completed:
                        cs = chunk_order[expected_index]
                        c_end, recs = completed.pop(cs)
                        pending_records.extend(recs)

                        if pending_intervals and pending_intervals[-1][1] + 1 == cs:
                            last_start, _ = pending_intervals[-1]
                            pending_intervals[-1] = (last_start, c_end)
                        else:
                            pending_intervals.append((cs, c_end))

                        if on_progress is not None:
                            on_progress(cs, c_end, len(recs))

                        expected_index += 1
                        chunks_since_flush += 1

                        if chunks_since_flush >= flush_every_n_chunks:
                            flush()

                    if on_queue_status is not None:
                        blocking_chunk_start = (
                            chunk_order[expected_index] if expected_index < len(chunk_order) else None
                        )
                        on_queue_status(len(completed), blocking_chunk_start)
        finally:
            # A chunk that exhausted its retries raises out of
            # future.result() above; cancel whatever hasn't started yet
            # rather than let the pool keep firing more requests. Flush
            # here too (not only on the success path above) — an
            # exception propagating past this block must never discard
            # chunks that already succeeded since the last flush.
            for f in futures:
                f.cancel()
            with lock:
                flush()
            # Persisted regardless of whether this run finished cleanly
            # or was cut short — whatever level the limiter actually
            # settled at is the useful starting point for a resume,
            # not just the happy-path ending level.
            write_concurrency_limit(out_path, limiter.limit)

    return total_written
