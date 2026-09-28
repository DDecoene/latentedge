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

from latentedge.ingest import rate_limiter as _rate_limiter
from latentedge.ingest.rate_limiter import Cancelled, RateLimiter
from latentedge.ingest.progress import (
    Interval,
    add_interval,
    read_progress,
    read_rate_ceiling,
    read_rate_limit,
    uncovered_gaps,
    write_progress,
    write_rate_ceiling,
    write_rate_limit,
)
from latentedge.ingest.rpc_logs import RateLimitError, fetch_swaps
from latentedge.schema import SwapRecord
from latentedge.store import write_swaps

# Kept equal to rpc_logs.ETH_GETLOGS_RANGE_CAP (the provider's per-call
# eth_getLogs limit) for now. fetch_swaps sub-chunks internally and paces
# every one of those sub-calls through the shared RateLimiter (see
# rpc_logs.py), so a larger chunk no longer bursts unpaced requests the
# way it used to — growing this is a separate, later change, not blocked
# by pacing anymore.
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
DEFAULT_CONCURRENCY_COOLDOWN_SECONDS = _rate_limiter.DEFAULT_COOLDOWN_SECONDS
# A conservative starting ceiling on real requests/sec against the
# provider — proactive pacing, not just a reaction to 429s already
# happening. Tunable per provider tier via LATENTEDGE_INGEST_MAX_RPS.
DEFAULT_MAX_RPS = 5.0
# A rate limit is an expected, temporary condition on a free-tier provider
# — not evidence of a real bug — so it must never be the reason an
# unattended, hours-long ingest run gives up partway through. Rate-limited
# chunks retry indefinitely (unlike max_retries below, which still bounds
# genuine errors so a real bug fails fast instead of looping forever), with
# backoff capped at this ceiling rather than growing unboundedly.
DEFAULT_MAX_RATE_LIMIT_BACKOFF_SECONDS = 120.0

FetchFn = Callable[..., list[SwapRecord]]


class IngestCancelled(Exception):
    """A cancel_event was set mid-run (e.g. the TUI's ctrl+q handler) —
    distinct from a real error so a caller can tell "the user asked to
    stop" apart from "something broke", and report accordingly. Carries
    however many records this call had already written before stopping.
    """

    def __init__(self, total_written: int = 0) -> None:
        super().__init__("ingest cancelled")
        self.total_written = total_written


def _interruptible_sleep(seconds: float, cancel_event: threading.Event | None) -> None:
    """time.sleep(seconds), but a set cancel_event wakes it immediately
    instead of letting a chunk wait out a full (up to 120s) rate-limit
    backoff before noticing a user-requested stop.
    """
    if cancel_event is not None:
        if cancel_event.wait(seconds):
            raise IngestCancelled()
    else:
        time.sleep(seconds)


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
    rate_limiter: RateLimiter,
    on_retry: OnRetryFn | None = None,
    on_status: Callable[[str], None] | None = None,
    max_rate_limit_backoff_seconds: float = DEFAULT_MAX_RATE_LIMIT_BACKOFF_SECONDS,
    cancel_event: threading.Event | None = None,
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
        if cancel_event is not None and cancel_event.is_set():
            raise IngestCancelled()
        if on_status is not None:
            on_status("fetching")
        try:
            result = fetch_fn(
                pool_address, from_block, to_block, client, rpc_url,
                rate_limiter=rate_limiter, cancel_event=cancel_event,
            )
        except Cancelled:
            raise IngestCancelled() from None
        except Exception as exc:  # RpcLogsError et al — real transient failures
            is_rate_limited = isinstance(exc, RateLimitError)
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
                _interruptible_sleep(sleep_seconds, cancel_event)
                continue

            failure_attempt += 1
            if failure_attempt >= max_retries:
                raise
            sleep_seconds = backoff_seconds * (2 ** (failure_attempt - 1))
            if on_retry is not None:
                on_retry(from_block, to_block, failure_attempt, max_retries, sleep_seconds, str(exc))
            _interruptible_sleep(sleep_seconds, cancel_event)
        else:
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
    max_rps: float = DEFAULT_MAX_RPS,
    flush_every_n_chunks: int = DEFAULT_FLUSH_EVERY_N_CHUNKS,
    concurrency_cooldown_seconds: float = DEFAULT_CONCURRENCY_COOLDOWN_SECONDS,
    max_rate_limit_backoff_seconds: float = DEFAULT_MAX_RATE_LIMIT_BACKOFF_SECONDS,
    on_progress: Callable[[int, int, int], None] | None = None,
    on_retry: OnRetryFn | None = None,
    on_queue_status: Callable[[int, int | None], None] | None = None,
    on_worker_status: Callable[[int, int, int, str], None] | None = None,
    on_rate_change: Callable[[float], None] | None = None,
    on_ceiling_change: Callable[[float], None] | None = None,
    fetch_fn: FetchFn = fetch_swaps,
    cancel_event: threading.Event | None = None,
) -> int:
    """Ingest [from_block, to_block] concurrently, in chunks, flushing to
    disk (and recording newly-covered intervals in the resumable
    progress file) every flush_every_n_chunks chunks — never per-chunk,
    which would mean rewriting the whole output file hundreds of
    thousands of times over a real year-long pull. Returns the total
    number of swap records written in this call. Any sub-range of
    [from_block, to_block] already present in a prior run's progress is
    skipped, never re-fetched.

    A caller (the TUI's ctrl+q handler) can set cancel_event to stop
    early — in-flight chunks are given a chance to finish or notice the
    cancellation quickly (rather than being abandoned mid-flight), then
    whatever completed gets flushed exactly like a normal run, and
    IngestCancelled is raised (carrying the partial total_written) so
    the caller can tell a deliberate stop apart from a real failure.
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

    persisted_rate = read_rate_limit(out_path)
    persisted_ceiling = read_rate_ceiling(out_path)
    # A self-raised ceiling from a prior run is real, measured headroom —
    # never resume below the config default even if it's somehow higher
    # (e.g. LATENTEDGE_INGEST_MAX_RPS was raised since that run).
    ceiling = max(persisted_ceiling, max_rps) if persisted_ceiling is not None else max_rps
    # A floor, not a fixed resume point — see RateLimiter's docstring and
    # AdaptiveConcurrencyLimiter's history for why: never resume below
    # half the ceiling regardless of how low a prior run bottomed out.
    start_rate = max(persisted_rate, ceiling / 2) if persisted_rate is not None else None
    rate_limiter = RateLimiter(
        ceiling=ceiling,
        cooldown_seconds=concurrency_cooldown_seconds,
        start_rate=start_rate,
        on_change=on_rate_change,
        on_ceiling_change=on_ceiling_change,
    )
    # A resumed run can start below its ceiling (the floor-clamped
    # persisted rate above), or with a ceiling already raised past
    # max_rps — without this, a caller (e.g. the TUI) has no way to know
    # either actual starting point and would assume the config defaults,
    # misreporting the rate, its direction of change, and the ceiling.
    if on_rate_change is not None and rate_limiter.rate != ceiling:
        on_rate_change(rate_limiter.rate)
    if on_ceiling_change is not None and rate_limiter.ceiling != max_rps:
        on_ceiling_change(rate_limiter.ceiling)

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
            max_retries, retry_backoff_seconds, rate_limiter, retry_hook, status_hook,
            max_rate_limit_backoff_seconds, cancel_event,
        )

        if on_worker_status is not None:
            on_worker_status(slot, chunk_start, chunk_end, "idle")
        return chunk_start, chunk_end, records

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_chunk, cs): cs for cs in chunk_order}
        cancelled = False
        try:
            for future in as_completed(futures):
                try:
                    chunk_start, chunk_end, records = future.result()
                except IngestCancelled:
                    # One cancelled chunk means every not-yet-started
                    # chunk will raise the same way in turn — no point
                    # waiting for the rest of as_completed to churn
                    # through them one at a time.
                    cancelled = True
                    break

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
            # or was cut short — whatever rate/ceiling the limiter
            # actually settled at is the useful starting point for a
            # resume, not just the happy-path ending level.
            write_rate_limit(out_path, rate_limiter.rate)
            write_rate_ceiling(out_path, rate_limiter.ceiling)

    if cancelled:
        raise IngestCancelled(total_written)
    return total_written
