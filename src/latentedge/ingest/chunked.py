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

from latentedge.ingest.concurrency import AdaptiveConcurrencyLimiter, ReleaseOutcome
from latentedge.ingest.progress import Interval, add_interval, read_progress, uncovered_gaps, write_progress
from latentedge.ingest.rpc_logs import RateLimitError, fetch_swaps
from latentedge.schema import SwapRecord
from latentedge.store import write_swaps

DEFAULT_CHUNK_SIZE = 10  # Alchemy's free-tier eth_getLogs cap; verified against the real service
DEFAULT_MAX_RETRIES = 5
DEFAULT_RETRY_BACKOFF_SECONDS = 2.0
# A rate limit retried on the same short schedule as a generic transient
# error just re-triggers the same limit (observed for real against
# Alchemy's free tier) — back off substantially longer for it.
RATE_LIMIT_BACKOFF_MULTIPLIER = 5.0
# Lower than earlier default (8): fewer concurrent workers means fewer
# simultaneous requests competing for the same per-second compute-unit
# budget, observed for real to matter more than backoff tuning alone.
DEFAULT_MAX_WORKERS = 4
# Lower than earlier default (200): a chunk that ultimately can't
# recover from a rate limit no longer loses unflushed progress (see
# ingest_range's finally-block flush), but flushing more often still
# bounds how much work a mid-run interruption could repeat on resume.
DEFAULT_FLUSH_EVERY_N_CHUNKS = 50

FetchFn = Callable[[str, int, int, httpx.Client, str], list[SwapRecord]]


OnRetryFn = Callable[[int, int, int, int, float, str], None]


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
) -> list[SwapRecord]:
    last_error: Exception | None = None
    for attempt in range(max_retries):
        limiter.acquire()
        try:
            result = fetch_fn(pool_address, from_block, to_block, client, rpc_url)
        except Exception as exc:  # RpcLogsError et al — real transient failures
            outcome: ReleaseOutcome = "rate_limited" if isinstance(exc, RateLimitError) else "failed"
            new_limit, changed = limiter.release(outcome)
            if changed and on_concurrency_change is not None:
                on_concurrency_change(new_limit)

            last_error = exc
            if attempt < max_retries - 1:
                sleep_seconds = backoff_seconds * (2**attempt)
                if isinstance(exc, RateLimitError):
                    sleep_seconds *= RATE_LIMIT_BACKOFF_MULTIPLIER
                if on_retry is not None:
                    on_retry(from_block, to_block, attempt + 1, max_retries, sleep_seconds, str(exc))
                time.sleep(sleep_seconds)
        else:
            new_limit, changed = limiter.release("success")
            if changed and on_concurrency_change is not None:
                on_concurrency_change(new_limit)
            return result
    assert last_error is not None
    raise last_error


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

    limiter = AdaptiveConcurrencyLimiter(ceiling=max_workers)

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
        if on_worker_status is not None:
            on_worker_status(slot, chunk_start, chunk_end, "fetching")

        def report_retry(cs: int, ce: int, attempt: int, retries: int, sleep_seconds: float, error_message: str) -> None:
            if on_retry is not None:
                on_retry(cs, ce, attempt, retries, sleep_seconds, error_message)
            if on_worker_status is not None:
                on_worker_status(slot, cs, ce, f"retry {attempt}/{retries}, waiting {sleep_seconds:.1f}s")

        retry_hook = report_retry if (on_retry is not None or on_worker_status is not None) else None
        records = _fetch_chunk_with_retries(
            fetch_fn, pool_address, chunk_start, chunk_end, client, rpc_url,
            max_retries, retry_backoff_seconds, limiter, retry_hook, on_concurrency_change,
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

    return total_written
