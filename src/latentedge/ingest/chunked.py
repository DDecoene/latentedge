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
"""

import json
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

from latentedge.ingest.rpc_logs import fetch_swaps
from latentedge.schema import SwapRecord
from latentedge.store import write_swaps

DEFAULT_CHUNK_SIZE = 10  # Alchemy's free-tier eth_getLogs cap; verified against the real service
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_SECONDS = 2.0
DEFAULT_MAX_WORKERS = 8
DEFAULT_FLUSH_EVERY_N_CHUNKS = 200

FetchFn = Callable[[str, int, int, httpx.Client, str], list[SwapRecord]]


def _progress_path(out_path: Path) -> Path:
    return Path(str(out_path) + ".progress.json")


def read_progress(out_path: Path) -> int | None:
    """The last fully-flushed block number, or None if no prior run."""
    path = _progress_path(out_path)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    last_completed_block: int = data["last_completed_block"]
    return last_completed_block


def write_progress(out_path: Path, last_completed_block: int) -> None:
    _progress_path(out_path).write_text(json.dumps({"last_completed_block": last_completed_block}))


def _fetch_chunk_with_retries(
    fetch_fn: FetchFn,
    pool_address: str,
    from_block: int,
    to_block: int,
    client: httpx.Client,
    rpc_url: str,
    max_retries: int,
    backoff_seconds: float,
) -> list[SwapRecord]:
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            return fetch_fn(pool_address, from_block, to_block, client, rpc_url)
        except Exception as exc:  # RpcLogsError et al — real transient failures
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(backoff_seconds * (2**attempt))
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
    fetch_fn: FetchFn = fetch_swaps,
) -> int:
    """Ingest [from_block, to_block] concurrently, in chunks, flushing to
    disk (and advancing the resumable progress watermark) every
    flush_every_n_chunks chunks — never per-chunk, which would mean
    rewriting the whole output file hundreds of thousands of times over
    a real year-long pull. Returns the total number of swap records
    written in this call. Resumes automatically from a prior run's
    progress file if the requested range overlaps it.
    """
    resume_from = read_progress(out_path)
    start_block = from_block
    if resume_from is not None and resume_from + 1 > start_block:
        start_block = resume_from + 1

    chunk_starts = list(range(start_block, to_block + 1, chunk_size))
    if not chunk_starts:
        return 0

    lock = threading.Lock()
    # chunk_start -> (chunk_end, records), for chunks that finished but
    # haven't yet been folded into the contiguous, flush-eligible run.
    completed: dict[int, tuple[int, list[SwapRecord]]] = {}
    next_watermark_start = start_block

    pending_records: list[SwapRecord] = []
    pending_last_block: int | None = None
    chunks_since_flush = 0
    total_written = 0

    def flush() -> None:
        nonlocal pending_records, pending_last_block, chunks_since_flush, total_written
        if pending_last_block is None:
            return
        if pending_records:
            write_swaps(pending_records, out_path)
            total_written += len(pending_records)
        write_progress(out_path, pending_last_block)
        pending_records = []
        chunks_since_flush = 0

    def process_chunk(chunk_start: int) -> tuple[int, int, list[SwapRecord]]:
        chunk_end = min(chunk_start + chunk_size - 1, to_block)
        records = _fetch_chunk_with_retries(
            fetch_fn, pool_address, chunk_start, chunk_end, client, rpc_url, max_retries, retry_backoff_seconds
        )
        return chunk_start, chunk_end, records

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_chunk, cs): cs for cs in chunk_starts}
        try:
            for future in as_completed(futures):
                chunk_start, chunk_end, records = future.result()

                with lock:
                    completed[chunk_start] = (chunk_end, records)

                    # Fold in whatever consecutive run is now available,
                    # starting from the current watermark — chunks can
                    # finish out of order, but the watermark (and what we
                    # flush/report) only ever advances through a
                    # contiguous, gap-free prefix.
                    while next_watermark_start in completed:
                        c_end, recs = completed.pop(next_watermark_start)
                        pending_records.extend(recs)
                        pending_last_block = c_end
                        chunks_since_flush += 1

                        if on_progress is not None:
                            on_progress(next_watermark_start, c_end, len(recs))

                        next_watermark_start = c_end + 1

                        if chunks_since_flush >= flush_every_n_chunks:
                            flush()
        finally:
            # A chunk that exhausted its retries raises out of
            # future.result() above; cancel whatever hasn't started yet
            # rather than let the pool keep firing more requests.
            for f in futures:
                f.cancel()

    with lock:
        flush()

    return total_written
