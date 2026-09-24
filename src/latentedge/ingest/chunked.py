"""Chunked, resumable ingestion over a large block range.

Real archive RPC providers cap eth_getLogs to a limited block range per
call, and a multi-hour pull (e.g. a year of history) needs to survive
transient errors and resume after a crash without re-fetching everything
from scratch.
"""

import json
import time
from collections.abc import Callable
from pathlib import Path

import httpx

from latentedge.ingest.rpc_logs import fetch_swaps
from latentedge.schema import SwapRecord
from latentedge.store import write_swaps

DEFAULT_CHUNK_SIZE = 2000
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_SECONDS = 2.0

FetchFn = Callable[[str, int, int, httpx.Client, str], list[SwapRecord]]


def _progress_path(out_path: Path) -> Path:
    return Path(str(out_path) + ".progress.json")


def read_progress(out_path: Path) -> int | None:
    """The last fully-completed block number, or None if no prior run."""
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
    on_progress: Callable[[int, int, int], None] | None = None,
    fetch_fn: FetchFn = fetch_swaps,
) -> int:
    """Ingest [from_block, to_block] in chunks, writing and recording
    progress after each one. Returns the total number of swap records
    written across all chunks in this call (not counting chunks skipped
    because a prior run already completed them). Resumes automatically:
    if a progress file from an earlier run covers part of the requested
    range, that part is skipped rather than re-fetched.
    """
    resume_from = read_progress(out_path)
    start_block = from_block
    if resume_from is not None and resume_from + 1 > start_block:
        start_block = resume_from + 1

    total_written = 0
    chunk_start = start_block
    while chunk_start <= to_block:
        chunk_end = min(chunk_start + chunk_size - 1, to_block)

        records = _fetch_chunk_with_retries(
            fetch_fn, pool_address, chunk_start, chunk_end, client, rpc_url, max_retries, retry_backoff_seconds
        )

        if records:
            write_swaps(records, out_path)
            total_written += len(records)

        write_progress(out_path, chunk_end)

        if on_progress is not None:
            on_progress(chunk_start, chunk_end, len(records))

        chunk_start = chunk_end + 1

    return total_written
