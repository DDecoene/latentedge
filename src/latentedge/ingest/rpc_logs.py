"""Ingestion: decode Swap events directly via eth_getLogs.

Swap(address indexed sender, address indexed recipient, int256 amount0,
     int256 amount1, uint160 sqrtPriceX96, uint128 liquidity, int24 tick)
"""

import threading
from typing import TYPE_CHECKING, Any

import httpx

from latentedge.schema import SwapRecord

if TYPE_CHECKING:
    from latentedge.ingest.rate_limiter import RateLimiter

# keccak256("Swap(address,address,int256,int256,uint160,uint128,int24)")
SWAP_TOPIC = "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"

# Alchemy's free-tier eth_getLogs cap; verified against the real service.
# This bounds only the getLogs sub-calls fetch_swaps makes internally —
# callers pass whatever from_block/to_block span they want and fetch_swaps
# sub-chunks it, so the (much larger) block-timestamp batch below isn't
# forced down to this same tiny granularity.
ETH_GETLOGS_RANGE_CAP = 10


class RpcLogsError(Exception):
    pass


class RateLimitError(RpcLogsError):
    """The provider rejected a request for exceeding its rate/throughput
    limit — either an HTTP 429, or (observed on Alchemy) an HTTP 200
    whose JSON-RPC batch response carries a {"code": 429, ...} error on
    individual entries. Retried with a longer backoff than other
    errors, since a short retry just re-triggers the same limit.
    """


def _rpc_call(
    client: httpx.Client,
    rpc_url: str,
    method: str,
    params: list[Any],
    rate_limiter: "RateLimiter | None" = None,
    cancel_event: threading.Event | None = None,
) -> Any:
    if rate_limiter is not None:
        rate_limiter.acquire(cancel_event)
    response = client.post(rpc_url, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
    if response.status_code == 429:
        if rate_limiter is not None:
            rate_limiter.release("rate_limited")
        raise RateLimitError(f"RPC rate limited (HTTP 429): {response.text}")
    if response.status_code != 200:
        if rate_limiter is not None:
            rate_limiter.release("failed")
        raise RpcLogsError(f"RPC returned HTTP {response.status_code}: {response.text}")
    payload = response.json()
    if "error" in payload:
        if rate_limiter is not None:
            rate_limiter.release("failed")
        raise RpcLogsError(f"RPC error: {payload['error']}")
    if rate_limiter is not None:
        rate_limiter.release("success")
    return payload["result"]


def get_latest_block(
    client: httpx.Client,
    rpc_url: str,
    rate_limiter: "RateLimiter | None" = None,
    cancel_event: threading.Event | None = None,
) -> int:
    result: str = _rpc_call(client, rpc_url, "eth_blockNumber", [], rate_limiter=rate_limiter, cancel_event=cancel_event)
    return int(result, 16)


def describe_error(rpc_url: str, exc: Exception) -> str:
    """Turn a network/RPC exception into a message a person can act on,
    instead of a raw traceback full of errno numbers and stack frames.
    """
    if isinstance(exc, httpx.ConnectError):
        return f"Could not reach {rpc_url} — check your internet connection and the RPC URL."
    if isinstance(exc, httpx.TimeoutException):
        return f"Timed out talking to {rpc_url} — the endpoint may be slow or unreachable right now."
    if isinstance(exc, RateLimitError):
        return f"{rpc_url} is rate-limiting requests — retrying with a longer backoff; if this keeps happening, try a lower --max-workers or a paid RPC tier."
    if isinstance(exc, RpcLogsError):
        return str(exc)
    if isinstance(exc, httpx.HTTPError):
        return f"Network error talking to {rpc_url}: {exc}"
    return str(exc)


BLOCK_BATCH_SIZE = 100


def _batch_fetch_blocks(
    block_numbers: list[int],
    client: httpx.Client,
    rpc_url: str,
    rate_limiter: "RateLimiter | None" = None,
    cancel_event: threading.Event | None = None,
) -> dict[int, tuple[int, int]]:
    """Fetch (timestamp, base_fee_wei) for each block number in as few
    HTTP round-trips as possible — a JSON-RPC batch request per
    BLOCK_BATCH_SIZE blocks, rather than one call per block. A year of
    history for an actively-traded pool can mean fetching timestamps for
    roughly a million unique blocks; one-at-a-time would take days.
    """
    result: dict[int, tuple[int, int]] = {}
    unique_blocks = list(dict.fromkeys(block_numbers))  # de-dupe, keep order

    for i in range(0, len(unique_blocks), BLOCK_BATCH_SIZE):
        batch = unique_blocks[i : i + BLOCK_BATCH_SIZE]
        payload = [
            {"jsonrpc": "2.0", "id": block_number, "method": "eth_getBlockByNumber", "params": [hex(block_number), False]}
            for block_number in batch
        ]
        if rate_limiter is not None:
            rate_limiter.acquire(cancel_event)
        response = client.post(rpc_url, json=payload)
        if response.status_code == 429:
            if rate_limiter is not None:
                rate_limiter.release("rate_limited")
            raise RateLimitError(f"RPC rate limited (HTTP 429): {response.text}")
        if response.status_code != 200:
            if rate_limiter is not None:
                rate_limiter.release("failed")
            raise RpcLogsError(f"RPC returned HTTP {response.status_code}: {response.text}")

        responses = response.json()
        by_id = {entry["id"]: entry for entry in responses}

        for block_number in batch:
            entry = by_id.get(block_number)
            if entry is None:
                if rate_limiter is not None:
                    rate_limiter.release("failed")
                raise RpcLogsError(f"batch response missing block {block_number}")
            if "error" in entry:
                if entry["error"].get("code") == 429:
                    if rate_limiter is not None:
                        rate_limiter.release("rate_limited")
                    raise RateLimitError(f"RPC rate limited for block {block_number}: {entry['error']}")
                if rate_limiter is not None:
                    rate_limiter.release("failed")
                raise RpcLogsError(f"RPC error for block {block_number}: {entry['error']}")

            block = entry["result"]
            timestamp = int(block["timestamp"], 16)
            base_fee_wei = int(block["baseFeePerGas"], 16) if "baseFeePerGas" in block else 0
            result[block_number] = (timestamp, base_fee_wei)

        if rate_limiter is not None:
            rate_limiter.release("success")

    return result


def get_block_at_or_after_timestamp(
    client: httpx.Client,
    rpc_url: str,
    target_timestamp: int,
    floor_block: int,
    head_block: int,
    rate_limiter: "RateLimiter | None" = None,
    cancel_event: threading.Event | None = None,
) -> int:
    """Binary search [floor_block, head_block] for the earliest block
    whose timestamp is >= target_timestamp — an exact, verifiable
    anchor for a "--days" window, instead of estimating from a constant
    average block time that drifts from the chain's real block times.
    Clamps to floor_block or head_block when target_timestamp falls
    outside the range those bounds actually cover.
    """
    lo, hi = floor_block, head_block
    while lo < hi:
        mid = (lo + hi) // 2
        timestamp, _ = _batch_fetch_blocks([mid], client, rpc_url, rate_limiter=rate_limiter, cancel_event=cancel_event)[mid]
        if timestamp >= target_timestamp:
            hi = mid
        else:
            lo = mid + 1
    return lo


def _decode_int(hex_str: str, bits: int, signed: bool) -> int:
    value = int(hex_str, 16)
    if signed and value >= (1 << (bits - 1)):
        value -= 1 << bits
    return value


def _decode_swap_data(data_hex: str) -> tuple[float, float, int, int, int]:
    data = data_hex[2:]  # strip 0x
    words = [data[i : i + 64] for i in range(0, len(data), 64)]
    amount0 = _decode_int(words[0], 256, signed=True)
    amount1 = _decode_int(words[1], 256, signed=True)
    sqrt_price_x96 = _decode_int(words[2], 256, signed=False)
    liquidity = _decode_int(words[3], 256, signed=False)
    tick = _decode_int(words[4], 256, signed=True)
    return float(amount0), float(amount1), sqrt_price_x96, liquidity, tick


def fetch_swaps(
    pool_address: str,
    from_block: int,
    to_block: int,
    client: httpx.Client,
    rpc_url: str,
    eth_getlogs_range_cap: int = ETH_GETLOGS_RANGE_CAP,
    rate_limiter: "RateLimiter | None" = None,
    cancel_event: threading.Event | None = None,
) -> list[SwapRecord]:
    # eth_getLogs itself must stay within the provider's tiny per-call
    # range cap, but that cap has nothing to do with how many blocks'
    # worth of timestamps get batched into one eth_getBlockByNumber
    # round-trip below — sub-chunking only the getLogs half here, over
    # whatever (larger) span the caller asked for, means a caller
    # requesting e.g. 100 blocks makes 10 getLogs calls but still only
    # 1 block-timestamp batch call instead of 10.
    logs: list[dict[str, Any]] = []
    for sub_from in range(from_block, to_block + 1, eth_getlogs_range_cap):
        sub_to = min(sub_from + eth_getlogs_range_cap - 1, to_block)
        logs.extend(
            _rpc_call(
                client,
                rpc_url,
                "eth_getLogs",
                [
                    {
                        "address": pool_address,
                        "topics": [SWAP_TOPIC],
                        "fromBlock": hex(sub_from),
                        "toBlock": hex(sub_to),
                    }
                ],
                rate_limiter=rate_limiter,
                cancel_event=cancel_event,
            )
        )

    # Both the timestamp and base fee come from the same block fetch —
    # eth_getBlockByNumber's response already carries baseFeePerGas, so
    # there's no need for a second round-trip per block (see
    # ingest.pool_state.backfill_base_fee, kept as a fallback utility for
    # records sourced without it, e.g. pre-London blocks with no base fee).
    # Batched rather than one call per block: a year of history for an
    # actively-traded pool can touch close to a million unique blocks.
    unique_block_numbers = [int(log["blockNumber"], 16) for log in logs]
    block_cache = _batch_fetch_blocks(unique_block_numbers, client, rpc_url, rate_limiter=rate_limiter, cancel_event=cancel_event)

    records: list[SwapRecord] = []
    for log in logs:
        block_number = int(log["blockNumber"], 16)
        timestamp, base_fee_wei = block_cache[block_number]

        amount0, amount1, sqrt_price_x96, liquidity, tick = _decode_swap_data(log["data"])

        records.append(
            SwapRecord(
                block_number=block_number,
                timestamp=timestamp,
                tx_hash=log["transactionHash"],
                log_index=int(log["logIndex"], 16),
                sqrt_price_x96=sqrt_price_x96,
                tick=tick,
                liquidity=liquidity,
                amount0=amount0,
                amount1=amount1,
                base_fee_wei=base_fee_wei,
            )
        )

    return records
