"""Backfills per-block base fee via a JSON-RPC archive node.

Liquidity is populated directly from the raw Swap event during RPC
ingestion (see ingest.rpc_logs) and is never re-fetched here.
"""

from typing import Any

import httpx

from latentedge.schema import SwapRecord


class PoolStateError(Exception):
    pass


def _rpc_call(client: httpx.Client, rpc_url: str, method: str, params: list[Any]) -> dict[str, Any]:
    response = client.post(rpc_url, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
    if response.status_code != 200:
        raise PoolStateError(f"RPC returned HTTP {response.status_code}: {response.text}")
    payload = response.json()
    if "error" in payload:
        raise PoolStateError(f"RPC error: {payload['error']}")
    result: dict[str, Any] = payload["result"]
    return result


def backfill_base_fee(records: list[SwapRecord], client: httpx.Client, rpc_url: str) -> list[SwapRecord]:
    updated: list[SwapRecord] = []
    block_cache: dict[int, int] = {}

    for record in records:
        if record.block_number not in block_cache:
            block_hex = hex(record.block_number)
            block = _rpc_call(client, rpc_url, "eth_getBlockByNumber", [block_hex, False])
            block_cache[record.block_number] = int(block["baseFeePerGas"], 16)

        updated.append(record.model_copy(update={"base_fee_wei": block_cache[record.block_number]}))
    return updated
