"""Ingestion: decode Swap events directly via eth_getLogs.

Swap(address indexed sender, address indexed recipient, int256 amount0,
     int256 amount1, uint160 sqrtPriceX96, uint128 liquidity, int24 tick)
"""

from typing import Any

import httpx

from latentedge.schema import SwapRecord

# keccak256("Swap(address,address,int256,int256,uint160,uint128,int24)")
SWAP_TOPIC = "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"


class RpcLogsError(Exception):
    pass


def _rpc_call(client: httpx.Client, rpc_url: str, method: str, params: list[Any]) -> Any:
    response = client.post(rpc_url, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
    if response.status_code != 200:
        raise RpcLogsError(f"RPC returned HTTP {response.status_code}: {response.text}")
    payload = response.json()
    if "error" in payload:
        raise RpcLogsError(f"RPC error: {payload['error']}")
    return payload["result"]


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


def fetch_swaps(pool_address: str, from_block: int, to_block: int, client: httpx.Client, rpc_url: str) -> list[SwapRecord]:
    logs = _rpc_call(
        client,
        rpc_url,
        "eth_getLogs",
        [
            {
                "address": pool_address,
                "topics": [SWAP_TOPIC],
                "fromBlock": hex(from_block),
                "toBlock": hex(to_block),
            }
        ],
    )

    records: list[SwapRecord] = []
    # Cache both the timestamp and base fee from a single block fetch —
    # eth_getBlockByNumber's response already carries baseFeePerGas, so
    # there's no need for a second round-trip per block (see
    # ingest.pool_state.backfill_base_fee, kept as a fallback utility for
    # records sourced without it, e.g. pre-London blocks with no base fee).
    block_cache: dict[int, tuple[int, int]] = {}

    for log in logs:
        block_number = int(log["blockNumber"], 16)
        if block_number not in block_cache:
            block = _rpc_call(client, rpc_url, "eth_getBlockByNumber", [log["blockNumber"], False])
            timestamp = int(block["timestamp"], 16)
            base_fee_wei = int(block["baseFeePerGas"], 16) if "baseFeePerGas" in block else 0
            block_cache[block_number] = (timestamp, base_fee_wei)
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
