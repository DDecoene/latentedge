import httpx
import pytest

from latentedge import config
from latentedge.ingest.rpc_logs import RateLimitError, RpcLogsError, describe_error, fetch_swaps, get_latest_block
from latentedge.ingest.rpc_logs import _batch_fetch_blocks, _rpc_call

RPC_URL = "https://ethereum.publicnode.com"


def _latest_block_number(client: httpx.Client) -> int:
    response = client.post(RPC_URL, json={"jsonrpc": "2.0", "id": 1, "method": "eth_blockNumber", "params": []})
    return int(response.json()["result"], 16)


def test_fetch_swaps_decodes_real_logs():
    # Free public RPC providers gate deep historical (archive-depth)
    # eth_getLogs behind a paid token, but serve recent-block queries
    # freely — a recent window is enough to prove the decoder works
    # against the real service, which is this test's actual job.
    # A 500-block lookback (~100 minutes) proved flaky in practice — it
    # sits close enough to the free tier's retention boundary that it
    # intermittently gets archive-gated (observed twice). A much smaller
    # lookback stays safely inside the free window.
    #
    # Also, "latest" is fetched from a load-balanced multi-backend
    # service, and the specific backend that later serves eth_getLogs can
    # lag behind the one that answered eth_blockNumber, rejecting a
    # to_block at the literal bleeding edge ("block range extends beyond
    # current head block", observed once). A small buffer behind the tip
    # gives a lagging replica room to have caught up.
    with httpx.Client(timeout=30.0) as client:
        latest = _latest_block_number(client)
        to_block = latest - 5
        from_block = to_block - 50
        records = fetch_swaps(config.POOL_ADDRESS, from_block=from_block, to_block=to_block, client=client, rpc_url=RPC_URL)

    assert len(records) > 0
    for r in records:
        assert r.liquidity > 0
        assert r.sqrt_price_x96 > 0
        assert r.tx_hash.startswith("0x")
        assert from_block <= r.block_number <= to_block
        # base_fee_wei comes from the same eth_getBlockByNumber call
        # already made for the timestamp — no separate backfill pass
        # needed for records sourced this way (avoids fetching each
        # block twice).
        assert r.base_fee_wei > 0


def test_batch_fetch_blocks_matches_individually_fetched_results():
    # A year-long pull needs to fetch timestamps/base-fees for roughly a
    # million unique blocks — one HTTP round-trip per block would take
    # days. This confirms batched fetches return the same data as the
    # already-proven single-call path, for real blocks against the real
    # service.
    with httpx.Client(timeout=30.0) as client:
        latest = _latest_block_number(client)
        block_numbers = [latest - 5, latest - 4, latest - 3]

        batched = _batch_fetch_blocks(block_numbers, client, RPC_URL)

        for block_number in block_numbers:
            individual_response = client.post(
                RPC_URL,
                json={"jsonrpc": "2.0", "id": 1, "method": "eth_getBlockByNumber", "params": [hex(block_number), False]},
            )
            individual_block = individual_response.json()["result"]
            expected_timestamp = int(individual_block["timestamp"], 16)
            expected_base_fee = int(individual_block["baseFeePerGas"], 16)

            assert batched[block_number] == (expected_timestamp, expected_base_fee)


def test_get_latest_block_returns_a_plausible_recent_block_number():
    with httpx.Client(timeout=30.0) as client:
        latest = get_latest_block(client, RPC_URL)
        reference = _latest_block_number(client)

    # Two separate calls a moment apart won't return the exact same
    # block on a live chain — assert they're close instead of equal.
    assert abs(latest - reference) < 20


def _mock_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_fetch_swaps_sub_chunks_eth_getlogs_but_batches_blocks_in_one_call():
    # from_block=0..29 with a cap of 10 must issue 3 eth_getLogs calls
    # (one per 10-block sub-range) but only 1 eth_getBlockByNumber batch
    # call for every unique block across the whole span, not 3 — this is
    # the whole point of decoupling the getLogs cap from the block-batch
    # granularity.
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        body = _json.loads(request.content)
        if isinstance(body, list):
            calls.append("eth_getBlockByNumber_batch")
            return httpx.Response(
                200,
                json=[
                    {
                        "jsonrpc": "2.0",
                        "id": entry["id"],
                        "result": {"timestamp": hex(entry["id"] * 12), "baseFeePerGas": "0x1"},
                    }
                    for entry in body
                ],
            )

        calls.append(body["method"])
        from_block = int(body["params"][0]["fromBlock"], 16)
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "result": [
                    {
                        "address": "0xpool",
                        "blockNumber": hex(from_block),
                        "transactionHash": "0x" + "1" * 64,
                        "logIndex": "0x0",
                        "data": "0x" + "0" * 64 * 5,
                    }
                ],
            },
        )

    with _mock_client(handler) as client:
        records = fetch_swaps("0xpool", from_block=0, to_block=29, client=client, rpc_url=RPC_URL)

    assert calls.count("eth_getLogs") == 3
    assert calls.count("eth_getBlockByNumber_batch") == 1
    assert len(records) == 3


def test_rpc_call_raises_rate_limit_error_on_http_429():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="rate limited")

    with _mock_client(handler) as client:
        with pytest.raises(RateLimitError):
            _rpc_call(client, RPC_URL, "eth_blockNumber", [])


def test_batch_fetch_blocks_raises_rate_limit_error_on_http_429():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="rate limited")

    with _mock_client(handler) as client:
        with pytest.raises(RateLimitError):
            _batch_fetch_blocks([1, 2, 3], client, RPC_URL)


def test_batch_fetch_blocks_raises_rate_limit_error_on_embedded_429_code():
    # Real observed behavior (Alchemy): the HTTP status is 200, but an
    # individual entry in the JSON-RPC batch response carries a
    # {"code": 429, ...} error when compute-unit throughput is exceeded.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[
                {"jsonrpc": "2.0", "id": 1, "error": {"code": 429, "message": "compute units exceeded"}},
            ],
        )

    with _mock_client(handler) as client:
        with pytest.raises(RateLimitError):
            _batch_fetch_blocks([1], client, RPC_URL)


def test_describe_error_explains_connection_failure_in_plain_language():
    exc = httpx.ConnectError("[Errno 8] nodename nor servname provided, or not known")
    message = describe_error(RPC_URL, exc)

    assert "internet connection" in message
    assert RPC_URL in message
    # No raw errno/traceback jargon leaking into the user-facing message.
    assert "Errno" not in message


def test_describe_error_explains_timeout_in_plain_language():
    exc = httpx.TimeoutException("timed out")
    message = describe_error(RPC_URL, exc)

    assert "slow" in message or "timed out" in message.lower()
    assert RPC_URL in message


def test_describe_error_explains_rate_limit_in_plain_language():
    exc = RateLimitError("RPC rate limited (HTTP 429): ...")
    message = describe_error(RPC_URL, exc)

    assert "rate" in message.lower()


def test_describe_error_passes_through_other_rpc_errors_as_is():
    exc = RpcLogsError("RPC error: {'code': -32602, 'message': 'block range extends beyond current head block'}")
    message = describe_error(RPC_URL, exc)

    assert "block range extends beyond current head block" in message
