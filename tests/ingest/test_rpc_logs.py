import httpx

from latentedge import config
from latentedge.ingest.rpc_logs import fetch_swaps, get_latest_block
from latentedge.ingest.rpc_logs import _batch_fetch_blocks

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
