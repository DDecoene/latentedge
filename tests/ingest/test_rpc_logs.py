import httpx

from latentedge import config
from latentedge.ingest.rpc_logs import fetch_swaps

RPC_URL = "https://ethereum.publicnode.com"


def _latest_block_number(client: httpx.Client) -> int:
    response = client.post(RPC_URL, json={"jsonrpc": "2.0", "id": 1, "method": "eth_blockNumber", "params": []})
    return int(response.json()["result"], 16)


def test_fetch_swaps_decodes_real_logs():
    # Free public RPC providers gate deep historical (archive-depth)
    # eth_getLogs behind a paid token, but serve recent-block queries
    # freely — a recent window is enough to prove the decoder works
    # against the real service, which is this test's actual job.
    with httpx.Client(timeout=30.0) as client:
        latest = _latest_block_number(client)
        from_block = latest - 500
        records = fetch_swaps(config.POOL_ADDRESS, from_block=from_block, to_block=latest, client=client, rpc_url=RPC_URL)

    assert len(records) > 0
    for r in records:
        assert r.liquidity > 0
        assert r.sqrt_price_x96 > 0
        assert r.tx_hash.startswith("0x")
        assert from_block <= r.block_number <= latest
