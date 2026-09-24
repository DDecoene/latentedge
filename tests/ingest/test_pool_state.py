import httpx

from latentedge.ingest.pool_state import backfill_base_fee
from latentedge.schema import SwapRecord

RPC_URL = "https://ethereum.publicnode.com"


def _bare_record(block_number: int, timestamp: int) -> SwapRecord:
    return SwapRecord(
        block_number=block_number,
        timestamp=timestamp,
        tx_hash="0x" + "0" * 64,
        log_index=0,
        sqrt_price_x96=1,
        tick=0,
        liquidity=123,  # already populated by Task 6; must survive untouched
        amount0=0.0,
        amount1=0.0,
        base_fee_wei=0,
    )


def test_backfill_populates_base_fee_and_preserves_liquidity():
    # A real, past mainnet block known to be well within archive range.
    record = _bare_record(block_number=17_500_000, timestamp=1_688_000_000)
    with httpx.Client(timeout=30.0) as client:
        result = backfill_base_fee([record], client, rpc_url=RPC_URL)

    assert len(result) == 1
    assert result[0].base_fee_wei > 0
    assert result[0].liquidity == 123
