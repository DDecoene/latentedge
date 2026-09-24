from pathlib import Path

import pandas as pd

from latentedge.schema import SwapRecord
from latentedge.store import read_swaps, write_swaps


def _record(tx_hash: str, log_index: int, timestamp: int) -> SwapRecord:
    return SwapRecord(
        block_number=100,
        timestamp=timestamp,
        tx_hash=tx_hash,
        log_index=log_index,
        sqrt_price_x96=1 << 96,
        tick=0,
        liquidity=10**18,
        amount0=1000.0,
        amount1=-0.3,
        base_fee_wei=20_000_000_000,
    )


def test_write_and_read_round_trip(tmp_path: Path):
    path = tmp_path / "swaps.parquet"
    write_swaps([_record("0xabc", 0, 100), _record("0xabc", 1, 101)], path)
    df = read_swaps(path)
    assert len(df) == 2
    assert list(df["timestamp"]) == [100, 101]


def test_write_dedupes_on_tx_hash_and_log_index(tmp_path: Path):
    path = tmp_path / "swaps.parquet"
    write_swaps([_record("0xabc", 0, 100)], path)
    write_swaps([_record("0xabc", 0, 100), _record("0xdef", 0, 102)], path)
    df = read_swaps(path)
    assert len(df) == 2
    assert set(df["tx_hash"]) == {"0xabc", "0xdef"}
