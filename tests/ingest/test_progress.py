import json
from pathlib import Path

from latentedge.ingest.progress import (
    add_interval,
    read_progress,
    read_rate_ceiling,
    read_rate_limit,
    uncovered_gaps,
    write_progress,
    write_rate_ceiling,
    write_rate_limit,
)
from latentedge.schema import SwapRecord
from latentedge.store import write_swaps


def test_read_progress_returns_empty_list_when_no_file_exists(tmp_path: Path):
    assert read_progress(tmp_path / "swaps.parquet") == []


def _record(block_number: int) -> SwapRecord:
    return SwapRecord(
        block_number=block_number, timestamp=block_number * 12,
        tx_hash=f"0x{block_number:064x}", log_index=0,
        sqrt_price_x96=1 << 96, tick=0, liquidity=10**18,
        amount0=1000.0, amount1=-0.3, base_fee_wei=20_000_000_000,
    )


def test_read_progress_migrates_the_legacy_single_watermark_format(tmp_path: Path):
    # Real progress files written before this change use
    # {"last_completed_block": N} with no recorded start — a run that
    # crashes on this instead of a clean migration would force deleting
    # real progress data and re-fetching blocks already on disk.
    out_path = tmp_path / "swaps.parquet"
    write_swaps([_record(100), _record(150)], out_path)
    (tmp_path / "swaps.parquet.progress.json").write_text(json.dumps({"last_completed_block": 150}))

    assert read_progress(out_path) == [(100, 150)]


def test_write_then_read_progress_round_trips(tmp_path: Path):
    out_path = tmp_path / "swaps.parquet"
    write_progress(out_path, [(0, 99), (200, 299)])
    assert read_progress(out_path) == [(0, 99), (200, 299)]


def test_read_rate_limit_returns_none_when_no_file_exists(tmp_path: Path):
    assert read_rate_limit(tmp_path / "swaps.parquet") is None


def test_write_then_read_rate_limit_round_trips(tmp_path: Path):
    out_path = tmp_path / "swaps.parquet"
    write_rate_limit(out_path, 3.5)
    assert read_rate_limit(out_path) == 3.5


def test_read_rate_ceiling_returns_none_when_no_file_exists(tmp_path: Path):
    assert read_rate_ceiling(tmp_path / "swaps.parquet") is None


def test_write_then_read_rate_ceiling_round_trips(tmp_path: Path):
    out_path = tmp_path / "swaps.parquet"
    write_rate_ceiling(out_path, 12.5)
    assert read_rate_ceiling(out_path) == 12.5


def test_rate_and_ceiling_are_independently_writable_without_clobbering_each_other(tmp_path: Path):
    out_path = tmp_path / "swaps.parquet"
    write_rate_limit(out_path, 3.5)
    write_rate_ceiling(out_path, 12.5)
    assert read_rate_limit(out_path) == 3.5
    assert read_rate_ceiling(out_path) == 12.5

    write_rate_limit(out_path, 4.0)
    assert read_rate_limit(out_path) == 4.0
    assert read_rate_ceiling(out_path) == 12.5


def test_add_interval_merges_overlapping_ranges():
    assert add_interval([(0, 99)], 50, 149) == [(0, 149)]


def test_add_interval_merges_adjacent_ranges():
    # 100 immediately follows 99 — must merge into one interval instead
    # of leaving two touching-but-separate entries.
    assert add_interval([(0, 99)], 100, 149) == [(0, 149)]


def test_add_interval_keeps_disjoint_ranges_separate():
    assert add_interval([(0, 99)], 200, 299) == [(0, 99), (200, 299)]


def test_add_interval_bridges_a_gap_between_two_existing_ranges():
    assert add_interval([(0, 99), (200, 299)], 100, 199) == [(0, 299)]


def test_add_interval_into_empty_list():
    assert add_interval([], 10, 20) == [(10, 20)]


def test_uncovered_gaps_with_no_existing_progress():
    assert uncovered_gaps([], 0, 99) == [(0, 99)]


def test_uncovered_gaps_fully_covered_returns_nothing():
    assert uncovered_gaps([(0, 99)], 0, 99) == []


def test_uncovered_gaps_covered_in_the_middle():
    assert uncovered_gaps([(40, 59)], 0, 99) == [(0, 39), (60, 99)]


def test_uncovered_gaps_covered_at_the_edges():
    assert uncovered_gaps([(0, 19), (80, 99)], 0, 99) == [(20, 79)]


def test_uncovered_gaps_ignores_intervals_outside_the_requested_range():
    assert uncovered_gaps([(200, 299)], 0, 99) == [(0, 99)]


def test_uncovered_gaps_returns_nothing_for_an_empty_range():
    assert uncovered_gaps([], 10, 5) == []
