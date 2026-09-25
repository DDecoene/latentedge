import json
from pathlib import Path

from latentedge.ingest.progress import (
    add_interval,
    extend_window_for_new_blocks,
    read_concurrency_limit,
    read_progress,
    uncovered_gaps,
    write_concurrency_limit,
    write_progress,
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


def test_read_concurrency_limit_returns_none_when_no_file_exists(tmp_path: Path):
    assert read_concurrency_limit(tmp_path / "swaps.parquet") is None


def test_write_then_read_concurrency_limit_round_trips(tmp_path: Path):
    out_path = tmp_path / "swaps.parquet"
    write_concurrency_limit(out_path, 3)
    assert read_concurrency_limit(out_path) == 3


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


def test_extend_window_returns_naive_from_when_already_enough_new_blocks():
    result = extend_window_for_new_blocks([], naive_from=100, naive_to=199, desired_new_blocks=100, floor_block=0)
    assert result == 100


def test_extend_window_walks_back_when_naive_window_fully_covered():
    intervals = [(100, 199)]
    result = extend_window_for_new_blocks(intervals, naive_from=100, naive_to=199, desired_new_blocks=100, floor_block=0)
    assert result == 0
    assert sum(e - s + 1 for s, e in uncovered_gaps(intervals, result, 199)) == 100


def test_extend_window_walks_past_a_covered_stretch_in_the_extension_zone():
    intervals = [(80, 99)]  # covered stretch sits below naive_from
    result = extend_window_for_new_blocks(intervals, naive_from=100, naive_to=199, desired_new_blocks=150, floor_block=0)
    assert result == 30
    assert sum(e - s + 1 for s, e in uncovered_gaps(intervals, result, 199)) == 150


def test_extend_window_clamps_at_floor_block_when_not_enough_new_blocks_exist():
    result = extend_window_for_new_blocks([], naive_from=100, naive_to=199, desired_new_blocks=1000, floor_block=50)
    assert result == 50


def test_extend_window_never_returns_below_floor_block_when_naive_from_is_already_below_it():
    # naive_from can sit below floor_block on its own (e.g. an
    # oversized --days value) — the contract is "never below
    # floor_block", not "never below floor_block only when we had to
    # extend to get there".
    result = extend_window_for_new_blocks([], naive_from=50, naive_to=199, desired_new_blocks=100, floor_block=100)
    assert result == 100
