"""Pure interval arithmetic for tracking which block ranges have
already been ingested for a given output file. No I/O beyond reading
and writing the interval list itself, no threading — this is the
single source of truth `ingest_range` and the CLI both consult so a
block already recorded here is never requested again.
"""

import json
from pathlib import Path

Interval = tuple[int, int]


def _progress_path(out_path: Path) -> Path:
    return Path(str(out_path) + ".progress.json")


def read_progress(out_path: Path) -> list[Interval]:
    path = _progress_path(out_path)
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return [(pair[0], pair[1]) for pair in data["ingested"]]


def write_progress(out_path: Path, intervals: list[Interval]) -> None:
    _progress_path(out_path).write_text(json.dumps({"ingested": [list(pair) for pair in intervals]}))


def add_interval(intervals: list[Interval], new_from: int, new_to: int) -> list[Interval]:
    merged: list[Interval] = []
    placed = False
    cur_from, cur_to = new_from, new_to
    for start, end in sorted(intervals):
        if end < cur_from - 1:
            merged.append((start, end))
        elif start > cur_to + 1:
            if not placed:
                merged.append((cur_from, cur_to))
                placed = True
            merged.append((start, end))
        else:
            cur_from = min(cur_from, start)
            cur_to = max(cur_to, end)
    if not placed:
        merged.append((cur_from, cur_to))
    return merged


def uncovered_gaps(intervals: list[Interval], from_block: int, to_block: int) -> list[Interval]:
    if from_block > to_block:
        return []
    gaps: list[Interval] = []
    cursor = from_block
    for start, end in sorted(intervals):
        if end < from_block or start > to_block:
            continue
        clipped_start = max(start, from_block)
        clipped_end = min(end, to_block)
        if clipped_start > cursor:
            gaps.append((cursor, clipped_start - 1))
        cursor = max(cursor, clipped_end + 1)
    if cursor <= to_block:
        gaps.append((cursor, to_block))
    return gaps


def extend_window_for_new_blocks(
    intervals: list[Interval],
    naive_from: int,
    naive_to: int,
    desired_new_blocks: int,
    floor_block: int,
) -> int:
    """The smallest from_block <= naive_from (never below floor_block)
    such that [from_block, naive_to] contains at least
    desired_new_blocks blocks not already covered by `intervals` — or
    floor_block if even the full [floor_block, naive_to] range doesn't
    have that many.
    """
    new_in_naive = sum(end - start + 1 for start, end in uncovered_gaps(intervals, naive_from, naive_to))
    remaining_needed = desired_new_blocks - new_in_naive
    if remaining_needed <= 0:
        return naive_from
    if naive_from <= floor_block:
        return floor_block

    gaps_below = uncovered_gaps(intervals, floor_block, naive_from - 1)
    from_block = floor_block
    for start, end in reversed(gaps_below):
        gap_size = end - start + 1
        if gap_size >= remaining_needed:
            from_block = end - remaining_needed + 1
            remaining_needed = 0
            break
        remaining_needed -= gap_size
    return from_block
