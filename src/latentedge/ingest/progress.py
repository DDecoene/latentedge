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


def _concurrency_path(out_path: Path) -> Path:
    return Path(str(out_path) + ".concurrency.json")


def read_concurrency_limit(out_path: Path) -> int | None:
    """The worker-concurrency limit a prior ingest_range run settled on
    for this output file, if any — lets a resumed run start from a level
    already known to avoid rate limiting instead of the full ceiling
    (which just re-earns the same throttle-down again).
    """
    path = _concurrency_path(out_path)
    if not path.exists():
        return None
    return int(json.loads(path.read_text())["limit"])


def write_concurrency_limit(out_path: Path, limit: int) -> None:
    _concurrency_path(out_path).write_text(json.dumps({"limit": limit}))


def read_progress(out_path: Path) -> list[Interval]:
    path = _progress_path(out_path)
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    if "ingested" in data:
        return [(pair[0], pair[1]) for pair in data["ingested"]]

    # Legacy single-watermark format (pre-interval progress tracking)
    # records no start block — derive it from the output file's actual
    # minimum block rather than assuming coverage back to genesis, which
    # would make earlier, never-fetched ranges look falsely
    # already-ingested and silently skip real missing history.
    from latentedge.store import read_swaps

    last_completed_block: int = data["last_completed_block"]
    earliest_block = int(read_swaps(out_path)["block_number"].min())
    return [(earliest_block, last_completed_block)]


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


def internal_gaps(intervals: list[Interval]) -> list[Interval]:
    """Gaps strictly between already-ingested intervals — history that
    some prior run skipped over (e.g. one that jumped straight to an
    explicit --from-block/--to-block without ever checking what came
    before it, rather than resuming from the existing watermark). Empty
    when there's at most one interval, since there's nothing for a gap
    to sit between.
    """
    if len(intervals) < 2:
        return []
    ordered = sorted(intervals)
    return uncovered_gaps(ordered, ordered[0][0], ordered[-1][1])


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
    naive_from = max(naive_from, floor_block)

    # Never let the naive near-head window sit disconnected from the
    # most recently ingested block — otherwise a window that already
    # contains enough new blocks on its own (e.g. a quick --days 1 run
    # after a much longer gap since the last run) would return
    # immediately without ever touching older coverage, permanently
    # stranding everything in between as a gap nothing else ever goes
    # back to look for. Pulling naive_from back to touch it means the
    # gap becomes part of what this window needs to cover, so the
    # ordinary walk-back logic below closes it like any other shortfall.
    if intervals:
        latest_covered_end = max(end for _, end in intervals)
        if latest_covered_end + 1 < naive_from:
            naive_from = max(latest_covered_end + 1, floor_block)

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
