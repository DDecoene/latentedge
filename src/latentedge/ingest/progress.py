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


def _rate_path(out_path: Path) -> Path:
    return Path(str(out_path) + ".rate.json")


def _read_rate_state(out_path: Path) -> dict:
    path = _rate_path(out_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _write_rate_state(out_path: Path, **updates: float) -> None:
    # Read-modify-write rather than overwrite — rate and ceiling are
    # written independently (chunked.py persists both at the end of a
    # run), and each write must not clobber whichever field it isn't
    # updating.
    state = _read_rate_state(out_path)
    state.update(updates)
    _rate_path(out_path).write_text(json.dumps(state))


def read_rate_limit(out_path: Path) -> float | None:
    """The req/s rate a prior ingest_range run settled on for this output
    file, if any — lets a resumed run start from a rate already known to
    avoid rate limiting instead of the full ceiling (which just re-earns
    the same throttle-down again).
    """
    state = _read_rate_state(out_path)
    return float(state["rate"]) if "rate" in state else None


def write_rate_limit(out_path: Path, rate: float) -> None:
    _write_rate_state(out_path, rate=rate)


def read_rate_ceiling(out_path: Path) -> float | None:
    """The req/s ceiling a prior ingest_range run discovered for this
    output file, if any — a self-raised ceiling from sustained clean
    throughput is real, measured headroom; a fresh run should resume
    probing from there rather than re-discovering it from the config
    default every time.
    """
    state = _read_rate_state(out_path)
    return float(state["ceiling"]) if "ceiling" in state else None


def write_rate_ceiling(out_path: Path, ceiling: float) -> None:
    _write_rate_state(out_path, ceiling=ceiling)


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
    """Gaps strictly between already-ingested intervals. Empty when
    there's at most one interval, since there's nothing for a gap to sit
    between.
    """
    if len(intervals) < 2:
        return []
    ordered = sorted(intervals)
    return uncovered_gaps(ordered, ordered[0][0], ordered[-1][1])
