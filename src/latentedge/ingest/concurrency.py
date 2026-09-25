"""An AIMD-style concurrency gate for ingest_range's worker threads.

A fixed --max-workers has to be hand-tuned per RPC provider/tier, and
whatever value works when the endpoint is healthy is too aggressive the
moment it starts rate-limiting (observed for real against Alchemy's free
tier: most workers stuck retrying simultaneously). This gate lets the
ThreadPoolExecutor keep its full thread count, but caps how many of those
threads may have a request in flight at once — additively creeping that
cap back up toward the ceiling on sustained success, and halving it the
instant a rate limit is reported.
"""

import threading
from typing import Literal

DEFAULT_SUCCESSES_BEFORE_INCREASE = 20

ReleaseOutcome = Literal["success", "failed", "rate_limited"]


class AdaptiveConcurrencyLimiter:
    def __init__(self, ceiling: int, successes_before_increase: int = DEFAULT_SUCCESSES_BEFORE_INCREASE) -> None:
        self._ceiling = max(1, ceiling)
        self._successes_before_increase = successes_before_increase
        self._limit = self._ceiling
        self._in_flight = 0
        self._consecutive_successes = 0
        self._cond = threading.Condition()

    @property
    def limit(self) -> int:
        with self._cond:
            return self._limit

    def acquire(self) -> None:
        """Block until fewer than the current limit are in flight."""
        with self._cond:
            while self._in_flight >= self._limit:
                self._cond.wait()
            self._in_flight += 1

    def release(self, outcome: ReleaseOutcome) -> tuple[int, bool]:
        """Record an attempt's outcome and adjust the limit accordingly.

        Returns (new_limit, changed) so a caller can report a real
        change without spamming a callback on every single release.
        """
        with self._cond:
            self._in_flight -= 1
            changed = False
            if outcome == "rate_limited":
                new_limit = max(1, self._limit // 2)
                if new_limit != self._limit:
                    self._limit = new_limit
                    changed = True
                self._consecutive_successes = 0
            elif outcome == "failed":
                # A non-rate-limit failure isn't evidence the endpoint is
                # overloaded, but it isn't a clean success either —
                # interrupt the streak without shrinking capacity.
                self._consecutive_successes = 0
            else:
                self._consecutive_successes += 1
                if self._consecutive_successes >= self._successes_before_increase and self._limit < self._ceiling:
                    self._limit += 1
                    self._consecutive_successes = 0
                    changed = True
            self._cond.notify_all()
            return self._limit, changed
