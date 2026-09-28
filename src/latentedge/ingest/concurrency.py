"""An AIMD-style concurrency gate for ingest_range's worker threads.

A fixed --max-workers has to be hand-tuned per RPC provider/tier, and
whatever value works when the endpoint is healthy is too aggressive the
moment it starts rate-limiting (observed for real against Alchemy's free
tier: most workers stuck retrying simultaneously). This gate lets the
ThreadPoolExecutor keep its full thread count, but caps how many of those
threads may have a request in flight at once — additively creeping that
cap back up toward the ceiling on sustained success, and halving it the
instant a rate limit is reported.

Halving the cap alone isn't enough: it only affects future acquire()
calls, so the other (limit - 1) requests already in flight or already
past the gate when the 429 lands keep firing immediately and tend to
re-trigger the same limit. A brief shared cooldown after a rate-limit
event pauses every acquire() (not just the caller that got limited)
until the burst has had time to clear.

start_limit lets a caller resume at the concurrency level a prior run
settled on, instead of always retrying the full ceiling and re-earning
the same throttle-down from scratch on every resume.
"""

import threading
import time
from typing import Literal

DEFAULT_SUCCESSES_BEFORE_INCREASE = 20
DEFAULT_COOLDOWN_SECONDS = 2.0

ReleaseOutcome = Literal["success", "failed", "rate_limited"]

# How often acquire()'s wait loop wakes on its own to recheck cancel_event,
# even with nothing to notify it — the only way a blocked worker notices a
# user-requested stop without waiting out a full cooldown or the limiter
# gate first.
_CANCEL_POLL_SECONDS = 0.5


class Cancelled(Exception):
    """A cancel_event was set while a worker was blocked in acquire()."""


class AdaptiveConcurrencyLimiter:
    def __init__(
        self,
        ceiling: int,
        successes_before_increase: int = DEFAULT_SUCCESSES_BEFORE_INCREASE,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        start_limit: int | None = None,
    ) -> None:
        self._ceiling = max(1, ceiling)
        self._successes_before_increase = successes_before_increase
        self._cooldown_seconds = cooldown_seconds
        self._limit = max(1, min(start_limit, self._ceiling)) if start_limit else self._ceiling
        self._in_flight = 0
        self._consecutive_successes = 0
        self._cooldown_until = 0.0
        self._cond = threading.Condition()

    @property
    def limit(self) -> int:
        with self._cond:
            return self._limit

    def acquire(self, cancel_event: threading.Event | None = None) -> None:
        """Block until fewer than the current limit are in flight and
        any post-rate-limit cooldown has elapsed.

        Waits use a short timeout (rather than an unbounded
        threading.Condition.wait()) purely so a set cancel_event is
        noticed within _CANCEL_POLL_SECONDS instead of only on the next
        release() — otherwise a worker blocked here during a user-
        requested stop would hang until some other, unrelated worker
        happens to release and notify.
        """
        with self._cond:
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    raise Cancelled()
                remaining_cooldown = self._cooldown_until - time.monotonic()
                if remaining_cooldown > 0:
                    self._cond.wait(timeout=min(remaining_cooldown, _CANCEL_POLL_SECONDS))
                    continue
                if self._in_flight >= self._limit:
                    self._cond.wait(timeout=_CANCEL_POLL_SECONDS)
                    continue
                break
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
                self._cooldown_until = time.monotonic() + self._cooldown_seconds
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
