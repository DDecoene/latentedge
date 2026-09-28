"""A proactive, learning rate limiter for ingest's outbound RPC calls.

Alchemy (and similar providers) throttle by requests over time, not by how
many connections are open at once — a fixed concurrency cap can still
burst well past a safe req/s the instant several permitted callers each
fire in the same instant. This paces individual acquire() calls to at most
one every 1/rate seconds (a strict schedule, not a bursty token bucket
that would let a full second's worth of calls fire the moment it refills),
and adjusts rate itself from real 429s: halved immediately on a rate
limit, climbed back gently on sustained success. See
docs/specs/2026-09-28-proactive-rate-limiting-design.md.
"""

import threading
import time
from collections.abc import Callable
from typing import Literal

DEFAULT_SUCCESSES_BEFORE_INCREASE = 20
DEFAULT_COOLDOWN_SECONDS = 2.0
DEFAULT_MIN_RPS = 0.5
# Gentle, not a snap back to the pre-throttle rate — a provider that just
# rate-limited us shouldn't be re-approached at full speed the moment a
# few requests succeed.
RATE_CLIMB_FACTOR = 1.1
# Each rate limit makes the next climb require more proof of health;
# RECOVERY_THRESHOLD_DECAY unwinds that extra caution gradually as climbs
# actually happen, rather than either resetting it instantly or leaving
# it permanently inflated by one incident hours ago.
RECOVERY_THRESHOLD_BACKOFF = 1.5
RECOVERY_THRESHOLD_DECAY = 2.0
MAX_SUCCESSES_BEFORE_INCREASE = 100

ReleaseOutcome = Literal["success", "failed", "rate_limited"]

# How often acquire()'s wait loop wakes on its own to recheck cancel_event,
# even with nothing to notify it — the only way a blocked worker notices a
# user-requested stop without waiting out a full cooldown or pacing
# interval first.
_CANCEL_POLL_SECONDS = 0.5


class Cancelled(Exception):
    """A cancel_event was set while a worker was blocked in acquire()."""


class RateLimiter:
    def __init__(
        self,
        ceiling: float,
        successes_before_increase: int = DEFAULT_SUCCESSES_BEFORE_INCREASE,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        start_rate: float | None = None,
        on_change: Callable[[float], None] | None = None,
    ) -> None:
        self._ceiling = max(DEFAULT_MIN_RPS, ceiling)
        self._base_successes_before_increase = successes_before_increase
        self._successes_before_increase: float = successes_before_increase
        self._cooldown_seconds = cooldown_seconds
        self._on_change = on_change
        self._rate = max(DEFAULT_MIN_RPS, min(start_rate, self._ceiling)) if start_rate else self._ceiling
        self._consecutive_successes = 0
        self._cooldown_until = 0.0
        self._next_allowed_time = time.monotonic()
        self._cond = threading.Condition()

    @property
    def rate(self) -> float:
        with self._cond:
            return self._rate

    def acquire(self, cancel_event: threading.Event | None = None) -> None:
        """Block until this call's scheduled slot arrives and any
        post-rate-limit cooldown has elapsed, then claim the next slot.

        Uses a strict schedule (one slot every 1/rate seconds) rather
        than a bursty token bucket — a bucket that lets a full second's
        worth of calls fire the moment it refills would reproduce the
        exact burst-then-throttle pattern this limiter exists to avoid.
        Slot-claiming happens while holding _cond's lock, so concurrent
        callers can never claim the same slot.
        """
        with self._cond:
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    raise Cancelled()
                now = time.monotonic()
                remaining_cooldown = self._cooldown_until - now
                if remaining_cooldown > 0:
                    self._cond.wait(timeout=min(remaining_cooldown, _CANCEL_POLL_SECONDS))
                    continue
                wait_seconds = self._next_allowed_time - now
                if wait_seconds > 0:
                    self._cond.wait(timeout=min(wait_seconds, _CANCEL_POLL_SECONDS))
                    continue
                self._next_allowed_time = max(now, self._next_allowed_time) + (1.0 / self._rate)
                return

    def release(self, outcome: ReleaseOutcome) -> None:
        """Record an attempt's outcome and adjust rate accordingly,
        invoking on_change(new_rate) itself when rate actually changes —
        callers don't need to propagate a return value, since gating now
        happens several call-layers away (inside rpc_logs.py) from
        anywhere that could consume one.
        """
        new_rate: float | None = None
        with self._cond:
            if outcome == "rate_limited":
                now = time.monotonic()
                # Every request already in flight when a provider starts
                # 429ing reports its own "rate_limited" outcome — without
                # this check, a burst of N in-flight requests would halve
                # the rate and back off the recovery threshold N times for
                # what is really one throttle event. A report arriving
                # while the previous one's cooldown is still active is
                # folded into that same event: it extends the cooldown
                # (the burst may still be landing) without compounding the
                # rate cut or the threshold backoff again.
                if now < self._cooldown_until:
                    self._consecutive_successes = 0
                    self._cooldown_until = now + self._cooldown_seconds
                else:
                    candidate = max(DEFAULT_MIN_RPS, self._rate / 2)
                    if candidate != self._rate:
                        self._rate = candidate
                        new_rate = candidate
                    self._consecutive_successes = 0
                    self._successes_before_increase = min(
                        self._successes_before_increase * RECOVERY_THRESHOLD_BACKOFF,
                        MAX_SUCCESSES_BEFORE_INCREASE,
                    )
                    self._cooldown_until = now + self._cooldown_seconds
            elif outcome == "failed":
                self._consecutive_successes = 0
            else:
                self._consecutive_successes += 1
                if self._consecutive_successes >= self._successes_before_increase and self._rate < self._ceiling:
                    self._rate = min(self._rate * RATE_CLIMB_FACTOR, self._ceiling)
                    new_rate = self._rate
                    self._consecutive_successes = 0
                    self._successes_before_increase = max(
                        self._successes_before_increase - RECOVERY_THRESHOLD_DECAY,
                        self._base_successes_before_increase,
                    )
            self._cond.notify_all()
        if new_rate is not None and self._on_change is not None:
            self._on_change(new_rate)
