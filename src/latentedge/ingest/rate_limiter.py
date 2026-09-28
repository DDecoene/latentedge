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
# A fixed ceiling only protects against a limit nobody has measured — it
# can't tell you the real one is higher. Once the rate has been sitting
# cleanly *at* the ceiling for a long streak (no 429s at all, not even
# ones that got absorbed into a cooldown), the ceiling itself is raised so
# the climb logic keeps probing upward. If a 429 does eventually land near
# the new ceiling, the rate settles into equilibrium below it (halving on
# 429, climbing on success) and is no longer "at ceiling" — so the raise
# streak never restarts and the search self-terminates at the real limit,
# no hard upper bound required.
CEILING_RAISE_SUCCESS_MULTIPLIER = 5
CEILING_RAISE_FACTOR = 1.5
# Falling back to a flat half of whatever rate just got throttled can
# undershoot a level that was proven to work moments earlier — reverting
# to the last rate that survived a full climb-success-streak keeps
# recovery close to the real limit instead of needlessly crawling back up
# from far below it. Each time that known-good rate gets overshot again,
# the climb step itself is narrowed (never below MIN_CLIMB_FACTOR) so
# repeated approaches to the same danger zone get more cautious and the
# oscillation converges instead of repeating at the same amplitude
# forever.
CLIMB_FACTOR_DECAY = 0.5
MIN_CLIMB_FACTOR = 1.01
# The known-good revert above only applies once a climb has proven a
# *lower* rate safe. Getting throttled again with no such point below the
# current rate — the very first throttle ever, or a throttle right at the
# rate that was just reverted to — used to fall back to a flat halving
# every time, undoing the whole point of converging: a burst of throttles
# with no intervening successful climb would still saw the rate down hard
# on every single one. This fraction now starts at a full halving but
# gentles toward MIN_BACKOFF_FRACTION with each consecutive throttle that
# isn't preceded by a new proven climb, so repeated throttling at the same
# danger zone backs off by less and less instead of by half every time.
INITIAL_BACKOFF_FRACTION = 0.5
MIN_BACKOFF_FRACTION = 0.1
BACKOFF_FRACTION_DECAY = 0.7
# The gentling fallback above still conflated two different situations: a
# genuinely fresh limiter with zero evidence (a big first cut is the right
# move), and a limiter that has already climbed successfully at least
# once and is now getting throttled again right at its last proven point
# (no successful climb happened in between) — the edge is close, evidenced
# by 8.3 -> 5.4 in a real run instead of the expected 8.3 -> 8.2 -> 8.1.
# Once any climb has ever succeeded, a throttle with nothing strictly
# lower to revert to should nudge down by a small fixed fraction instead —
# repeated small multiplicative steps naturally shrink in absolute size as
# the rate falls, without needing their own decay schedule.
FINE_STEP_FRACTION = 0.02

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
        on_ceiling_change: Callable[[float], None] | None = None,
    ) -> None:
        self._ceiling = max(DEFAULT_MIN_RPS, ceiling)
        self._base_successes_before_increase = successes_before_increase
        self._successes_before_increase: float = successes_before_increase
        self._ceiling_raise_successes = successes_before_increase * CEILING_RAISE_SUCCESS_MULTIPLIER
        self._cooldown_seconds = cooldown_seconds
        self._on_change = on_change
        self._on_ceiling_change = on_ceiling_change
        self._rate = max(DEFAULT_MIN_RPS, min(start_rate, self._ceiling)) if start_rate else self._ceiling
        # The rate right before its most recent proven climb — the last
        # value known to have survived a full success streak, so a real
        # throttle can fall back to it instead of halving blindly.
        self._last_stable_rate: float | None = None
        self._climb_factor = RATE_CLIMB_FACTOR
        self._backoff_fraction = INITIAL_BACKOFF_FRACTION
        self._consecutive_successes = 0
        self._consecutive_successes_at_ceiling = 0
        self._cooldown_until = 0.0
        self._next_allowed_time = time.monotonic()
        self._cond = threading.Condition()

    @property
    def rate(self) -> float:
        with self._cond:
            return self._rate

    @property
    def ceiling(self) -> float:
        with self._cond:
            return self._ceiling

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
        new_ceiling: float | None = None
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
                self._consecutive_successes_at_ceiling = 0
                if now < self._cooldown_until:
                    self._consecutive_successes = 0
                    self._cooldown_until = now + self._cooldown_seconds
                else:
                    if self._last_stable_rate is not None and self._last_stable_rate < self._rate:
                        candidate = max(DEFAULT_MIN_RPS, self._last_stable_rate)
                        self._climb_factor = max(
                            MIN_CLIMB_FACTOR, 1.0 + (self._climb_factor - 1.0) * CLIMB_FACTOR_DECAY
                        )
                    elif self._last_stable_rate is not None:
                        # Already proven a climb before, just not one
                        # strictly below the current rate — we're right at
                        # the edge, so probe down in a small step rather
                        # than slashing back toward the last big cut.
                        candidate = max(DEFAULT_MIN_RPS, self._rate * (1.0 - FINE_STEP_FRACTION))
                        self._climb_factor = max(
                            MIN_CLIMB_FACTOR, 1.0 + (self._climb_factor - 1.0) * CLIMB_FACTOR_DECAY
                        )
                    else:
                        # No evidence at all yet — the very first throttle
                        # before any successful climb. A big first cut is
                        # the right move here.
                        candidate = max(DEFAULT_MIN_RPS, self._rate * (1.0 - self._backoff_fraction))
                        self._backoff_fraction = max(
                            MIN_BACKOFF_FRACTION, self._backoff_fraction * BACKOFF_FRACTION_DECAY
                        )
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
                self._consecutive_successes_at_ceiling = 0
            else:
                self._consecutive_successes += 1
                if self._rate < self._ceiling:
                    self._consecutive_successes_at_ceiling = 0
                    if self._consecutive_successes >= self._successes_before_increase:
                        self._last_stable_rate = self._rate
                        self._rate = min(self._rate * self._climb_factor, self._ceiling)
                        new_rate = self._rate
                        self._consecutive_successes = 0
                        self._successes_before_increase = max(
                            self._successes_before_increase - RECOVERY_THRESHOLD_DECAY,
                            self._base_successes_before_increase,
                        )
                else:
                    # Already at the ceiling with nothing knocking it back
                    # down — a long enough clean streak here means there's
                    # probably real headroom above it.
                    self._consecutive_successes_at_ceiling += 1
                    if self._consecutive_successes_at_ceiling >= self._ceiling_raise_successes:
                        self._ceiling = self._ceiling * CEILING_RAISE_FACTOR
                        new_ceiling = self._ceiling
                        self._consecutive_successes_at_ceiling = 0
            self._cond.notify_all()
        if new_rate is not None and self._on_change is not None:
            self._on_change(new_rate)
        if new_ceiling is not None and self._on_ceiling_change is not None:
            self._on_ceiling_change(new_ceiling)
