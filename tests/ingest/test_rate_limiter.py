import threading
import time

import pytest

from latentedge.ingest.rate_limiter import RateLimiter, Cancelled, DEFAULT_MIN_RPS, BACKOFF_FRACTION_DECAY


def test_limiter_starts_at_the_ceiling():
    limiter = RateLimiter(ceiling=4.0)
    assert limiter.rate == 4.0


def test_limiter_halves_on_rate_limited_release():
    changes: list[float] = []
    limiter = RateLimiter(ceiling=4.0, cooldown_seconds=0, on_change=changes.append)
    limiter.release("rate_limited")
    assert limiter.rate == 2.0
    assert changes == [2.0]


def test_limiter_does_not_compound_multiple_rate_limits_within_the_same_cooldown_window():
    # Regression test (final review finding): every request already in
    # flight when a provider starts 429ing reports its own "rate_limited"
    # outcome. Without deduping, a burst of N in-flight requests halves
    # the rate N times and inflates the recovery threshold N times for
    # what is really one throttle event, turning a brief 429 into many
    # minutes of near-floor throughput. A second rate_limited report
    # arriving while the first one's cooldown is still active must be
    # treated as the same event: it extends the cooldown but does not
    # halve the rate or back off the threshold again.
    changes: list[float] = []
    limiter = RateLimiter(ceiling=4.0, cooldown_seconds=10.0, on_change=changes.append)
    limiter.release("rate_limited")  # 4.0 -> 2.0
    limiter.release("rate_limited")  # same event (still within cooldown) — no further halving
    limiter.release("rate_limited")  # still the same event
    assert limiter.rate == 2.0
    assert changes == [2.0]


def test_limiter_never_drops_below_the_floor():
    limiter = RateLimiter(ceiling=4.0, cooldown_seconds=0)
    for _ in range(10):
        limiter.release("rate_limited")
    assert limiter.rate == DEFAULT_MIN_RPS


def test_limiter_does_not_invoke_on_change_when_already_at_the_floor():
    changes: list[float] = []
    limiter = RateLimiter(ceiling=DEFAULT_MIN_RPS, cooldown_seconds=0, on_change=changes.append)
    limiter.release("rate_limited")
    assert limiter.rate == DEFAULT_MIN_RPS
    assert changes == []


def test_limiter_climbs_gently_after_enough_consecutive_successes():
    changes: list[float] = []
    limiter = RateLimiter(ceiling=4.0, successes_before_increase=3, cooldown_seconds=0, on_change=changes.append)
    limiter.release("rate_limited")  # 4.0 -> 2.0, threshold 3 -> 4.5
    changes.clear()
    for _ in range(4):
        limiter.release("success")
    assert changes == []  # 4 successes still short of the backed-off 4.5 threshold
    limiter.release("success")  # 5th consecutive success clears it
    assert changes == [pytest.approx(2.2)]  # 2.0 * RATE_CLIMB_FACTOR (1.1)
    assert limiter.rate == pytest.approx(2.2)


def test_limiter_never_exceeds_its_current_ceiling():
    # 4 successes stays under the ceiling-raise threshold (successes_before_increase=1
    # * CEILING_RAISE_SUCCESS_MULTIPLIER=5), isolating this assertion from the
    # self-raising-ceiling behavior covered separately below.
    limiter = RateLimiter(ceiling=2.0, successes_before_increase=1, cooldown_seconds=0)
    for _ in range(4):
        limiter.release("success")
    assert limiter.rate == pytest.approx(2.0)


def test_limiter_plain_failure_resets_the_success_streak_without_shrinking_rate():
    changes: list[float] = []
    limiter = RateLimiter(ceiling=4.0, successes_before_increase=2, cooldown_seconds=0, on_change=changes.append)
    limiter.release("rate_limited")  # 4.0 -> 2.0
    changes.clear()
    limiter.release("success")  # 1 consecutive success
    limiter.release("failed")  # streak reset, no shrink
    assert changes == []
    for _ in range(3):  # threshold is now 3 (2 * 1.5), needs 3 consecutive
        limiter.release("success")
    assert changes == [pytest.approx(2.2)]


def test_limiter_gradually_decays_the_backed_off_success_threshold_across_climbs():
    # Regression test for "smarter recovery": the success-streak
    # requirement inflated by a rate limit must come back down in steps
    # as the limiter proves it's healthy again, not snap straight back to
    # its pre-throttle value after a single climb.
    changes: list[float] = []
    limiter = RateLimiter(ceiling=1000.0, successes_before_increase=10, cooldown_seconds=0, on_change=changes.append)
    limiter.release("rate_limited")  # rate 1000.0 -> 500.0, threshold 10 -> 15.0
    changes.clear()

    for _ in range(14):
        limiter.release("success")
    assert changes == []  # 14 < 15
    limiter.release("success")  # 15th clears it
    assert changes == [pytest.approx(550.0)]  # 500.0 * RATE_CLIMB_FACTOR (1.1)
    changes.clear()  # threshold decays 15.0 -> 13.0 (still above base 10)

    for _ in range(12):
        limiter.release("success")
    assert changes == []  # 12 < 13
    limiter.release("success")  # 13th clears it
    assert changes == [pytest.approx(605.0)]  # 550.0 * 1.1


def test_limiter_reverts_to_the_last_known_good_rate_instead_of_halving_after_a_proven_climb():
    # Regression test: falling back to a flat half of whatever rate just
    # got throttled (6.655 / 2 =~ 3.33) undershoots a level (6.05) that
    # was proven to work moments earlier. The limiter must fall back to
    # that known-good rate instead.
    limiter = RateLimiter(ceiling=100.0, start_rate=5.0, successes_before_increase=1, cooldown_seconds=0)
    limiter.release("success")  # last-known-good 5.0, rate -> 5.5
    limiter.release("success")  # last-known-good 5.5, rate -> 6.05
    limiter.release("success")  # last-known-good 6.05, rate -> 6.655
    limiter.release("rate_limited")
    assert limiter.rate == pytest.approx(6.05)


def test_limiter_climbs_more_cautiously_after_reverting_to_a_known_good_rate():
    # Regression test: after overshooting a known-good rate and reverting
    # to it, the next climb step must be smaller than RATE_CLIMB_FACTOR —
    # repeated approaches to the same danger zone should get more
    # cautious, dampening the oscillation instead of repeating it at the
    # same amplitude forever.
    limiter = RateLimiter(ceiling=1000.0, start_rate=5.0, successes_before_increase=1, cooldown_seconds=0)
    limiter.release("success")  # rate 5.0 -> 5.5
    limiter.release("rate_limited")  # reverts to 5.0 (known-good), climb factor 1.1 -> 1.05

    # successes_before_increase backed off from 1 to 1.5 by the throttle above.
    limiter.release("success")  # consecutive 1, below 1.5
    limiter.release("success")  # consecutive 2, clears it — climbs at the narrowed factor
    assert limiter.rate == pytest.approx(5.0 * 1.05)


def test_limiter_halves_on_the_first_throttle_with_no_known_good_rate_proven_yet():
    # The revert-to-known-good path only applies once a climb has actually
    # proven a lower rate was safe — a fresh limiter has no such evidence
    # yet, so its very first throttle still falls back to a full halving.
    limiter = RateLimiter(ceiling=4.0, cooldown_seconds=0)
    limiter.release("rate_limited")
    assert limiter.rate == pytest.approx(2.0)


def test_limiter_gentles_the_fallback_cut_on_repeated_throttles_with_zero_evidence():
    # Regression test: a burst of throttles hitting a fresh limiter that
    # has never once climbed successfully (zero evidence of any safe
    # rate) used to fall back to a flat halving every single time —
    # sawing the rate down hard on every throttle in the burst instead of
    # converging. The fallback cut must gentle with each consecutive
    # throttle like this, not repeat the same aggressive halving forever.
    limiter = RateLimiter(ceiling=4.0, cooldown_seconds=0)
    limiter.release("rate_limited")  # first throttle: full halving, 4.0 -> 2.0
    assert limiter.rate == pytest.approx(2.0)
    limiter.release("rate_limited")  # second, with no intervening climb: gentler cut
    assert limiter.rate == pytest.approx(2.0 * (1 - 0.5 * BACKOFF_FRACTION_DECAY))
    second_rate = limiter.rate
    limiter.release("rate_limited")  # third: gentler still
    assert limiter.rate > second_rate * 0.5  # nowhere near another halving


def test_limiter_steps_down_finely_once_throttled_right_at_a_previously_proven_rate():
    # Regression test for a real run: rate climbed cleanly to 8.3, then a
    # throttle dropped it straight to 5.4 (a ~35% cut) instead of easing
    # down toward the edge (8.3 -> ~8.2 -> ~8.1 -> ...). Once at least one
    # climb has ever succeeded, getting throttled with nothing *strictly
    # lower* proven (i.e. right at the last proven rate, since nothing
    # climbed further since the last revert) means the edge is close —
    # the response must be a small step down, not the same big cut used
    # for a totally fresh limiter with zero evidence.
    limiter = RateLimiter(ceiling=100.0, start_rate=5.0, successes_before_increase=1, cooldown_seconds=0)
    limiter.release("success")  # last-known-good 5.0, rate -> 5.5
    limiter.release("rate_limited")  # reverts to 5.0 (known-good), climb factor narrows

    # No successful climb has happened since the revert — last_stable_rate
    # (5.0) now equals the current rate (5.0), so this is the "right at
    # the edge" case, not a fresh-limiter one.
    limiter.release("rate_limited")
    assert limiter.rate == pytest.approx(5.0 * (1 - 0.02))  # a small 2% step, not another big cut


def test_limiter_raises_its_own_ceiling_after_a_long_clean_streak_at_it():
    # successes_before_increase=1 -> ceiling_raise_successes = 5. Rate
    # starts at the ceiling (no start_rate given), so every success from
    # the first one lands in the "already at ceiling" branch.
    ceiling_changes: list[float] = []
    limiter = RateLimiter(
        ceiling=2.0, successes_before_increase=1, cooldown_seconds=0,
        on_ceiling_change=ceiling_changes.append,
    )
    for _ in range(4):
        limiter.release("success")
    assert ceiling_changes == []
    assert limiter.ceiling == 2.0
    limiter.release("success")  # 5th clean success at the ceiling
    assert ceiling_changes == [pytest.approx(3.0)]  # 2.0 * CEILING_RAISE_FACTOR (1.5)
    assert limiter.ceiling == pytest.approx(3.0)


def test_limiter_resumes_climbing_toward_a_newly_raised_ceiling():
    limiter = RateLimiter(ceiling=2.0, successes_before_increase=1, cooldown_seconds=0)
    for _ in range(5):
        limiter.release("success")  # raises ceiling to 3.0, rate still 2.0
    assert limiter.rate == pytest.approx(2.0)
    assert limiter.ceiling == pytest.approx(3.0)
    limiter.release("success")  # rate is now below the new ceiling — climbs again
    assert limiter.rate == pytest.approx(2.2)


def test_limiter_stops_raising_the_ceiling_once_a_real_rate_limit_creates_equilibrium_below_it():
    # A rate limit near the ceiling knocks the rate below it — the
    # at-ceiling streak must not have survived that, so a caller that
    # merely oscillates around the new, lower equilibrium never
    # re-triggers a ceiling raise from a handful of ordinary successes.
    ceiling_changes: list[float] = []
    limiter = RateLimiter(
        ceiling=2.0, successes_before_increase=1, cooldown_seconds=0,
        on_ceiling_change=ceiling_changes.append,
    )
    for _ in range(4):
        limiter.release("success")  # 4 clean successes at the ceiling, 1 short of raising
    limiter.release("rate_limited")  # knocks the rate down, resets the at-ceiling streak
    for _ in range(4):
        limiter.release("success")  # climbs back toward 2.0 but doesn't reach/hold it yet
    assert ceiling_changes == []


def test_limiter_never_raises_the_ceiling_while_still_climbing_toward_it():
    ceiling_changes: list[float] = []
    limiter = RateLimiter(
        ceiling=1000.0, start_rate=1.0, successes_before_increase=1, cooldown_seconds=0,
        on_ceiling_change=ceiling_changes.append,
    )
    for _ in range(50):
        limiter.release("success")  # 1.0 * 1.1**50 =~ 117.4, still well below the ceiling
    assert ceiling_changes == []
    assert limiter.ceiling == 1000.0


def test_limiter_starts_from_a_persisted_rate_instead_of_the_ceiling():
    limiter = RateLimiter(ceiling=8.0, start_rate=2.0)
    assert limiter.rate == 2.0


def test_limiter_clamps_a_persisted_rate_that_exceeds_the_current_ceiling():
    limiter = RateLimiter(ceiling=2.0, start_rate=8.0)
    assert limiter.rate == 2.0


def test_limiter_paces_successive_acquires_to_the_configured_interval():
    limiter = RateLimiter(ceiling=20.0, cooldown_seconds=0)  # interval = 0.05s
    start = time.monotonic()
    for _ in range(4):
        limiter.acquire()
    elapsed = time.monotonic() - start
    assert elapsed >= 0.14  # 3 gaps of ~0.05s, with a little scheduling slack


def test_limiter_does_not_delay_acquire_when_calls_are_already_spaced_out():
    limiter = RateLimiter(ceiling=1.0, cooldown_seconds=0)  # interval = 1s
    limiter.acquire()
    time.sleep(1.05)
    start = time.monotonic()
    limiter.acquire()
    assert time.monotonic() - start < 0.1


def test_limiter_paces_concurrent_acquires_from_multiple_threads():
    limiter = RateLimiter(ceiling=50.0, cooldown_seconds=0)  # interval = 0.02s
    timestamps: list[float] = []
    lock = threading.Lock()

    def worker() -> None:
        limiter.acquire()
        with lock:
            timestamps.append(time.monotonic())

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=2.0)

    timestamps.sort()
    gaps = [b - a for a, b in zip(timestamps, timestamps[1:])]
    assert len(gaps) == 9
    assert all(gap >= 0.018 for gap in gaps)  # allow a little scheduling slack below 0.02s


def test_limiter_pauses_every_acquire_for_a_cooldown_after_a_rate_limit():
    limiter = RateLimiter(ceiling=1000.0, cooldown_seconds=0.15)
    limiter.acquire()
    limiter.release("rate_limited")

    resumed = threading.Event()

    def acquire_after_cooldown() -> None:
        limiter.acquire()
        resumed.set()

    thread = threading.Thread(target=acquire_after_cooldown)
    thread.start()
    try:
        time.sleep(0.05)
        assert not resumed.is_set()
        thread.join(timeout=1.0)
        assert resumed.is_set()
    finally:
        thread.join(timeout=1.0)


def test_limiter_does_not_reset_the_cooldown_on_a_plain_failure():
    limiter = RateLimiter(ceiling=1000.0, cooldown_seconds=0)
    limiter.acquire()
    limiter.release("failed")
    start = time.monotonic()
    limiter.acquire()
    assert time.monotonic() - start < 0.1


def test_limiter_acquire_raises_cancelled_immediately_if_already_set():
    limiter = RateLimiter(ceiling=1000.0, cooldown_seconds=0)
    cancel_event = threading.Event()
    cancel_event.set()
    with pytest.raises(Cancelled):
        limiter.acquire(cancel_event)


def test_limiter_acquire_wakes_promptly_on_cancel_event_instead_of_waiting_for_the_interval():
    limiter = RateLimiter(ceiling=0.1, cooldown_seconds=0)  # ceiling floors to DEFAULT_MIN_RPS (0.5) -> interval = 2s
    limiter.acquire()  # schedules the next slot ~2s out

    cancel_event = threading.Event()
    raised = threading.Event()

    def acquire_blocked() -> None:
        try:
            limiter.acquire(cancel_event)
        except Cancelled:
            raised.set()

    thread = threading.Thread(target=acquire_blocked)
    thread.start()
    try:
        time.sleep(0.05)
        assert not raised.is_set()  # genuinely waiting on the interval, not a fluke pass
        cancel_event.set()
        thread.join(timeout=1.0)
        assert raised.is_set()
    finally:
        thread.join(timeout=1.0)
