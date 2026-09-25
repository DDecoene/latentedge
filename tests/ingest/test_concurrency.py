import threading
import time

from latentedge.ingest.concurrency import AdaptiveConcurrencyLimiter


def test_limiter_starts_at_the_ceiling():
    limiter = AdaptiveConcurrencyLimiter(ceiling=4)
    assert limiter.limit == 4


def test_limiter_halves_on_rate_limited_release():
    limiter = AdaptiveConcurrencyLimiter(ceiling=4, cooldown_seconds=0)
    limiter.acquire()
    new_limit, changed = limiter.release("rate_limited")
    assert new_limit == 2
    assert changed is True


def test_limiter_never_drops_below_one():
    limiter = AdaptiveConcurrencyLimiter(ceiling=4, cooldown_seconds=0)
    for _ in range(5):
        limiter.acquire()
        limiter.release("rate_limited")
    assert limiter.limit == 1


def test_limiter_reports_unchanged_when_already_at_floor():
    limiter = AdaptiveConcurrencyLimiter(ceiling=1, cooldown_seconds=0)
    limiter.acquire()
    new_limit, changed = limiter.release("rate_limited")
    assert new_limit == 1
    assert changed is False


def test_limiter_grows_by_one_after_enough_consecutive_successes():
    limiter = AdaptiveConcurrencyLimiter(ceiling=4, successes_before_increase=3, cooldown_seconds=0)
    limiter.acquire()
    limiter.release("rate_limited")  # 4 -> 2
    for _ in range(3):
        limiter.acquire()
        new_limit, changed = limiter.release("success")
    assert new_limit == 3
    assert changed is True


def test_limiter_never_exceeds_its_ceiling():
    limiter = AdaptiveConcurrencyLimiter(ceiling=2, successes_before_increase=1, cooldown_seconds=0)
    for _ in range(10):
        limiter.acquire()
        new_limit, _ = limiter.release("success")
    assert new_limit == 2


def test_limiter_plain_failure_resets_the_success_streak_without_shrinking():
    limiter = AdaptiveConcurrencyLimiter(ceiling=4, successes_before_increase=2, cooldown_seconds=0)
    limiter.acquire()
    limiter.release("rate_limited")  # 4 -> 2
    limiter.acquire()
    limiter.release("success")  # 1 consecutive success
    limiter.acquire()
    new_limit, changed = limiter.release("failed")  # streak reset, no shrink
    assert new_limit == 2
    assert changed is False
    for _ in range(2):
        limiter.acquire()
        new_limit, changed = limiter.release("success")
    assert new_limit == 3  # only counts the 2 successes after the reset
    assert changed is True


def test_limiter_acquire_blocks_extra_callers_beyond_the_current_limit():
    limiter = AdaptiveConcurrencyLimiter(ceiling=4, cooldown_seconds=0)
    limiter.acquire()
    limiter.release("rate_limited")  # limit now 2

    limiter.acquire()
    limiter.acquire()  # both permits now held; limit is exhausted

    third_acquired = threading.Event()

    def acquire_third() -> None:
        limiter.acquire()
        third_acquired.set()

    thread = threading.Thread(target=acquire_third)
    thread.start()
    try:
        time.sleep(0.05)
        assert not third_acquired.is_set()  # blocked: no free permit
        limiter.release("success")  # frees one permit
        thread.join(timeout=1.0)
        assert third_acquired.is_set()
    finally:
        thread.join(timeout=1.0)


def test_limiter_starts_from_a_persisted_limit_instead_of_the_ceiling():
    limiter = AdaptiveConcurrencyLimiter(ceiling=8, start_limit=2)
    assert limiter.limit == 2


def test_limiter_clamps_a_persisted_limit_that_exceeds_the_current_ceiling():
    # --max-workers may be lowered between runs; a stale persisted limit
    # from a more permissive run must never exceed the new ceiling.
    limiter = AdaptiveConcurrencyLimiter(ceiling=2, start_limit=8)
    assert limiter.limit == 2


def test_limiter_pauses_every_acquire_for_a_cooldown_after_a_rate_limit():
    limiter = AdaptiveConcurrencyLimiter(ceiling=4, cooldown_seconds=0.15)
    limiter.acquire()
    limiter.release("rate_limited")  # frees the permit but starts a cooldown

    resumed = threading.Event()

    def acquire_after_cooldown() -> None:
        limiter.acquire()
        resumed.set()

    thread = threading.Thread(target=acquire_after_cooldown)
    thread.start()
    try:
        time.sleep(0.05)
        # Cooldown hasn't elapsed yet, even though a permit is free.
        assert not resumed.is_set()
        thread.join(timeout=1.0)
        assert resumed.is_set()
    finally:
        thread.join(timeout=1.0)


def test_limiter_does_not_reset_the_cooldown_on_a_plain_failure():
    limiter = AdaptiveConcurrencyLimiter(ceiling=4, cooldown_seconds=0)
    limiter.acquire()
    limiter.release("failed")
    # No cooldown was armed (only rate_limited arms one), so this must
    # return immediately regardless of cooldown_seconds.
    start = time.monotonic()
    limiter.acquire()
    assert time.monotonic() - start < 0.1
