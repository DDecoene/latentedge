import pytest

from latentedge.safety_guard import GuardState, SafetyGuard, record_trade_result

DAY_SECONDS = 86_400


def test_zero_or_negative_predicted_return_sizes_no_trade():
    guard = SafetyGuard(max_position_fraction=0.1, daily_loss_limit_fraction=0.05, full_size_return=0.02)
    state = GuardState(equity_usd=10_000.0, daily_loss_usd=0.0, current_day=0, locked_out=False)

    size, _ = guard.size_position(state, predicted_return=0.0, timestamp=0)
    assert size == 0.0

    size, _ = guard.size_position(state, predicted_return=-0.01, timestamp=0)
    assert size == 0.0


def test_nan_predicted_return_sizes_no_trade():
    # Regression test: `predicted_return <= 0` is False for NaN (every
    # comparison with NaN is False), so a naive check would let a NaN
    # prediction through and size a real position on garbage input.
    guard = SafetyGuard(max_position_fraction=0.1, daily_loss_limit_fraction=0.05, full_size_return=0.02)
    state = GuardState(equity_usd=10_000.0, daily_loss_usd=0.0, current_day=0, locked_out=False)

    size, _ = guard.size_position(state, predicted_return=float("nan"), timestamp=0)
    assert size == 0.0


def test_positive_predicted_return_sizes_a_fraction_of_equity():
    guard = SafetyGuard(max_position_fraction=0.1, daily_loss_limit_fraction=0.05, full_size_return=0.02)
    state = GuardState(equity_usd=10_000.0, daily_loss_usd=0.0, current_day=0, locked_out=False)

    size, _ = guard.size_position(state, predicted_return=0.02, timestamp=0)
    assert 0 < size <= 1_000.0  # at most 10% of equity


def test_size_scales_with_predicted_return_magnitude_up_to_a_cap():
    # Spec 3.5: the safety-guard uses the predicted return's magnitude
    # directly for position sizing, not just a directional signal — a
    # small predicted edge should size smaller than a large one, not
    # both get the same max_position_fraction-sized trade.
    guard = SafetyGuard(max_position_fraction=0.1, daily_loss_limit_fraction=0.05, full_size_return=0.02)
    state = GuardState(equity_usd=10_000.0, daily_loss_usd=0.0, current_day=0, locked_out=False)

    small_edge, _ = guard.size_position(state, predicted_return=0.005, timestamp=0)  # 1/4 of full_size_return
    full_edge, _ = guard.size_position(state, predicted_return=0.02, timestamp=0)  # at full_size_return
    beyond_full_edge, _ = guard.size_position(state, predicted_return=0.5, timestamp=0)  # far beyond

    assert small_edge == pytest.approx(250.0)  # 10% * $10,000 * (0.005/0.02)
    assert full_edge == pytest.approx(1_000.0)  # 10% * $10,000, fully sized
    assert beyond_full_edge == pytest.approx(1_000.0)  # capped, never exceeds max_position_fraction


def test_daily_loss_limit_locks_out_trading():
    state = GuardState(equity_usd=10_000.0, daily_loss_usd=0.0, current_day=0, locked_out=False)
    state = record_trade_result(state, pnl_usd=-600.0, daily_loss_limit_fraction=0.05)  # -6% > 5% limit
    assert state.locked_out is True

    guard = SafetyGuard(max_position_fraction=0.1, daily_loss_limit_fraction=0.05, full_size_return=0.02)
    size, _ = guard.size_position(state, predicted_return=0.05, timestamp=100)
    assert size == 0.0


def test_lockout_resets_on_new_day():
    state = GuardState(equity_usd=10_000.0, daily_loss_usd=600.0, current_day=0, locked_out=True)
    guard = SafetyGuard(max_position_fraction=0.1, daily_loss_limit_fraction=0.05, full_size_return=0.02)

    next_day_timestamp = DAY_SECONDS + 10
    size, new_state = guard.size_position(state, predicted_return=0.02, timestamp=next_day_timestamp)

    assert new_state.locked_out is False
    assert new_state.daily_loss_usd == 0.0
    assert new_state.current_day == 1
    assert size > 0.0


def test_lockout_resets_across_multiple_day_boundaries_in_sequence():
    # Regression test: a previous version of this guard never reset and
    # silently froze a long-running backtest partway through. Simulate a
    # sequence of days, each starting locked out from the day before,
    # crossing several boundaries in a row.
    #
    # Losses are a fraction of current equity (not a fixed dollar amount)
    # so the account decays but never goes bankrupt over 90 days — a fixed
    # per-day loss would exhaust a $10,000 account by day ~17, which is a
    # separate, correct "no trade when bankrupt" behavior, not the reset
    # bug this test exists to catch.
    state = GuardState(equity_usd=10_000.0, daily_loss_usd=0.0, current_day=0, locked_out=False)
    guard = SafetyGuard(max_position_fraction=0.1, daily_loss_limit_fraction=0.05, full_size_return=0.02)

    for day in range(1, 91):
        loss = 0.06 * state.equity_usd  # 6% loss, above the 5% lockout threshold
        state = record_trade_result(state, pnl_usd=-loss, daily_loss_limit_fraction=0.05)
        assert state.locked_out is True

        timestamp = day * DAY_SECONDS + 10
        size, state = guard.size_position(state, predicted_return=0.02, timestamp=timestamp)

        assert state.locked_out is False, f"guard stayed locked at day {day}"
        assert size > 0.0, f"guard produced no trade at day {day}"
