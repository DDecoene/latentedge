from pydantic import BaseModel

DAY_SECONDS = 86_400


class GuardState(BaseModel):
    equity_usd: float
    daily_loss_usd: float
    current_day: int
    locked_out: bool


def _day_for_timestamp(timestamp: int) -> int:
    return timestamp // DAY_SECONDS


def roll_to_day(state: GuardState, timestamp: int) -> GuardState:
    """Advance the guard's day-tracking to the day `timestamp` falls on,
    resetting the daily-loss tally and any lockout if that's a new day.
    Public so callers settling a trade at its actual exit time (which may
    be a later day than the entry) can attribute the result to the
    correct day before crediting it."""
    day = _day_for_timestamp(timestamp)
    if day == state.current_day:
        return state
    return state.model_copy(update={"current_day": day, "daily_loss_usd": 0.0, "locked_out": False})


class SafetyGuard(BaseModel):
    max_position_fraction: float
    daily_loss_limit_fraction: float
    # The predicted return at (or above) which a trade sizes at the full
    # max_position_fraction. Below this, size scales down proportionally
    # with the prediction's magnitude — per spec 3.5, the guard uses the
    # predicted return's magnitude directly for sizing, not just its sign.
    full_size_return: float

    def size_position(self, state: GuardState, predicted_return: float, timestamp: int) -> tuple[float, GuardState]:
        state = roll_to_day(state, timestamp)

        # `not (predicted_return > 0)` rather than `<= 0` also catches
        # NaN, which fails every comparison (NaN <= 0 is False) — the
        # signal client should already reject NaN before it gets here,
        # but the guard defends itself too rather than relying on that.
        if state.locked_out or not (predicted_return > 0) or state.equity_usd <= 0:
            return 0.0, state

        confidence = min(predicted_return / self.full_size_return, 1.0)
        size_usd = state.equity_usd * self.max_position_fraction * confidence
        return size_usd, state


def record_trade_result(state: GuardState, pnl_usd: float, daily_loss_limit_fraction: float) -> GuardState:
    new_equity = state.equity_usd + pnl_usd
    new_daily_loss = state.daily_loss_usd + max(0.0, -pnl_usd)
    limit_usd = state.equity_usd * daily_loss_limit_fraction
    locked_out = new_daily_loss >= limit_usd

    return state.model_copy(update={"equity_usd": new_equity, "daily_loss_usd": new_daily_loss, "locked_out": locked_out})
