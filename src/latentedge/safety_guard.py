from pydantic import BaseModel

DAY_SECONDS = 86_400


class GuardState(BaseModel):
    equity_usd: float
    daily_loss_usd: float
    current_day: int
    locked_out: bool


def _day_for_timestamp(timestamp: int) -> int:
    return timestamp // DAY_SECONDS


def _roll_to_day_if_needed(state: GuardState, timestamp: int) -> GuardState:
    day = _day_for_timestamp(timestamp)
    if day == state.current_day:
        return state
    return state.model_copy(update={"current_day": day, "daily_loss_usd": 0.0, "locked_out": False})


class SafetyGuard(BaseModel):
    max_position_fraction: float
    daily_loss_limit_fraction: float

    def size_position(self, state: GuardState, predicted_return: float, timestamp: int) -> tuple[float, GuardState]:
        state = _roll_to_day_if_needed(state, timestamp)

        if state.locked_out or predicted_return <= 0 or state.equity_usd <= 0:
            return 0.0, state

        size_usd = state.equity_usd * self.max_position_fraction
        return size_usd, state


def record_trade_result(state: GuardState, pnl_usd: float, daily_loss_limit_fraction: float) -> GuardState:
    new_equity = state.equity_usd + pnl_usd
    new_daily_loss = state.daily_loss_usd + max(0.0, -pnl_usd)
    limit_usd = state.equity_usd * daily_loss_limit_fraction
    locked_out = new_daily_loss >= limit_usd

    return state.model_copy(update={"equity_usd": new_equity, "daily_loss_usd": new_daily_loss, "locked_out": locked_out})
