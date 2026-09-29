from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from latentedge import config
from latentedge.costs import estimate_gas_cost_usd, estimate_slippage_fraction
from latentedge.uniswap_math import sqrt_price_x96_to_weth_usdc_price

PROGRESS_EVERY = 1000
FEE_FRACTION = config.FEE_TIER_BPS / 10_000
DEFAULT_BARRIER_STDS = 2.0


@dataclass(frozen=True)
class LabelSettings:
    """How bars are labeled: how long a trade may run, and how wide the
    take-profit/stop-loss band is in standard deviations of the pool's
    1-bar return. Recorded with the model, since a backtest must relabel
    with the settings the model was trained under."""

    horizon_seconds: int = config.LABEL_HORIZON_SECONDS
    barrier_stds: float = DEFAULT_BARRIER_STDS


def _returns(entry_price: float, exit_price: float, entry_swap: dict, exit_swap: dict) -> tuple[float, float]:
    """(gross, net) return of one round trip: gross is the price move alone,
    net also pays the pool fees, slippage and gas."""
    notional = config.REFERENCE_NOTIONAL_USD

    raw_return = (exit_price - entry_price) / entry_price
    gross_pnl = notional * raw_return

    fee_cost = 2 * notional * FEE_FRACTION

    entry_slippage = estimate_slippage_fraction(notional, entry_swap["liquidity"], entry_swap["sqrt_price_x96"])
    exit_slippage = estimate_slippage_fraction(notional, exit_swap["liquidity"], exit_swap["sqrt_price_x96"])
    slippage_cost = notional * (entry_slippage + exit_slippage)

    entry_gas = estimate_gas_cost_usd(entry_swap["base_fee_wei"], config.GAS_USED_PER_SWAP, entry_price)
    exit_gas = estimate_gas_cost_usd(exit_swap["base_fee_wei"], config.GAS_USED_PER_SWAP, exit_price)
    gas_cost = entry_gas + exit_gas

    net_pnl = gross_pnl - fee_cost - slippage_cost - gas_cost
    return raw_return, net_pnl / notional


def sort_swaps(swaps: pd.DataFrame) -> pd.DataFrame:
    """The one canonical swap ordering. Swaps in the same block share a
    timestamp, so ties are broken by on-chain position (block, log index)
    when available and otherwise by arrival order (stable sort) — the
    label's stored swap indices and the backtest's replay both index into
    this ordering, so it must be deterministic.
    """
    keys = ["timestamp"] + [c for c in ("block_number", "log_index") if c in swaps.columns]
    return swaps.sort_values(keys, kind="stable").reset_index(drop=True)


def label_bars(
    bars: pd.DataFrame,
    swaps: pd.DataFrame,
    tp_sl_fraction: float,
    on_progress: Callable[[int, int], None] | None = None,
    horizon_seconds: int = config.LABEL_HORIZON_SECONDS,
) -> pd.DataFrame:
    swaps = sort_swaps(swaps)
    timestamps = swaps["timestamp"].to_numpy()
    prices = swaps["sqrt_price_x96"].apply(sqrt_price_x96_to_weth_usdc_price).to_numpy(dtype="float64")
    liquidity = swaps["liquidity"].tolist()
    sqrt_prices = swaps["sqrt_price_x96"].tolist()
    base_fees = swaps["base_fee_wei"].tolist()

    def swap_at(i: int) -> dict:
        return {"liquidity": liquidity[i], "sqrt_price_x96": sqrt_prices[i], "base_fee_wei": base_fees[i]}

    net_returns: list[float] = []
    gross_returns: list[float] = []
    excluded: list[bool] = []
    reasons: list[str | None] = []
    entry_indices: list[int] = []
    exit_indices: list[int] = []

    history_end = timestamps[-1] if len(timestamps) else -1
    bar_starts = bars["bar_start"].tolist()
    total = len(bar_starts)

    for n, t in enumerate(bar_starts):
        if on_progress is not None and n % PROGRESS_EVERY == 0:
            on_progress(n, total)
        horizon_end = t + horizon_seconds

        # The swaps are time-sorted, so each bar's entry and forward
        # window are located by binary search instead of rescanning the
        # whole history per bar.
        entry_idx = int(np.searchsorted(timestamps, t, side="left"))
        if entry_idx >= len(timestamps) or timestamps[entry_idx] > horizon_end:
            net_returns.append(float("nan"))
            gross_returns.append(float("nan"))
            excluded.append(True)
            reasons.append("no_entry_fill")
            entry_indices.append(-1)
            exit_indices.append(-1)
            continue

        entry_price = prices[entry_idx]
        fwd_start = int(np.searchsorted(timestamps, timestamps[entry_idx], side="right"))
        fwd_end = int(np.searchsorted(timestamps, horizon_end, side="right"))

        exit_idx = None
        if fwd_end > fwd_start:
            moves = np.abs((prices[fwd_start:fwd_end] - entry_price) / entry_price)
            hits = np.flatnonzero(moves >= tp_sl_fraction)
            if hits.size:
                exit_idx = fwd_start + int(hits[0])

        # No barrier triggered within available data. If the full horizon
        # hasn't actually been observed yet, a barrier might still trigger
        # beyond what we've ingested — the time-limit exit can't be
        # trusted without that data, so exclude rather than guess.
        if exit_idx is None and history_end < horizon_end:
            net_returns.append(float("nan"))
            gross_returns.append(float("nan"))
            excluded.append(True)
            reasons.append("incomplete_horizon")
            entry_indices.append(-1)
            exit_indices.append(-1)
            continue

        if exit_idx is None:
            exit_idx = fwd_end - 1 if fwd_end > fwd_start else entry_idx

        gross, net = _returns(entry_price, prices[exit_idx], swap_at(entry_idx), swap_at(exit_idx))
        gross_returns.append(gross)
        net_returns.append(net)
        excluded.append(False)
        reasons.append(None)
        entry_indices.append(entry_idx)
        exit_indices.append(exit_idx)

    if on_progress is not None:
        on_progress(total, total)

    result = bars.copy()
    result["net_return"] = net_returns
    # The same round trip before any cost: what a model that only knew the
    # direction of the move could hope to predict.
    result["gross_return"] = gross_returns
    result["excluded"] = excluded
    result["reason"] = reasons
    # Row positions in the time-sorted swaps, so a backtest replays the
    # exact fills the label was computed from. -1 marks an excluded bar.
    result["entry_swap_idx"] = entry_indices
    result["exit_swap_idx"] = exit_indices
    return result
