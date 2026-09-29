from collections.abc import Callable

import numpy as np
import pandas as pd

from latentedge import config
from latentedge.costs import estimate_gas_cost_usd, estimate_slippage_fraction
from latentedge.uniswap_math import sqrt_price_x96_to_weth_usdc_price

PROGRESS_EVERY = 1000
FEE_FRACTION = config.FEE_TIER_BPS / 10_000


def _net_return(entry_price: float, exit_price: float, entry_swap: dict, exit_swap: dict) -> float:
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
    return net_pnl / notional


def label_bars(
    bars: pd.DataFrame,
    swaps: pd.DataFrame,
    tp_sl_fraction: float,
    on_progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    swaps = swaps.sort_values("timestamp").reset_index(drop=True)
    timestamps = swaps["timestamp"].to_numpy()
    prices = swaps["sqrt_price_x96"].apply(sqrt_price_x96_to_weth_usdc_price).to_numpy(dtype="float64")
    liquidity = swaps["liquidity"].tolist()
    sqrt_prices = swaps["sqrt_price_x96"].tolist()
    base_fees = swaps["base_fee_wei"].tolist()

    def swap_at(i: int) -> dict:
        return {"liquidity": liquidity[i], "sqrt_price_x96": sqrt_prices[i], "base_fee_wei": base_fees[i]}

    net_returns: list[float] = []
    excluded: list[bool] = []
    reasons: list[str | None] = []

    history_end = timestamps[-1] if len(timestamps) else -1
    bar_starts = bars["bar_start"].tolist()
    total = len(bar_starts)

    for n, t in enumerate(bar_starts):
        if on_progress is not None and n % PROGRESS_EVERY == 0:
            on_progress(n, total)
        horizon_end = t + config.LABEL_HORIZON_SECONDS

        # The swaps are time-sorted, so each bar's entry and forward
        # window are located by binary search instead of rescanning the
        # whole history per bar.
        entry_idx = int(np.searchsorted(timestamps, t, side="left"))
        if entry_idx >= len(timestamps) or timestamps[entry_idx] > horizon_end:
            net_returns.append(float("nan"))
            excluded.append(True)
            reasons.append("no_entry_fill")
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
            excluded.append(True)
            reasons.append("incomplete_horizon")
            continue

        if exit_idx is None:
            exit_idx = fwd_end - 1 if fwd_end > fwd_start else entry_idx

        net_returns.append(_net_return(entry_price, prices[exit_idx], swap_at(entry_idx), swap_at(exit_idx)))
        excluded.append(False)
        reasons.append(None)

    if on_progress is not None:
        on_progress(total, total)

    result = bars.copy()
    result["net_return"] = net_returns
    result["excluded"] = excluded
    result["reason"] = reasons
    return result
