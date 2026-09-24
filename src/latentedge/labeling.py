import pandas as pd

from latentedge import config
from latentedge.costs import estimate_gas_cost_usd, estimate_slippage_fraction
from latentedge.uniswap_math import sqrt_price_x96_to_weth_usdc_price

FEE_FRACTION = config.FEE_TIER_BPS / 10_000


def _net_return(entry_price: float, exit_price: float, entry_swap: pd.Series, exit_swap: pd.Series) -> float:
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


def label_bars(bars: pd.DataFrame, swaps: pd.DataFrame, tp_sl_fraction: float) -> pd.DataFrame:
    swaps = swaps.sort_values("timestamp").reset_index(drop=True)
    swaps["price_usdc_per_weth"] = swaps["sqrt_price_x96"].apply(sqrt_price_x96_to_weth_usdc_price)

    net_returns: list[float] = []
    excluded: list[bool] = []
    reasons: list[str | None] = []

    history_end = swaps["timestamp"].max() if not swaps.empty else -1

    for _, bar in bars.iterrows():
        t = bar["bar_start"]
        horizon_end = t + config.LABEL_HORIZON_SECONDS

        entry_candidates = swaps[swaps["timestamp"] >= t]
        if entry_candidates.empty or entry_candidates.iloc[0]["timestamp"] > horizon_end:
            net_returns.append(float("nan"))
            excluded.append(True)
            reasons.append("no_entry_fill")
            continue

        entry_swap = entry_candidates.iloc[0]
        entry_price = entry_swap["price_usdc_per_weth"]

        forward = swaps[(swaps["timestamp"] > entry_swap["timestamp"]) & (swaps["timestamp"] <= horizon_end)]

        exit_swap = None
        for _, candidate in forward.iterrows():
            move = (candidate["price_usdc_per_weth"] - entry_price) / entry_price
            if abs(move) >= tp_sl_fraction:
                exit_swap = candidate
                break

        # No barrier triggered within available data. If the full horizon
        # hasn't actually been observed yet, a barrier might still trigger
        # beyond what we've ingested — the time-limit exit can't be
        # trusted without that data, so exclude rather than guess.
        if exit_swap is None and history_end < horizon_end:
            net_returns.append(float("nan"))
            excluded.append(True)
            reasons.append("incomplete_horizon")
            continue

        if exit_swap is None:
            exit_swap = forward.iloc[-1] if not forward.empty else entry_swap

        exit_price = exit_swap["price_usdc_per_weth"]
        net_returns.append(_net_return(entry_price, exit_price, entry_swap, exit_swap))
        excluded.append(False)
        reasons.append(None)

    result = bars.copy()
    result["net_return"] = net_returns
    result["excluded"] = excluded
    result["reason"] = reasons
    return result
