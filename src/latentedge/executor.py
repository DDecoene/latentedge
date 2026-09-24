from typing import Any

from latentedge import config
from latentedge.costs import estimate_gas_cost_usd, estimate_slippage_fraction

FEE_FRACTION = config.FEE_TIER_BPS / 10_000


def simulate_fill(size_usd: float, entry_price: float, exit_price: float, entry_swap: dict[str, Any], exit_swap: dict[str, Any]) -> float:
    if size_usd <= 0:
        return 0.0

    raw_return = (exit_price - entry_price) / entry_price
    gross_pnl = size_usd * raw_return

    fee_cost = 2 * size_usd * FEE_FRACTION

    entry_slippage = estimate_slippage_fraction(size_usd, entry_swap["liquidity"], entry_swap["sqrt_price_x96"])
    exit_slippage = estimate_slippage_fraction(size_usd, exit_swap["liquidity"], exit_swap["sqrt_price_x96"])
    slippage_cost = size_usd * (entry_slippage + exit_slippage)

    entry_gas = estimate_gas_cost_usd(entry_swap["base_fee_wei"], config.GAS_USED_PER_SWAP, entry_price)
    exit_gas = estimate_gas_cost_usd(exit_swap["base_fee_wei"], config.GAS_USED_PER_SWAP, exit_price)
    gas_cost = entry_gas + exit_gas

    return gross_pnl - fee_cost - slippage_cost - gas_cost
